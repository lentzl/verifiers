"""Validate and materialize model-authored portable Agent Skills."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import verifiers.v1 as vf

PACKAGE_PATTERN = re.compile(
    r"<skill_package>\s*(.*?)\s*</skill_package>", re.DOTALL | re.IGNORECASE
)
NAME_PATTERN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
PORTABLE_SKILLS_DIR = "/task/agent-skills"
MAX_SKILL_MD_CHARS = 8_000
MAX_FILE_CHARS = 12_000
MAX_TOTAL_FILE_CHARS = 24_000
MAX_FILES = 8
ALLOWED_ROOTS = {"assets", "references", "scripts"}
FORBIDDEN_IMPORTS = {"http", "requests", "socket", "subprocess", "urllib"}
FORBIDDEN_CALLS = {"__import__", "compile", "eval", "exec"}
FORBIDDEN_ATTRIBUTE_CALLS = {("os", "popen"), ("os", "system")}


class SkillPackageError(ValueError):
    """The authored package cannot safely satisfy the portable skill contract."""


@dataclass(frozen=True)
class SkillCandidate:
    name: str
    skill_md: str
    files: dict[str, str]


def _require_text(payload: dict[str, Any], key: str, limit: int) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SkillPackageError(f"{key!r} must be a non-empty string")
    if len(value) > limit:
        raise SkillPackageError(f"{key!r} exceeds the {limit}-character limit")
    return value.strip() + "\n"


def _frontmatter(skill_md: str) -> dict[str, str]:
    lines = skill_md.splitlines()
    if len(lines) < 4 or lines[0].strip() != "---":
        raise SkillPackageError("SKILL.md must begin with YAML frontmatter")
    try:
        end = next(i for i, line in enumerate(lines[1:], 1) if line.strip() == "---")
    except StopIteration as exc:
        raise SkillPackageError("SKILL.md frontmatter has no closing '---'") from exc
    values: dict[str, str] = {}
    for line in lines[1:end]:
        if ":" not in line:
            raise SkillPackageError(f"invalid SKILL.md frontmatter line: {line!r}")
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("\"'")
    unknown = set(values) - {"description", "name"}
    if unknown:
        raise SkillPackageError(
            f"SKILL.md uses non-portable frontmatter fields: {sorted(unknown)}"
        )
    if not "\n".join(lines[end + 1 :]).strip():
        raise SkillPackageError(
            "SKILL.md must include usage guidance after frontmatter"
        )
    return values


def _validate_script(path: str, source: str) -> None:
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        raise SkillPackageError(f"{path} is invalid Python: {exc.msg}") from exc

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = (
                [alias.name for alias in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
            )
            roots = {module.split(".", 1)[0] for module in modules}
            forbidden = roots & FORBIDDEN_IMPORTS
            if forbidden:
                raise SkillPackageError(
                    f"{path} imports forbidden module {sorted(forbidden)[0]!r}"
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in FORBIDDEN_CALLS
        ):
            raise SkillPackageError(f"{path} calls forbidden {node.func.id}()")
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id, node.func.attr) in FORBIDDEN_ATTRIBUTE_CALLS
        ):
            raise SkillPackageError(
                f"{path} calls forbidden {node.func.value.id}.{node.func.attr}()"
            )


def _validate_files(payload: dict[str, Any]) -> dict[str, str]:
    value = payload.get("files")
    if not isinstance(value, dict) or not value:
        raise SkillPackageError("'files' must be a non-empty path-to-text object")
    if len(value) > MAX_FILES:
        raise SkillPackageError(f"'files' exceeds the {MAX_FILES}-file limit")

    files: dict[str, str] = {}
    total_chars = 0
    for raw_path, raw_source in value.items():
        if not isinstance(raw_path, str) or not isinstance(raw_source, str):
            raise SkillPackageError("every file path and content must be a string")
        path = PurePosixPath(raw_path)
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 2
            or path.parts[0] not in ALLOWED_ROOTS
        ):
            raise SkillPackageError(
                f"{raw_path!r} must be a relative path under "
                "scripts/, references/, or assets/"
            )
        normalized = str(path)
        if normalized in files:
            raise SkillPackageError(f"duplicate file path {normalized!r}")
        if len(raw_source) > MAX_FILE_CHARS:
            raise SkillPackageError(
                f"{normalized!r} exceeds the {MAX_FILE_CHARS}-character limit"
            )
        total_chars += len(raw_source)
        files[normalized] = raw_source.rstrip() + "\n"

    if total_chars > MAX_TOTAL_FILE_CHARS:
        raise SkillPackageError(
            f"'files' exceeds the {MAX_TOTAL_FILE_CHARS}-character total limit"
        )
    scripts = [path for path in files if path.startswith("scripts/")]
    if not scripts:
        raise SkillPackageError("the skill needs at least one file under scripts/")
    non_python = [path for path in scripts if not path.endswith(".py")]
    if non_python:
        raise SkillPackageError(
            f"this curriculum rung accepts Python scripts only: {non_python[0]!r}"
        )
    for path in scripts:
        _validate_script(path, files[path])
    return files


def parse_candidate(reply: str) -> SkillCandidate:
    match = PACKAGE_PATTERN.search(reply)
    if match is None:
        raise SkillPackageError("reply must contain one <skill_package> JSON object")
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise SkillPackageError(f"skill package is invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SkillPackageError("skill package JSON must be an object")
    unknown = set(payload) - {"files", "name", "skill_md"}
    if unknown:
        raise SkillPackageError(f"unknown skill package fields: {sorted(unknown)}")

    name = payload.get("name")
    if (
        not isinstance(name, str)
        or len(name) > 64
        or NAME_PATTERN.fullmatch(name) is None
    ):
        raise SkillPackageError(
            "name must be at most 64 lowercase letters, digits, or single hyphens"
        )
    skill_md = _require_text(payload, "skill_md", MAX_SKILL_MD_CHARS)
    metadata = _frontmatter(skill_md)
    if metadata.get("name") != name:
        raise SkillPackageError("SKILL.md frontmatter name must match the package name")
    description = metadata.get("description", "")
    if not description or len(description) > 1024:
        raise SkillPackageError(
            "SKILL.md frontmatter description must contain 1-1024 characters"
        )
    files = _validate_files(payload)
    if not any(path in skill_md for path in files if path.startswith("scripts/")):
        raise SkillPackageError("SKILL.md must explain how to run a bundled script")
    return SkillCandidate(name=name, skill_md=skill_md, files=files)


def render_candidate(candidate: SkillCandidate) -> dict[str, bytes]:
    return {
        "SKILL.md": candidate.skill_md.encode(),
        **{path: source.encode() for path, source in candidate.files.items()},
    }


async def install_candidate(runtime: vf.Runtime, candidate: SkillCandidate) -> str:
    root = f"{PORTABLE_SKILLS_DIR}/{candidate.name}"
    for relative_path, content in render_candidate(candidate).items():
        await runtime.write(f"{root}/{relative_path}", content)
    return root
