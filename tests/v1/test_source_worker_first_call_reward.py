import pytest
from source_worker_first_call_v1.reward import (
    CellEvidence,
    is_designated_source_inspector_task,
    score_first_call,
)

AST_PATHS = ("/workspace/sample/alpha.py", "/workspace/sample/beta.py")
CONFIG_PATHS = ("/workspace/sample/service.toml", "/workspace/sample/features.env")


def test_direct_ast_reads_receive_partial_first_call_credit() -> None:
    code = """import ast
from pathlib import Path
left = ast.parse(Path('/workspace/sample/alpha.py').read_text())
right = ast.parse(Path('/workspace/sample/beta.py').read_text())
value = sum(isinstance(node, ast.FunctionDef) for tree in (left, right) for node in ast.walk(tree))
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.correct_file_api == 1
    assert result.score == pytest.approx(0.3)


def test_two_hop_loop_proves_each_path_is_read() -> None:
    code = """import ast
from pathlib import Path
paths = ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')
trees = []
for path in paths:
    trees.append(ast.parse(Path(path).read_text()))
value = sum(isinstance(node, ast.FunctionDef) for tree in trees for node in ast.walk(tree))
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.correct_file_api == 1


def test_unused_path_literals_do_not_count_as_reads() -> None:
    code = """import ast
from pathlib import Path
claimed = ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')
tree = ast.parse(Path('/workspace/sample/other.py').read_text())
value = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.exception_free_first_call == 0
    assert result.correct_file_api == 0


@pytest.mark.parametrize(
    "toml_expression",
    (
        "tomllib.loads(Path('/workspace/sample/service.toml').read_text())",
        "tomllib.load(open('/workspace/sample/service.toml', 'rb'))",
    ),
)
def test_config_requires_tomllib_env_split_true_and_nonconstant_compute(
    toml_expression: str,
) -> None:
    code = f"""import tomllib
from pathlib import Path
config = {toml_expression}
features = dict(line.split('=', 1) for line in Path('/workspace/sample/features.env').read_text().splitlines())
value = config['runtime']['workers'] * config['runtime']['timeout_seconds'] + sum(item == 'true' for item in features.values())
"""
    result = score_first_call(
        family="specialist_source_config",
        required_paths=CONFIG_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.correct_file_api == 1
    assert result.score == pytest.approx(0.3)


@pytest.mark.parametrize("loader", ("json.loads", "yaml.safe_load"))
def test_config_rejects_json_and_yaml_substitutes(loader: str) -> None:
    module = loader.split(".")[0]
    code = f"""import {module}
from pathlib import Path
config = {loader}(Path('/workspace/sample/service.toml').read_text())
features = dict(line.split('=', 1) for line in Path('/workspace/sample/features.env').read_text().splitlines())
value = len(config) + sum(item == 'true' for item in features.values())
"""
    result = score_first_call(
        family="specialist_source_config",
        required_paths=CONFIG_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.exception_free_first_call == 0
    assert result.correct_file_api == 0


@pytest.mark.parametrize(
    "computation",
    (
        "value = len(trees)",
        "value = sum(1 for tree in trees for node in ast.walk(tree))",
    ),
)
def test_ast_requires_walk_and_structural_target(computation: str) -> None:
    code = f"""import ast
from pathlib import Path
trees = [ast.parse(Path(path).read_text()) for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
{computation}
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.exception_free_first_call == 0
    assert result.correct_file_api == 0


@pytest.mark.parametrize(
    "code",
    (
        "",
        "rlm.list_subagents()",
        "await agent_observe('child')",
        "print('conversation log')",
        "goal = 'inspect source eventually'",
    ),
)
def test_silence_and_control_plane_inspection_earn_no_positive_credit(
    code: str,
) -> None:
    cells = () if not code else (CellEvidence(code, "ok"),)
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=cells,
        delivered_bodies=(),
    )
    assert result.exception_free_first_call == 0
    assert result.correct_file_api == 0
    assert result.score == pytest.approx(0.0)


def test_reads_without_family_computation_earn_no_positive_credit() -> None:
    code = """from pathlib import Path
texts = [Path(path).read_text() for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
"""
    result = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    assert result.exception_free_first_call == 0
    assert result.score == pytest.approx(0.0)


def test_partial_correct_source_work_outranks_silence() -> None:
    code = """import ast
from pathlib import Path
trees = [ast.parse(Path(path).read_text()) for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
value = sum(isinstance(node, ast.FunctionDef) for tree in trees for node in ast.walk(tree))
"""
    partial = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(code, "ok"),),
        delivered_bodies=(),
    )
    silence = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(),
        delivered_bodies=(),
    )
    assert partial.exception_free_first_call == 1
    assert partial.correct_file_api == 1
    assert partial.score == pytest.approx(0.3)
    assert partial.score > silence.score


def test_later_exact_retry_cannot_outrank_correct_first_call() -> None:
    correct = """import ast
from pathlib import Path
trees = [ast.parse(Path(path).read_text()) for path in ('/workspace/sample/alpha.py', '/workspace/sample/beta.py')]
value = sum(isinstance(node, ast.FunctionDef) for tree in trees for node in ast.walk(tree))
"""
    baseline = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(CellEvidence(correct, "ok"),),
        delivered_bodies=(),
    )
    recovered = score_first_call(
        family="specialist_source_ast",
        required_paths=AST_PATHS,
        expected_value=30,
        cells=(
            CellEvidence("value = 29", "29"),
            CellEvidence(
                "await agent_message.send('{\"value\":30}', receiver_role='parent')",
                "Agent message sent: agentmsg_456",
            ),
        ),
        delivered_bodies=('{"value":30}',),
    )
    assert baseline.score == pytest.approx(0.3)
    assert recovered.exact_oracle_value == 1
    assert recovered.atomic_compact_parent_send == 0
    assert recovered.score < baseline.score


def test_only_typed_source_inspector_assignment_is_designated() -> None:
    typed = """[task from parent]
[selected terminal capability]
expert_id=source_inspector
session_role=terminal_worker
Read both files.
"""
    assert is_designated_source_inspector_task(typed)
    assert not is_designated_source_inspector_task("[task from parent]\nInspect files")
    assert not is_designated_source_inspector_task(
        typed.replace("expert_id=source_inspector", "expert_id=table_analyst")
    )
