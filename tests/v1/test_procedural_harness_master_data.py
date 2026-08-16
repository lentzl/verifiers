import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = (
    Path(__file__).parents[2]
    / "datasets"
    / "procedural_harness_master_v1"
    / "generate.py"
)
SPEC = importlib.util.spec_from_file_location("procedural_harness_master_v1_generator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_generation_is_deterministic() -> None:
    assert MODULE.generate_episode("train_gen", 17) == MODULE.generate_episode(
        "train_gen", 17
    )


def test_split_surfaces_are_separated() -> None:
    train = [MODULE.generate_episode("train_gen", i) for i in range(96)]
    valid = [MODULE.generate_episode("valid_gen", i) for i in range(96)]
    ood = [MODULE.generate_episode("ood_gen", i) for i in range(96)]
    train_resources = {
        family
        for row in train + valid
        for family in row["metadata"]["resource_families"]
        if family != "verification_manifest"
    }
    ood_resources = {
        family
        for row in ood
        for family in row["metadata"]["resource_families"]
        if family != "verification_manifest"
    }
    assert train_resources <= set(MODULE.TRAIN_RESOURCES)
    assert ood_resources <= set(MODULE.OOD_RESOURCES)
    assert train_resources.isdisjoint(ood_resources)
    assert {
        row["metadata"]["instruction_style"] for row in train
    }.isdisjoint({row["metadata"]["instruction_style"] for row in valid})
    assert {
        row["metadata"]["instruction_style"] for row in train + valid
    }.isdisjoint({row["metadata"]["instruction_style"] for row in ood})


def test_every_row_has_hidden_conjunctive_contract() -> None:
    for split in ("train_gen", "valid_gen", "ood_gen"):
        for index in range(64):
            row = MODULE.generate_episode(split, index)
            MODULE.validate_row(row)
            public = json.dumps(row["public"], sort_keys=True)
            assert "reasoning_content" not in public
            assert "final_answer" not in public
            assert "trajectory_contract" not in public
            assert all(row["oracle"]["trajectory_contract"]["hard_gate"].values())


def test_ood_contains_direct_controls_and_new_compositions() -> None:
    rows = [MODULE.generate_episode("ood_gen", i) for i in range(80)]
    families = {row["metadata"]["episode_family"] for row in rows}
    assert {"direct", "triple", "reclaim"} <= families


def test_reclaim_forbids_parent_access_before_failure() -> None:
    row = next(
        row
        for i in range(80)
        if (row := MODULE.generate_episode("ood_gen", i))["metadata"][
            "episode_family"
        ]
        == "reclaim"
    )
    contract = row["oracle"]["trajectory_contract"]
    assert row["oracle"]["fault_plan"]["type"] == "inject_child_failure"
    assert "coordinator_read_child_owned_before_reclaim" in contract["forbidden_atoms"]
    assert {
        "before": "explicit_reclaim",
        "after": "coordinator_read_after_reclaim",
    } in contract["ordering"]


def test_train_windows_change_but_frozen_eval_does_not(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    MODULE.materialize(first, 24, 24, 24, 0, 20260816)
    MODULE.materialize(second, 24, 24, 24, 24, 20260816)
    a = json.loads((first / "manifest.json").read_text())
    b = json.loads((second / "manifest.json").read_text())
    assert a["splits"]["valid_gen"]["sha256"] == b["splits"]["valid_gen"]["sha256"]
    assert a["splits"]["ood_gen"]["sha256"] == b["splits"]["ood_gen"]["sha256"]
    assert a["splits"]["train_gen"]["sha256"] != b["splits"]["train_gen"]["sha256"]
