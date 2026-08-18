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
SPEC = importlib.util.spec_from_file_location(
    "procedural_harness_master_v1_generator", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_generation_is_deterministic() -> None:
    assert MODULE.generate_episode("train_gen", 17) == MODULE.generate_episode(
        "train_gen", 17
    )


def test_atomic_curriculum_generation_is_deterministic_and_oracle_hidden() -> None:
    for rung in (
        "atomic_state",
        "atomic_send",
        "atomic_child_request",
        "atomic_followup",
        "atomic_parallel",
    ):
        for split in ("train_gen", "valid_gen", "ood_gen"):
            row = MODULE.generate_curriculum_episode(rung, split, 17)
            assert row == MODULE.generate_curriculum_episode(rung, split, 17)
            MODULE.validate_row(row)
            assert row["generator_version"] == MODULE.CURRICULUM_VERSION
            assert row["metadata"]["curriculum_rung"] == rung
            assert row["metadata"]["episode_family"] == rung
            assert row["public"]["workspace_files"] == {}
            assert row["oracle"]["resource_ownership"] == {}
            assert all(
                child["resource_path"] is None for child in row["oracle"]["children"]
            )
            assert "reasoning_content" not in json.dumps(row)
            assert "trajectory_contract" not in json.dumps(row["public"])
            assert "final_answer" not in json.dumps(row["public"])


def test_atomic_state_public_answer_contract_matches_oracle() -> None:
    row = MODULE.generate_curriculum_episode("atomic_state", "train_gen", 17)
    prompt = row["public"]["user_prompt"]
    answer = row["oracle"]["final_answer"]

    assert "marker must be the original retained value" in prompt
    assert "result must be the printed sum" in prompt
    assert answer["marker"] == next(iter(row["oracle"]["coordinator_state"].values()))


def test_atomic_child_request_exposes_only_the_observable_prefix_contract() -> None:
    row = MODULE.generate_curriculum_episode("atomic_child_request", "train_gen", 17)
    prompt = row["public"]["user_prompt"]
    contract = row["oracle"]["trajectory_contract"]

    assert "initial rlm prompt" in prompt
    assert "need multiplier" in prompt
    assert "send_followup" not in " ".join(contract["required_atoms"])
    assert contract["cardinality"]["parent_to_child_message"] == 0
    assert contract["cardinality"]["child_to_parent_message"] == 1


def test_natural_curriculum_is_deterministic_hidden_and_non_prescriptive() -> None:
    forbidden = set(MODULE.NATURAL_USER_PROMPT_FORBIDDEN)
    for rung in ("natural_n1", "natural_n2"):
        for split in ("train_gen", "valid_gen", "ood_gen"):
            for index in range(24):
                row = MODULE.generate_curriculum_episode(rung, split, index)
                assert row == MODULE.generate_curriculum_episode(rung, split, index)
                MODULE.validate_row(row)
                prompt = row["public"]["user_prompt"].lower()
                assert not {term for term in forbidden if term in prompt}
                assert row["public"]["workspace_files"]
                assert row["oracle"]["resource_ownership"]
                assert row["metadata"]["natural_stage"] == (
                    "N1" if rung == "natural_n1" else "N2"
                )
                assert row["metadata"]["semantic_family"]
                assert row["metadata"]["graph_variant"]
                assert "trajectory_contract" not in json.dumps(row["public"])
                assert "reasoning_content" not in json.dumps(row)


def test_natural_curriculum_holds_out_semantic_families() -> None:
    by_split = {
        split: {
            MODULE.generate_curriculum_episode("natural_n2", split, index)["metadata"][
                "semantic_family"
            ]
            for index in range(96)
        }
        for split in ("train_gen", "valid_gen", "ood_gen")
    }

    assert len(by_split["train_gen"]) == len(MODULE.NATURAL_SCENARIOS["train_gen"])
    assert len(by_split["valid_gen"]) == len(MODULE.NATURAL_SCENARIOS["valid_gen"])
    assert len(by_split["ood_gen"]) == len(MODULE.NATURAL_SCENARIOS["ood_gen"])
    assert by_split["train_gen"].isdisjoint(by_split["valid_gen"])
    assert by_split["train_gen"].isdisjoint(by_split["ood_gen"])
    assert by_split["valid_gen"].isdisjoint(by_split["ood_gen"])


def test_natural_train_window_covers_every_semantic_family_and_wording_axis() -> None:
    for rung in ("natural_n1", "natural_n2"):
        rows = [
            MODULE.generate_curriculum_episode(rung, "train_gen", index)
            for index in range(len(MODULE.NATURAL_SCENARIOS["train_gen"]))
        ]

        assert {row["metadata"]["semantic_family"] for row in rows} == {
            scenario.key for scenario in MODULE.NATURAL_SCENARIOS["train_gen"]
        }
        assert len({row["metadata"]["instruction_style"] for row in rows}) >= 2


def test_natural_n1_varies_the_composition_graph() -> None:
    variants = {
        MODULE.generate_curriculum_episode("natural_n1", "train_gen", index)[
            "metadata"
        ]["graph_variant"]
        for index in range(16)
    }

    assert variants == {
        "child_plus_private_state",
        "child_plus_local_work_and_private_state",
    }


def test_natural_n2_holds_out_a_composition_graph_for_ood() -> None:
    train_variants = {
        MODULE.generate_curriculum_episode("natural_n2", "train_gen", index)[
            "metadata"
        ]["graph_variant"]
        for index in range(24)
    }
    valid_variants = {
        MODULE.generate_curriculum_episode("natural_n2", "valid_gen", index)[
            "metadata"
        ]["graph_variant"]
        for index in range(24)
    }
    ood_variants = {
        MODULE.generate_curriculum_episode("natural_n2", "ood_gen", index)["metadata"][
            "graph_variant"
        ]
        for index in range(24)
    }

    assert train_variants == valid_variants == {"staged_private_parameter_cycle"}
    assert ood_variants == {
        "staged_private_parameter_cycle",
        "staged_private_parameter_cycle_with_local_work",
    }
    assert "staged_private_parameter_cycle_with_local_work" not in train_variants


def test_default_generation_does_not_select_curriculum() -> None:
    row = MODULE.generate_episode("train_gen", 17)
    assert "curriculum_rung" not in row["metadata"]
    assert row["metadata"]["episode_family"] in MODULE.TRAIN_FAMILIES


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
    assert {row["metadata"]["instruction_style"] for row in train}.isdisjoint(
        {row["metadata"]["instruction_style"] for row in valid}
    )
    assert {row["metadata"]["instruction_style"] for row in train + valid}.isdisjoint(
        {row["metadata"]["instruction_style"] for row in ood}
    )


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
        if (row := MODULE.generate_episode("ood_gen", i))["metadata"]["episode_family"]
        == "reclaim"
    )
    contract = row["oracle"]["trajectory_contract"]
    assert row["oracle"]["fault_plan"]["type"] == "inject_child_failure"
    assert "coordinator_read_child_owned_before_reclaim" in contract["forbidden_atoms"]
    assert {
        "before": "explicit_reclaim",
        "after": "coordinator_read_after_reclaim",
    } in contract["ordering"]


def test_verify_prompt_distinguishes_digest_evidence_from_final_result() -> None:
    row = MODULE.generate_episode("train_gen", 5)

    assert row["metadata"]["episode_family"] == "verify"
    assert "digest is verification evidence only" in row["public"]["user_prompt"]
    assert "do not put the digest in the final JSON" in row["public"]["user_prompt"]
    assert (
        row["oracle"]["final_answer"]["child"]
        == row["oracle"]["final_answer"]["result"]
    )
    assert row["oracle"]["final_answer"]["verified"] is True


def test_train_windows_change_but_frozen_eval_does_not(tmp_path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    MODULE.materialize(first, 24, 24, 24, 0, 20260816)
    MODULE.materialize(second, 24, 24, 24, 24, 20260816)
    a = json.loads((first / "manifest.json").read_text())
    b = json.loads((second / "manifest.json").read_text())
    assert a["splits"]["valid_gen"]["sha256"] == b["splits"]["valid_gen"]["sha256"]
    assert a["splits"]["ood_gen"]["sha256"] == b["splits"]["ood_gen"]["sha256"]
    assert a["splits"]["train_gen"]["sha256"] != b["splits"]["train_gen"]["sha256"]
