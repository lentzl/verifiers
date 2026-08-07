import base64
import json
import os
import random
import subprocess
import sys

import pytest
from ipython_foundations_v1.document_control import (
    generate_document_control_scenario,
)
from ipython_foundations_v1.document_recovery import PARSER_FIXTURES
from ipython_foundations_v1.file_processing import (
    FILE_PROCESSING_FIXTURES,
    generate_file_processing_scenario,
)
from ipython_foundations_v1.python_recovery_cases import RECOVERY_KINDS
from ipython_foundations_v1.taskset import (
    PDFTOTEXT_COMPAT,
    IpythonFoundationsConfig,
    IpythonFoundationsTaskset,
    _behavior,
    _partial_score,
    _round_prompt,
)

import verifiers.v1 as vf
from verifiers.v1.graph import MessageNode
from verifiers.v1.types import AssistantMessage, ToolCall, ToolMessage, UserMessage


def _trace(calls):
    nodes = []
    parent = None
    for segment, code, output in calls:
        while sum(isinstance(node.message, UserMessage) for node in nodes) <= segment:
            nodes.append(
                MessageNode(
                    parent=parent,
                    message=UserMessage(content=f"request-{segment}"),
                    sampled=False,
                )
            )
            parent = len(nodes) - 1
        call_id = f"call-{len(nodes)}"
        nodes.append(
            MessageNode(
                parent=parent,
                message=AssistantMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=call_id,
                            name="ipython",
                            arguments=json.dumps({"code": code}),
                        )
                    ],
                ),
                sampled=True,
            )
        )
        parent = len(nodes) - 1
        nodes.append(
            MessageNode(
                parent=parent,
                message=ToolMessage(tool_call_id=call_id, content=output),
                sampled=False,
            )
        )
        parent = len(nodes) - 1
    return vf.Trace(
        id="ipython-foundations-test",
        agent=vf.AgentInfo(config=vf.AgentConfig()),
        task=vf.TraceTask(type="IpythonFoundationsTask", data=vf.TaskData(idx=0)),
        nodes=nodes,
    )


def test_taskset_balances_families_and_holds_out_variants():
    train = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(split="train", instances_per_template=1)
    ).load()
    evaluation = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(split="eval", instances_per_template=1)
    ).load()

    assert len(train) == 32
    assert len(evaluation) == 18
    assert {task.data.family for task in train} == {
        "completion",
        "assignment",
        "state",
        "recovery",
        "subprocess",
        "document_recovery",
        "file_processing",
        "document_control",
    }
    assert {task.data.template_variant for task in train} == {0, 1, 2, 3}
    assert {
        task.data.template_variant
        for task in evaluation
        if task.data.family != "recovery"
    } == {4, 5}
    assert {
        task.data.template_variant
        for task in evaluation
        if task.data.family == "recovery"
    } == {4, 5, 6, 7}
    assert all(
        len(task.data.rounds)
        == (
            1
            if task.data.family
            in {"completion", "file_processing", "document_control"}
            else 3
        )
        for task in [*train, *evaluation]
    )


def test_round_limit_isolates_single_request_assignment_rung():
    tasks = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(
            families=("assignment",),
            rounds_per_task=1,
            instances_per_template=1,
        )
    ).load()

    assert len(tasks) == 4
    assert all(len(task.data.rounds) == 1 for task in tasks)


def test_file_backed_tasks_name_their_mounted_inputs():
    tasks = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(instances_per_template=1)
    ).load()

    for family in ("assignment", "state"):
        task = next(task for task in tasks if task.data.family == family)
        first_round = task.data.rounds[0]
        assert set(first_round.files) == {
            f"/workspace/inbox/{'values' if family == 'assignment' else 'records'}.json"
        }
        assert next(iter(first_round.files)) in first_round.instruction


def test_completion_stream_requires_one_result_then_immediate_answer():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(instances_per_template=1)
        ).load()
        if task.data.family == "completion"
    )

    assert len(task.data.rounds) == 1
    assert "one IPython call" in task.data.rounds[0].instruction
    assert "return" in task.data.rounds[0].explicit_operation


def test_state_stream_removes_source_and_requires_later_notebook_reuse():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(instances_per_template=1)
        ).load()
        if task.data.family == "state"
    )

    assert task.data.rounds[0].remove_after == ("/workspace/inbox/records.json",)
    assert not task.data.rounds[1].files
    assert not task.data.rounds[2].files
    assert "retained `records`" in task.data.rounds[2].instruction


def test_subprocess_stream_preserves_path_and_requires_error_directed_repair():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(instances_per_template=1)
        ).load()
        if task.data.family == "subprocess"
    )

    path = task.data.rounds[0].answer
    assert "Title:" not in task.data.rounds[0].files[path]
    assert task.data.rounds[1].answer["returncode"] == 1
    assert "'-text'" in task.data.rounds[1].answer["stderr"]
    assert not task.data.rounds[1].files
    assert not task.data.rounds[2].files
    assert "output path" in task.data.rounds[2].instruction
    assert "raw PDF bytes" in task.data.rounds[2].instruction


def test_document_recovery_mixes_source_forms_and_parser_profiles():
    tasks = [
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(
                families=("document_recovery",), instances_per_template=1
            )
        ).load()
    ]

    assert {task.data.source_kind for task in tasks} == {
        "direct_path",
        "structured_download",
    }
    assert all(len(task.data.rounds) == 3 for task in tasks)
    assert all(len(task.data.rounds[0].files) == 1 for task in tasks)
    assert all(not task.data.rounds[1].files for task in tasks)
    assert all(not task.data.rounds[2].files for task in tasks)
    assert any("PyMuPDF" in task.data.rounds[1].instruction for task in tasks)
    assert any("pdfminer" in task.data.rounds[1].instruction for task in tasks)
    assert all("live traceback" in task.data.rounds[1].instruction for task in tasks)
    assert all("page_text" in task.data.rounds[2].instruction for task in tasks)
    structured = next(
        task for task in tasks if task.data.source_kind == "structured_download"
    )
    mounted_content = next(iter(structured.data.rounds[0].files.values()))
    assert structured.data.rounds[0].answer["bytes"] == len(mounted_content)


def test_document_parser_fixtures_expose_distribution_import_and_public_api(tmp_path):
    for relative_path, content in PARSER_FIXTURES.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = tmp_path / "report.pdf"
    source.write_text("VGl0bGU6IFJlcG9ydAo=")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata as metadata; import fitz; "
                "from pdfminer.high_level import extract_text; "
                "assert metadata.distribution('pymupdf').read_text('top_level.txt').strip() == 'fitz'; "
                "assert fitz.open('report.pdf')[0].get_text() == 'Title: Report\\n'; "
                "assert extract_text('report.pdf', page_numbers=[0]) == 'Title: Report\\n'"
            ),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_file_processing_matrix_covers_types_and_observed_failures():
    tasks = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(
            families=("file_processing",), instances_per_template=5
        )
    ).load()

    assert len(tasks) == 20
    assert {task.data.file_kind for task in tasks} == {
        "text",
        "markdown",
        "csv",
        "json",
        "pdf",
        "docx",
        "unknown",
    }
    assert {
        "missing_download_key",
        "structured_result_is_dict",
        "missing_pdf_stream",
        "missing_pdf_import",
        "page_object_not_text",
        "pdfminer_public_api",
        "pdftotext_missing",
        "wrong_encoding",
        "wrong_parser",
        "malformed_csv",
        "invalid_json",
        "unknown_format",
        "scanned_pdf",
        "password_protected_pdf",
    } <= {task.data.failure_kind for task in tasks}
    assert all(len(task.data.rounds) == 1 for task in tasks)
    assert all("assistant -> ipython" in task.data.demonstration for task in tasks)
    assert all(
        "download" in task.data.rounds[0].instruction
        or task.data.source_kind == "direct_path"
        for task in tasks
    )


def test_file_processing_fixtures_expose_pypdf_and_docx_apis(tmp_path):
    for relative_path, content in FILE_PROCESSING_FIXTURES.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    pdf = tmp_path / "report.pdf"
    pdf.write_text("VGl0bGU6IFJlcG9ydAo=")
    docx = tmp_path / "report.docx"
    docx.write_text("VGl0bGU6IFJlcG9ydAo=")

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from PyPDF2 import PdfReader; from docx import Document; "
                "assert PdfReader('report.pdf').pages[0].extract_text() == 'Title: Report\\n'; "
                "assert Document('report.docx').paragraphs[0].text == 'Title: Report'"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_document_control_fixture_exposes_len_failure_and_all_page_repair(tmp_path):
    for relative_path, content in FILE_PROCESSING_FIXTURES.items():
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    source = tmp_path / "review.pdf"
    source.write_text(
        base64.b64encode(
            b"Title: Review\n\fFinding: Not approved.\n\fContext: Limited.\n"
        ).decode()
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from PyPDF2 import PdfReader; reader = PdfReader('review.pdf'); "
                "assert len(reader.pages) == 3; "
                "assert 'Not approved' in '\\n'.join("
                "page.extract_text() for page in reader.pages); "
                "\ntry: len(reader)\nexcept TypeError: pass\n"
                "else: raise AssertionError('len(reader) unexpectedly succeeded')"
            ),
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_every_file_processing_expert_trajectory_satisfies_process_contract():
    for instance in range(5):
        for variant in range(4):
            rng = random.Random((20260806 * 1_000_003) + (variant * 10_007) + instance)
            scenario = generate_file_processing_scenario(variant, instance, rng)
            trace = _trace(
                [(0, call.code, call.output) for call in scenario.expert_calls]
            )

            behavior = _behavior(
                trace,
                "file_processing",
                "document_path",
                expected_segments=1,
                source_kind=scenario.source_kind,
                file_kind=scenario.file_kind,
                failure_kind=scenario.failure_kind,
                expected_output_marker=scenario.expected_output_marker,
                terminal_status=scenario.terminal_status,
            )

            assert behavior["process_aligned"] == 1.0, scenario.failure_kind
            assert behavior["processing_outcome_observed"] == 1.0


def test_document_control_matrix_pairs_claim_polarity_and_real_failures():
    tasks = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(
            families=("document_control",), instances_per_template=6
        )
    ).load()

    assert len(tasks) == 24
    assert {task.data.failure_kind for task in tasks} == {
        "document_control_len_reader",
        "document_control_pages_api",
        "document_control_page_join",
        "document_control_page_attribute",
    }
    statuses = {task.data.rounds[0].answer["status"] for task in tasks}
    assert statuses == {
        "approved",
        "not_approved",
        "variance_detected",
        "no_variance",
    }
    assert all(
        "exactly one JSON object" in task.data.rounds[0].instruction for task in tasks
    )
    assert all("negation" in task.data.rounds[0].instruction for task in tasks)


def test_every_document_control_expert_trajectory_proves_repair_outcome():
    for instance in range(6):
        for variant in range(4):
            rng = random.Random((20260806 * 1_000_003) + (variant * 10_007) + instance)
            scenario = generate_document_control_scenario(variant, instance, rng)
            trace = _trace(
                [(0, call.code, call.output) for call in scenario.expert_calls]
            )

            behavior = _behavior(
                trace,
                "document_control",
                "full_text",
                expected_segments=1,
                source_kind="structured_download",
                file_kind="pdf",
                failure_kind=scenario.failure_kind,
                expected_output_marker=scenario.expected_output_marker,
            )

            assert behavior["traceback_informed_change"] == 1.0
            assert behavior["repair_outcome_observed"] == 1.0
            assert behavior["full_document_text_extracted"] == 1.0
            assert behavior["process_score"] == 1.0
            assert behavior["process_aligned"] == 1.0


def test_recovery_training_matrix_covers_every_real_error_kind():
    recovery_tasks = [
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(instances_per_template=1)
        ).load()
        if task.data.family == "recovery"
    ]
    rounds = [round_ for task in recovery_tasks for round_ in task.data.rounds]

    assert {round_.recovery_kind for round_ in rounds} == set(RECOVERY_KINDS)
    assert all("real" in round_.instruction.lower() for round_ in rounds)
    assert all(
        "Traceback (most recent call last)" not in round_.instruction
        for round_ in rounds
    )
    assert all(not round_.remove_after for round_ in rounds)
    unavailable = next(
        round_ for round_ in rounds if round_.recovery_kind == "unavailable_dependency"
    )
    assert unavailable.answer["status"] == "unavailable"
    assert "do not retry" in unavailable.instruction

    evaluation = [
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(split="eval", instances_per_template=1)
        ).load()
        if task.data.family == "recovery"
    ]
    evaluation_kinds = {
        round_.recovery_kind for task in evaluation for round_ in task.data.rounds
    }
    assert evaluation_kinds == set(RECOVERY_KINDS)


def test_pdftotext_fixture_exposes_real_failure_and_stdout_repair(tmp_path):
    executable = tmp_path / "pdftotext"
    executable.write_text(PDFTOTEXT_COMPAT)
    executable.chmod(0o755)
    source = tmp_path / "report.pdf"
    source.write_text("VGl0bGU6IFJlcG9ydAo=")

    failed = subprocess.run(
        [executable, "-layout", "-text", source],
        capture_output=True,
        text=True,
        check=False,
    )
    repaired = subprocess.run(
        [executable, "-layout", source, "-"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert failed.returncode == 1
    assert failed.stdout == ""
    assert "'-text'" in failed.stderr
    assert repaired.returncode == 0
    assert repaired.stdout == "Title: Report\n"
    assert repaired.stderr == ""


def test_explicit_scaffolding_describes_operations_without_leaking_answers():
    tasks = IpythonFoundationsTaskset(
        IpythonFoundationsConfig(instruction_level="explicit", instances_per_template=1)
    ).load()

    for task in tasks[:3]:
        prompt = _round_prompt(task, 0, None)
        assert "Foundation exercise:" in prompt
        assert task.data.rounds[0].explicit_operation in prompt
        assert json.dumps(task.data.rounds[0].answer) not in prompt


def test_guided_scaffolding_names_invariant_without_supplying_code():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(
                families=("assignment",),
                instruction_level="guided",
                instances_per_template=1,
                rounds_per_task=1,
            )
        ).load()
    )

    prompt = _round_prompt(task, 0, None)
    assert "Foundation hint:" in prompt
    assert "two separate IPython calls" in prompt
    assert task.data.rounds[0].explicit_operation not in prompt
    assert json.dumps(task.data.rounds[0].answer) not in prompt


def test_self_distillation_demonstration_is_teacher_only():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(
                families=("assignment",),
                instruction_level="standard",
                instances_per_template=1,
                rounds_per_task=1,
            )
        ).load()
    )

    prompt = _round_prompt(task, 0, None)
    assert "assistant -> ipython:" in task.data.demonstration
    assert "<empty output>" in task.data.demonstration
    assert task.data.state_variable in task.data.demonstration
    assert json.dumps(task.data.rounds[0].answer) in task.data.demonstration
    assert task.data.demonstration not in prompt
    assert task.data.rounds[0].explicit_operation not in prompt
    assert json.dumps(task.data.rounds[0].answer) not in prompt


def test_state_self_distillation_demonstrates_cross_request_reuse():
    task = next(
        task
        for task in IpythonFoundationsTaskset(
            IpythonFoundationsConfig(
                families=("state",),
                instruction_level="standard",
                instances_per_template=1,
                rounds_per_task=3,
            )
        ).load()
    )

    assert task.data.demonstration.count("Expert trajectory:") == 3
    assert task.data.demonstration.count("records") >= 3
    assert "Path('/workspace/inbox/records.json')" in task.data.demonstration
    assert "for row in records:" in task.data.demonstration
    assert "max(totals.values())" in task.data.demonstration


@pytest.mark.parametrize(
    ("family", "variable", "calls"),
    [
        (
            "completion",
            "result",
            [(0, "sum([2, 3])", "5")],
        ),
        (
            "assignment",
            "values",
            [
                (0, "values = [2, 3]", ""),
                (0, "sum(values)", "5"),
            ],
        ),
        (
            "state",
            "records",
            [
                (0, "records = [{'amount': 2}]\nlen(records)", "1"),
                (1, "sum(row['amount'] for row in records)", "2"),
            ],
        ),
        (
            "recovery",
            "rows",
            [
                (
                    0,
                    "rows = [{'amount': 2}]\nsum(row['value'] for row in rows)",
                    "Traceback: KeyError: 'value'",
                ),
                (0, "rows[0]", "{'amount': 2}"),
                (0, "sum(row['amount'] for row in rows)", "2"),
            ],
        ),
        (
            "subprocess",
            "pdf_path",
            [
                (
                    0,
                    "from pathlib import Path\npdf_path = '/workspace/inbox/report.pdf'\nPath(pdf_path).exists()",
                    "True",
                ),
                (
                    1,
                    "result = subprocess.run(['pdftotext', '-layout', '-text', pdf_path], capture_output=True, text=True)\n{'returncode': result.returncode, 'stdout': result.stdout, 'stderr': result.stderr}",
                    "{'returncode': 1, 'stdout': '', 'stderr': \"I/O Error: Couldn't open file '-text'\"}",
                ),
                (
                    2,
                    "result = subprocess.run(['pdftotext', '-layout', pdf_path, '-'], capture_output=True, text=True)\n(result.returncode, result.stdout, result.stderr)",
                    "(0, 'Title: Report', '')",
                ),
            ],
        ),
        (
            "document_recovery",
            "document_path",
            [
                (
                    0,
                    "download = {'path': '/workspace/inbox/report.pdf'}\nprint(type(download), sorted(download))\ndocument_path = download['path']\nPath(document_path).exists()",
                    "<class 'dict'> ['path']\\nTrue",
                ),
                (
                    1,
                    "import PyMuPDF\ndocument = PyMuPDF.PdfReader(document_path)",
                    "Traceback: ModuleNotFoundError: No module named 'PyMuPDF'",
                ),
                (
                    1,
                    "import importlib.metadata as metadata\nmetadata.distribution('pymupdf').read_text('top_level.txt')\nimport fitz, inspect\ndir(fitz)\ninspect.signature(fitz.open)\ndocument = fitz.open(document_path)\npage_text = document[0].get_text()\npage_text",
                    "Title: Report\\nSubject: testing\\nFinding: complete.\\n",
                ),
                (
                    2,
                    "lines = page_text.splitlines()\n{'title': lines[0], 'subject': lines[1], 'finding': lines[2]}",
                    "{'title': 'Title: Report', 'subject': 'Subject: testing', 'finding': 'Finding: complete.'}",
                ),
            ],
        ),
    ],
)
def test_process_alignment_recognizes_family_specific_notebook_semantics(
    family, variable, calls
):
    behavior = _behavior(
        _trace(calls),
        family,
        variable,
        source_kind=("structured_download" if family == "document_recovery" else None),
    )

    assert behavior["process_aligned"] == 1.0
    assert behavior["process_score"] == 1.0
    assert behavior["state_reused"] == float(family != "completion")
    assert behavior["identical_consecutive_calls"] == 0.0


def test_subprocess_loop_and_raw_byte_fallback_are_not_rewarded():
    failed = "result = subprocess.run(['pdftotext', '-text', pdf_path], capture_output=True, text=True)\n(result.returncode, result.stdout, result.stderr)"
    trace = _trace(
        [
            (0, "pdf_path = '/workspace/inbox/report.pdf'", ""),
            (1, failed, "(1, '', \"I/O Error: Couldn't open file '-text'\")"),
            (1, failed, "(1, '', \"I/O Error: Couldn't open file '-text'\")"),
            (1, "Path(pdf_path).read_bytes().decode()", "UnicodeDecodeError"),
        ]
    )

    behavior = _behavior(trace, "subprocess", "pdf_path")

    assert behavior["subprocess_result_observed"] == 1.0
    assert behavior["identical_consecutive_calls"] == 1.0
    assert behavior["subprocess_failure_retries"] == 1.0
    assert behavior["raw_pdf_fallback_used"] == 1.0
    assert 0.0 < behavior["process_score"] < 0.5
    assert behavior["process_aligned"] == 0.0


def test_document_recovery_penalizes_repeated_errors_and_reacquisition():
    failed = "import PyMuPDF; PyMuPDF.PdfReader(document_path)"
    error = "Traceback: ModuleNotFoundError: No module named 'PyMuPDF'"
    trace = _trace(
        [
            (
                0,
                "document_path = '/workspace/inbox/report.pdf'\nprint(type(document_path), Path(document_path).exists())",
                "<class 'str'> True",
            ),
            (1, failed, error),
            (1, failed, error),
            (1, "await omnigent_list_files()", "[]"),
            (1, "Path(document_path).read_bytes().decode()", "UnicodeDecodeError"),
        ]
    )

    behavior = _behavior(
        trace,
        "document_recovery",
        "document_path",
        expected_segments=3,
        source_kind="direct_path",
    )

    assert behavior["repeated_error_signatures"] >= 1.0
    assert behavior["document_extra_errors"] >= 1.0
    assert behavior["file_acquisition_calls"] == 1.0
    assert behavior["raw_pdf_fallback_used"] == 1.0
    assert behavior["process_score"] < 0.1
    assert behavior["process_aligned"] == 0.0


def test_file_processing_rewards_inspection_path_reuse_and_text_extraction():
    trace = _trace(
        [
            (
                0,
                "download = {'path': '/workspace/inbox/report.pdf'}\ntype(download), sorted(download)",
                "(<class 'dict'>, ['path'])",
            ),
            (
                0,
                "document_path = download['path']\ndocument_path",
                "'/workspace/inbox/report.pdf'",
            ),
            (
                0,
                "from PyPDF2 import PdfReader\nreader = PdfReader()",
                "Traceback: TypeError: PdfReader.__init__() missing 1 required positional argument: 'stream'",
            ),
            (0, "reader = PdfReader(document_path)\nlen(reader.pages)", "2"),
            (0, "first_page = reader.pages[0]\nfirst_page", "<PyPDF2.Page object>"),
            (
                0,
                "first_page_text = first_page.extract_text()\nfirst_page_text",
                "Title: Report\nFinding: complete.\n",
            ),
        ]
    )

    behavior = _behavior(
        trace,
        "file_processing",
        "document_path",
        expected_segments=1,
        source_kind="structured_download",
        file_kind="pdf",
        failure_kind="missing_pdf_stream",
        expected_output_marker="Title: Report",
    )

    assert behavior["structured_result_inspected"] == 1.0
    assert behavior["download_path_selected"] == 1.0
    assert behavior["path_reused_for_parser"] == 1.0
    assert behavior["traceback_informed_change"] == 1.0
    assert behavior["extracted_output_observed"] == 1.0
    assert behavior["process_score"] == 1.0
    assert behavior["process_aligned"] == 1.0


def test_document_control_accepts_direct_use_of_inspected_download_path():
    trace = _trace(
        [
            (
                0,
                "download = {'path': '/workspace/inbox/report.pdf'}\n"
                "type(download), sorted(download)",
                "(<class 'dict'>, ['path'])",
            ),
            (0, "from PyPDF2 import PdfReader", ""),
            (0, "reader = PdfReader(download['path'])", ""),
            (
                0,
                "page_count = len(reader)",
                "Traceback: TypeError: object of type 'PdfReader' has no len()",
            ),
            (
                0,
                "page_count = len(reader.pages)\n"
                "page_texts = [page.extract_text() for page in reader.pages]\n"
                "full_text = '\\n'.join(page_texts)\n"
                "page_count, full_text",
                "(3, 'Finding: No material variance was detected.')",
            ),
        ]
    )

    behavior = _behavior(
        trace,
        "document_control",
        "full_text",
        expected_segments=1,
        source_kind="structured_download",
        file_kind="pdf",
        failure_kind="document_control_len_reader",
        expected_output_marker="No material variance was detected.",
    )

    assert behavior["download_path_selected"] == 1.0
    assert behavior["path_reused_for_parser"] == 1.0
    assert behavior["process_score"] == 1.0
    assert behavior["process_aligned"] == 1.0


def test_file_processing_accepts_evidenced_terminal_failure_without_retry():
    trace = _trace(
        [
            (
                0,
                "download = {'path': '/workspace/inbox/data.json'}\ntype(download), sorted(download)",
                "(<class 'dict'>, ['path'])",
            ),
            (0, "document_path = download['path']", ""),
            (
                0,
                "import json\nrecords = json.load(open(document_path))",
                "Traceback: JSONDecodeError: Expecting value: line 1 column 1 (char 0)",
            ),
        ]
    )

    behavior = _behavior(
        trace,
        "file_processing",
        "document_path",
        expected_segments=1,
        source_kind="structured_download",
        file_kind="json",
        failure_kind="invalid_json",
        terminal_status="invalid_json",
    )

    assert behavior["terminal_failure_observed"] == 1.0
    assert behavior["file_feedback_handled"] == 1.0
    assert behavior["file_processing_extra_errors"] == 0.0
    assert behavior["process_score"] == 1.0
    assert behavior["process_aligned"] == 1.0


def test_document_control_does_not_reward_change_without_successful_repair():
    trace = _trace(
        [
            (
                0,
                "download = {'path': '/workspace/inbox/report.pdf'}\n"
                "type(download), sorted(download)",
                "(<class 'dict'>, ['path'])",
            ),
            (0, "document_path = download['path']\ndocument_path", "'report.pdf'"),
            (0, "from PyPDF2 import PdfReader", ""),
            (
                0,
                "reader = PdfReader(document_path)\ntype(reader)",
                "<class 'PyPDF2.PdfReader'>",
            ),
            (
                0,
                "page_count = len(reader)\npage_count",
                "Traceback: TypeError: object of type 'PdfReader' has no len()",
            ),
            (0, "page_count = len(reader.pages)\npage_count", "3"),
        ]
    )

    behavior = _behavior(
        trace,
        "document_control",
        "full_text",
        expected_segments=1,
        source_kind="structured_download",
        file_kind="pdf",
        failure_kind="document_control_len_reader",
        expected_output_marker="The safety review did not approve deployment.",
    )

    assert behavior["traceback_informed_change"] == 1.0
    assert behavior["repair_outcome_observed"] == 0.0
    assert behavior["full_document_text_extracted"] == 0.0
    assert behavior["process_aligned"] == 0.0


def test_recovery_process_reward_requires_feedback_repair_in_every_segment():
    trace = _trace(
        [
            (
                0,
                "payload = [1]\npayload[1]",
                "Traceback: IndexError: list index out of range",
            ),
            (0, "payload[0]", "1"),
            (
                1,
                "payload = {'value': 2}\npayload['missing']",
                "Traceback: KeyError: 'missing'",
            ),
        ]
    )

    behavior = _behavior(trace, "recovery", "payload", expected_segments=2)

    assert behavior["recovery_error_segments"] == 2.0
    assert behavior["recovery_repaired_segments"] == 1.0
    assert behavior["recovery_round_coverage"] == pytest.approx(0.5)
    assert behavior["process_score"] == pytest.approx(0.5)
    assert behavior["process_aligned"] == 0.0


def test_completion_process_reward_decays_when_agent_keeps_calling_ipython():
    trace = _trace([(0, "sum([2, 3])", "5")] * 4)

    behavior = _behavior(trace, "completion", "result")

    assert behavior["successful_result_observed"] == 1.0
    assert behavior["ipython_call_efficiency"] == pytest.approx(0.25)
    assert behavior["process_score"] == pytest.approx(1 / 16)
    assert behavior["process_aligned"] == 0.0


def test_identical_empty_assignment_loop_is_not_rewarded():
    trace = _trace(
        [
            (0, "values = [2, 3]", ""),
            (0, "values = [2, 3]", ""),
            (0, "sum(values)", "5"),
        ]
    )

    behavior = _behavior(trace, "assignment", "values")

    assert behavior["silent_assignment_recovered"] == 1.0
    assert behavior["identical_consecutive_calls"] == 1.0
    assert behavior["process_score"] == pytest.approx(1 / 3)
    assert behavior["process_aligned"] == 0.0


def test_reassigning_state_does_not_count_as_reuse():
    trace = _trace(
        [
            (0, "values = [2, 3]", ""),
            (1, "values = [4, 5]\nsum(values)", "9"),
        ]
    )

    behavior = _behavior(trace, "state", "values")

    assert behavior["state_reused"] == 0.0
    assert behavior["cross_turn_state_reused"] == 0.0
    assert behavior["process_score"] == 0.0


def test_subprocess_process_score_requires_positive_milestones():
    trace = _trace([(0, "pdf_path = '/workspace/inbox/report.pdf'", "")])

    behavior = _behavior(trace, "subprocess", "pdf_path")

    assert behavior["raw_pdf_fallback_used"] == 0.0
    assert behavior["subprocess_failure_retries"] == 0.0
    assert behavior["process_score"] == 0.0


@pytest.mark.parametrize(
    ("actual", "expected", "score"),
    [
        ({"a": 1, "b": 2}, {"a": 1, "b": 2}, 1.0),
        ({"a": 1}, {"a": 1, "b": 2}, 0.5),
        (["a"], ["a", "b"], 0.5),
        (7, 7, 1.0),
        (None, 7, 0.0),
    ],
)
def test_partial_score_provides_dense_answer_credit(actual, expected, score):
    assert _partial_score(actual, expected) == pytest.approx(score)
