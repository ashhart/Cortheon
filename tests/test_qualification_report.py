"""Report assembly: content-free runs, minimal reproducers, promotion gates."""

import json

from qualification_support import (
    _cell,
    _condition_entries,
    _historical_condition_entries,
    _result,
    _write_manifest,
)

from cortheon.qualification_core.conditions import (
    AVAILABLE_CONDITIONS,
    CONTRASTS,
    HISTORICAL_CONDITIONS,
)
from cortheon.qualification_factory import (
    CellRun,
    _independent_pairing,
    _public_run,
    _reproducers,
    load_manifest,
    run_qualification,
)


def _complete_run(cell, case_ids, results, repeats):
    contrasts = {}
    deltas = {}
    invalid = {}
    for name, comparison in CONTRASTS.items():
        if comparison not in cell.condition_ids:
            continue
        selected = [row for row in results if row.condition in {"full", comparison}]
        paired, case_deltas, invalid_ids = _independent_pairing(
            selected,
            treatment="full",
            comparison=comparison,
            repeats=repeats,
            seed=7,
        )
        contrasts[name] = paired
        deltas[name] = case_deltas
        invalid[name] = invalid_ids
    primary = contrasts["full_vs_bare"]
    return CellRun(
        cell=cell,
        case_ids=tuple(case_ids),
        task_digests={case_id: f"task-{case_id}" for case_id in case_ids},
        results=results,
        pairing=primary,
        case_deltas=deltas["full_vs_bare"],
        invalid_case_ids=invalid["full_vs_bare"],
        repository_unchanged=True,
        environment_stable=True,
        runtime={
            "ok": True,
            "storage": "memory_only",
            "source_fingerprint": "runtime-fingerprint",
            "protocol_version": "1.0.0",
        },
        inference={"ok": True, "model_id": "small-model"},
        host_version="1",
        evaluator_runtime_source_fingerprint="runtime-fingerprint",
        evaluator_runtime_protocol="1.0.0",
        contrasts=contrasts,
        contrast_case_deltas=deltas,
        contrast_invalid_case_ids=invalid,
        scheduled_repeats=repeats,
    )


def test_public_runs_never_retain_answers_or_error_text():
    run = _result(
        "a",
        0,
        "full",
        False,
        process_error="/private/repository/secret.py failed",
        final_text="the sensitive final answer",
    )

    encoded = json.dumps(_public_run(run))

    assert "sensitive final answer" not in encoded
    assert "private expected answer" not in encoded
    assert "/private/repository" not in encoded
    assert '"process_error": true' in encoded


def test_failure_reproducer_is_one_cell_case_repeat_and_content_free(
    monkeypatch,
    tmp_path,
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    manifests = workspace / "benchmarks"
    manifests.mkdir()
    manifest_path = _write_manifest(manifests)
    monkeypatch.chdir(workspace)
    manifest = load_manifest(manifest_path)
    failure = _result(
        "sealed-case",
        2,
        "full",
        False,
        final_text="sensitive answer",
    )
    run = CellRun(
        cell=_cell(repeats=3),
        case_ids=("sealed-case",),
        task_digests={"sealed-case": "digest"},
        results=[failure],
        pairing={},
        case_deltas={},
        invalid_case_ids=set(),
        repository_unchanged=True,
        environment_stable=True,
        runtime={"storage": "memory_only"},
        inference={"ok": True},
        host_version="1",
    )

    reproducers = _reproducers(manifest, [run])
    encoded = json.dumps(reproducers)

    assert len(reproducers) == 1
    assert reproducers[0]["minimal_scope"]["cases"] == 1
    assert "-- benchmarks/qualification.json" in reproducers[0]["command"]
    assert "--cell local-semantic" in reproducers[0]["command"]
    assert "--case-id sealed-case" in reproducers[0]["command"]
    assert "--repeat 2" in reproducers[0]["command"]
    assert "sensitive answer" not in encoded


def test_full_factory_refuses_causal_claim_without_old_planner(monkeypatch, tmp_path):
    manifest = load_manifest(_write_manifest(tmp_path))
    results = []
    for case_id in ("a", "b"):
        results.extend(
            _result(
                case_id,
                0,
                condition,
                condition == "full",
                telemetry=condition != "bare",
            )
            for condition in AVAILABLE_CONDITIONS
        )
    run = _complete_run(manifest.cells[0], ("a", "b"), results, (0,))
    monkeypatch.setattr(
        "cortheon.qualification_factory._run_cell",
        lambda *_args, **_kwargs: run,
    )
    monkeypatch.setattr(
        "cortheon.qualification_factory._repository_fingerprint",
        lambda _repository: "fingerprint",
    )
    monkeypatch.setattr(
        "cortheon.qualification_factory._git_revision",
        lambda _repository: "abc123",
    )

    report = run_qualification(manifest, progress=False)
    partial = run_qualification(
        manifest,
        cell_filter="local-semantic",
        case_filter="a",
        progress=False,
    )

    assert not report["promoted"]
    assert not report["causal_lift_claimed"]
    assert not report["promotion_gates"]["old_planner_available"]
    assert report["aggregate"]["contrasts"]["full_vs_bare"]["accounting"]["independent_cases"] == 2
    assert not partial["promoted"]
    assert not partial["promotion_gates"]["full_manifest_executed"]
    encoded = json.dumps(report)
    assert "private model output" not in encoded
    assert "private expected answer" not in encoded


def test_historical_contrast_uses_only_designated_opencode_cells(monkeypatch, tmp_path):
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            cells=[
                {
                    "id": "flagship",
                    "suite": "semantic",
                    "host": "opencode",
                    "provider": "Local",
                    "model_id": "small-model",
                    "cases": 2,
                    "repeats": 1,
                    "historical_comparison": True,
                    "conditions": _historical_condition_entries(),
                },
                {
                    "id": "cross-host",
                    "suite": "semantic",
                    "host": "pi",
                    "provider": "Local",
                    "model_id": "small-model",
                    "cases": 2,
                    "repeats": 1,
                    "conditions": _condition_entries(),
                },
            ],
        )
    )
    runs = {}
    for cell in manifest.cells:
        results = [
            _result(
                case_id,
                0,
                condition,
                condition == "full",
                telemetry=condition != "bare",
            )
            for case_id in ("a", "b")
            for condition in cell.condition_ids
        ]
        runs[cell.cell_id] = _complete_run(cell, ("a", "b"), results, (0,))
    monkeypatch.setattr(
        "cortheon.qualification_factory._run_cell",
        lambda _manifest, cell, **_kwargs: runs[cell.cell_id],
    )
    monkeypatch.setattr(
        "cortheon.qualification_factory._repository_fingerprint",
        lambda _repository: "fingerprint",
    )
    monkeypatch.setattr(
        "cortheon.qualification_factory._git_revision",
        lambda _repository: "abc123",
    )

    report = run_qualification(manifest, progress=False)

    historical = report["aggregate"]["contrasts"]["full_vs_old_planner"]
    assert historical["available"] is True
    assert historical["accounting"]["cell_case_exposures"] == 2
    assert report["promotion_gates"]["historical_comparison_preregistered"] is True
    assert report["promotion_gates"]["old_planner_available"] is True
    cross_host = next(item for item in report["cells"] if item["configuration"]["host"] == "pi")
    assert cross_host["gates"]["old_planner_available"] is True
    assert "old_planner" not in {item["id"] for item in cross_host["configuration"]["conditions"]}
    assert tuple(runs["flagship"].cell.condition_ids) == HISTORICAL_CONDITIONS


def test_one_case_repeated_six_times_cannot_reach_the_promotion_floor(
    monkeypatch,
    tmp_path,
):
    manifest = load_manifest(
        _write_manifest(
            tmp_path,
            cells=[
                {
                    "id": "local-semantic",
                    "suite": "semantic",
                    "host": "opencode",
                    "provider": "Local",
                    "model_id": "small-model",
                    "cases": 2,
                    "repeats": 6,
                    "conditions": _condition_entries(),
                }
            ],
        )
    )
    results = [
        _result(
            "one",
            repeat,
            condition,
            condition == "full",
            telemetry=condition != "bare",
        )
        for repeat in range(6)
        for condition in AVAILABLE_CONDITIONS
    ]
    run = _complete_run(
        manifest.cells[0],
        ("one",),
        results,
        tuple(range(6)),
    )
    monkeypatch.setattr("cortheon.qualification_factory._run_cell", lambda *_a, **_k: run)
    monkeypatch.setattr(
        "cortheon.qualification_factory._repository_fingerprint",
        lambda _repository: "fingerprint",
    )
    monkeypatch.setattr(
        "cortheon.qualification_factory._git_revision",
        lambda _repository: "abc123",
    )

    report = run_qualification(manifest, progress=False)

    accounting = report["aggregate"]["contrasts"]["full_vs_bare"]["accounting"]
    assert accounting["repeat_pairs"] == 6
    assert accounting["independent_cases"] == 1
    assert report["promotion_gates"]["independent_case_floor"] is False
    assert report["promoted"] is False
