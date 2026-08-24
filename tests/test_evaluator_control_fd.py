"""Hostile tests for the evaluator-owned adapter control channel."""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from cognitive_http_cases_common import running_server

from cortheon.benchmark_core.condition_execution import (
    AppliedCondition,
    condition_control_payload,
    prepare_condition,
)
from cortheon.benchmark_core.execution_provenance import (
    ExecutionPolicy,
    execute_host_process,
)
from cortheon.qualification_core.conditions import execution_profile

ROOT = Path(__file__).parents[1]
CONTROL_KEYS = {
    "CORTHEON_EVALUATOR_PROFILE",
    "CORTHEON_COGNITIVE_TOKEN",
    "CORTHEON_EVALUATOR_MAX_STEPS",
    "CORTHEON_AUTO_ENABLE",
    "CORTHEON_BENCHMARK_CAPTURE_CANDIDATE",
    "CORTHEON_MAX_HOST_TOOL_CALLS",
    "CORTHEON_CONTROL_FD",
}


def _profile(condition: str, nonce: str = "3" * 32) -> dict:
    profile = execution_profile(condition, "a" * 64)
    profile["nonce"] = nonce
    return profile


def _payload(profile: dict, host: str) -> bytes:
    payload = condition_control_payload(
        AppliedCondition(profile=profile, nonce=profile["nonce"]),
        token="test-secret-token",
        host=host,
        treatment=True,
        max_steps=4,
        max_host_tool_calls=12,
    )
    assert payload is not None
    return payload


def _command(script: str, strip_types: bool) -> list[str]:
    command = ["node"]
    if strip_types:
        command.append("--experimental-strip-types")
    return [*command, "--input-type=module", "-e", script]


def _run_fd(
    script: str,
    profile: dict,
    *,
    host: str,
    strip_types: bool = False,
    raw_payload: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    read_fd, write_fd = os.pipe()
    environment = {key: value for key, value in os.environ.items() if key not in CONTROL_KEYS}
    environment["CORTHEON_CONTROL_FD"] = str(read_fd)
    try:
        process = subprocess.Popen(
            _command(script, strip_types),
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=ROOT,
            pass_fds=(read_fd,),
        )
        os.close(read_fd)
        read_fd = -1
        view = memoryview(raw_payload or _payload(profile, host))
        while view:
            view = view[os.write(write_fd, view) :]
        os.close(write_fd)
        write_fd = -1
        stdout, stderr = process.communicate(timeout=15)
    finally:
        if read_fd >= 0:
            os.close(read_fd)
        if write_fd >= 0:
            os.close(write_fd)
    return subprocess.CompletedProcess(process.args, process.returncode, stdout, stderr)


def _run_fd_json(*args, **kwargs) -> dict:
    completed = _run_fd(*args, **kwargs)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("module", "host", "strip_types"),
    [
        ("./src/cortheon/pi_core/protocol.ts", "pi", True),
        ("./src/cortheon/opencode_core/state.js", "opencode", False),
    ],
)
def test_real_execution_pipe_applies_the_disabled_operator_and_fd_receipt(
    module: str,
    host: str,
    strip_types: bool,
) -> None:
    profile = _profile("without_cross_source_derivation")
    script = f"""
      const api = await import('{module}');
      const receipt = api.adapterEvaluationProfile()?.adapter_receipt;
      console.log(JSON.stringify({{type: 'control_probe', enabled: api.operatorEnabled('cross_source_derivation'), transport: receipt?.control_transport, nonce: receipt?.nonce}}));
    """
    capture = execute_host_process(
        _command(script, strip_types),
        cwd=ROOT,
        env={key: value for key, value in os.environ.items() if key not in CONTROL_KEYS},
        host=host,
        policy=ExecutionPolicy(4, 16, 10.0, 8_192, 512),
        control_payload=_payload(profile, host),
    )
    assert capture.returncode == 0, capture.stderr
    assert json.loads(capture.stdout) == {
        "type": "control_probe",
        "enabled": False,
        "transport": "fd",
        "nonce": "3" * 32,
    }


def test_host_ignoring_control_pipe_is_captured_instead_of_crashing_evaluator() -> None:
    profile = _profile("full")
    capture = execute_host_process(
        ["/bin/sh", "-c", "exit 0"],
        cwd=ROOT,
        env={key: value for key, value in os.environ.items() if key not in CONTROL_KEYS},
        host="pi",
        policy=ExecutionPolicy(4, 16, 2.0, 8_192, 512),
        control_payload=_payload(profile, "pi"),
    )
    assert capture.returncode == 0
    assert capture.stdout == ""


@pytest.mark.parametrize(
    ("module", "host", "strip_types"),
    [
        ("./src/cortheon/pi_core/protocol.ts", "pi", True),
        ("./src/cortheon/opencode_core/state.js", "opencode", False),
    ],
)
def test_tool_shell_cannot_recover_payload_from_parent_launch_or_fd(
    module: str,
    host: str,
    strip_types: bool,
) -> None:
    script = f"""
      import {{spawnSync}} from 'node:child_process';
      import {{fstatSync}} from 'node:fs';
      const descriptor = process.env.CORTHEON_CONTROL_FD;
      const api = await import('{module}');
      let controlFdOpen = true;
      try {{ fstatSync(Number(descriptor)); }} catch {{ controlFdOpen = false; }}
      const receipt = api.adapterEvaluationProfile()?.adapter_receipt;
      const launch = spawnSync('/bin/sh', ['-c', 'ps eww -p $PPID'], {{encoding: 'utf8'}}).stdout;
      const childProbe = spawnSync(process.execPath, ['--input-type=module', '-e',
        'console.log(JSON.stringify({{env: process.env.CORTHEON_CONTROL_FD ?? null}}));'], {{encoding: 'utf8'}});
      const token = api.configuredRuntimeToken?.() ?? api.initialEnvironment?.token;
      console.log(JSON.stringify({{
        launchHasToken: launch.includes(token),
        launchHasNonce: launch.includes(receipt.nonce),
        launchHasConfig: launch.includes(receipt.config_sha256),
        controlFdOpen,
        child: JSON.parse(childProbe.stdout),
        enabled: api.operatorEnabled('cross_source_derivation'),
        transport: receipt.control_transport,
      }}));
    """
    observed = _run_fd_json(
        script,
        _profile("without_cross_source_derivation"),
        host=host,
        strip_types=strip_types,
    )
    assert observed == {
        "launchHasToken": False,
        "launchHasNonce": False,
        "launchHasConfig": False,
        "controlFdOpen": False,
        "child": {"env": None},
        "enabled": False,
        "transport": "fd",
    }


@pytest.mark.parametrize(
    ("module", "host", "strip_types"),
    [
        ("./src/cortheon/pi_core/protocol.ts", "pi", True),
        ("./src/cortheon/opencode_core/state.js", "opencode", False),
    ],
)
@pytest.mark.parametrize("raw_payload", [b"not-json", b"x" * 16_385])
def test_malformed_and_oversized_fd_payloads_fail_closed(
    module: str,
    host: str,
    strip_types: bool,
    raw_payload: bytes,
) -> None:
    completed = _run_fd(
        f"await import('{module}');",
        _profile("full"),
        host=host,
        strip_types=strip_types,
        raw_payload=raw_payload,
    )
    assert completed.returncode != 0
    assert "invalid evaluator control descriptor or payload" in completed.stderr


@pytest.mark.parametrize(
    ("module", "strip_types"),
    [
        ("./src/cortheon/pi_core/protocol.ts", True),
        ("./src/cortheon/opencode_core/state.js", False),
    ],
)
def test_missing_or_malformed_descriptor_never_produces_an_fd_receipt(
    module: str,
    strip_types: bool,
) -> None:
    script = f"const api=await import('{module}'); console.log(JSON.stringify(api.adapterEvaluationProfile?.() ?? null));"
    missing = subprocess.run(
        _command(script, strip_types),
        cwd=ROOT,
        env={key: value for key, value in os.environ.items() if key not in CONTROL_KEYS},
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(missing.stdout) is None
    malformed_environment = {
        key: value for key, value in os.environ.items() if key not in CONTROL_KEYS
    }
    malformed_environment["CORTHEON_CONTROL_FD"] = "not-a-descriptor"
    malformed = subprocess.run(
        _command(f"await import('{module}');", strip_types),
        cwd=ROOT,
        env=malformed_environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert malformed.returncode != 0
    assert "invalid evaluator control descriptor or payload" in malformed.stderr


def test_concurrent_processes_cannot_cross_read_control_payloads() -> None:
    def observe(index: int) -> dict:
        profile = _profile("without_cross_source_derivation", f"{index:032x}")
        return _run_fd_json(
            """
              import {adapterEvaluationProfile, operatorEnabled} from './src/cortheon/opencode_core/state.js';
              const receipt = adapterEvaluationProfile()?.adapter_receipt;
              console.log(JSON.stringify({nonce: receipt?.nonce, enabled: operatorEnabled('cross_source_derivation'), transport: receipt?.control_transport}));
            """,
            profile,
            host="opencode",
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = list(pool.map(observe, range(1, 9)))
    assert observed == [
        {"nonce": f"{index:032x}", "enabled": False, "transport": "fd"} for index in range(1, 9)
    ]


def test_condition_setup_scrubs_every_ambient_control_key() -> None:
    environment = dict.fromkeys(CONTROL_KEYS, "stale")
    prepare_condition(environment, execution_profile("full", "a" * 64), treatment=True)
    assert not CONTROL_KEYS.intersection(environment)


@pytest.mark.parametrize("limit", [0, 65, "12", None])
def test_control_payload_rejects_unbound_tool_limits(limit: object) -> None:
    profile = _profile("full")
    with pytest.raises(ValueError, match="tool-call limit"):
        condition_control_payload(
            AppliedCondition(profile=profile, nonce=profile["nonce"]),
            token="token",
            host="pi",
            treatment=True,
            max_steps=4,
            max_host_tool_calls=limit,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("limit", [0, 1_025, "4", None])
def test_control_payload_rejects_unbound_step_limits(limit: object) -> None:
    profile = _profile("full")
    with pytest.raises(ValueError, match="step limit"):
        condition_control_payload(
            AppliedCondition(profile=profile, nonce=profile["nonce"]),
            token="token",
            host="pi",
            treatment=True,
            max_steps=limit,  # type: ignore[arg-type]
            max_host_tool_calls=12,
        )


def test_fd_adapter_profile_survives_real_http_start_and_receipt() -> None:
    with running_server(token="test-secret-token") as (server, base):
        script = f"""
          import {{adapterEvaluationProfile, initialEnvironment}} from './src/cortheon/opencode_core/state.js';
          const headers = {{'content-type': 'application/json', authorization: `Bearer ${{initialEnvironment.token}}`}};
          const profile = adapterEvaluationProfile();
          const started = await fetch('{base}/v1/start', {{method: 'POST', headers, body: JSON.stringify({{goal: 'Inspect two documents.', evaluation_profile: profile}})}}).then(async response => ({{status: response.status, body: await response.json()}}));
          const receipt = await fetch('{base}/v1/evaluation-receipt', {{method: 'POST', headers, body: JSON.stringify({{nonce: profile.nonce}})}}).then(async response => ({{status: response.status, body: await response.json()}}));
          await fetch('{base}/v1/abandon', {{method: 'POST', headers, body: JSON.stringify({{session_id: started.body.session.session_id}})}});
          console.log(JSON.stringify({{startStatus: started.status, receiptStatus: receipt.status, transport: receipt.body.adapter_receipt.control_transport, nonce: receipt.body.adapter_receipt.nonce}}));
        """
        observed = _run_fd_json(
            script,
            _profile("without_cross_source_derivation"),
            host="opencode",
        )
        assert observed == {
            "startStatus": 200,
            "receiptStatus": 200,
            "transport": "fd",
            "nonce": "3" * 32,
        }
        assert server.runtime.active_sessions == 0
