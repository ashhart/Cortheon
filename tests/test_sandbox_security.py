"""Security and containment tests for ``flash_forge.sandbox``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from flash_forge.sandbox import run_python, sandbox_backend
from flash_forge.skill_loader import SKILL_DIR, SkillPack


_IS_MACOS_SEATBELT = sys.platform == "darwin" and os.path.isfile("/usr/bin/sandbox-exec")


def test_sandbox_backend_detects_macos_seatbelt_on_darwin(monkeypatch):
    monkeypatch.delenv("CORTHEON_UNSAFE_NO_SANDBOX", raising=False)
    if _IS_MACOS_SEATBELT:
        assert sandbox_backend() == "macos-seatbelt"
    else:
        assert sandbox_backend() in ("linux-bwrap", "subprocess-rlimit")


def test_sandbox_backend_opt_out_flag_and_env(monkeypatch):
    monkeypatch.delenv("CORTHEON_UNSAFE_NO_SANDBOX", raising=False)
    assert sandbox_backend(unsafe_no_sandbox=True) == "unsafe-disabled"

    monkeypatch.setenv("CORTHEON_UNSAFE_NO_SANDBOX", "1")
    assert sandbox_backend() == "unsafe-disabled"


def test_sandbox_backend_linux_bwrap_and_rlimit_fallback(monkeypatch):
    monkeypatch.delenv("CORTHEON_UNSAFE_NO_SANDBOX", raising=False)
    monkeypatch.setattr("flash_forge.sandbox.sys.platform", "linux")
    monkeypatch.setattr("flash_forge.sandbox.shutil.which", lambda name: "/usr/bin/bwrap" if name == "bwrap" else None)
    assert sandbox_backend() == "linux-bwrap"

    monkeypatch.setattr("flash_forge.sandbox.shutil.which", lambda _name: None)
    assert sandbox_backend() == "subprocess-rlimit"


@pytest.mark.skipif(not _IS_MACOS_SEATBELT, reason="Requires macOS Seatbelt (/usr/bin/sandbox-exec)")
def test_sandbox_blocks_outbound_network_sockets():
    tcp_code = (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 80), timeout=1)\n"
        "    print('UNEXPECTED_CONNECT')\n"
        "except PermissionError:\n"
        "    print('BLOCKED_TCP_PERMISSION')\n"
        "except OSError:\n"
        "    print('BLOCKED_TCP_OSERROR')\n"
    )
    res = run_python(tcp_code, timeout_s=5.0)
    assert res.ok, f"Unexpected failure: {res.stderr}"
    assert "BLOCKED_TCP_" in res.stdout
    assert "UNEXPECTED_CONNECT" not in res.stdout

    udp_code = (
        "import socket\n"
        "try:\n"
        "    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)\n"
        "    s.sendto(b'ping', ('1.1.1.1', 53))\n"
        "    print('UNEXPECTED_UDP')\n"
        "except (PermissionError, OSError):\n"
        "    print('BLOCKED_UDP')\n"
    )
    res_udp = run_python(udp_code, timeout_s=5.0)
    assert res_udp.ok
    assert "BLOCKED_UDP" in res_udp.stdout


@pytest.mark.skipif(not _IS_MACOS_SEATBELT, reason="Requires macOS Seatbelt (/usr/bin/sandbox-exec)")
def test_sandbox_blocks_file_writes_outside_ephemeral_tmpdir(tmp_path: Path):
    escape_in_pytest_tmp = tmp_path / "outside_escape.txt"
    code = (
        "from pathlib import Path\n"
        f"targets = [Path.cwd().parent / 'ff_escape_parent.txt', Path({str(escape_in_pytest_tmp)!r})]\n"
        "blocked = 0\n"
        "for t in targets:\n"
        "    try:\n"
        "        t.write_text('escape', encoding='utf-8')\n"
        "        print('ESCAPED:', t)\n"
        "    except PermissionError:\n"
        "        blocked += 1\n"
        "print(f'BLOCKED={blocked}')\n"
    )
    res = run_python(code, timeout_s=5.0)
    assert res.ok, f"Unexpected failure: {res.stderr}"
    assert "BLOCKED=2" in res.stdout
    assert "ESCAPED:" not in res.stdout
    assert not escape_in_pytest_tmp.exists()


@pytest.mark.skipif(not _IS_MACOS_SEATBELT, reason="Requires macOS Seatbelt (/usr/bin/sandbox-exec)")
def test_sandbox_blocks_reading_host_ssh_if_present():
    ssh_dir = Path.home() / ".ssh"
    if not ssh_dir.exists():
        pytest.skip("~/.ssh does not exist on this host")
    code = (
        "import os\n"
        "from pathlib import Path\n"
        "try:\n"
        "    os.listdir(Path.home() / '.ssh')\n"
        "    print('READ_SSH_ALLOWED')\n"
        "except PermissionError:\n"
        "    print('SSH_DENIED')\n"
    )
    res = run_python(code, timeout_s=5.0)
    assert res.ok, f"Unexpected failure: {res.stderr}"
    assert res.stdout.strip() == "SSH_DENIED"


@pytest.mark.skipif(not _IS_MACOS_SEATBELT, reason="Requires macOS Seatbelt (/usr/bin/sandbox-exec)")
def test_sandbox_blocks_sensitive_credential_paths_in_home(tmp_path: Path, monkeypatch):
    fake_home = tmp_path / "fake_home"
    (fake_home / ".ssh").mkdir(parents=True)
    (fake_home / ".ssh" / "id_test").write_text("DUMMY_KEY", encoding="utf-8")
    (fake_home / ".aws").mkdir(parents=True)
    (fake_home / ".aws" / "credentials").write_text("DUMMY_AWS", encoding="utf-8")
    (fake_home / ".netrc").write_text("machine example login u password p", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))

    code = (
        "from pathlib import Path\n"
        "h = Path.home()\n"
        "targets = [h / '.ssh' / 'id_test', h / '.aws' / 'credentials', h / '.netrc']\n"
        "denied = 0\n"
        "for t in targets:\n"
        "    try:\n"
        "        t.read_text()\n"
        "        print('LEAKED:', t.name)\n"
        "    except PermissionError:\n"
        "        denied += 1\n"
        "print(f'DENIED={denied}')\n"
    )
    res = run_python(code, timeout_s=5.0)
    assert res.ok, f"Unexpected failure: {res.stderr}"
    assert "DENIED=3" in res.stdout
    assert "LEAKED:" not in res.stdout


@pytest.mark.skipif(not _IS_MACOS_SEATBELT, reason="Requires macOS Seatbelt (/usr/bin/sandbox-exec)")
def test_sandbox_blocks_agent_harness_homes_in_home(tmp_path: Path, monkeypatch):
    """Harness home directories hold credentials and transcripts.

    Cortheon is usually driven *by* one of these harnesses, so a solver script
    that could read them could exfiltrate the credentials the sandbox exists to
    protect.
    """
    fake_home = tmp_path / "fake_home"
    (fake_home / ".dsh").mkdir(parents=True)
    (fake_home / ".dsh" / "credentials.json").write_text("DUMMY_DSH", encoding="utf-8")
    (fake_home / ".claude").mkdir(parents=True)
    (fake_home / ".claude" / ".credentials.json").write_text("DUMMY_CLAUDE", encoding="utf-8")
    (fake_home / ".cursor").mkdir(parents=True)
    (fake_home / ".cursor" / "mcp.json").write_text("DUMMY_CURSOR", encoding="utf-8")
    monkeypatch.setenv("HOME", str(fake_home))

    code = (
        "from pathlib import Path\n"
        "h = Path.home()\n"
        "targets = [h / '.dsh' / 'credentials.json', "
        "h / '.claude' / '.credentials.json', h / '.cursor' / 'mcp.json']\n"
        "denied = 0\n"
        "for t in targets:\n"
        "    try:\n"
        "        t.read_text()\n"
        "        print('LEAKED:', t.name)\n"
        "    except PermissionError:\n"
        "        denied += 1\n"
        "print(f'DENIED={denied}')\n"
    )
    res = run_python(code, timeout_s=5.0)
    assert res.ok, f"Unexpected failure: {res.stderr}"
    assert "DENIED=3" in res.stdout
    assert "LEAKED:" not in res.stdout


def test_sandbox_allows_sympy_numpy_z3_fractions_and_local_writes():
    code = (
        "from fractions import Fraction\n"
        "from pathlib import Path\n"
        "import tempfile\n"
        "import numpy as np\n"
        "import sympy as sp\n"
        "import z3\n"
        "\n"
        "x = sp.Symbol('x')\n"
        "poly = sp.expand((x + 1)**3)\n"
        "arr_sum = int(np.arange(1, 11).sum())\n"
        "s = z3.Solver()\n"
        "k = z3.Int('k')\n"
        "s.add(k * 3 == 21)\n"
        "assert s.check() == z3.sat\n"
        "k_val = s.model()[k].as_long()\n"
        "frac = Fraction(6, 8)\n"
        "local = Path('scratch_output.txt')\n"
        "local.write_text(f'{poly}|{arr_sum}|{k_val}|{frac}', encoding='utf-8')\n"
        "with tempfile.TemporaryDirectory() as td:\n"
        "    (Path(td) / 'nested.txt').write_text('ok', encoding='utf-8')\n"
        "    assert (Path(td) / 'nested.txt').read_text() == 'ok'\n"
        "print(local.read_text(encoding='utf-8'))\n"
    )
    res = run_python(code, timeout_s=10.0)
    assert res.ok, f"Legitimate math script failed: {res.stderr}"
    assert res.stdout.strip() == "x**3 + 3*x**2 + 3*x + 1|55|7|3/4"


def test_sandbox_timeout_kills_runaway_infinite_loop():
    res = run_python("while True:\n    pass\n", timeout_s=0.4)
    assert not res.ok
    assert res.timed_out is True
    assert res.elapsed_s < 4.0


def test_sandbox_scrubs_secret_env_vars(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-dummy-test-value")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-dummy-secret-value")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ant-dummy-secret-value")
    code = (
        "import os\n"
        "leaked = [k for k in ('OPENAI_API_KEY', 'AWS_SECRET_ACCESS_KEY', 'ANTHROPIC_API_KEY') if k in os.environ]\n"
        "print('LEAKED=' + ','.join(leaked))\n"
    )
    res = run_python(code, timeout_s=5.0)
    assert res.ok, f"Env scrub check failed: {res.stderr}"
    assert res.stdout.strip() == "LEAKED="


def test_sandbox_rejects_extra_files_path_traversal():
    with pytest.raises(ValueError, match="outside sandbox temp dir"):
        run_python("print('ok')", extra_files={"../escape.py": "x = 1\n"})


@pytest.mark.skipif(os.name != "posix", reason="POSIX RLIMIT_FSIZE check")
def test_sandbox_enforces_fsize_rlimit():
    code = (
        "try:\n"
        "    with open('big.bin', 'wb') as f:\n"
        "        f.seek(70 * 1024 * 1024)\n"
        "        f.write(b'x')\n"
        "    print('WROTE_OVER_LIMIT')\n"
        "except OSError:\n"
        "    print('FSIZE_BLOCKED')\n"
    )
    res = run_python(code, timeout_s=5.0)
    assert "WROTE_OVER_LIMIT" not in res.stdout
    # Either caught as OSError or terminated by SIGXFSZ
    assert ("FSIZE_BLOCKED" in res.stdout) or (not res.ok)


def test_skill_pack_single_source_of_truth_in_package():
    assert SKILL_DIR.name == "_skill"
    assert SKILL_DIR.parent.name == "flash_forge"
    assert (SKILL_DIR / "cards").is_dir()
    assert (SKILL_DIR / "exemplars.json").is_file()
    repo_root = SKILL_DIR.parent.parent
    assert not (repo_root / "skill").exists()
    pack = SkillPack()
    assert len(pack.card("pal_exact")) > 40
