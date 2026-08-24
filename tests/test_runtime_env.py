import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

from cortheon.runtime_env import RuntimeEnvPool, extract_call_probes, run_bind_probe
from cortheon.verifier import _venv_python


class ExtractCallProbesTests(unittest.TestCase):
    def test_module_attribute_calls_and_kwargs(self) -> None:
        code = "import httpx\nhttpx.request('GET', url, retries=3, timeout=5)\n"
        self.assertEqual(
            extract_call_probes(code, "httpx"),
            [{"path": "request", "kwargs": ["retries", "timeout"]}],
        )

    def test_module_alias_and_chained_attributes(self) -> None:
        code = "import httpx as hx\nhx.transports.HTTPTransport(retries=2)\n"
        self.assertEqual(
            extract_call_probes(code, "httpx"),
            [{"path": "transports.HTTPTransport", "kwargs": ["retries"]}],
        )

    def test_from_import_bindings_with_alias(self) -> None:
        code = "from httpx import HTTPTransport as T\nT(retries=1)\n"
        self.assertEqual(
            extract_call_probes(code, "httpx"),
            [{"path": "HTTPTransport", "kwargs": ["retries"]}],
        )

    def test_instance_calls_and_other_packages_skipped(self) -> None:
        code = (
            "import httpx\nimport yaml\n"
            "client = httpx.Client()\n"
            "client.get(url)\n"  # instance method: not module-rooted
            "yaml.safe_load(x)\n"  # different package root
        )
        self.assertEqual(extract_call_probes(code, "httpx"), [{"path": "Client", "kwargs": []}])

    def test_duplicates_removed_and_syntax_error_empty(self) -> None:
        code = "import httpx\nhttpx.get(u)\nhttpx.get(u)\n"
        self.assertEqual(extract_call_probes(code, "httpx"), [{"path": "get", "kwargs": []}])
        self.assertEqual(extract_call_probes("def broken(:", "httpx"), [])


class RunBindProbeTests(unittest.TestCase):
    """Real interpreter, stdlib target: runtime truth without network."""

    def test_resolved_and_bindable_kwarg(self) -> None:
        results = run_bind_probe(sys.executable, "json", [{"path": "dumps", "kwargs": ["indent"]}])
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["resolved"])
        self.assertTrue(results[0]["signature_known"])
        self.assertEqual(results[0]["unexpected_kwargs"], [])

    def test_live_signature_rejects_fake_kwarg(self) -> None:
        # textwrap.indent has no **kwargs, so a fake keyword must be rejected.
        # (json.dumps would bind anything — it forwards **kw; that acceptance
        # is runtime truth too, and exactly why static-only checking is not
        # the last word.)
        results = run_bind_probe(
            sys.executable, "textwrap", [{"path": "indent", "kwargs": ["zzz_fake_kwarg"]}]
        )
        self.assertEqual(results[0]["unexpected_kwargs"], ["zzz_fake_kwarg"])

    def test_missing_symbol_unresolved(self) -> None:
        results = run_bind_probe(sys.executable, "json", [{"path": "no_such_thing", "kwargs": []}])
        self.assertFalse(results[0]["resolved"])

    def test_class_constructor_binding(self) -> None:
        results = run_bind_probe(
            sys.executable,
            "json",
            [
                {"path": "JSONDecoder", "kwargs": ["strict"]},
                {"path": "JSONDecoder", "kwargs": ["zzz"]},
            ],
        )
        self.assertEqual(results[0]["unexpected_kwargs"], [])
        self.assertEqual(results[1]["unexpected_kwargs"], ["zzz"])

    def test_unimportable_module_returns_none_not_findings(self) -> None:
        self.assertIsNone(
            run_bind_probe(sys.executable, "zzz_no_such_module", [{"path": "x", "kwargs": []}])
        )

    def test_empty_probes_short_circuit(self) -> None:
        self.assertEqual(run_bind_probe(sys.executable, "json", []), [])


class RuntimeEnvPoolTests(unittest.TestCase):
    def test_ready_env_reused_without_subprocess(self) -> None:
        with mock.patch("cortheon.runtime_env.subprocess.run") as run:
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                pool = RuntimeEnvPool(tmp)
                env_dir = Path(tmp) / "httpx-1.0.0"
                bin_dir = "Scripts" if sys.platform.startswith("win") else "bin"
                (env_dir / bin_dir).mkdir(parents=True)
                (env_dir / bin_dir / "python").touch()
                (env_dir / ".ready").touch()

                python = pool.python_for("httpx", "1.0.0")

                self.assertIsNotNone(python)
                run.assert_not_called()

    def test_build_failure_cached_for_process(self) -> None:
        calls: list[list[str]] = []

        def failing_run(cmd, **kwargs):
            calls.append(list(cmd))
            raise subprocess.CalledProcessError(1, cmd)

        with mock.patch("cortheon.runtime_env.subprocess.run", side_effect=failing_run):
            import tempfile

            with tempfile.TemporaryDirectory() as tmp:
                pool = RuntimeEnvPool(tmp)
                self.assertIsNone(pool.python_for("ghost", "1.0"))
                first_count = len(calls)
                self.assertIsNone(pool.python_for("ghost", "1.0"))
                self.assertEqual(len(calls), first_count)  # no rebuild storm


class RuntimeEnvPoolAsyncTests(unittest.TestCase):
    """The usability contract: a cold build must never block the request path."""

    def _fake_pool(self, tmp: str) -> RuntimeEnvPool:
        """A pool whose build is a fast, fake no-op that writes the .ready marker."""
        pool = RuntimeEnvPool(tmp)

        def fake_build(package: str, version: str):
            env_dir = Path(tmp) / f"{package}-{version}"
            bin_dir = "Scripts" if sys.platform.startswith("win") else "bin"
            (env_dir / bin_dir).mkdir(parents=True)
            (env_dir / bin_dir / "python").touch()
            (env_dir / ".ready").touch()
            return _venv_python(env_dir)

        pool._build_sync = fake_build  # type: ignore[method-assign]
        return pool

    def test_ready_false_before_build_true_after(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            self.assertFalse(pool.ready("httpx", "1.0.0"))
            pool.python_for("httpx", "1.0.0")
            self.assertTrue(pool.ready("httpx", "1.0.0"))

    def test_wait_false_does_not_block_and_triggers_background_build(self) -> None:
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            # Cold env + wait=False: returns None immediately (no blocking),
            # and starts a background build.
            self.assertIsNone(pool.python_for("httpx", "1.0.0", wait=False))
            self.assertTrue(pool.is_building("httpx", "1.0.0") or pool.ready("httpx", "1.0.0"))
            # The background thread finishes the fake build quickly.
            for _ in range(50):
                if pool.ready("httpx", "1.0.0"):
                    break
                time.sleep(0.01)
            self.assertTrue(pool.ready("httpx", "1.0.0"))

    def test_prewarm_kicks_off_background_builds(self) -> None:
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            kicked = pool.prewarm([("httpx", "1.0.0"), ("rich", "13.0.0")])
            self.assertEqual(kicked, 2)
            # Both eventually become ready without any caller blocking.
            for _ in range(50):
                if pool.ready("httpx", "1.0.0") and pool.ready("rich", "13.0.0"):
                    break
                time.sleep(0.01)
            self.assertTrue(pool.ready("httpx", "1.0.0"))
            self.assertTrue(pool.ready("rich", "13.0.0"))

    def test_prewarm_skips_already_ready_envs(self) -> None:
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            pool.python_for("httpx", "1.0.0")  # build it now
            kicked = pool.prewarm([("httpx", "1.0.0"), ("rich", "13.0.0")])
            self.assertEqual(kicked, 1)  # only the not-yet-ready one
            # Let the one background build finish before the tempdir is removed.
            for _ in range(50):
                if pool.ready("rich", "13.0.0"):
                    break
                time.sleep(0.01)

    def test_bind_check_wait_false_returns_none_on_cold_env(self) -> None:
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            # Non-empty probes reach python_for(wait=False), which returns None
            # for a cold env instead of blocking.
            self.assertIsNone(
                pool.bind_check(
                    "httpx", "1.0.0", "httpx", [{"path": "Client", "kwargs": []}], wait=False
                )
            )
            # Let the background build finish before the tempdir is removed.
            for _ in range(50):
                if pool.ready("httpx", "1.0.0"):
                    break
                time.sleep(0.01)


class PythonForSpecsTests(unittest.TestCase):
    """Multi-package envs for the behavioral rung: one env per spec set."""

    def _fake_pool(self, tmp: str) -> RuntimeEnvPool:
        """A pool whose multi-spec build is a fast no-op writing .ready."""
        pool = RuntimeEnvPool(tmp)

        def fake_build_specs(key: str, specs: list[str]):
            env_dir = Path(tmp) / key
            bin_dir = "Scripts" if sys.platform.startswith("win") else "bin"
            (env_dir / bin_dir).mkdir(parents=True)
            (env_dir / bin_dir / "python").touch()
            (env_dir / ".ready").touch()
            return _venv_python(env_dir)

        pool._build_sync_specs = fake_build_specs  # type: ignore[method-assign]
        return pool

    def test_specs_key_is_order_independent(self) -> None:
        a = RuntimeEnvPool._specs_key(["httpx==0.28.1", "rich"])
        b = RuntimeEnvPool._specs_key(["rich", "httpx==0.28.1"])
        self.assertEqual(a, b)

    def test_specs_key_differs_for_different_specs(self) -> None:
        self.assertNotEqual(
            RuntimeEnvPool._specs_key(["httpx"]),
            RuntimeEnvPool._specs_key(["httpx", "rich"]),
        )

    def test_empty_specs_yields_stdlib_env(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            python = pool.python_for_specs([])
            self.assertIsNotNone(python)
            self.assertTrue(pool._ready_key(pool._specs_key([])))

    def test_multi_package_env_built_and_reused(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            key = pool._specs_key(["httpx==0.28.1", "rich"])
            self.assertFalse(pool._ready_key(key))
            python = pool.python_for_specs(["httpx==0.28.1", "rich"])
            self.assertIsNotNone(python)
            self.assertTrue(pool._ready_key(key))
            # Second call reuses the same env without rebuilding.
            again = pool.python_for_specs(["httpx==0.28.1", "rich"])
            self.assertEqual(python, again)

    def test_wait_false_returns_none_and_builds_in_background(self) -> None:
        import tempfile
        import time

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            self.assertIsNone(pool.python_for_specs(["httpx"], wait=False))
            key = pool._specs_key(["httpx"])
            with pool._lock:
                building = key in pool._building
            self.assertTrue(building or pool._ready_key(key))
            for _ in range(50):
                if pool._ready_key(key):
                    break
                time.sleep(0.01)
            self.assertTrue(pool._ready_key(key))

    def test_single_package_path_still_uses_unchanged_key_format(self) -> None:
        # Regression guard: the refactor must not change the single-package key,
        # or existing on-disk envs stop being reused (the paste noted this trap).
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            pool = self._fake_pool(tmp)
            pool.python_for("httpx", "1.0.0")
            # Key is 'httpx-1.0.0', NOT 'httpx==1.0.0'.
            self.assertTrue((Path(tmp) / "httpx-1.0.0" / ".ready").exists())


if __name__ == "__main__":
    unittest.main()
