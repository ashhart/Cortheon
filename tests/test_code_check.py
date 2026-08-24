import unittest

from cortheon.code_check import check_api_usage, extract_code_blocks
from cortheon.models import ApiSymbol


def symbol(qualname: str, kind: str = "method", signature: str | None = None) -> ApiSymbol:
    name = qualname.split(".")[-1]
    return ApiSymbol(
        name=name,
        kind=kind,
        module="httpx._client",
        qualname=qualname,
        signature=signature if signature is not None else f"{name}(self)",
        file_path="httpx/_client.py",
        line=1,
        docstring=None,
    )


# A minimal httpx-like symbol table: Client/stream/iter_bytes exist; the
# hallucinated stream_to_file and save do not.
HTTPX_SYMBOLS = [
    symbol("httpx.Client", kind="class"),
    symbol("httpx._client.Client.stream"),
    symbol("httpx._client.Client.get"),
    symbol("httpx._client.Client.__init__"),
    symbol("httpx._models.Response.iter_bytes"),
]


# Symbol table where Client.__init__ takes proxy= (current) but not proxies= (removed).
HTTPX_KW_SYMBOLS = [
    symbol("httpx._client.Client", kind="class", signature="class Client(BaseClient)"),
    symbol(
        "httpx._client.Client.__init__",
        signature="__init__(self, *, auth=None, proxy=None, timeout=None, verify=True) -> None",
    ),
    symbol("httpx._client.Client.stream", signature="stream(self, method, url) -> Response"),
]


class KeywordCheckTests(unittest.TestCase):
    def test_dead_parameter_is_blocked(self) -> None:
        code = "import httpx\nclient = httpx.Client(proxies='http://p', timeout=30)\n"
        report = check_api_usage(code, "httpx", HTTPX_KW_SYMBOLS)

        self.assertEqual(report.verdict, "block")
        finding = next(f for f in report.findings if f.kind == "unknown_argument")
        self.assertIn("proxies", finding.reason)
        self.assertIn("proxy", finding.reason)  # the real one is suggested

    def test_current_parameter_passes(self) -> None:
        code = "import httpx\nclient = httpx.Client(proxy='http://p', timeout=30)\n"
        report = check_api_usage(code, "httpx", HTTPX_KW_SYMBOLS)

        self.assertEqual(report.verdict, "allow")

    def test_kwargs_accepting_signature_suppresses_flag(self) -> None:
        syms = [
            symbol("pkg.Thing", kind="class", signature="class Thing(object)"),
            symbol("pkg.Thing.__init__", signature="__init__(self, **kwargs) -> None"),
        ]
        report = check_api_usage("import pkg\npkg.Thing(anything=1)\n", "pkg", syms)
        self.assertEqual(report.verdict, "allow")


class FeedbackQualityTests(unittest.TestCase):
    def test_legacy_namespace_symbols_are_never_suggested(self) -> None:
        syms = [
            symbol(
                "pydantic.deprecated.class_validators.validator",
                kind="function",
                signature="validator(*fields)",
            ),
            symbol(
                "pydantic.v1.validators.int_validator",
                kind="function",
                signature="int_validator(v)",
            ),
            symbol(
                "pydantic.functional_validators.field_validator",
                kind="function",
                signature="field_validator(*fields)",
            ),
        ]
        code = "from pydantic import validator\n@validator('email')\ndef lower(cls, v):\n    return v\n"
        report = check_api_usage(code, "pydantic", syms)

        finding = next(f for f in report.findings if f.kind == "deprecated_symbol")
        self.assertIn("field_validator", finding.reason)
        self.assertNotIn("int_validator", finding.reason)

    def test_rejected_kwarg_names_the_callable_that_accepts_it(self) -> None:
        syms = [
            symbol("httpx._client.Client", kind="class", signature="class Client(BaseClient)"),
            symbol(
                "httpx._client.Client.__init__",
                signature="__init__(self, *, timeout=None, transport=None, verify=True) -> None",
            ),
            symbol(
                "httpx._transports.default.HTTPTransport",
                kind="class",
                signature="class HTTPTransport(BaseTransport)",
            ),
            symbol(
                "httpx._transports.default.HTTPTransport.__init__",
                signature="__init__(self, *, verify=True, retries=0) -> None",
            ),
        ]
        code = "import httpx\nclient = httpx.Client(timeout=30, retries=3)\n"
        report = check_api_usage(code, "httpx", syms)

        finding = next(f for f in report.findings if f.kind == "unknown_argument")
        self.assertIn("transport", finding.reason)  # full accepted list, not a truncated slice
        self.assertIn("HTTPTransport", finding.reason)  # cross-hint names the right home


class RepairAndBridgeTests(unittest.TestCase):
    def test_deprecation_rename_repairs_stuck_code(self) -> None:
        from cortheon.code_check import apply_deprecation_renames

        syms = [
            symbol("pydantic.BaseModel", kind="class", signature="class BaseModel(object)"),
            symbol(
                "pydantic.deprecated.class_validators.validator",
                kind="function",
                signature="validator(*fields)",
            ),
            symbol(
                "pydantic.functional_validators.field_validator",
                kind="function",
                signature="field_validator(*fields)",
            ),
        ]
        code = (
            "from pydantic import BaseModel, validator\n"
            "class Account(BaseModel):\n"
            "    email: str\n"
            "    @validator('email')\n"
            "    def lower(cls, v):\n"
            "        return v.lower()\n"
        )
        repaired, renames = apply_deprecation_renames(code, syms)

        self.assertEqual(renames, {"validator": "field_validator"})
        self.assertIn("@field_validator('email')", repaired)
        self.assertIn("from pydantic import BaseModel, field_validator", repaired)
        self.assertNotIn("@validator", repaired)
        self.assertEqual(check_api_usage(repaired, "pydantic", syms).verdict, "allow")

    def test_composition_bridge_prefers_name_affine_param(self) -> None:
        syms = [
            symbol("httpx._client.Client", kind="class", signature="class Client(BaseClient)"),
            symbol(
                "httpx._client.Client.__init__",
                signature=(
                    "__init__(self, *, timeout: TimeoutTypes = None, "
                    "mounts: Mapping[str, BaseTransport | None] = None, "
                    "transport: BaseTransport | None = None) -> None"
                ),
            ),
            symbol(
                "httpx._transports.default.HTTPTransport",
                kind="class",
                signature="class HTTPTransport(BaseTransport)",
            ),
            symbol(
                "httpx._transports.default.HTTPTransport.__init__",
                signature="__init__(self, *, verify=True, retries: int = 0) -> None",
            ),
        ]
        report = check_api_usage("import httpx\nc = httpx.Client(retries=3)\n", "httpx", syms)

        finding = next(f for f in report.findings if f.kind == "unknown_argument")
        self.assertIn("transport=HTTPTransport(retries=...)", finding.reason)


class DeprecatedUsageTests(unittest.TestCase):
    def test_deprecated_decorator_is_blocked(self) -> None:
        syms = [
            symbol("pydantic.BaseModel", kind="class", signature="class BaseModel(object)"),
            symbol(
                "pydantic.deprecated.class_validators.validator",
                kind="function",
                signature="validator(*fields)",
            ),
            symbol(
                "pydantic.functional_validators.field_validator",
                kind="function",
                signature="field_validator(*fields)",
            ),
        ]
        syms[1].deprecated = True  # validator is deprecated in current pydantic
        code = (
            "from pydantic import BaseModel, validator\n"
            "class Account(BaseModel):\n"
            "    email: str\n"
            "    @validator('email')\n"
            "    def lower(cls, v):\n"
            "        return v.lower()\n"
        )
        report = check_api_usage(code, "pydantic", syms)

        self.assertEqual(report.verdict, "block")
        self.assertTrue(
            any(
                f.kind == "deprecated_symbol" and f.attribute == "validator"
                for f in report.findings
            )
        )

    def test_current_replacement_passes(self) -> None:
        syms = [
            symbol("pydantic.BaseModel", kind="class", signature="class BaseModel(object)"),
            symbol(
                "pydantic.functional_validators.field_validator",
                kind="function",
                signature="field_validator(*fields)",
            ),
        ]
        code = (
            "from pydantic import BaseModel, field_validator\n"
            "class Account(BaseModel):\n"
            "    email: str\n"
            "    @field_validator('email')\n"
            "    def lower(cls, v):\n"
            "        return v.lower()\n"
        )
        self.assertEqual(check_api_usage(code, "pydantic", syms).verdict, "allow")


class CodeCheckTests(unittest.TestCase):
    def test_flags_hallucinated_method_used_on_instance(self) -> None:
        code = (
            "import httpx\n"
            "client = httpx.Client()\n"
            "with client.stream_to_file('GET', url) as r:\n"
            "    pass\n"
        )
        report = check_api_usage(code, "httpx", HTTPX_SYMBOLS)

        self.assertTrue(report.parsed)
        self.assertEqual(report.verdict, "block")
        self.assertEqual([f.attribute for f in report.findings], ["stream_to_file"])
        self.assertEqual(report.findings[0].line, 3)

    def test_real_method_passes(self) -> None:
        code = (
            "import httpx\n"
            "with httpx.Client() as client:\n"
            "    with client.stream('GET', url) as r:\n"
            "        r.iter_bytes()\n"
        )
        report = check_api_usage(code, "httpx", HTTPX_SYMBOLS)

        self.assertEqual(report.verdict, "allow")
        self.assertEqual(report.findings, [])
        # Both client.stream and the with-bound r.iter_bytes are checked and real.
        self.assertGreaterEqual(report.checked_calls, 2)

    def test_hallucinated_method_on_with_bound_name_is_caught(self) -> None:
        # The exact Llama-3.2-3B failure mode: a fake method on a `with ... as`
        # bound client that plain assignment tracking would have missed.
        code = (
            "import httpx\nwith httpx.Client() as client:\n    client.stream_to_file('GET', url)\n"
        )
        report = check_api_usage(code, "httpx", HTTPX_SYMBOLS)

        self.assertEqual(report.verdict, "block")
        self.assertEqual([f.attribute for f in report.findings], ["stream_to_file"])

    def test_hallucinated_terminal_method_in_chain_is_caught_without_import(self) -> None:
        code = (
            "with httpx.Client() as client:\n"
            "    response = client.get(url)\n"
            "    response.stream().write_to_file('output.bin')\n"
        )
        report = check_api_usage(code, "httpx", HTTPX_SYMBOLS)

        self.assertEqual(report.verdict, "block")
        self.assertEqual([f.attribute for f in report.findings], ["write_to_file"])

    def test_unparseable_code_blocks(self) -> None:
        report = check_api_usage("def broken(:\n    pass", "httpx", HTTPX_SYMBOLS)
        self.assertFalse(report.parsed)
        self.assertEqual(report.verdict, "block")

    def test_calls_on_unrelated_names_are_ignored(self) -> None:
        code = "import os\nos.definitely_fake_thing()\n"
        report = check_api_usage(code, "httpx", HTTPX_SYMBOLS)
        # os is not the package under check; no false positive.
        self.assertEqual(report.verdict, "allow")
        self.assertEqual(report.checked_calls, 0)

    def test_from_import_binding_is_checked(self) -> None:
        code = "from httpx import Client\nc = Client()\nc.stream_to_file('GET', url)\n"
        report = check_api_usage(code, "httpx", HTTPX_SYMBOLS)
        self.assertEqual([f.attribute for f in report.findings], ["stream_to_file"])

    def test_extract_code_blocks_prefers_fenced(self) -> None:
        answer = "Here is the code:\n```python\nimport httpx\nhttpx.get('x')\n```\nDone."
        blocks = extract_code_blocks(answer)
        self.assertEqual(len(blocks), 1)
        self.assertIn("httpx.get", blocks[0])

    def test_extract_falls_back_to_bare_code(self) -> None:
        self.assertEqual(extract_code_blocks("x = 1\n"), ["x = 1"])
        self.assertEqual(extract_code_blocks("this is just prose, no code."), [])


if __name__ == "__main__":
    unittest.main()
