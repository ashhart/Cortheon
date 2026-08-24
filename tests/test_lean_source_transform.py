"""Direct tests for the build-output compaction pass.

The transformer may only remove bytes with no runtime meaning (comments and
layout). Docstrings, function annotations, module/class annotations - which
dataclasses read at runtime - and every import, including unaliased dotted
imports like ``import collections.abc``, must survive untouched so that
installed-wheel introspection (``help()``, ``__doc__``,
``inspect.signature``, ``__annotations__``, ``typing.get_type_hints``)
matches repository source. Release builds may omit metadata belonging only
to private definitions and private implementation modules.
"""

from __future__ import annotations

import collections.abc
import dataclasses
import importlib.util
import inspect
import sys
import textwrap
import typing
from pathlib import Path
from types import ModuleType
from typing import Any

from build_support.lean_source import compact_python_module, drop_typescript_declarations


def _compact(tmp_path: Path, source: str) -> str:
    path = tmp_path / "lean.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    compact_python_module(path)
    return path.read_text(encoding="utf-8")


def _load(tmp_path: Path, name: str, source: str) -> ModuleType:
    path = tmp_path / f"lean-mod-{name}.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    compact_python_module(path)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_comments_are_removed_and_code_survives(tmp_path: Path) -> None:
    compacted = _compact(
        tmp_path,
        """
        # A comment with no runtime meaning.
        x = 3  # trailing comment
        """,
    )
    assert "comment" not in compacted
    assert "x = 3" in compacted


def test_docstrings_are_preserved(tmp_path: Path) -> None:
    module = _load(
        tmp_path,
        "lean_docstrings",
        '''
        """Module docstring."""
        class C:
            """Class docstring."""
            def method(self):
                """Method docstring."""
                return 1
        ''',
    )
    assert module.__doc__ == "Module docstring."
    assert module.C.__doc__ == "Class docstring."
    assert module.C.method.__doc__ == "Method docstring."


def test_release_compaction_drops_only_private_metadata(tmp_path: Path) -> None:
    path = tmp_path / "private_core.py"
    path.write_text(
        textwrap.dedent(
            '''
            """Private implementation module."""
            def public(value: int) -> int:
                """Public callable metadata."""
                return _private(value)
            class Public:
                def __init__(self, value: int) -> None:
                    self.value = value
            def _private(value: int) -> int:
                """Maintainer-only metadata."""
                return value + 1
            '''
        ),
        encoding="utf-8",
    )
    compact_python_module(
        path,
        strip_private_metadata=True,
        private_module=True,
    )
    spec = importlib.util.spec_from_file_location("private_core", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__doc__ is None
    assert module.public.__doc__ == "Public callable metadata."
    assert module._private.__doc__ is None
    assert str(inspect.signature(module._private)) == "(value)"
    assert str(inspect.signature(module.public)) == "(value: int) -> int"
    assert str(inspect.signature(module.Public)) == "(value: int) -> None"


def test_build_only_compaction_drops_all_callable_metadata(tmp_path: Path) -> None:
    path = tmp_path / "build_only.py"
    path.write_text(
        '"""Build-only module."""\n'
        "def public(value: int) -> int:\n"
        '    """Not part of an installed API."""\n'
        "    return value + 1\n",
        encoding="utf-8",
    )
    compact_python_module(
        path,
        strip_private_metadata=True,
        private_module=True,
        strip_all_callable_metadata=True,
    )
    spec = importlib.util.spec_from_file_location("build_only", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.__doc__ is None
    assert module.public.__doc__ is None
    assert str(inspect.signature(module.public)) == "(value)"
    assert module.public(2) == 3


def test_built_typescript_drops_only_type_declarations(tmp_path: Path) -> None:
    path = tmp_path / "adapter.ts"
    path.write_text(
        "interface Receipt { value: string; nested: { ok: boolean }; }\n"
        "type Mode =\n  | 'full'\n  | 'bare';\n"
        "const receipt: Receipt = {value: 'x', nested: {ok: true}};\n",
        encoding="utf-8",
    )
    drop_typescript_declarations(path)
    compacted = path.read_text(encoding="utf-8")
    assert "interface Receipt" not in compacted
    assert "type Mode" not in compacted
    assert "const receipt: Receipt" in compacted


def test_function_annotations_are_preserved(tmp_path: Path) -> None:
    compacted = _compact(
        tmp_path,
        """
        from __future__ import annotations
        def annotated(a: int, *args: int, b: str = 'x', **kw: int) -> int:
            return a
        """,
    )
    assert "-> int" in compacted
    assert "a: int" in compacted and "b: str" in compacted

    module = _load(
        tmp_path,
        "lean_annotations",
        """
        from __future__ import annotations
        def annotated(a: int, b: str = 'x') -> int:
            return a
        """,
    )
    hints = typing.get_type_hints(module.annotated)
    assert hints == {"a": int, "b": str, "return": int}
    # The module keeps ``from __future__ import annotations``, so
    # inspect.signature renders the annotations as their string forms -
    # exactly as it does for the untransformed repository source.
    signature = inspect.signature(module.annotated)
    assert str(signature) == "(a: 'int', b: 'str' = 'x') -> 'int'"


def test_local_annotations_are_removed_without_changing_values(tmp_path: Path) -> None:
    compacted = _compact(
        tmp_path,
        """
        def calculate() -> tuple[int, int]:
            first: int = 2
            second: int
            second = 3
            return first, second
        """,
    )
    assert "first: int" not in compacted
    assert "second: int" in compacted

    module = _load(
        tmp_path,
        "lean_locals",
        """
        def calculate() -> tuple[int, int]:
            first: int = 2
            second: int
            second = 3
            return first, second
        """,
    )
    assert module.calculate() == (2, 3)


def test_class_annotations_preserve_dataclass_fields(tmp_path: Path) -> None:
    module = _load(
        tmp_path,
        "lean_dataclass",
        """
        from __future__ import annotations
        from dataclasses import dataclass, field
        from typing import Any
        @dataclass
        class Sample:
            name: str
            count: int = 0
            extra: Any = field(default_factory=dict)
        """,
    )
    sample: Any = module.Sample(name="n", count=2)
    assert [f.name for f in dataclasses.fields(module.Sample)] == [
        "name",
        "count",
        "extra",
    ]
    assert sample.extra == {}
    assert typing.get_type_hints(module.Sample)["extra"] is Any


def test_no_import_is_ever_pruned(tmp_path: Path) -> None:
    compacted = _compact(
        tmp_path,
        """
        from __future__ import annotations
        import collections.abc
        import json
        import os as unused_alias
        from collections import OrderedDict
        from typing import Any, Optional, Sequence
        x = 1
        """,
    )
    assert "import collections.abc" in compacted
    assert "import json" in compacted
    assert "unused_alias" in compacted
    assert "OrderedDict" in compacted
    assert "Any" in compacted and "Optional" in compacted and "Sequence" in compacted


def test_unaliased_dotted_import_keeps_module_attribute_access(
    tmp_path: Path,
) -> None:
    module = _load(
        tmp_path,
        "lean_dotted_import",
        """
        import collections.abc
        value = collections.abc.Iterable
        """,
    )
    assert module.value is collections.abc.Iterable
