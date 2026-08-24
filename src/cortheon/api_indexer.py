from __future__ import annotations

import ast
import io
import tarfile
import zipfile
from dataclasses import asdict
from pathlib import PurePosixPath

from cortheon.cache import SYMBOLS_SCHEMA_VERSION, FactCache
from cortheon.connectors.http import ConnectorError, JsonHttpClient
from cortheon.models import (
    ApiEvidenceReport,
    ApiSymbol,
    DistributionArtifact,
    Evidence,
    PackageMetadata,
    SupportLevel,
    utc_now,
)


class ApiEvidenceExtractor:
    def __init__(
        self,
        client: JsonHttpClient | None = None,
        max_artifact_bytes: int = 30_000_000,
        max_python_files: int = 800,
        cache: FactCache | None = None,
    ) -> None:
        self.client = client or JsonHttpClient(timeout_seconds=30)
        self.max_artifact_bytes = max_artifact_bytes
        self.max_python_files = max_python_files
        self.cache = cache or FactCache()

    def load_symbols(
        self,
        metadata: PackageMetadata,
    ) -> tuple[DistributionArtifact | None, list[ApiSymbol], list[str]]:
        artifact = choose_artifact(metadata.artifacts)
        errors: list[str] = []
        if not artifact:
            errors.append(
                "No source distribution or wheel artifact was available for the selected version."
            )
            return None, [], errors
        if artifact.size and artifact.size > self.max_artifact_bytes:
            errors.append(
                f"Selected artifact {artifact.filename} is {artifact.size} bytes, above limit {self.max_artifact_bytes}."
            )
            return artifact, [], errors

        # A pinned artifact's symbol table is immutable, so a cache hit is as
        # trustworthy as a fresh download — and instant.
        cache_key = symbols_cache_key(metadata, artifact)
        cached = self.cache.get(*cache_key)
        if cached is not None:
            try:
                return artifact, [ApiSymbol(**item) for item in cached], errors
            except TypeError:
                pass  # schema drift; fall through to a fresh extraction

        try:
            response = self.client.get(artifact.url)
        except ConnectorError as exc:
            errors.append(str(exc))
            return artifact, [], errors

        if len(response.body) > self.max_artifact_bytes:
            errors.append(
                f"Downloaded artifact {artifact.filename} is {len(response.body)} bytes, above limit {self.max_artifact_bytes}."
            )
            return artifact, [], errors

        try:
            symbols = extract_symbols_from_archive(
                artifact.filename, response.body, self.max_python_files
            )
        except (tarfile.TarError, zipfile.BadZipFile, UnicodeDecodeError, SyntaxError) as exc:
            errors.append(
                f"Could not parse artifact {artifact.filename}: {type(exc).__name__}: {exc}"
            )
            return artifact, [], errors
        self.cache.put([asdict(symbol) for symbol in symbols], *cache_key)
        return artifact, symbols, errors

    def retrieve(self, metadata: PackageMetadata, query: str) -> ApiEvidenceReport:
        artifact, symbols, errors = self.load_symbols(metadata)
        evidence: list[Evidence] = []
        if errors:
            return self._report(metadata, query, artifact, symbols, evidence, errors)
        if artifact is None:
            errors.append("No source artifact was available for API inspection.")
            return self._report(metadata, query, artifact, symbols, evidence, errors)

        matches = match_symbols(symbols, query)
        if matches:
            evidence.append(
                Evidence(
                    claim=(
                        f"Source artifact {artifact.filename} for {metadata.name} {metadata.version} "
                        f"defines {len(matches)} symbol(s) matching {query!r}."
                    ),
                    source_type="source_artifact_ast",
                    source_url=artifact.url,
                    package=metadata.name,
                    version=metadata.version,
                    support=SupportLevel.VERIFIED,
                    details={
                        "artifact_filename": artifact.filename,
                        "query": query,
                        "match_count": len(matches),
                        "total_symbols": len(symbols),
                    },
                )
            )
        else:
            evidence.append(
                Evidence(
                    claim=(
                        f"Source artifact {artifact.filename} for {metadata.name} {metadata.version} "
                        f"did not define a public symbol matching {query!r}."
                    ),
                    source_type="source_artifact_ast",
                    source_url=artifact.url,
                    package=metadata.name,
                    version=metadata.version,
                    support=SupportLevel.FAILED,
                    details={
                        "artifact_filename": artifact.filename,
                        "query": query,
                        "total_symbols": len(symbols),
                    },
                )
            )
        return self._report(metadata, query, artifact, symbols, evidence, errors)

    def _report(
        self,
        metadata: PackageMetadata,
        query: str,
        artifact: DistributionArtifact | None,
        symbols: list[ApiSymbol],
        evidence: list[Evidence],
        errors: list[str],
    ) -> ApiEvidenceReport:
        matches = match_symbols(symbols, query)
        return ApiEvidenceReport(
            package=metadata.name,
            version=metadata.version,
            query=query,
            artifact_filename=artifact.filename if artifact else None,
            artifact_url=artifact.url if artifact else None,
            extracted_at=utc_now(),
            total_symbols=len(symbols),
            matches=matches,
            evidence=evidence,
            errors=errors,
        )


def symbols_cache_key(metadata: PackageMetadata, artifact: DistributionArtifact) -> tuple[str, ...]:
    digest = (
        artifact.digests.get("sha256") or artifact.digests.get("md5") or str(artifact.size or "")
    )
    return (
        "symbols",
        SYMBOLS_SCHEMA_VERSION,
        metadata.name.lower(),
        metadata.version,
        artifact.filename,
        digest,
    )


def choose_artifact(artifacts: list[DistributionArtifact]) -> DistributionArtifact | None:
    source_exts = (".tar.gz", ".zip", ".tar.bz2", ".tgz")
    source = [
        item
        for item in artifacts
        if item.package_type == "sdist" or item.filename.endswith(source_exts)
    ]
    if source:
        return sorted(source, key=lambda item: item.size or 0)[0]
    wheels = [
        item
        for item in artifacts
        if item.package_type == "bdist_wheel" or item.filename.endswith(".whl")
    ]
    if wheels:
        return sorted(wheels, key=lambda item: item.size or 0)[0]
    return artifacts[0] if artifacts else None


def extract_symbols_from_archive(
    filename: str,
    body: bytes,
    max_python_files: int,
) -> list[ApiSymbol]:
    if filename.endswith(".whl") or filename.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            files = [
                (name, archive.read(name))
                for name in archive.namelist()
                if _is_python_source_path(name)
            ][:max_python_files]
    else:
        files = []
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:*") as archive:
            for member in archive.getmembers():
                if len(files) >= max_python_files:
                    break
                if not member.isfile() or not _is_python_source_path(member.name):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                files.append((member.name, extracted.read()))

    symbols: list[ApiSymbol] = []
    for path, content in files:
        source = content.decode("utf-8")
        module = module_name_from_path(path)
        if not module:
            continue
        tree = ast.parse(source, filename=path)
        symbols.extend(extract_symbols_from_ast(tree, module, path))
    return symbols


def extract_symbols_from_ast(tree: ast.AST, module: str, path: str) -> list[ApiSymbol]:
    symbols: list[ApiSymbol] = []
    for node in getattr(tree, "body", []):
        if isinstance(node, ast.ClassDef) and public_name(node.name):
            qualname = f"{module}.{node.name}"
            symbols.append(
                ApiSymbol(
                    name=node.name,
                    kind="class",
                    module=module,
                    qualname=qualname,
                    signature=class_signature(node),
                    file_path=path,
                    line=node.lineno,
                    docstring=first_doc_line(ast.get_docstring(node)),
                    deprecated=class_is_deprecated(node),
                )
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and public_name(
                    child.name
                ):
                    kind = "async_method" if isinstance(child, ast.AsyncFunctionDef) else "method"
                    symbols.append(
                        ApiSymbol(
                            name=child.name,
                            kind=kind,
                            module=module,
                            qualname=f"{qualname}.{child.name}",
                            signature=function_signature(child),
                            file_path=path,
                            line=child.lineno,
                            docstring=first_doc_line(ast.get_docstring(child)),
                            deprecated=function_is_deprecated(child),
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and public_name(node.name):
            kind = "async_function" if isinstance(node, ast.AsyncFunctionDef) else "function"
            symbols.append(
                ApiSymbol(
                    name=node.name,
                    kind=kind,
                    module=module,
                    qualname=f"{module}.{node.name}",
                    signature=function_signature(node),
                    file_path=path,
                    line=node.lineno,
                    docstring=first_doc_line(ast.get_docstring(node)),
                    deprecated=function_is_deprecated(node),
                )
            )
    return symbols


def class_is_deprecated(node: ast.ClassDef) -> bool:
    # Classes only look at their own docstring/decorators; a deprecated method
    # must not mark the whole class.
    return _doc_marks_deprecated(ast.get_docstring(node)) or _decorators_mark_deprecated(
        node.decorator_list
    )


def function_is_deprecated(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    if _doc_marks_deprecated(ast.get_docstring(node)) or _decorators_mark_deprecated(
        node.decorator_list
    ):
        return True
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        func_name = _dotted_name(child.func)
        if not func_name or func_name.split(".")[-1] != "warn":
            continue
        arguments = list(child.args) + [keyword.value for keyword in child.keywords]
        for argument in arguments:
            argument_name = _dotted_name(argument)
            if argument_name and argument_name.split(".")[-1].endswith("DeprecationWarning"):
                return True
    return False


def _doc_marks_deprecated(docstring: str | None) -> bool:
    return bool(docstring) and "deprecat" in docstring[:400].lower()


def _decorators_mark_deprecated(decorators: list[ast.expr]) -> bool:
    for decorator in decorators:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = _dotted_name(target)
        if name and "deprecated" in name.lower():
            return True
    return False


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def match_symbols(symbols: list[ApiSymbol], query: str) -> list[ApiSymbol]:
    normalized = query.strip().lower()
    if not normalized:
        return []
    exact: list[ApiSymbol] = []
    partial: list[ApiSymbol] = []
    for symbol in symbols:
        names = {
            symbol.name.lower(),
            symbol.qualname.lower(),
            f"{symbol.module}.{symbol.name}".lower(),
        }
        if normalized in names or any(name.endswith(f".{normalized}") for name in names):
            exact.append(symbol)
        elif normalized in symbol.qualname.lower():
            partial.append(symbol)
    return (exact + partial)[:25]


def module_name_from_path(path: str) -> str | None:
    parts = list(PurePosixPath(path).parts)
    if not parts or parts[-1] == "setup.py":
        return None
    if _should_skip(parts):
        return None
    if len(parts) > 1 and ("-" in parts[0] or parts[0].endswith((".egg-info", ".dist-info"))):
        parts = parts[1:]
    if "src" in parts:
        parts = parts[parts.index("src") + 1 :]
    if not parts or parts[-1].endswith(".py") is False:
        return None
    parts[-1] = parts[-1][:-3]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    parts = [part for part in parts if part and not part.endswith(".dist-info")]
    if not parts:
        return None
    if any(part in {"__pycache__", "site-packages"} for part in parts):
        return None
    return ".".join(parts)


def function_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    parameters: list[str] = []
    positional = list(node.args.posonlyargs) + list(node.args.args)
    default_offset = len(positional) - len(node.args.defaults)
    for index, arg in enumerate(positional):
        default = node.args.defaults[index - default_offset] if index >= default_offset else None
        parameters.append(format_arg(arg, default))
    if node.args.vararg:
        parameters.append("*" + format_arg(node.args.vararg, None))
    elif node.args.kwonlyargs:
        parameters.append("*")
    for arg, default in zip(node.args.kwonlyargs, node.args.kw_defaults, strict=True):
        parameters.append(format_arg(arg, default))
    if node.args.kwarg:
        parameters.append("**" + format_arg(node.args.kwarg, None))
    signature = f"{node.name}({', '.join(parameters)})"
    if node.returns:
        signature += f" -> {safe_unparse(node.returns)}"
    return signature


def class_signature(node: ast.ClassDef) -> str:
    bases = [safe_unparse(base) for base in node.bases]
    return f"class {node.name}({', '.join(bases)})" if bases else f"class {node.name}"


def format_arg(arg: ast.arg, default: ast.AST | None) -> str:
    text = arg.arg
    if arg.annotation:
        text += f": {safe_unparse(arg.annotation)}"
    if default is not None:
        text += f" = {safe_unparse(default)}"
    return text


def safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "..."


def first_doc_line(docstring: str | None) -> str | None:
    if not docstring:
        return None
    for line in docstring.splitlines():
        cleaned = line.strip()
        if cleaned:
            return cleaned[:240]
    return None


def public_name(name: str) -> bool:
    # __init__ is deliberately public: constructor keyword arguments are the
    # most commonly hallucinated API surface, so signature diffs must see them.
    return not name.startswith("_") or name in {
        "__init__",
        "__call__",
        "__enter__",
        "__exit__",
        "__aiter__",
        "__anext__",
    }


def _is_python_source_path(path: str) -> bool:
    parts = list(PurePosixPath(path).parts)
    return path.endswith(".py") and not _should_skip(parts)


def _should_skip(parts: list[str]) -> bool:
    lowered = {part.lower() for part in parts}
    return bool(
        lowered
        & {
            "tests",
            "test",
            "testing",
            "docs",
            "doc",
            "examples",
            "benchmarks",
            "benchmark",
            ".github",
        }
    )
