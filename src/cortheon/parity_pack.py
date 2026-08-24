"""Issue and verify authenticated external challenge packs for cortheon-bench.

Stable import surface: every definition now lives in the repository-only
``cortheon.parity_pack_core`` package and is re-exported here (public and
private names alike) so existing imports keep working.

``datetime`` is a rebinding anchor: the sealing clock resolves it through this
module at call time, so substituting a frozen clock here still steers every
timestamp a pack carries.
"""

from datetime import datetime

from cortheon.parity_pack_core.cli import build_parser, main
from cortheon.parity_pack_core.contract import write_release_contract
from cortheon.parity_pack_core.keys import _canonical_signed_payload
from cortheon.parity_pack_core.seal import seal_case_pack
from cortheon.parity_pack_core.verify import verify_case_pack

__all__ = [
    "build_parser",
    "main",
    "seal_case_pack",
    "verify_case_pack",
    "write_release_contract",
]

# The private helper the pre-split module exposed as an attribute, and the
# clock callers rebind. Naming them here keeps the re-exports live -- an
# unused import would be removed as dead code -- without widening ``__all__``.
_COMPATIBILITY_EXPORTS = (
    _canonical_signed_payload,
    datetime,
)

if __name__ == "__main__":
    raise SystemExit(main())
