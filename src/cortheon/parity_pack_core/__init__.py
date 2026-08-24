"""Repository-only implementation of the evaluator challenge-pack tool.

``cortheon.parity_pack`` is a thin compatibility facade over this package.
Each module here owns one responsibility of issuing and checking a pack: the
late-bound clock, evaluator secret material and the HMAC seal, held-out case
selection, manifest assembly and its public projection, the sealing
orchestration, pack verification, release-contract authoring, and the
``cortheon-pack`` command line.

Nothing in this package may import the facade statically; ``_compat`` resolves
it lazily so facade-level rebindings keep steering the tool.
"""
