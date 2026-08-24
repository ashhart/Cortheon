"""Repository-only implementation of the frontier-parity release contract.

``cortheon.parity`` is a thin compatibility facade over this package. Each
module here owns one responsibility of the contract evaluation: the failure
type and registered host universes, value coercion, the evaluation context,
public task projection and scheduling, contract validation, pre-registration
and blinding, contender identity, execution binding, outcome gates, paired
comparison statistics, cost metering, and decision assembly.

Nothing in this package may import the facade statically; ``_compat`` resolves
it lazily so facade-level rebindings keep steering the gates.
"""
