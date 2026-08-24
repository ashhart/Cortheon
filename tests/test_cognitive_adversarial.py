import unittest

import cognitive_adversarial_cases_atomicity as _atomicity
import cognitive_adversarial_cases_common as _common
import cognitive_adversarial_cases_completion_evidence as _completion_evidence
import cognitive_adversarial_cases_completion_hostile as _completion_hostile
import cognitive_adversarial_cases_completion_integrity as _completion_integrity
import cognitive_adversarial_cases_concurrency as _concurrency
import cognitive_adversarial_cases_host_receipts as _host_receipts
import cognitive_adversarial_cases_protocol as _protocol
import cognitive_adversarial_cases_randomized as _randomized

hypothesis = _common.hypothesis


class CognitiveAtomicityTests(_atomicity.CognitiveAtomicityTests):
    pass


class CognitiveCompletionContractTests(
    _completion_evidence.CompletionEvidenceMixin,
    _completion_integrity.CompletionIntegrityMixin,
    _completion_hostile.CompletionHostileMixin,
):
    pass


class CognitiveHostReceiptHardeningTests(_host_receipts.CognitiveHostReceiptHardeningTests):
    pass


class CognitiveConcurrencyTests(_concurrency.CognitiveConcurrencyTests):
    pass


class CognitiveProtocolFuzzTests(_protocol.CognitiveProtocolFuzzTests):
    pass


class CognitiveRandomizedStateTests(_randomized.CognitiveRandomizedStateTests):
    pass


if __name__ == "__main__":
    unittest.main()
