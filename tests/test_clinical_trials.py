import unittest

from cortheon.artifact_assessment import assess_artifacts
from cortheon.clinical_trials import ClinicalTrialsGovDiscovery


class ClinicalTrialsTests(unittest.TestCase):
    def test_clinical_trials_connector_returns_trial_artifact(self) -> None:
        discovery = ClinicalTrialsGovDiscovery(client=FakeClinicalTrialsClient())

        artifacts, evidence, errors = discovery.search("cancer immunotherapy clinical trial", 1)

        self.assertEqual(errors, [])
        self.assertEqual(len(artifacts), 1)
        artifact = artifacts[0]
        self.assertEqual(artifact.kind, "clinical_trial")
        self.assertEqual(artifact.provider, "clinicaltrials_gov")
        self.assertEqual(artifact.metadata["nct_id"], "NCT00000001")
        self.assertEqual(artifact.metadata["overall_status"], "RECRUITING")
        self.assertIn("Pembrolizumab", artifact.metadata["interventions"])
        self.assertEqual(evidence[0].source_type, "clinicaltrials_gov_search")

    def test_clinical_trial_artifact_assessment(self) -> None:
        discovery = ClinicalTrialsGovDiscovery(client=FakeClinicalTrialsClient())
        artifacts, _, _ = discovery.search("cancer immunotherapy clinical trial", 1)

        assessment = assess_artifacts("cancer immunotherapy clinical trial", artifacts)[0]

        self.assertEqual(assessment.decision, "inspect_trial")
        self.assertIn("Trial status is RECRUITING.", assessment.reasons)
        self.assertTrue(any("not proof of efficacy" in risk for risk in assessment.risks))


class FakeClinicalTrialsClient:
    def get_json(self, url):
        return {
            "totalCount": 1,
            "studies": [
                {
                    "protocolSection": {
                        "identificationModule": {
                            "nctId": "NCT00000001",
                            "briefTitle": "Cancer Immunotherapy Trial",
                        },
                        "statusModule": {
                            "overallStatus": "RECRUITING",
                        },
                        "designModule": {
                            "studyType": "INTERVENTIONAL",
                            "phases": ["PHASE2"],
                        },
                        "conditionsModule": {
                            "conditions": ["Cancer"],
                        },
                        "armsInterventionsModule": {
                            "interventions": [
                                {
                                    "name": "Pembrolizumab",
                                }
                            ],
                        },
                        "eligibilityModule": {
                            "eligibilityCriteria": "Adults with eligible cancer.",
                        },
                    }
                }
            ],
        }


if __name__ == "__main__":
    unittest.main()
