import unittest

from cortheon.tasks import find_profile


class TaskProfileTests(unittest.TestCase):
    def test_rest_api_profile_matches(self) -> None:
        profile = find_profile("Build a REST API for a Python service")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "python_rest_api")
        self.assertEqual(profile.candidates[0].package, "fastapi")

    def test_asyncclient_does_not_match_cli_substring(self) -> None:
        profile = find_profile("Use httpx.AsyncClient.fake_stream_now in production")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "async_http_client")
        self.assertEqual(profile.candidates[0].package, "httpx")

    def test_cli_profile_requires_word_boundary(self) -> None:
        profile = find_profile("Build a command line importer")

        self.assertIsNotNone(profile)
        self.assertEqual(profile.name, "cli_app")


if __name__ == "__main__":
    unittest.main()
