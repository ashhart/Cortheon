import unittest
from unittest import mock

from cortheon.connectors.http import ConnectorError, JsonHttpClient


class HttpClientErrorWrappingTests(unittest.TestCase):
    def test_response_size_is_bounded_before_json_decoding(self) -> None:
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.geturl.return_value = "https://example.org/large"
        response.read.return_value = b"x" * 1025
        client = JsonHttpClient(timeout_seconds=1, max_response_bytes=1024)

        with (
            mock.patch.object(client._opener, "open", return_value=response),
            self.assertRaisesRegex(ConnectorError, "exceeded 1024"),
        ):
            client.get_json("https://example.org/large")

        response.read.assert_called_once_with(1025)

    def test_read_timeout_becomes_connector_error(self) -> None:
        client = JsonHttpClient(timeout_seconds=1)
        with (
            mock.patch.object(
                client._opener,
                "open",
                side_effect=TimeoutError("read timed out"),
            ),
            self.assertRaises(ConnectorError) as ctx,
        ):
            client.get("https://example.org/slow")

        self.assertIn("TimeoutError", str(ctx.exception))

    def test_remote_disconnect_becomes_connector_error(self) -> None:
        import http.client

        client = JsonHttpClient(timeout_seconds=1)
        with (
            mock.patch.object(
                client._opener,
                "open",
                side_effect=http.client.RemoteDisconnected("closed"),
            ),
            self.assertRaises(ConnectorError),
        ):
            client.get("https://example.org/drop")

    def test_rate_limit_retries_once_then_wraps(self) -> None:
        import io
        import urllib.error

        def make_429():
            return urllib.error.HTTPError(
                "https://example.org/limited",
                429,
                "Too Many Requests",
                {"Retry-After": "0.5"},  # type: ignore[arg-type]
                io.BytesIO(b""),
            )

        client = JsonHttpClient(timeout_seconds=1)
        with (
            mock.patch.object(
                client._opener,
                "open",
                side_effect=[make_429(), make_429()],
            ) as fake,
            mock.patch("cortheon.connectors.http.time.sleep") as fake_sleep,
            self.assertRaises(ConnectorError) as ctx,
        ):
            client.get("https://example.org/limited")

        self.assertEqual(fake.call_count, 2)
        fake_sleep.assert_called_once_with(0.5)
        self.assertIn("429", str(ctx.exception))

    def test_non_http_and_credentialed_urls_are_rejected(self) -> None:
        client = JsonHttpClient(timeout_seconds=1)

        with self.assertRaisesRegex(ConnectorError, "absolute HTTP"):
            client.get("file:///etc/passwd")
        with self.assertRaisesRegex(ConnectorError, "credentials"):
            client.get("https://user:password@example.org/")


if __name__ == "__main__":
    unittest.main()
