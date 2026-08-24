import unittest

from cortheon.web_crawler import (
    HtmlTextExtractor,
    authority_score,
    classify_source,
    normalize_url,
)


class WebCrawlerTests(unittest.TestCase):
    def test_html_extractor_collects_title_text_and_links(self) -> None:
        parser = HtmlTextExtractor("https://example.org/root/")
        parser.feed(
            """
            <html>
              <head><title>Research Page</title><script>ignore()</script></head>
              <body>
                <h1>Artificial life benchmark</h1>
                <a href="/paper">Paper</a>
              </body>
            </html>
            """
        )
        parser.close()

        self.assertEqual(parser.title, "Research Page")
        self.assertIn("Artificial life benchmark", " ".join(parser.text_parts))
        self.assertEqual(parser.links, ["https://example.org/paper"])

    def test_classifies_health_authority_as_high_authority(self) -> None:
        url = "https://www.nih.gov/research-training/example"

        self.assertEqual(
            classify_source(url, "NIH research", "clinical references"), "official_health_authority"
        )
        self.assertGreater(authority_score(url, "NIH research", "clinical references"), 0.9)

    def test_normalize_url_filters_assets_and_fragments(self) -> None:
        self.assertEqual(
            normalize_url("/docs#section", base="https://example.org/start"),
            "https://example.org/docs",
        )
        self.assertIsNone(normalize_url("https://example.org/image.png"))


if __name__ == "__main__":
    unittest.main()
