from __future__ import annotations

from html.parser import HTMLParser
from typing import ClassVar

from cortheon.docs_reader_core._compat import facade


class DocsHtmlParser(HTMLParser):
    """HTML parser tuned for documentation sites.

    Unlike the crawler's extractor it preserves <pre> blocks verbatim so code
    examples survive with their newlines. Noise tags (script/style) drop
    everything; chrome tags (nav/footer/aside) drop text but keep link hrefs,
    because MkDocs/Sphinx sites put their table of contents inside <nav>.
    """

    NOISE_TAGS: ClassVar[set[str]] = {"script", "style", "noscript", "svg"}
    CHROME_TAGS: ClassVar[set[str]] = {"nav", "footer", "aside"}
    HEADING_TAGS: ClassVar[set[str]] = {"h1", "h2", "h3"}

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.title: str | None = None
        self.headings: list[str] = []
        self.text_parts: list[str] = []
        self.code_blocks: list[str] = []
        self.links: list[tuple[str, str]] = []
        self._noise_depth = 0
        self._chrome_depth = 0
        self._pre_depth = 0
        self._pre_parts: list[str] = []
        self._in_title = False
        self._title_parts: list[str] = []
        self._heading_parts: list[str] | None = None
        self._link_href: str | None = None
        self._link_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.NOISE_TAGS:
            self._noise_depth += 1
            return
        if self._noise_depth:
            return
        if tag in self.CHROME_TAGS:
            self._chrome_depth += 1
        if tag == "a":
            href = dict(attrs).get("href")
            self._link_href = facade().normalize_url(href, base=self.base_url) if href else None
            self._link_parts = []
            return
        if self._chrome_depth:
            return
        if tag == "pre":
            self._pre_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag in self.HEADING_TAGS:
            self._heading_parts = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in self.NOISE_TAGS:
            self._noise_depth = max(0, self._noise_depth - 1)
            return
        if self._noise_depth:
            return
        if tag == "a":
            if self._link_href:
                self.links.append(
                    (self._link_href, facade().normalize_space(" ".join(self._link_parts))[:120])
                )
            self._link_href = None
            self._link_parts = []
            return
        if tag in self.CHROME_TAGS:
            self._chrome_depth = max(0, self._chrome_depth - 1)
            return
        if self._chrome_depth:
            return
        if tag == "pre" and self._pre_depth:
            self._pre_depth -= 1
            if not self._pre_depth:
                block = "".join(self._pre_parts).strip()
                self._pre_parts = []
                if block:
                    self.code_blocks.append(block)
        elif tag == "title":
            self._in_title = False
            self.title = facade().normalize_space(" ".join(self._title_parts))[:240] or None
        elif tag in self.HEADING_TAGS and self._heading_parts is not None:
            heading = facade().normalize_space(" ".join(self._heading_parts))
            self._heading_parts = None
            if heading:
                self.headings.append(heading[:160])
                self.text_parts.append(heading)

    def handle_data(self, data: str) -> None:
        if self._noise_depth:
            return
        if self._link_href is not None and data.strip():
            self._link_parts.append(data.strip())
        if self._chrome_depth:
            return
        if self._pre_depth:
            # Syntax highlighters split code into per-token spans; fragments
            # must join verbatim so only the page's own newlines survive.
            self._pre_parts.append(data)
            return
        if not data.strip():
            return
        cleaned = data.strip()
        if self._in_title:
            self._title_parts.append(cleaned)
            return
        if self._heading_parts is not None:
            self._heading_parts.append(cleaned)
        self.text_parts.append(cleaned)
