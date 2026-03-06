from __future__ import annotations

from datetime import datetime
from typing import Any

import feedparser
import requests

ARXIV_API = "https://export.arxiv.org/api/query"


def normalize(text: str) -> str:
    return " ".join((text or "").split())


def search_arxiv(query: str, max_results: int = 20) -> list[dict[str, Any]]:
    params = {
        "search_query": query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }
    headers = {"User-Agent": "paper-digest-agent/0.2"}

    response = requests.get(ARXIV_API, params=params, headers=headers, timeout=30)
    response.raise_for_status()

    feed = feedparser.parse(response.text)
    papers: list[dict[str, Any]] = []

    for entry in feed.entries:
        published = datetime.fromisoformat(entry.published.replace("Z", "+00:00"))

        authors = []
        for author in getattr(entry, "authors", []):
            name = getattr(author, "name", "").strip()
            if name:
                authors.append(name)

        pdf = None
        for link in getattr(entry, "links", []):
            href = getattr(link, "href", None)
            link_type = getattr(link, "type", "")
            title = getattr(link, "title", "")
            if href and (link_type == "application/pdf" or title == "pdf"):
                pdf = href
                break

        papers.append(
            {
                "id": entry.id,
                "title": normalize(entry.title),
                "abstract": normalize(entry.summary),
                "authors": authors,
                "published": published,
                "url": entry.id,
                "pdf": pdf,
                "source": "arXiv",
            }
        )

    return papers
