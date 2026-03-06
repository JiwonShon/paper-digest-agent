from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree as ET

import requests

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def normalize(text: str) -> str:
    return " ".join((text or "").split())


def _clean_query(query: str) -> str:
    return query.replace("all:", "")


def _parse_month(text: str | None) -> int:
    if not text:
        return 1
    text = text.strip()
    if text.isdigit():
        return max(1, min(12, int(text)))
    return MONTHS.get(text[:3].lower(), 1)


def _node_text(node: ET.Element | None) -> str:
    if node is None:
        return ""
    return "".join(node.itertext()).strip()


def _parse_pub_date(article: ET.Element) -> datetime:
    candidates = [
        article.find(".//PubMedPubDate[@PubStatus='pubmed']"),
        article.find(".//ArticleDate"),
        article.find(".//JournalIssue/PubDate"),
    ]

    for node in candidates:
        if node is None:
            continue
        year = _node_text(node.find("Year"))
        if not year.isdigit():
            continue
        month = _parse_month(_node_text(node.find("Month")))
        day_text = _node_text(node.find("Day"))
        day = int(day_text) if day_text.isdigit() else 1
        try:
            return datetime(int(year), month, day, tzinfo=timezone.utc)
        except ValueError:
            continue

    return datetime.now(timezone.utc)


def search_pubmed(query: str, max_results: int = 20) -> list[dict[str, Any]]:
    params = {
        "db": "pubmed",
        "term": _clean_query(query),
        "retmax": max_results,
        "retmode": "json",
        "sort": "pub date",
    }
    search = requests.get(BASE + "esearch.fcgi", params=params, timeout=30)
    search.raise_for_status()

    id_list = search.json().get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    fetch = requests.get(
        BASE + "efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(id_list), "retmode": "xml"},
        timeout=30,
    )
    fetch.raise_for_status()

    root = ET.fromstring(fetch.text)
    papers: list[dict[str, Any]] = []

    for article in root.findall(".//PubmedArticle"):
        pmid = _node_text(article.find(".//PMID"))
        title = normalize(_node_text(article.find(".//ArticleTitle")))
        abstract_nodes = article.findall(".//Abstract/AbstractText")
        abstract = normalize(" ".join(_node_text(n) for n in abstract_nodes))
        authors = []

        for author in article.findall(".//AuthorList/Author"):
            last = _node_text(author.find("LastName"))
            fore = _node_text(author.find("ForeName"))
            collective = _node_text(author.find("CollectiveName"))
            name = " ".join(x for x in [fore, last] if x).strip() or collective
            if name:
                authors.append(name)

        doi = ""
        for article_id in article.findall(".//ArticleId"):
            if article_id.attrib.get("IdType") == "doi":
                doi = (article_id.text or "").strip()
                break

        papers.append(
            {
                "id": pmid or doi or title.lower(),
                "doi": doi or None,
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "published": _parse_pub_date(article),
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else "",
                "pdf": None,
                "source": "PubMed",
            }
        )

    return papers
