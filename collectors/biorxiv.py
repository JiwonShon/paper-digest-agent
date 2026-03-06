from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

API = "https://api.biorxiv.org/details/biorxiv"

_TOKEN_RE = re.compile(r'all:"[^"]+"|"[^"]+"|\(|\)|\bAND\b|\bOR\b|[^\s()]+', re.IGNORECASE)


def normalize(text: str) -> str:
    return " ".join((text or "").split())


def _clean_term(token: str) -> str:
    token = token.strip()
    if token.startswith("all:"):
        token = token[4:]
    if token.startswith('"') and token.endswith('"'):
        token = token[1:-1]
    return token.strip().lower()


def _tokenize(query: str) -> list[str]:
    return [tok for tok in _TOKEN_RE.findall(query) if tok.strip()]


def _to_rpn(tokens: list[str]) -> list[str]:
    prec = {"OR": 1, "AND": 2}
    output: list[str] = []
    ops: list[str] = []

    for tok in tokens:
        up = tok.upper()
        if up in ("AND", "OR"):
            while ops and ops[-1] != "(" and prec[ops[-1]] >= prec[up]:
                output.append(ops.pop())
            ops.append(up)
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                output.append(ops.pop())
            if ops and ops[-1] == "(":
                ops.pop()
        else:
            output.append(tok)

    while ops:
        output.append(ops.pop())
    return output


def _match_query(text: str, query: str) -> bool:
    text = text.lower()
    tokens = _tokenize(query)
    if not tokens:
        return True

    rpn = _to_rpn(tokens)
    stack: list[bool] = []

    for tok in rpn:
        if tok in ("AND", "OR"):
            if len(stack) < 2:
                return False
            b = stack.pop()
            a = stack.pop()
            stack.append(a and b if tok == "AND" else a or b)
        else:
            term = _clean_term(tok)
            if not term:
                stack.append(True)
            else:
                stack.append(term in text)

    return stack[-1] if stack else True


def search_biorxiv(
    query: str, max_results: int = 20, lookback_hours: int = 168
) -> list[dict[str, Any]]:
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=max(2, lookback_hours // 24 + 2))

    papers: list[dict[str, Any]] = []
    cursor = 0

    while len(papers) < max_results:
        url = f"{API}/{start_date.isoformat()}/{end_date.isoformat()}/{cursor}/json"
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()

        batch = payload.get("collection", [])
        if not batch:
            break

        for item in batch:
            title = normalize(item.get("title", ""))
            abstract = normalize(item.get("abstract", ""))
            haystack = f"{title} {abstract}"

            if not title or not abstract or not _match_query(haystack, query):
                continue

            doi = item.get("doi", "").strip()
            version = item.get("version", "").strip()
            authors = [a.strip() for a in item.get("authors", "").split(";") if a.strip()]
            published = item.get("date")

            papers.append(
                {
                    "id": doi or item.get("rel_doi") or title.lower(),
                    "doi": doi or item.get("rel_doi"),
                    "title": title,
                    "abstract": abstract,
                    "authors": authors,
                    "published": published,
                    "url": f"https://doi.org/{doi}" if doi else "",
                    "pdf": (
                        f"https://www.biorxiv.org/content/10.1101/{doi}v{version}.full.pdf"
                        if doi and version
                        else None
                    ),
                    "source": "bioRxiv",
                }
            )
            if len(papers) >= max_results:
                break

        cursor += 100

    return papers
