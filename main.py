from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from openai import OpenAI

from collectors.arxiv import search_arxiv
from collectors.biorxiv import search_biorxiv
from collectors.pubmed import search_pubmed

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "keywords.json"
SEEN_PATH = ROOT / "data" / "seen_papers.json"
OUTPUTS_DIR = ROOT / "outputs"

SEOUL_TZ = ZoneInfo("Asia/Seoul")
UTC = timezone.utc


@dataclass
class Paper:
    paper_id: str
    title: str
    abstract: str
    authors: list[str]
    published: datetime
    abs_url: str
    pdf_url: str | None
    source: str
    matched_keywords: list[str]

    @property
    def author_text(self) -> str:
        if not self.authors:
            return "N/A"
        if len(self.authors) <= 6:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:6]) + ", et al."


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        text = text.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
            return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            pass

        # YYYY-MM-DD
        try:
            dt = datetime.strptime(text[:10], "%Y-%m-%d")
            return dt.replace(tzinfo=UTC)
        except ValueError:
            return None
    return None


def load_seen_state() -> dict[str, Any]:
    raw = load_json(SEEN_PATH, {"papers": {}, "updated_at": None})
    papers = raw.get("papers", {})
    if isinstance(papers, list):
        papers = {paper_id: datetime.now(SEOUL_TZ).date().isoformat() for paper_id in papers}
    return {"updated_at": raw.get("updated_at"), "papers": papers}


def prune_seen_state(state: dict[str, Any], keep_days: int) -> dict[str, Any]:
    cutoff = datetime.now(SEOUL_TZ).date() - timedelta(days=keep_days)
    pruned: dict[str, str] = {}

    for paper_id, seen_on in state.get("papers", {}).items():
        try:
            seen_date = datetime.fromisoformat(seen_on).date()
        except ValueError:
            continue
        if seen_date >= cutoff:
            pruned[paper_id] = seen_on

    return {"updated_at": datetime.now(SEOUL_TZ).isoformat(), "papers": pruned}


def canonical_paper_id(item: dict[str, Any]) -> str | None:
    for key in ("doi", "id", "url", "title"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def to_paper(item: dict[str, Any], label: str) -> Paper | None:
    title = normalize_text(item.get("title", ""))
    abstract = normalize_text(item.get("abstract", ""))
    published = parse_datetime(item.get("published"))
    paper_id = canonical_paper_id(item)

    if not title or not abstract or published is None or not paper_id:
        return None

    authors = item.get("authors") or []
    if isinstance(authors, str):
        authors = [a.strip() for a in authors.split(";") if a.strip()]
    if not isinstance(authors, list):
        authors = []

    return Paper(
        paper_id=paper_id,
        title=title,
        abstract=abstract,
        authors=authors,
        published=published,
        abs_url=item.get("url", ""),
        pdf_url=item.get("pdf"),
        source=item.get("source", "unknown"),
        matched_keywords=[label],
    )


def collect_papers(config: dict[str, Any]) -> list[Paper]:
    lookback_hours = int(config.get("lookback_hours", 168))
    max_results_per_keyword = int(config.get("max_results_per_keyword", 20))
    sleep_seconds = float(config.get("sleep_seconds", 1.5))
    seen_keep_days = int(config.get("seen_keep_days", 90))
    cutoff = datetime.now(UTC) - timedelta(hours=lookback_hours)

    seen_state = prune_seen_state(load_seen_state(), seen_keep_days)
    seen_ids = set(seen_state["papers"].keys())

    by_id: dict[str, Paper] = {}

    for keyword in config.get("keywords", []):
        label = keyword["label"]
        query = keyword["query"]

        results: list[dict[str, Any]] = []

        try:
            results.extend(search_arxiv(query=query, max_results=max_results_per_keyword))
        except Exception as e:
            print(f"[WARN] arXiv failed for {label}: {e}")

        try:
            results.extend(
                search_biorxiv(
                    query=query,
                    max_results=max_results_per_keyword,
                    lookback_hours=lookback_hours,
                )
            )
        except Exception as e:
            print(f"[WARN] bioRxiv failed for {label}: {e}")

        try:
            results.extend(search_pubmed(query=query, max_results=max_results_per_keyword))
        except Exception as e:
            print(f"[WARN] PubMed failed for {label}: {e}")

        for item in results:
            paper = to_paper(item, label)
            if paper is None:
                continue
            if paper.published < cutoff:
                continue
            if paper.paper_id in seen_ids:
                continue

            existing = by_id.get(paper.paper_id)
            if existing is None:
                by_id[paper.paper_id] = paper
            else:
                if label not in existing.matched_keywords:
                    existing.matched_keywords.append(label)
                # Prefer non-arXiv URL/PDF if current entry has them
                if existing.source == "arXiv" and paper.source != "arXiv":
                    existing.source = paper.source
                    existing.abs_url = paper.abs_url or existing.abs_url
                    existing.pdf_url = paper.pdf_url or existing.pdf_url
                if len(paper.abstract) > len(existing.abstract):
                    existing.abstract = paper.abstract
                if len(paper.authors) > len(existing.authors):
                    existing.authors = paper.authors
                if paper.published > existing.published:
                    existing.published = paper.published

        time.sleep(sleep_seconds)

    papers = list(by_id.values())
    papers.sort(
        key=lambda p: (len(p.matched_keywords), p.published.timestamp()),
        reverse=True,
    )
    return papers


def summarize_paper(client: OpenAI, model: str, paper: Paper) -> str:
    prompt = f"""
당신은 생명과학 논문 브리핑을 작성하는 리서치 어시스턴트다.
반드시 아래 title과 abstract만 근거로 한국어로 요약하라.
abstract에 없는 실험 결과, 수치, 장점, 한계를 추정해서 쓰지 마라.
출력 형식은 정확히 아래 bullet 5개만 사용한다.

- 한줄요약: ...
- 핵심문제: ...
- 접근법: ...
- 핵심기여: ...
- 연구메모: ...

Title: {paper.title}
Abstract: {paper.abstract}
""".strip()

    response = client.responses.create(model=model, input=prompt)
    return response.output_text.strip()


def fallback_summary(paper: Paper, error: Exception) -> str:
    abstract_preview = paper.abstract[:500] + ("..." if len(paper.abstract) > 500 else "")
    return "\n".join(
        [
            f"- 한줄요약: 요약 생성 실패 ({type(error).__name__})",
            f"- 핵심문제: 원문 abstract를 확인하세요.",
            f"- 접근법: 원문 abstract를 확인하세요.",
            f"- 핵심기여: 원문 abstract를 확인하세요.",
            f"- 연구메모: {abstract_preview}",
        ]
    )


def render_digest_markdown(
    config: dict[str, Any], papers: list[Paper], summaries: dict[str, str]
) -> str:
    now_kst = datetime.now(SEOUL_TZ)
    title_date = now_kst.strftime("%Y-%m-%d")

    lines: list[str] = [
        f"# Daily Paper Digest - {title_date}",
        "",
        "## Tracking keywords",
        "",
    ]

    for item in config.get("keywords", []):
        lines.append(f"- **{item['label']}**: `{item['query']}`")

    lines.extend(["", "## Selected papers", ""])

    if not papers:
        lines.append("지난 탐색 구간에 새로 잡힌 논문이 없습니다.")
        return "\n".join(lines) + "\n"

    for idx, paper in enumerate(papers, start=1):
        published_text = paper.published.astimezone(SEOUL_TZ).strftime("%Y-%m-%d %H:%M KST")
        keyword_text = ", ".join(sorted(paper.matched_keywords))

        lines.extend(
            [
                f"### {idx}. {paper.title}",
                "",
                f"- Source: {paper.source}",
                f"- Matched keywords: {keyword_text}",
                f"- Published: {published_text}",
                f"- Authors: {paper.author_text}",
                f"- Link: {paper.abs_url}",
                f"- PDF: {paper.pdf_url or 'N/A'}",
                "",
                summaries[paper.paper_id],
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def update_seen_state(papers: list[Paper], keep_days: int) -> None:
    state = prune_seen_state(load_seen_state(), keep_days)
    today = datetime.now(SEOUL_TZ).date().isoformat()

    for paper in papers:
        state["papers"][paper.paper_id] = today

    state["updated_at"] = datetime.now(SEOUL_TZ).isoformat()
    save_json(SEEN_PATH, state)


def write_outputs(content: str) -> None:
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    today_name = datetime.now(SEOUL_TZ).strftime("%Y-%m-%d") + ".md"
    dated_path = OUTPUTS_DIR / today_name
    latest_path = OUTPUTS_DIR / "latest.md"

    dated_path.write_text(content, encoding="utf-8")
    latest_path.write_text(content, encoding="utf-8")


def main() -> None:
    config = load_json(CONFIG_PATH, {})
    papers = collect_papers(config)
    top_n = int(config.get("top_n", 8))
    selected = papers[:top_n]

    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
    api_key = os.getenv("OPENAI_API_KEY")

    summaries: dict[str, str] = {}

    if api_key:
        client = OpenAI(api_key=api_key)
        for paper in selected:
            try:
                summaries[paper.paper_id] = summarize_paper(client, model, paper)
            except Exception as e:
                print(f"[WARN] summary failed for {paper.title}: {e}")
                summaries[paper.paper_id] = fallback_summary(paper, e)
    else:
        for paper in selected:
            summaries[paper.paper_id] = fallback_summary(
                paper, RuntimeError("OPENAI_API_KEY not set")
            )

    digest = render_digest_markdown(config, selected, summaries)
    write_outputs(digest)
    update_seen_state(selected, keep_days=int(config.get("seen_keep_days", 90)))


if __name__ == "__main__":
    main()
