"""This file scrapes the public TNG Digital help centre, in English, Malay and Chinese."""

from __future__ import annotations
import csv
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from .config import Config
from .constants import TNGD_FAQ_CATEGORY_ID, TNGD_FAQ_URL, TNGD_HELP_BASE
from .logging_utils import LOG
from .text import clean_text, html_to_text, normalize_question, sha1


class ScrapeError(RuntimeError):
    pass


def _http_get_json(url: str, cfg: Config, retries: int = 3) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": cfg.user_agent,
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # pragma: no cover - network dependent
            last = exc
            sleep = cfg.scrape_delay * (2**attempt)
            LOG.warning("GET %s failed (%s); retrying in %.1fs", url, exc, sleep)
            time.sleep(sleep)
    raise ScrapeError(f"GET {url} failed after {retries} attempts: {last}")


# Help centre languages, only English feeds the knowledge base the system answers from
LOCALES = ("en-my", "ms-my", "zh-my")


def kb_filename(cfg: Config, locale: str) -> str:
    """File a language is saved to: tngd_faq.json for English, tngd_faq_ms.json for Malay."""
    if locale == "en-my":
        return cfg.kb_file
    stem, _, ext = cfg.kb_file.rpartition(".")
    return f"{stem}_{locale.split('-')[0]}.{ext}"


def scrape_tngd_faq(
    cfg: Config, *, max_articles: int = 0, locale: str = "en-my"
) -> list[dict[str, str]]:
    """Fetch the public TNGD FAQ category, in one language, into the KB schema."""
    if locale not in LOCALES:
        raise ValueError(f"unknown locale {locale!r}, expected one of {LOCALES}")
    base = f"{TNGD_HELP_BASE}/api/v2/help_center/{locale}"

    LOG.info("Fetching section names ...")
    sections: dict[int, str] = {}
    page = 1
    while True:
        data = _http_get_json(
            f"{base}/categories/{TNGD_FAQ_CATEGORY_ID}/sections.json"
            f"?per_page={cfg.scrape_page_size}&page={page}",
            cfg,
        )
        for sec in data.get("sections", []):
            sections[int(sec["id"])] = clean_text(sec.get("name", "")) or "General"
        if not data.get("next_page"):
            break
        page += 1
        time.sleep(cfg.scrape_delay)
    LOG.info("Found %d sections", len(sections))

    records: list[dict[str, str]] = []
    seen: set = set()
    page = 1
    limit = max_articles or cfg.scrape_max_articles
    while True:
        data = _http_get_json(
            f"{base}/categories/{TNGD_FAQ_CATEGORY_ID}/articles.json"
            f"?per_page={cfg.scrape_page_size}&page={page}",
            cfg,
        )
        articles = data.get("articles", [])
        if not articles:
            break
        for art in articles:
            if art.get("draft"):
                continue
            question = clean_text(art.get("title", ""))
            answer = clean_text(html_to_text(art.get("body") or ""))
            url = clean_text(art.get("html_url", "")) or TNGD_FAQ_URL
            if not question or not answer or len(answer) < 20:
                continue
            key = sha1(normalize_question(question), url)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "question": question,
                    "answer": answer,
                    "url": url,
                    "category": sections.get(int(art.get("section_id") or 0), "General"),
                }
            )
            if limit and len(records) >= limit:
                break
        LOG.info("page %d/%s -> %d records so far", page, data.get("page_count", "?"), len(records))
        if (limit and len(records) >= limit) or not data.get("next_page"):
            break
        page += 1
        time.sleep(cfg.scrape_delay)

    if not records:
        raise ScrapeError("scraper produced no records - the API shape may have changed")
    LOG.info("Scraped %d FAQ articles", len(records))
    return records


def write_kb(
    records: Sequence[dict[str, str]],
    cfg: Config,
    *,
    also_csv: bool = True,
    locale: str = "en-my",
) -> Path:
    """Save records as JSON, plus CSV unless turned off, to the file for their language."""
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    json_path = cfg.data_dir / kb_filename(cfg, locale)
    json_path.write_text(json.dumps(list(records), ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Wrote %s (%d records)", json_path, len(records))
    if also_csv:
        csv_path = json_path.with_suffix(".csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=["question", "answer", "url", "category"])
            writer.writeheader()
            writer.writerows(records)
        LOG.info("Wrote %s", csv_path)
    return json_path
