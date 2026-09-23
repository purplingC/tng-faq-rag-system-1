"""This file loads the FAQ, cleans each record and removes duplicates."""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from .config import Config
from .constants import TNGD_FAQ_URL
from .logging_utils import LOG
from .models import FaqDoc, IngestionReport
from .resources import load_seed_records
from .text import clean_text, html_to_text, normalize_question, sha1


def _coerce_record(raw: Any) -> dict[str, str] | None:
    """Accept the several shapes an FAQ record arrives in and normalise them."""
    if not isinstance(raw, dict):
        return None
    q = clean_text(raw.get("question") or raw.get("title") or raw.get("q") or "")
    a_raw = raw.get("answer") or raw.get("body") or raw.get("a") or ""
    a = clean_text(html_to_text(a_raw) if "<" in str(a_raw) else a_raw)
    url = clean_text(raw.get("url") or raw.get("html_url") or raw.get("source") or "")
    cat = clean_text(raw.get("category") or raw.get("section") or raw.get("topic") or "General")
    if not q or not a:
        return None
    return {"question": q, "answer": a, "url": url or TNGD_FAQ_URL, "category": cat or "General"}


def load_documents(
    cfg: Config, path: Path | None = None, use_seed: bool = False, language: str = "en"
) -> tuple[list[FaqDoc], IngestionReport]:
    """Load the KB: explicit path > scraped file > embedded seed."""
    report = IngestionReport(
        source="", raw=0, accepted=0, rejected_empty=0, rejected_dupe=0, truncated=0
    )

    raw_records: list[Any]
    candidate = None if use_seed else (path or (cfg.data_dir / cfg.kb_file))
    if candidate and Path(candidate).exists():
        with open(candidate, encoding="utf-8") as fh:
            raw_records = json.load(fh)
        report["source"] = str(candidate)
    else:
        raw_records = load_seed_records(language)
        report["source"] = "packaged seed KB"

    if isinstance(raw_records, dict):  # tolerate {"articles": [...]} wrappers
        for key in ("articles", "records", "data", "faq", "items"):
            if isinstance(raw_records.get(key), list):
                raw_records = raw_records[key]
                break
    if not isinstance(raw_records, list):
        raise ValueError(f"KB at {report['source']} is not a list of records")

    report["raw"] = len(raw_records)
    seen: dict[str, str] = {}
    docs: list[FaqDoc] = []

    for raw in raw_records:
        rec = _coerce_record(raw)
        if rec is None:
            report["rejected_empty"] += 1
            continue
        # Dedupe on normalised question and url, since articles repeat across sections
        key = sha1(normalize_question(rec["question"]), rec["url"])
        if key in seen:
            report["rejected_dupe"] += 1
            continue
        seen[key] = rec["question"]
        doc_id = sha1(rec["url"], normalize_question(rec["question"]))[:16]
        docs.append(FaqDoc(doc_id=doc_id, **rec))

    report["accepted"] = len(docs)
    if not docs:
        raise ValueError(f"No usable FAQ records found in {report['source']}")
    LOG.info(
        "Ingested %d docs from %s (raw=%d, empty=%d, dupes=%d)",
        report["accepted"],
        report["source"],
        report["raw"],
        report["rejected_empty"],
        report["rejected_dupe"],
    )
    return docs, report
