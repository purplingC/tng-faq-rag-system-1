"""This file tests FAQ loading, record cleaning and duplicate removal."""

from __future__ import annotations
import json
from pathlib import Path
import pytest
from tngd_faq_rag.ingestion import _coerce_record, load_documents
from tngd_faq_rag.resources import load_seed_records


class TestSeedCorpus:
    def test_ships_with_the_package(self):
        records = load_seed_records()
        assert len(records) == 30

    def test_matches_the_assignment_schema(self):
        for record in load_seed_records():
            assert set(record) == {"question", "answer", "url", "category"}
            assert all(record.values())

    def test_urls_point_at_the_official_help_centre(self):
        for record in load_seed_records():
            assert record["url"].startswith("https://support.tngdigital.com.my/")


class TestCoercion:
    def test_accepts_zendesk_field_names(self):
        record = _coerce_record(
            {
                "title": "Q?",
                "body": "<p>An answer.</p>",
                "html_url": "https://example.test/a",
                "section": "Cat",
            }
        )
        assert record["question"] == "Q?"
        assert record["answer"] == "An answer."

    def test_rejects_records_without_content(self):
        assert _coerce_record({"question": "Q?", "answer": "   "}) is None
        assert _coerce_record("not a dict") is None

    def test_defaults_missing_category_and_url(self):
        record = _coerce_record({"question": "Q?", "answer": "A."})
        assert record["category"] == "General"
        assert record["url"].startswith("https://")


class TestLoading:
    def test_falls_back_to_the_seed_corpus(self, cfg):
        docs, report = load_documents(cfg, use_seed=True)
        assert len(docs) == 30
        assert report["source"] == "packaged seed KB"
        assert report["rejected_empty"] == 0

    def test_reads_an_explicit_file(self, cfg, tmp_path):
        path = tmp_path / "kb.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "question": "Q1?",
                        "answer": "A1.",
                        "url": "https://e.test/1",
                        "category": "C",
                    },
                    {
                        "question": "Q2?",
                        "answer": "A2.",
                        "url": "https://e.test/2",
                        "category": "C",
                    },
                ]
            ),
            encoding="utf-8",
        )
        docs, report = load_documents(cfg, path)
        assert len(docs) == 2
        assert str(path) in report["source"]

    def test_deduplicates_republished_articles(self, cfg, tmp_path):
        """The live help centre publishes the same article under several sections."""
        entry = {"question": "Q?", "answer": "A.", "url": "https://e.test/1", "category": "C"}
        path = tmp_path / "dupes.json"
        path.write_text(
            json.dumps([entry, dict(entry), dict(entry, category="Other")]), encoding="utf-8"
        )
        docs, report = load_documents(cfg, path)
        assert len(docs) == 1
        assert report["rejected_dupe"] == 2

    def test_empty_knowledge_base_is_an_error(self, cfg, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError):
            load_documents(cfg, path)


class TestCommittedKnowledgeBase:
    """The scraped FAQ ships in the repo, so a fresh clone answers real questions."""

    PATH = Path(__file__).resolve().parents[1] / "data" / "tngd_faq.json"

    def test_is_complete_and_well_formed(self, cfg):
        if not self.PATH.exists():
            pytest.skip("data/tngd_faq.json is not present")
        docs, report = load_documents(cfg, self.PATH)
        assert len(docs) >= 2000
        assert report["rejected_empty"] == 0
        assert all(d.url.startswith("https://support.tngdigital.com.my/") for d in docs)
        assert all(d.question and d.answer and d.category for d in docs)
