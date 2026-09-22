"""This file tests how the scraper parses pages, without using the network."""

from __future__ import annotations
import json
import pytest
from tngd_faq_rag.ingestion import _coerce_record
from tngd_faq_rag import scraper
from tngd_faq_rag.scraper import kb_filename, write_kb
from tngd_faq_rag.text import html_to_text

ZENDESK_ARTICLE = {
    "id": 123,
    "title": "What is TNG eWallet SOS Balance?",
    "body": "<p>SOS Balance is a feature for <b>toll payments</b>.</p><ul><li>No fees</li></ul>",
    "html_url": "https://support.tngdigital.com.my/hc/en-my/articles/123-What-is-SOS",
    "section_id": 456,
    "draft": False,
}


class TestArticleMapping:
    def test_maps_zendesk_fields_to_the_required_schema(self):
        record = _coerce_record(ZENDESK_ARTICLE)
        assert record["question"] == ZENDESK_ARTICLE["title"]
        assert record["url"] == ZENDESK_ARTICLE["html_url"]
        assert "toll payments" in record["answer"]

    def test_strips_html_but_keeps_list_content(self):
        text = html_to_text(ZENDESK_ARTICLE["body"])
        assert "No fees" in text
        assert "<b>" not in text


class TestOutput:
    def test_writes_json_and_csv(self, cfg, tmp_path):
        cfg.data_dir = tmp_path
        records = [{"question": "Q?", "answer": "A.", "url": "https://e.test/1", "category": "C"}]
        path = write_kb(records, cfg, also_csv=True)
        assert json.loads(path.read_text(encoding="utf-8")) == records
        assert path.with_suffix(".csv").exists()

    def test_each_language_gets_its_own_file(self, cfg, tmp_path):
        cfg.data_dir = tmp_path
        records = [{"question": "Q?", "answer": "A.", "url": "https://e.test/1", "category": "C"}]
        assert kb_filename(cfg, "en-my") == "tngd_faq.json"
        assert kb_filename(cfg, "ms-my") == "tngd_faq_ms.json"
        assert kb_filename(cfg, "zh-my") == "tngd_faq_zh.json"
        assert write_kb(records, cfg, also_csv=False, locale="ms-my").name == "tngd_faq_ms.json"

    def test_output_is_loadable_as_a_knowledge_base(self, cfg, tmp_path):
        from tngd_faq_rag.ingestion import load_documents

        cfg.data_dir = tmp_path
        records = [
            {"question": "Q?", "answer": "An answer.", "url": "https://e.test/1", "category": "C"}
        ]
        path = write_kb(records, cfg, also_csv=False)
        docs, _ = load_documents(cfg, path)
        assert len(docs) == 1


class TestLocales:
    """The scraper fetches any help centre language into its own file."""

    def test_requests_use_the_chosen_language(self, cfg, monkeypatch):
        urls: list[str] = []
        malay_article = {
            **ZENDESK_ARTICLE,
            "html_url": ZENDESK_ARTICLE["html_url"].replace("en-my", "ms-my"),
        }

        def fake_get(url, cfg, retries=3):
            urls.append(url)
            if "sections.json" in url:
                return {"sections": [{"id": 456, "name": "Baki SOS"}]}
            return {"articles": [malay_article]}

        monkeypatch.setattr(scraper, "_http_get_json", fake_get)
        records = scraper.scrape_tngd_faq(cfg, locale="ms-my")
        assert urls and all("/help_center/ms-my/" in u for u in urls)
        assert records[0]["category"] == "Baki SOS"

    def test_unknown_language_is_rejected(self, cfg):
        with pytest.raises(ValueError):
            scraper.scrape_tngd_faq(cfg, locale="fr-fr")


@pytest.mark.network
def test_live_help_centre_is_reachable():
    """Opt in with: pytest -m network"""
    from tngd_faq_rag.config import Config
    from tngd_faq_rag.scraper import scrape_tngd_faq

    records = scrape_tngd_faq(Config(), max_articles=5)
    assert len(records) == 5
    assert all(r["url"].startswith("https://support.tngdigital.com.my/") for r in records)
