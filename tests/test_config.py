"""This file tests settings and when a saved index gets rebuilt."""

from __future__ import annotations
import pytest
import tngd_faq_rag.text as text_module
from tngd_faq_rag.config import Config


def test_fingerprint_is_stable():
    assert Config().fingerprint() == Config().fingerprint()


def test_fingerprint_tracks_chunking_settings():
    a = Config()
    b = Config()
    b.max_chunk_tokens = a.max_chunk_tokens + 64
    assert a.fingerprint() != b.fingerprint()


def test_fingerprint_tracks_the_analysis_vocabulary(monkeypatch):
    """Editing the stopword list changes every token in the corpus."""
    before = Config().fingerprint()
    monkeypatch.setattr(text_module, "STOPWORDS", frozenset(text_module.STOPWORDS | {"zzz"}))
    monkeypatch.setattr("tngd_faq_rag.config.STOPWORDS", text_module.STOPWORDS)
    assert Config().fingerprint() != before


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("TNGD_ABSTAIN_THRESHOLD", "0.55")
    assert Config().abstain_threshold == 0.55


class TestApiKeyResolution:
    """Where an API key may come from, and which endpoint that implies."""

    KEY_VARS = ("LLM_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "LLM_API_BASE")

    @pytest.fixture(autouse=True)
    def _clean(self, monkeypatch):
        for name in self.KEY_VARS:
            monkeypatch.delenv(name, raising=False)

    def test_own_variable_is_read(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "own")
        assert Config().api_key == "own"

    @pytest.mark.parametrize("name", ["GEMINI_API_KEY", "GOOGLE_API_KEY"])
    def test_google_variables_are_accepted(self, monkeypatch, name):
        monkeypatch.setenv(name, "google-key")
        assert Config().api_key == "google-key"

    def test_our_variable_wins_over_googles(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "ours")
        monkeypatch.setenv("GEMINI_API_KEY", "theirs")
        assert Config().api_key == "ours"

    def test_gemini_variable_selects_the_gemini_endpoint(self, monkeypatch):
        """Not a convenience - a credential-disclosure guard."""
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        assert "generativelanguage.googleapis.com" in Config().api_base

    def test_our_variable_does_not_assume_a_provider(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "own")
        assert Config().api_base == "https://api.openai.com/v1"

    def test_explicit_base_always_wins(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "g")
        monkeypatch.setenv("LLM_API_BASE", "http://localhost:11434/v1")
        assert Config().api_base == "http://localhost:11434/v1"

    def test_no_key_configured(self):
        assert Config().api_key == ""

    def test_blank_variable_is_treated_as_unset(self, monkeypatch):
        monkeypatch.setenv("LLM_API_KEY", "   ")
        monkeypatch.setenv("GEMINI_API_KEY", "real")
        assert Config().api_key == "real"

    def test_llm_is_configured_honours_every_variable(self, monkeypatch):
        from tngd_faq_rag.llm import llm_is_configured

        assert not llm_is_configured()
        monkeypatch.setenv("GOOGLE_API_KEY", "x")
        assert llm_is_configured()
