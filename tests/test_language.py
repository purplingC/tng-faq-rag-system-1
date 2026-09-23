"""This file tests the Malay and English language guess."""

from __future__ import annotations
import pytest
from tngd_faq_rag.language import detect_language


class TestMalay:
    @pytest.mark.parametrize(
        "question",
        [
            "Macam mana nak reload eWallet?",
            "Apakah itu SOS Balance?",
            "Bagaimanakah cara menyemak baki saya?",
            "Boleh saya guna eWallet di luar negara?",
            "Berapakah harga tag RFID?",
            "Adakah terdapat had umur?",
        ],
    )
    def test_malay_questions(self, question):
        assert detect_language(question) == "ms"

    def test_kah_suffix_is_enough(self):
        # Bagaimanakah is not in the marker list, the suffix rule catches it
        assert detect_language("Bagaimanakah pembayaran diproses?") == "ms"


class TestEnglish:
    @pytest.mark.parametrize(
        "question",
        [
            "What is TNG eWallet SOS Balance?",
            "How do I reload my eWallet?",
            "Can I use my card overseas?",
            "Why must I verify my saved cards?",
        ],
    )
    def test_english_questions(self, question):
        assert detect_language(question) == "en"

    def test_unknown_wording_defaults_to_english(self):
        # A bare product name belongs to no language, so the default applies
        assert detect_language("CardMatch") == "en"
        assert detect_language("reload limit") == "en"

    def test_empty_input(self):
        assert detect_language("") == "en"
        assert detect_language("   ") == "en"
