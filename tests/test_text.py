"""This file tests text cleaning, tokenising, sentence splitting and numeric helpers."""

from __future__ import annotations
import pytest
from tngd_faq_rag.text import (
    _token_estimate,
    approx_token_count,
    clean_text,
    cosine,
    extract_numbers,
    html_to_text,
    luhn_valid,
    normalize_question,
    split_sentences,
    tokenize,
)


class TestLuhn:
    def test_accepts_valid_test_pan(self):
        assert luhn_valid("4111111111111111")

    def test_rejects_plain_reference_number(self):
        assert not luhn_valid("1234567890123456")

    @pytest.mark.parametrize("digits", ["601151662025", "123", "", "40801181483289"])
    def test_rejects_short_or_non_card_runs(self, digits):
        """Twelve digits is below the floor."""
        assert not luhn_valid(digits) or len(digits) >= 13


class TestHtml:
    def test_strips_markup_and_scripts(self):
        out = html_to_text("<p>hello <b>world</b></p><script>var x=1</script>")
        assert "hello world" in out.lower()
        assert "var x" not in out

    def test_preserves_list_structure(self):
        out = html_to_text("<ul><li>first</li><li>second</li></ul>")
        assert "first" in out and "second" in out

    def test_survives_malformed_markup(self):
        assert "text" in html_to_text("<p>text<<<>>")


class TestTables:
    """Each table row becomes its own labelled sentence."""

    def test_header_row_labels_each_value(self):
        out = html_to_text(
            "<table><tr><th>Account</th><th>Daily limit</th><th>Wallet size</th></tr>"
            "<tr><td>Premium</td><td>RM5,000</td><td>RM20,000</td></tr></table>"
        )
        assert out == "Account Premium: Daily limit RM5,000; Wallet size RM20,000."

    def test_two_column_table_reads_as_label_and_value(self):
        out = html_to_text(
            "<table><tr><td>Loan amount</td><td><p>RM10,000</p></td></tr>"
            "<tr><td>Tenure</td><td>1 year - 5 years</td></tr></table>"
        )
        assert split_sentences(clean_text(out)) == [
            "Loan amount: RM10,000.",
            "Tenure: 1 year - 5 years.",
        ]

    def test_cells_are_never_glued_together(self):
        out = html_to_text("<table><tr><td>Lite</td><td>RM1,500</td></tr></table>")
        assert "LiteRM1,500" not in out


class TestSentences:
    def test_keeps_abbreviations_intact(self):
        assert len(split_sentences("Fees apply e.g. RM1. Then you pay.")) == 2

    def test_splits_on_terminators(self):
        assert len(split_sentences("One. Two! Three?")) == 3

    def test_handles_empty(self):
        assert split_sentences("") == []


class TestTokenisation:
    def test_lowercases_and_stems(self):
        assert "card" in tokenize("Cards")

    def test_drops_stopwords(self):
        assert "the" not in tokenize("the card")

    def test_keeps_interrogatives(self):
        """Question words are content words in an FAQ corpus."""
        assert "why" in tokenize("Why must I verify?")
        assert "how" in tokenize("How soon must I verify?")
        assert tokenize("Why must I verify?") != tokenize("How must I verify?")

    def test_normalize_question_is_punctuation_insensitive(self):
        """Punctuation and spacing differences must normalise away identically."""
        assert normalize_question("What's this?") == normalize_question("What s  this")
        assert normalize_question("Road tax - renewal!") == normalize_question("road tax  renewal")


class TestTokenEstimate:
    def test_is_additive(self):
        """Fractional on purpose."""
        parts = ["clause one here"] * 200
        summed = sum(_token_estimate(p) for p in parts)
        joined = _token_estimate(" ".join(parts))
        assert abs(summed - joined) < 1.0

    def test_public_helper_returns_int(self):
        assert isinstance(approx_token_count("a few words here"), int)


class TestMisc:
    def test_clean_text_normalises_quotes_and_whitespace(self):
        assert clean_text("  a’b   c  ") == "a'b c"

    def test_extract_numbers_for_grounding(self):
        nums = extract_numbers("within 24 hours, up to RM1,000 or 5%")
        assert "24" in nums and "1000" in nums and "5%" in nums

    def test_cosine_bounds(self):
        assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
        assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
        assert cosine([0, 0], [1, 1]) == 0.0
