"""This file tests that answers are checked against their sources."""

from __future__ import annotations
import pytest
from tngd_faq_rag.guardrails import GroundingChecker

SOURCE = [
    "Reload your eWallet within 24 hours and any outstanding SOS Balance "
    "will be automatically deducted. There are no additional fees."
]


@pytest.fixture
def checker(system) -> GroundingChecker:
    return system.grounding


def test_supported_sentence_passes(checker):
    assert checker.check("Reload your eWallet within 24 hours.", SOURCE)["grounded"]


def test_unsupported_sentence_fails(checker):
    assert not checker.check("Napoleon was crowned emperor in 1804.", SOURCE)["grounded"]


def test_fabricated_figure_fails_numeric_check(checker):
    """A payments FAQ cannot tolerate an invented deadline."""
    assert not checker.check("Reload your eWallet within 7 days.", SOURCE)["grounded"]


def test_percentages_and_amounts_are_checked(checker):
    src = ["A 2% fee applies to each transfer."]
    assert checker.check("A 2% fee applies.", src)["grounded"]
    assert not checker.check("A 15% fee applies.", src)["grounded"]


def test_no_sources_means_not_grounded(checker):
    assert not checker.check("Anything at all.", [])["grounded"]


def test_citations_point_at_the_supporting_source(checker):
    sources = ["Completely unrelated content about trains.", SOURCE[0]]
    result = checker.check("Reload your eWallet within 24 hours.", sources)
    assert result["grounded"]
    assert result["citations"][0]["source_index"] == 1
