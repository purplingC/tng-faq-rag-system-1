"""This file tests the .env loader."""

from __future__ import annotations
import os
import pytest
from tngd_faq_rag.env import load_dotenv


@pytest.fixture
def env_file(tmp_path):
    def write(content: str):
        path = tmp_path / ".env"
        path.write_text(content, encoding="utf-8")
        return path

    return write


class TestParsing:
    def test_loads_simple_pairs(self, env_file, monkeypatch):
        monkeypatch.delenv("TNGD_TEST_A", raising=False)
        load_dotenv(env_file("TNGD_TEST_A=hello\n"))
        assert os.environ["TNGD_TEST_A"] == "hello"

    def test_ignores_comments_and_blank_lines(self, env_file, monkeypatch):
        monkeypatch.delenv("TNGD_TEST_B", raising=False)
        applied = load_dotenv(env_file("# a comment\n\n   \nTNGD_TEST_B=v\n"))
        assert applied == {"TNGD_TEST_B": "v"}

    def test_accepts_the_export_prefix(self, env_file, monkeypatch):
        """People copy lines straight out of shell instructions."""
        monkeypatch.delenv("TNGD_TEST_C", raising=False)
        load_dotenv(env_file("export TNGD_TEST_C=shell\n"))
        assert os.environ["TNGD_TEST_C"] == "shell"

    @pytest.mark.parametrize(
        "line,expected",
        [
            ('TNGD_TEST_D="quoted value"', "quoted value"),
            ("TNGD_TEST_D='single'", "single"),
            ("TNGD_TEST_D=bare # trailing comment", "bare"),
            ("TNGD_TEST_D=has=equals=signs", "has=equals=signs"),
            ("TNGD_TEST_D=   spaced   ", "spaced"),
            ('TNGD_TEST_D="keeps # inside quotes"', "keeps # inside quotes"),
        ],
    )
    def test_value_forms(self, env_file, monkeypatch, line, expected):
        monkeypatch.delenv("TNGD_TEST_D", raising=False)
        load_dotenv(env_file(line + "\n"))
        assert os.environ["TNGD_TEST_D"] == expected

    def test_malformed_lines_are_skipped_not_fatal(self, env_file, monkeypatch):
        monkeypatch.delenv("TNGD_TEST_E", raising=False)
        applied = load_dotenv(env_file("this is not a pair\n123BAD=x\nTNGD_TEST_E=ok\n"))
        assert applied == {"TNGD_TEST_E": "ok"}


class TestPrecedence:
    def test_the_real_environment_wins(self, env_file, monkeypatch):
        """A stale .env must never beat a value the user just exported."""
        monkeypatch.setenv("TNGD_TEST_F", "from-shell")
        load_dotenv(env_file("TNGD_TEST_F=from-file\n"))
        assert os.environ["TNGD_TEST_F"] == "from-shell"

    def test_override_is_available_when_asked_for(self, env_file, monkeypatch):
        monkeypatch.setenv("TNGD_TEST_G", "from-shell")
        load_dotenv(env_file("TNGD_TEST_G=from-file\n"), override=True)
        assert os.environ["TNGD_TEST_G"] == "from-file"


class TestFailureModes:
    def test_missing_file_is_harmless(self, tmp_path):
        assert load_dotenv(tmp_path / "nope.env") == {}

    def test_directory_instead_of_file_is_harmless(self, tmp_path):
        assert load_dotenv(tmp_path) == {}

    def test_returns_only_what_it_actually_set(self, env_file, monkeypatch):
        monkeypatch.setenv("TNGD_TEST_H", "already")
        monkeypatch.delenv("TNGD_TEST_I", raising=False)
        applied = load_dotenv(env_file("TNGD_TEST_H=x\nTNGD_TEST_I=y\n"))
        assert applied == {"TNGD_TEST_I": "y"}


class TestSecrets:
    def test_values_are_never_logged(self, env_file, monkeypatch, caplog):
        """Names only. A logged .env is an API key in your log aggregator."""
        monkeypatch.delenv("TNGD_TEST_SECRET", raising=False)
        with caplog.at_level("DEBUG", logger="tngd_faq_rag"):
            load_dotenv(env_file("TNGD_TEST_SECRET=AIzaSuperSecretValue\n"))
        assert "AIzaSuperSecretValue" not in caplog.text
        assert "TNGD_TEST_SECRET" in caplog.text
