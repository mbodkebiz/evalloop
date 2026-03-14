"""
cli.py tests — evalloop rescore command.
"""

from unittest.mock import patch

import pytest
from click.testing import CliRunner

from evalloop.cli import cli
from evalloop.scorer import Score


def _runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(n=3, tag="default"):
    return [
        {
            "id": i + 1,
            "output_text": f"output {i}",
            "task_tag": tag,
            "score": None,
            "score_flags": None,
            "confidence": None,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_rescore_no_calls(tmp_path):
    """rescore exits cleanly when DB has no calls."""
    db_path = str(tmp_path / "empty.db")
    result = _runner().invoke(cli, ["rescore", "--db", db_path])
    assert result.exit_code == 0
    assert "No calls found" in result.output


def test_rescore_uses_heuristics_when_no_keys(monkeypatch, tmp_path):
    """rescore routes to heuristics when no API keys are set."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    rows = _make_rows(2, tag="default")
    mock_score = Score(0.5, ["degraded_mode"], confidence=0.1)

    with (
        patch("evalloop.cli.DB") as MockDB,
        patch("evalloop.cli._heuristics_score", return_value=mock_score) as mock_h,
        patch("evalloop.cli.baseline_load", return_value=["good output"]),
    ):
        instance = MockDB.return_value
        instance.export.return_value = rows

        result = _runner().invoke(cli, ["rescore"])

    assert result.exit_code == 0
    assert "heuristics" in result.output
    assert mock_h.call_count == 2
    assert instance.update_score.call_count == 2


def test_rescore_uses_llm_judge_when_anthropic_key(monkeypatch, tmp_path):
    """rescore routes to LLM-as-judge when ANTHROPIC_API_KEY is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    rows = _make_rows(1, tag="qa")
    mock_score = Score(0.9, ["llm_judge"], confidence=0.9)

    with (
        patch("evalloop.cli.DB") as MockDB,
        patch("evalloop.cli._llm_judge_score", return_value=mock_score) as mock_j,
        patch("evalloop.cli._heuristics_score") as mock_h,
        patch("evalloop.cli.baseline_load", return_value=["a good answer"]),
    ):
        instance = MockDB.return_value
        instance.export.return_value = rows

        result = _runner().invoke(cli, ["rescore"])

    assert result.exit_code == 0
    assert "LLM-as-judge" in result.output
    mock_j.assert_called_once()
    mock_h.assert_not_called()
    instance.update_score.assert_called_once()


def test_rescore_uses_voyage_when_voyage_key(monkeypatch, tmp_path):
    """rescore routes to Voyage when VOYAGE_API_KEY is set."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-test")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    rows = _make_rows(1, tag="default")
    mock_score = Score(0.85, [], confidence=1.0)

    with (
        patch("evalloop.cli.DB") as MockDB,
        patch("evalloop.cli._voyage_score", return_value=mock_score) as mock_v,
        patch("evalloop.cli._llm_judge_score") as mock_j,
        patch("evalloop.cli.baseline_load", return_value=["baseline output"]),
    ):
        instance = MockDB.return_value
        instance.export.return_value = rows

        result = _runner().invoke(cli, ["rescore"])

    assert result.exit_code == 0
    assert "Voyage AI" in result.output
    mock_v.assert_called_once()
    mock_j.assert_not_called()
    instance.update_score.assert_called_once()


def test_rescore_tag_filter_passed_to_db(monkeypatch):
    """rescore passes --tag to db.export."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    with (
        patch("evalloop.cli.DB") as MockDB,
        patch("evalloop.cli._heuristics_score", return_value=Score(0.5, [])),
        patch("evalloop.cli.baseline_load", return_value=["x"]),
    ):
        instance = MockDB.return_value
        instance.export.return_value = _make_rows(1, tag="math")

        _runner().invoke(cli, ["rescore", "--tag", "math"])

    instance.export.assert_called_once_with(task_tag="math", limit=100_000)


def test_rescore_reports_counts(monkeypatch):
    """rescore prints updated/errors summary line."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    rows = _make_rows(3)
    with (
        patch("evalloop.cli.DB") as MockDB,
        patch("evalloop.cli._heuristics_score", return_value=Score(0.5, [])),
        patch("evalloop.cli.baseline_load", return_value=["good"]),
    ):
        instance = MockDB.return_value
        instance.export.return_value = rows

        result = _runner().invoke(cli, ["rescore"])

    assert "Updated: 3" in result.output
    assert "Errors: 0" in result.output


def test_rescore_handles_scorer_error_gracefully(monkeypatch):
    """rescore continues and reports errors when scorer raises."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    rows = _make_rows(2)
    with (
        patch("evalloop.cli.DB") as MockDB,
        patch("evalloop.cli._heuristics_score", side_effect=RuntimeError("boom")),
        patch("evalloop.cli.baseline_load", return_value=["good"]),
    ):
        instance = MockDB.return_value
        instance.export.return_value = rows

        result = _runner().invoke(cli, ["rescore"])

    assert result.exit_code == 0
    assert "Errors: 2" in result.output
    instance.update_score.assert_not_called()
