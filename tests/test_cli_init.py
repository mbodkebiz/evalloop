"""
cli.py tests — evalloop init wizard.
"""

from unittest.mock import patch

from click.testing import CliRunner

from evalloop.cli import cli


def _runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# Auto-configure paths (key already present)
# ---------------------------------------------------------------------------


def test_init_auto_configures_anthropic_when_key_detected(monkeypatch):
    """init auto-selects Anthropic backend when ANTHROPIC_API_KEY is set."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={"qa": 5}):
        result = _runner().invoke(cli, ["init"])

    assert result.exit_code == 0
    assert "Anthropic scoring ready" in result.output


def test_init_auto_configures_voyage_when_key_detected(monkeypatch):
    """init auto-selects Voyage backend when VOYAGE_API_KEY is set."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-testkey")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={"qa": 5}):
        result = _runner().invoke(cli, ["init"])

    assert result.exit_code == 0
    assert "Voyage AI scoring ready" in result.output


def test_init_voyage_takes_priority_over_anthropic(monkeypatch):
    """When both keys are set, init shows Voyage (higher priority)."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-testkey")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    with patch("evalloop.cli.install_all", return_value={}):
        result = _runner().invoke(cli, ["init"])

    assert result.exit_code == 0
    assert "Voyage AI scoring ready" in result.output


# ---------------------------------------------------------------------------
# Choice menu paths (no keys)
# ---------------------------------------------------------------------------


def test_init_shows_choice_menu_when_no_keys(monkeypatch):
    """init presents A/V/S menu when neither API key is configured."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={}):
        result = _runner().invoke(cli, ["init"], input="S\n")

    assert result.exit_code == 0
    assert "[A]" in result.output
    assert "[V]" in result.output
    assert "[S]" in result.output


def test_init_choice_a_shows_anthropic_signup_link(monkeypatch):
    """Choosing A shows the Anthropic API key signup URL."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={}):
        result = _runner().invoke(cli, ["init"], input="A\n")

    assert result.exit_code == 0
    assert "console.anthropic.com" in result.output


def test_init_choice_v_shows_voyage_signup_link(monkeypatch):
    """Choosing V shows the Voyage AI signup URL."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={}):
        result = _runner().invoke(cli, ["init"], input="V\n")

    assert result.exit_code == 0
    assert "voyageai.com" in result.output


def test_init_choice_s_confirms_heuristics_mode(monkeypatch):
    """Choosing S confirms heuristics-only mode."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={}):
        result = _runner().invoke(cli, ["init"], input="S\n")

    assert result.exit_code == 0
    assert "heuristics" in result.output.lower()


# ---------------------------------------------------------------------------
# Baseline installation — all paths call _init_install_baselines
# ---------------------------------------------------------------------------


def test_init_installs_baselines_on_anthropic_auto_path(monkeypatch):
    """Auto-configure Anthropic path installs baselines."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={"qa": 5, "code": 3}) as mock_install:
        result = _runner().invoke(cli, ["init"])

    mock_install.assert_called_once()
    assert result.exit_code == 0


def test_init_installs_baselines_on_voyage_auto_path(monkeypatch):
    """Auto-configure Voyage path installs baselines."""
    monkeypatch.setenv("VOYAGE_API_KEY", "pa-testkey")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={"qa": 5}) as mock_install:
        result = _runner().invoke(cli, ["init"])

    mock_install.assert_called_once()
    assert result.exit_code == 0


def test_init_installs_baselines_on_choice_a_path(monkeypatch):
    """Choice-A path installs baselines."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={"qa": 5}) as mock_install:
        _runner().invoke(cli, ["init"], input="A\n")

    mock_install.assert_called_once()


def test_init_installs_baselines_on_choice_v_path(monkeypatch):
    """Choice-V path installs baselines."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={"qa": 5}) as mock_install:
        _runner().invoke(cli, ["init"], input="V\n")

    mock_install.assert_called_once()


def test_init_installs_baselines_on_choice_s_path(monkeypatch):
    """Choice-S path installs baselines."""
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with patch("evalloop.cli.install_all", return_value={"qa": 5}) as mock_install:
        _runner().invoke(cli, ["init"], input="S\n")

    mock_install.assert_called_once()
