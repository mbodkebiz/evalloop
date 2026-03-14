"""
defaults.py tests — install, cold-start, tag inference.
"""

import pytest
import evalloop.baseline as bl
import evalloop.defaults as defaults_mod
from evalloop.defaults import DEFAULTS, infer_tag, install, install_all


@pytest.fixture(autouse=True)
def tmp_baseline_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(bl, "_BASELINE_DIR", str(tmp_path))
    monkeypatch.setattr("evalloop.baseline._BASELINE_DIR", str(tmp_path))
    yield


# ---------------------------------------------------------------------------
# install()
# ---------------------------------------------------------------------------

def test_install_known_tag_returns_count():
    n = install("qa")
    assert n == len(DEFAULTS["qa"])


def test_install_unknown_tag_returns_zero():
    assert install("nonexistent-tag") == 0


def test_install_populates_baseline():
    install("qa")
    loaded = bl.load("qa")
    assert len(loaded) == len(DEFAULTS["qa"])
    for example in DEFAULTS["qa"]:
        assert example in loaded


def test_install_skips_if_already_exists():
    install("qa")
    bl.add("my custom baseline", task_tag="qa")
    n = install("qa")  # should skip, not overwrite
    assert n == 0
    # custom baseline still there
    assert "my custom baseline" in bl.load("qa")


def test_install_overwrite_replaces_existing():
    install("qa")
    bl.add("my custom baseline", task_tag="qa")
    install("qa", overwrite=True)
    loaded = bl.load("qa")
    assert "my custom baseline" not in loaded
    assert len(loaded) == len(DEFAULTS["qa"])


# ---------------------------------------------------------------------------
# install_all()
# ---------------------------------------------------------------------------

def test_install_all_installs_every_tag():
    results = install_all()
    for tag in DEFAULTS:
        assert results[tag] == len(DEFAULTS[tag])
        assert len(bl.load(tag)) == len(DEFAULTS[tag])


def test_install_all_skips_existing():
    install_all()
    results = install_all()  # second call should skip all
    assert all(n == 0 for n in results.values())


# ---------------------------------------------------------------------------
# infer_tag()
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prompt,expected_tag", [
    ("You are a helpful assistant that summarizes documents.", "summarization"),
    ("Classify the sentiment of the following text.", "classification"),
    ("Write a Python function to sort a list.", "code"),
    ("You are a customer support agent for Acme Corp.", "customer-service"),
    ("Answer the user's question clearly and concisely.", "qa"),
])
def test_infer_tag(prompt, expected_tag):
    assert infer_tag(prompt) == expected_tag


def test_infer_tag_none_for_empty():
    assert infer_tag("") is None
    assert infer_tag(None) is None  # type: ignore[arg-type]


def test_infer_tag_none_for_unrecognized():
    assert infer_tag("You are a pirate.") is None


# ---------------------------------------------------------------------------
# DEFAULTS content sanity
# ---------------------------------------------------------------------------

def test_all_defaults_are_nonempty_strings():
    for tag, examples in DEFAULTS.items():
        assert len(examples) >= 3, f"[{tag}] needs at least 3 examples"
        for ex in examples:
            assert isinstance(ex, str) and ex.strip(), f"[{tag}] has empty example"
