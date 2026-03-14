"""
cli.py — evalloop status dashboard.

Commands:
  evalloop init                 Set up evalloop — choose scoring backend
  evalloop status               Show score trend for all task tags
  evalloop status --tag <name>  Show trend for a specific tag
  evalloop watch                Poll for regressions, alert to terminal
  evalloop export               Export captured calls to JSON or CSV
  evalloop baseline add <text>  Add a known-good output to a task tag
  evalloop baseline list        List all task tags with baselines
  evalloop baseline install     Install curated default baselines
  evalloop defaults             List available built-in task types
"""

from __future__ import annotations

import csv
import io
import json
import os
import time
from collections import Counter

import click

from evalloop.baseline import add as baseline_add, list_tags
from evalloop.db import DB
from evalloop.defaults import DEFAULTS, install as defaults_install, install_all


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _trend_bar(scores: list[float], width: int = 20) -> str:
    """ASCII sparkline of score history (oldest → newest)."""
    if not scores:
        return " " * width
    blocks = " ▁▂▃▄▅▆▇█"
    return "".join(blocks[min(8, int(s * 8))] for s in scores[-width:])


def _avg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _print_status(db: DB, tag: str, now: float) -> bool:
    """Print status block for a single tag. Returns True if regression detected."""
    day = 86_400
    rows = db.recent(task_tag=tag, limit=500)
    if not rows:
        return False

    scored = [r for r in rows if r["score"] is not None]
    recent_7d = [r for r in scored if now - r["ts"] < 7 * day]
    recent_1d = [r for r in scored if now - r["ts"] < day]

    scores_7d = [r["score"] for r in recent_7d]
    scores_1d = [r["score"] for r in recent_1d]
    all_scores = [r["score"] for r in scored]

    avg_7d = _avg(scores_7d)
    avg_1d = _avg(scores_1d)
    regression = avg_1d < avg_7d - 0.05 if scores_7d and scores_1d else False

    status_icon = "🔴" if regression else "🟢"
    click.echo(f"\n{status_icon}  [{tag}]")
    click.echo(f"   Calls captured : {len(rows)}")
    click.echo(f"   Scored         : {len(scored)}")
    click.echo(f"   Avg (7d)       : {avg_7d:.2f}")
    click.echo(f"   Avg (24h)      : {avg_1d:.2f}" if scores_1d else "   Avg (24h)      : —")

    if regression:
        drop = (avg_7d - avg_1d) * 100
        click.echo(f"   ⚠  Regression   : score dropped {drop:.1f}pp in last 24h")

    if all_scores:
        bar = _trend_bar(list(reversed(all_scores)))
        click.echo(f"   Trend (recent) : {bar}")

    # Surface top flags
    all_flags: list[str] = []
    for r in scored:
        if r["score_flags"]:
            try:
                all_flags.extend(json.loads(r["score_flags"]))
            except Exception:
                pass
    if all_flags:
        top = Counter(all_flags).most_common(3)
        flag_str = "  ".join(f"{f}({n})" for f, n in top)
        click.echo(f"   Top flags      : {flag_str}")

    return regression


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """evalloop — closed-loop eval monitoring for LLM-powered products."""


# ---------------------------------------------------------------------------
# init — setup wizard
# ---------------------------------------------------------------------------

_ANTHROPIC_SIGNUP = "https://console.anthropic.com/api-keys"
_VOYAGE_SIGNUP = "https://dash.voyageai.com/api-keys"


@cli.command()
def init() -> None:
    """Set up evalloop — detect API keys and choose a scoring backend."""
    click.echo("\nevalloop setup")
    click.echo("──────────────")

    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_voyage = bool(os.environ.get("VOYAGE_API_KEY"))

    tick = click.style("✓", fg="green")
    cross = click.style("✗", fg="red")

    click.echo(f"{'  ' + tick if has_anthropic else '  ' + cross} ANTHROPIC_API_KEY"
               + (" (detected)" if has_anthropic else ""))
    click.echo(f"{'  ' + tick if has_voyage else '  ' + cross} VOYAGE_API_KEY"
               + (" (detected)" if has_voyage else ""))
    click.echo()

    if has_voyage:
        click.echo(tick + " Voyage AI scoring ready — best semantic accuracy.")
        _init_install_baselines()
        _init_done()
        return

    if has_anthropic:
        click.echo(tick + " Anthropic scoring ready — LLM-as-judge via claude-haiku.")
        click.echo("  Tip: add VOYAGE_API_KEY for better semantic accuracy (optional).")
        _init_install_baselines()
        _init_done()
        return

    # Neither key found — present the choice
    click.echo("No scoring API keys found. Choose a backend:\n")
    click.echo("  [A] Anthropic  — LLM-as-judge via claude-haiku")
    click.echo("      ~$0.001 per 1000 scored calls")
    click.echo(f"      Get key → {_ANTHROPIC_SIGNUP}\n")
    click.echo("  [V] Voyage AI  — semantic embeddings, best accuracy")
    click.echo("      Free tier: 50M tokens/month")
    click.echo(f"      Get key → {_VOYAGE_SIGNUP}\n")
    click.echo("  [S] Skip       — heuristics only (no semantic scoring)\n")

    raw = click.prompt("Choice", default="A")
    choice = raw.strip().upper()[:1]

    if choice == "A":
        click.echo()
        click.echo("1. Get your Anthropic API key:")
        click.echo(f"   {_ANTHROPIC_SIGNUP}")
        click.echo()
        click.echo("2. Add it to your environment:")
        click.echo("   export ANTHROPIC_API_KEY=sk-ant-...")
        click.echo("   # or add to your .env file")
        _init_install_baselines()
    elif choice == "V":
        click.echo()
        click.echo("1. Get your Voyage AI key (free tier available):")
        click.echo(f"   {_VOYAGE_SIGNUP}")
        click.echo()
        click.echo("2. Add it to your environment:")
        click.echo("   export VOYAGE_API_KEY=pa-...")
        click.echo("   # or add to your .env file")
        _init_install_baselines()
    else:
        click.echo()
        click.echo("Running in heuristics-only mode.")
        click.echo("evalloop will still catch empty, truncated, and runaway outputs.")
        _init_install_baselines()

    click.echo()
    click.echo("Then wrap your client:")
    click.echo()
    click.echo("  from evalloop import wrap")
    click.echo("  import anthropic")
    click.echo()
    click.echo("  client = wrap(anthropic.Anthropic())")
    click.echo()


def _init_install_baselines() -> None:
    results = install_all(overwrite=False)
    installed = sum(1 for n in results.values() if n > 0)
    if installed:
        click.echo(f"  Installed default baselines for {installed} task type(s).")


def _init_done() -> None:
    click.echo()
    click.echo("You're set up. Wrap your client and start capturing:")
    click.echo()
    click.echo("  from evalloop import wrap")
    click.echo("  client = wrap(anthropic.Anthropic())")
    click.echo()
    click.echo("Then check your scores: evalloop status")
    click.echo()


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--tag", default=None, help="Filter to a specific task tag.")
@click.option("--db", "db_path", default=None, help="Path to evalloop DB.")
def status(tag: str | None, db_path: str | None) -> None:
    """Show score trends and regressions for captured LLM calls."""
    db = DB(db_path)
    tags = [tag] if tag else db.all_task_tags()

    if not tags:
        click.echo("No calls captured yet. Wrap your client:\n")
        click.echo("    from evalloop import wrap")
        click.echo("    client = wrap(anthropic.Anthropic())\n")
        return

    now = time.time()
    for t in tags:
        _print_status(db, t, now)

    click.echo()


@cli.command()
@click.option("--tag", default=None, help="Watch a specific task tag.")
@click.option("--db", "db_path", default=None, help="Path to evalloop DB.")
@click.option("--interval", default=60, show_default=True, help="Poll interval in seconds.")
def watch(tag: str | None, db_path: str | None, interval: int) -> None:
    """Poll for score regressions and alert to terminal. Ctrl-C to stop."""
    db = DB(db_path)
    click.echo(f"evalloop watch — polling every {interval}s. Ctrl-C to stop.\n")

    try:
        while True:
            now = time.time()
            tags = [tag] if tag else db.all_task_tags()
            had_regression = False
            for t in tags:
                reg = _print_status(db, t, now)
                if reg:
                    had_regression = True
            if not tags:
                click.echo("No calls captured yet. Waiting...")
            elif not had_regression:
                click.echo(f"  [ok] No regressions detected. Next check in {interval}s.")
            click.echo()
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\nWatch stopped.")


@cli.command()
@click.option("--tag", default=None, help="Export calls for a specific task tag.")
@click.option("--format", "fmt", default="json",
              type=click.Choice(["json", "csv"]), show_default=True,
              help="Output format.")
@click.option("--limit", default=1000, show_default=True, help="Max rows to export.")
@click.option("--db", "db_path", default=None, help="Path to evalloop DB.")
@click.option("--output", "-o", default=None, help="Output file (default: stdout).")
def export(tag: str | None, fmt: str, limit: int, db_path: str | None, output: str | None) -> None:
    """Export captured calls and scores to JSON or CSV."""
    db = DB(db_path)
    rows = db.export(task_tag=tag, limit=limit)

    if not rows:
        click.echo("No calls to export.", err=True)
        return

    if fmt == "json":
        content = json.dumps(rows, indent=2, default=str)
    else:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        content = buf.getvalue()

    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(content)
        click.echo(f"Exported {len(rows)} rows to {output}", err=True)
    else:
        click.echo(content)


@cli.group()
def baseline() -> None:
    """Manage known-good baseline examples."""


@baseline.command("add")
@click.argument("output_text")
@click.option("--tag", default="default", help="Task tag for this baseline.")
def baseline_add_cmd(output_text: str, tag: str) -> None:
    """Add a known-good output to a task tag's baseline."""
    baseline_add(output_text, task_tag=tag)
    click.echo(f"Added to baseline [{tag}].")


@baseline.command("list")
def baseline_list_cmd() -> None:
    """List all task tags that have baseline files."""
    tags = list_tags()
    if not tags:
        click.echo("No baselines yet. Add one:\n")
        click.echo('    evalloop baseline add "your good output" --tag my-task')
        click.echo("Or install built-in defaults:\n")
        click.echo("    evalloop baseline install\n")
        return
    for t in sorted(tags):
        click.echo(f"  {t}")


@baseline.command("install")
@click.option("--tag", default=None, help="Install defaults for a specific tag only.")
@click.option("--overwrite", is_flag=True, default=False, help="Overwrite existing baselines.")
def baseline_install_cmd(tag: str | None, overwrite: bool) -> None:
    """Install curated default baselines (solves cold-start problem)."""
    if tag:
        if tag not in DEFAULTS:
            click.echo(f"Unknown tag '{tag}'. Available: {', '.join(sorted(DEFAULTS))}")
            return
        n = defaults_install(tag, overwrite=overwrite)
        if n == 0:
            click.echo(f"[{tag}] already has baselines. Use --overwrite to replace.")
        else:
            click.echo(f"Installed {n} default examples for [{tag}].")
    else:
        results = install_all(overwrite=overwrite)
        installed = {t: n for t, n in results.items() if n > 0}
        skipped = {t: n for t, n in results.items() if n == 0}
        for t, n in sorted(installed.items()):
            click.echo(f"  [{t}] installed {n} examples")
        if skipped:
            click.echo(f"  Skipped (already exist): {', '.join(sorted(skipped))}  (use --overwrite)")
        if not installed:
            click.echo("All defaults already installed. Use --overwrite to replace.")


@cli.command()
def defaults() -> None:
    """List available built-in task types and their example count."""
    click.echo("\nBuilt-in task types:\n")
    for tag, examples in sorted(DEFAULTS.items()):
        click.echo(f"  {tag:<20} {len(examples)} examples")
    click.echo(f"\nInstall all:  evalloop baseline install")
    click.echo(f"Install one:  evalloop baseline install --tag qa\n")
