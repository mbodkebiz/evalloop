"""
cli.py — evalloop status dashboard.

Commands:
  evalloop status               Show score trend for all task tags
  evalloop status --tag <name>  Show trend for a specific tag
  evalloop baseline add <text>  Add a known-good output to a task tag
  evalloop baseline list        List all task tags with baselines
  evalloop baseline install     Install curated default baselines
  evalloop defaults             List available built-in task types
"""

from __future__ import annotations

import sys
import time

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """evalloop — closed-loop eval monitoring for LLM-powered products."""


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
        click.echo("    client = wrap(openai.OpenAI())\n")
        return

    now = time.time()
    day = 86_400

    for t in tags:
        rows = db.recent(task_tag=t, limit=500)
        if not rows:
            continue

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
        click.echo(f"\n{status_icon}  [{t}]")
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
        import json
        for r in scored:
            if r["score_flags"]:
                try:
                    all_flags.extend(json.loads(r["score_flags"]))
                except Exception:
                    pass
        if all_flags:
            from collections import Counter
            top = Counter(all_flags).most_common(3)
            flag_str = "  ".join(f"{f}({n})" for f, n in top)
            click.echo(f"   Top flags      : {flag_str}")

    click.echo()


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
