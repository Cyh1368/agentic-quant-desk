"""Shadow isolation enforcement tests (twitter-sentiment-plan-v2 §8).

(a) type-level: the portfolio engine never mentions or accepts ResearchForecast.
(b) import-graph: no transitive import path from portfolio/risk/execution to
    research modules, and research modules never import execution/portfolio.
(c) provenance: persisted research forecasts always carry research_only=1 and
    research code never references OrderIntent.
"""
from __future__ import annotations

import ast
import inspect
import sqlite3
import typing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
QUANTDESK = REPO_ROOT / "quantdesk"

FORBIDDEN_REACHABLE = {
    "quantdesk.common.research_schemas",
    "quantdesk.advisors.sentiment_llm",
    "quantdesk.advisors.sent_zscore_baseline",
    "quantdesk.data.twitter",
}

RESEARCH_SIDE_FILES = [
    QUANTDESK / "data" / "twitter.py",
    QUANTDESK / "advisors" / "sentiment_llm.py",
    QUANTDESK / "advisors" / "sent_zscore_baseline.py",
]

ORDER_INTENT_FREE_FILES = [
    QUANTDESK / "advisors" / "sentiment_llm.py",
    QUANTDESK / "advisors" / "sent_zscore_baseline.py",
    QUANTDESK / "ledger" / "research_store.py",
    QUANTDESK / "scoring" / "research_eval.py",
]


# --------------------------------------------------------------------------
# (a) type-level
# --------------------------------------------------------------------------

def test_portfolio_engine_source_never_mentions_research_forecast():
    source = (QUANTDESK / "portfolio" / "engine.py").read_text()
    assert "ResearchForecast" not in source


def test_portfolio_engine_signatures_never_use_research_forecast():
    import quantdesk.portfolio.engine as engine

    for name, obj in vars(engine).items():
        if inspect.isfunction(obj) or inspect.ismethod(obj):
            _assert_no_research_forecast_hints(obj, name)
        elif inspect.isclass(obj) and obj.__module__ == engine.__name__:
            for meth_name, meth in inspect.getmembers(obj, inspect.isfunction):
                _assert_no_research_forecast_hints(meth, f"{name}.{meth_name}")


def _assert_no_research_forecast_hints(fn, label: str) -> None:
    try:
        hints = typing.get_type_hints(fn)
    except Exception:
        hints = getattr(fn, "__annotations__", {})
    for hint in hints.values():
        assert "ResearchForecast" not in str(hint), label
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return
    assert "ResearchForecast" not in str(sig), label


# --------------------------------------------------------------------------
# (b) import-graph
# --------------------------------------------------------------------------

def _module_name_for(path: Path) -> str:
    rel = path.relative_to(REPO_ROOT).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _path_for_module(module: str) -> Path | None:
    base = REPO_ROOT / Path(*module.split("."))
    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    return None


def _imports_of(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    module_name = _module_name_for(path)
    out: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                # resolve relative import against this module's package
                pkg_parts = module_name.split(".")
                # drop the module leaf, then drop (level - 1) more packages
                pkg_parts = pkg_parts[: len(pkg_parts) - node.level]
                base = ".".join(pkg_parts)
                mod = f"{base}.{node.module}" if node.module else base
            else:
                mod = node.module or ""
            if mod:
                out.add(mod)
                # `from pkg import name` may import submodule pkg.name
                for alias in node.names:
                    out.add(f"{mod}.{alias.name}")
    return {m for m in out if m.startswith("quantdesk")}


def _reachable_modules(roots: list[Path]) -> set[str]:
    seen: set[str] = set()
    stack: list[str] = []
    for root in roots:
        name = _module_name_for(root)
        seen.add(name)
        stack.append(name)
    while stack:
        module = stack.pop()
        path = _path_for_module(module)
        if path is None:
            continue
        for imported in _imports_of(path):
            # normalize: keep only modules that resolve to real files
            candidate = imported
            while candidate and _path_for_module(candidate) is None:
                candidate = ".".join(candidate.split(".")[:-1])
            if candidate and candidate not in seen:
                seen.add(candidate)
                stack.append(candidate)
    return seen


def test_execution_side_never_reaches_research_modules():
    roots = []
    for pkg in ("portfolio", "risk", "execution"):
        roots.extend(sorted((QUANTDESK / pkg).rglob("*.py")))
    assert roots
    reachable = _reachable_modules(roots)
    forbidden_hit = reachable & FORBIDDEN_REACHABLE
    assert not forbidden_hit, f"forbidden research modules reachable: {forbidden_hit}"


def test_research_side_never_imports_execution_or_portfolio():
    existing = [p for p in RESEARCH_SIDE_FILES if p.exists()]
    if not existing:
        pytest.skip("no research-side modules present yet")
    for path in existing:
        for imported in _imports_of(path):
            assert not imported.startswith("quantdesk.execution"), path
            assert not imported.startswith("quantdesk.portfolio"), path


# --------------------------------------------------------------------------
# (c) provenance
# --------------------------------------------------------------------------

def test_research_store_persists_research_only_flag(tmp_path):
    from quantdesk.common.research_schemas import ResearchForecast
    from quantdesk.ledger.research_store import ResearchStore

    store = ResearchStore(tmp_path / "research.sqlite")
    forecast = ResearchForecast(
        forecast_id=uuid4(),
        advisor_id="crypto_sentiment_llm_v1",
        advisor_version="1.0",
        generated_at=datetime.now(timezone.utc),
        data_cutoff_at=datetime.now(timezone.utc),
        instrument_id="BTC",
        horizon=timedelta(hours=24),
        direction="long",
        abstain=False,
        probability_positive=0.6,
        expected_excess_return_bps=25.0,
        confidence=0.5,
        evidence_feature_ids=["feat1"],
        snapshot_id=uuid4(),
        model_run_id=None,
    )
    store.insert_forecast(forecast)
    store.close()

    conn = sqlite3.connect(store.db_path)
    row = conn.execute(
        "SELECT research_only FROM research_forecasts WHERE forecast_id = ?",
        (str(forecast.forecast_id),),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == 1


def test_research_modules_never_reference_order_intent():
    existing = [p for p in ORDER_INTENT_FREE_FILES if p.exists()]
    assert existing, "expected at least one research module to exist"
    for path in existing:
        source = path.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                assert node.id != "OrderIntent", path
            elif isinstance(node, ast.Attribute):
                assert node.attr != "OrderIntent", path
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                for alias in node.names:
                    assert alias.name != "OrderIntent", path
        # Substring backstop over code with string literals blanked out
        # (docstrings may legitimately *document* the "never OrderIntent"
        # invariant; code must never reference the name).
        code_only = _strip_string_constants(source, tree)
        assert "OrderIntent" not in code_only, path


def _strip_string_constants(source: str, tree: ast.Module) -> str:
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    chars = list(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            start = offsets[node.lineno - 1] + node.col_offset
            end = offsets[node.end_lineno - 1] + node.end_col_offset
            for i in range(start, end):
                chars[i] = " "
    return "".join(chars)
