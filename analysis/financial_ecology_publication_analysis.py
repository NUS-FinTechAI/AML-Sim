#!/usr/bin/env python3
"""Analyse the locked AML financial-ecology publication replication.

The primary D0/I1/A1 study has ten paired seeds per decision mode. The
secondary frozen-strategy C0/C2 three-market demonstration retains three paired
seeds. The analysis treats the seed as the replication unit and reports paired effects
with seed-level bootstrap intervals instead of treating agents or orders as
independent observations.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/aml-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / ".aml_runs"
OUTPUT_DIR = ROOT / "artifacts" / "financial_ecology_publication_v1" / "analysis"
SESSION_START = pd.Timestamp("2025-03-01T09:30:00+00:00")
SESSION_END = pd.Timestamp("2025-03-01T10:30:00+00:00")

LEGACY_TREATMENTS = ("d0", "i1", "a1")
MODES = ("frozen", "llm")
SECONDARY_MODES = ("frozen",)
PRIMARY_REPLICATES = tuple(range(1, 11))
SECONDARY_REPLICATES = (1, 2, 3)
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20261006
LEGACY_LABELS = {"d0": "D0 disconnected", "i1": "I1 information", "a1": "A1 linked"}
LINKED_LABELS = {"c0": "C0 control", "c2": "C2 linked"}
COLORS = {"d0": "#667085", "i1": "#1f8a70", "a1": "#d97706", "c0": "#667085", "c2": "#1f8a70"}
ASSET_COLORS = {"AAPL": "#0f766e", "AAPL_FUT": "#2563eb", "UST10Y": "#7c3aed"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=RUNS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def as_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def table(frame: pd.DataFrame) -> str:
    if not isinstance(frame.index, pd.RangeIndex) or frame.index.name is not None:
        frame = frame.reset_index()
    else:
        frame = frame.copy()
    headers = [str(column) for column in frame.columns]

    def render(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.3f}"
        return str(value)

    rows = [[render(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
        + ["| " + " | ".join(row) + " |" for row in rows]
    )


def run_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for treatment in LEGACY_TREATMENTS:
        for mode in MODES:
            for replicate in PRIMARY_REPLICATES:
                specs.append(
                    {
                        "study": "stock_future_micro",
                        "treatment": treatment,
                        "mode": mode,
                        "replicate": replicate,
                        "run_id": f"pub1_ecology_{treatment}_{mode}_r{replicate:02d}",
                        "micro_time": pd.Timestamp("2025-03-01T09:59:00+00:00"),
                        "micro_end": pd.Timestamp("2025-03-01T10:03:00+00:00"),
                        "macro_time": pd.NaT,
                    }
                )
    for treatment in ("c0", "c2"):
        for mode in SECONDARY_MODES:
            for replicate in SECONDARY_REPLICATES:
                specs.append(
                    {
                        "study": "stock_future_bond_two_shock",
                        "treatment": treatment,
                        "mode": mode,
                        "replicate": replicate,
                        "run_id": f"pub1_ecology_{treatment}_{mode}_r{replicate:02d}",
                        "micro_time": pd.Timestamp("2025-03-01T09:45:00+00:00"),
                        "micro_end": pd.Timestamp("2025-03-01T09:49:00+00:00"),
                        "macro_time": pd.Timestamp("2025-03-01T10:00:00+00:00"),
                    }
                )
    return specs


def market_path(summary: dict[str, Any]) -> pd.DataFrame:
    grid = pd.DataFrame({"timestamp": pd.date_range(SESSION_START, SESSION_END, freq="30s")})
    trades = pd.DataFrame(summary.get("trades", []))
    if trades.empty:
        grid["price"] = np.nan
        grid["volume"] = 0
        grid["trade_count"] = 0
        return grid
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], utc=True)
    trades["price"] = pd.to_numeric(trades["price"], errors="coerce")
    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0)
    trades = trades.sort_values("timestamp")
    last = trades.drop_duplicates("timestamp", keep="last")[["timestamp", "price"]]
    grid = pd.merge_asof(grid, last, on="timestamp", direction="backward")
    grid["price"] = grid["price"].ffill().fillna(as_float(summary.get("start_price")))
    grid["volume"] = grid["timestamp"].map(trades.groupby("timestamp")["quantity"].sum()).fillna(0)
    grid["trade_count"] = grid["timestamp"].map(trades.groupby("timestamp").size()).fillna(0)
    return grid


def price_at(path: pd.DataFrame, timestamp: pd.Timestamp) -> float:
    rows = path.loc[path["timestamp"] == timestamp, "price"]
    return as_float(rows.iloc[0]) if not rows.empty else float("nan")


def event_metrics(path: pd.DataFrame, event_start: pd.Timestamp, event_end: pd.Timestamp) -> dict[str, float]:
    pre = price_at(path, event_start - pd.Timedelta(seconds=30))
    end = price_at(path, event_end)
    recovery = price_at(path, min(event_start + pd.Timedelta(minutes=15), SESSION_END))
    active = path.loc[(path["timestamp"] >= event_start) & (path["timestamp"] < event_end)]
    prices = active["price"].replace(0, np.nan).dropna()
    returns = np.log(prices).diff().dropna()
    return {
        "pre_price": pre,
        "end_price": end,
        "event_return_pct": 100 * (end / pre - 1) if pre > 0 else np.nan,
        "fifteen_min_return_pct": 100 * (recovery / pre - 1) if pre > 0 else np.nan,
        "event_volume": float(active["volume"].sum()),
        "event_trade_count": int(active["trade_count"].sum()),
        "event_realized_vol_bps": float(returns.std(ddof=1) * 10_000) if len(returns) > 1 else np.nan,
    }


def agent_role(agent_id: str) -> str:
    if "market_maker" in agent_id:
        return "market_maker"
    if "institutional" in agent_id:
        return "institutional"
    if "retail" in agent_id:
        return "retail"
    if "arbitrage" in agent_id:
        return "arbitrageur"
    return "other"


def warning_counts(run_dir: Path) -> Counter[str]:
    result: Counter[str] = Counter()
    for path in (run_dir / "logs").rglob("*.log"):
        content = path.read_text(errors="replace")
        result["warnings"] += len(re.findall(r"\bWARNING\b", content, flags=re.I))
        result["self_trade_prevention"] += len(re.findall(r"Self-trade prevention", content, flags=re.I))
        result["stale_cancel"] += len(re.findall(r"non-existent order ID|failed to cancel order", content, flags=re.I))
        result["partial_market_orders"] += len(re.findall(r"partially unfilled", content, flags=re.I))
        result["tracebacks"] += len(re.findall(r"Traceback", content, flags=re.I))
    return result


def collect(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    paths: list[pd.DataFrame] = []
    event_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []

    for spec in run_specs():
        run_dir = runs_dir / spec["run_id"]
        reports = run_dir / "reports"
        required = [
            reports / "simulation_summary.json",
            reports / "ecology_market_summary.json",
            reports / "ecology_agent_response_summary.json",
            reports / "ecology_channel_ledger.json",
            reports / "ecology_decision_summary.json",
        ]
        missing = [str(path.relative_to(run_dir)) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"{spec['run_id']} is incomplete: {', '.join(missing)}")

        manifest = read_json(reports / "ecology_manifest.json")
        simulation = read_json(reports / "simulation_summary.json")
        market_summary = read_json(reports / "ecology_market_summary.json")["markets"]
        response_summary = read_json(reports / "ecology_agent_response_summary.json")
        ledger = read_json(reports / "ecology_channel_ledger.json")
        counters = warning_counts(run_dir)
        statuses = Counter(item.get("slow_loop_status") for item in response_summary.get("responses", []))
        if spec["mode"] == "llm" and statuses.get("failed", 0):
            raise ValueError(
                f"{spec['run_id']} contains {statuses['failed']} failed slow-loop calls"
            )
        quality_rows.append(
            {
                **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                "duration": simulation["simulation_info"].get("duration"),
                "assets": ",".join(simulation["simulation_info"].get("instruments", [])),
                "llm_agents": simulation.get("research_metrics", {}).get("llm_agents", 0),
                "slow_loop_completed": statuses.get("completed", 0),
                "slow_loop_rejected": statuses.get("rejected", 0),
                "slow_loop_failed": statuses.get("failed", 0),
                "strategy_changes": response_summary.get("strategy_change_count", 0),
                "event_deliveries": sum(ledger.get("event_delivery_counts", {}).values()),
                "relationship_deliveries": sum(
                    value for key, value in ledger.get("event_delivery_counts", {}).items() if "relationship" in key
                ),
                **counters,
            }
        )

        for asset, summary in market_summary.items():
            path = market_path(summary)
            path = path.assign(**{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}}, asset=asset)
            paths.append(path)
            for event_name, start, end in (
                ("micro", spec["micro_time"], spec["micro_end"]),
                ("macro", spec["macro_time"], min(spec["macro_time"] + pd.Timedelta(minutes=4), SESSION_END) if pd.notna(spec["macro_time"]) else pd.NaT),
            ):
                if pd.isna(start):
                    continue
                event_rows.append(
                    {
                        **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                        "asset": asset,
                        "event": event_name,
                        **event_metrics(path, start, end),
                        "total_trade_count": int(summary.get("trade_count", 0)),
                        "total_volume": float(summary.get("volume", 0)),
                        "session_return_pct": 100 * as_float(summary.get("return")),
                    }
                )

        for metrics_path in sorted((reports / "agents").glob("metrics_*.json")):
            agent_id = metrics_path.stem.removeprefix("metrics_")
            metrics = read_json(metrics_path)
            series_path = reports / "agents" / f"portfolio_timeseries_{agent_id}.json"
            series = read_json(series_path) if series_path.exists() else []
            agent_rows.append(
                {
                    **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                    "agent_id": agent_id,
                    "role": agent_role(agent_id),
                    "initial_value": as_float(series[0].get("value")) if series else np.nan,
                    "final_value": as_float(metrics.get("Last Portfolio Value")),
                    "roi_pct": 100 * as_float(metrics.get("ROI")),
                    "max_drawdown_pct": 100 * as_float(metrics.get("Max Drawdown")),
                    "sharpe": as_float(metrics.get("Sharpe Ratio")),
                    "total_pnl": sum(as_float(value, 0.0) for value in (metrics.get("Total P&L") or {}).values()),
                    "gross_exposure": as_float(metrics.get("Gross Exposure")),
                    "net_exposure": as_float(metrics.get("Net Exposure")),
                    "num_trades": int(as_float(metrics.get("Num Trades"), 0)),
                }
            )

        for item in response_summary.get("responses", []):
            response_rows.append(
                {
                    **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                    "agent_id": item.get("agent_id"),
                    "role": agent_role(str(item.get("agent_id"))),
                    "status": item.get("slow_loop_status"),
                    "strategy_changed": bool(item.get("strategy_changed")),
                    "event_context": bool(item.get("event_context_present")),
                    "active_event": bool(item.get("active_event_ids")),
                    "known_event": bool(item.get("known_event_ids")),
                    "risk_changed": item.get("risk_mode_before") != item.get("risk_mode_after"),
                }
            )

        for item in read_json(reports / "ecology_decision_summary.json").get("decisions", []):
            decision_rows.append(
                {
                    **{key: value for key, value in spec.items() if key not in {"micro_time", "micro_end", "macro_time"}},
                    "timestamp": pd.to_datetime(item.get("timestamp"), utc=True),
                    "decision_id": item.get("decision_id"),
                    "observed_basis_bps": as_float(item.get("observed_basis_bps")),
                    "abs_observed_basis_bps": abs(as_float(item.get("observed_basis_bps"))),
                    "execution_outcome": item.get("execution_outcome"),
                    "active_shock": bool(item.get("active_event_ids")),
                }
            )

    return (
        pd.concat(paths, ignore_index=True),
        pd.DataFrame(event_rows),
        pd.DataFrame(quality_rows),
        pd.DataFrame(agent_rows),
        pd.DataFrame(response_rows),
        pd.DataFrame(decision_rows),
    )


def save_legacy_paths(paths: pd.DataFrame, output_dir: Path) -> None:
    start = pd.Timestamp("2025-03-01T09:49:00+00:00")
    end = pd.Timestamp("2025-03-01T10:14:00+00:00")
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    shock = pd.Timestamp("2025-03-01T09:59:00+00:00")
    for row, mode in enumerate(MODES):
        for col, asset in enumerate(("AAPL", "AAPL_FUT")):
            ax = axes[row, col]
            subset = paths.loc[(paths.study == "stock_future_micro") & (paths["mode"] == mode) & (paths.asset == asset)].copy()
            for treatment in LEGACY_TREATMENTS:
                data = subset.loc[subset.treatment == treatment].copy()
                data["relative_min"] = (data.timestamp - shock).dt.total_seconds() / 60
                base = data.loc[data.timestamp == shock - pd.Timedelta(seconds=30), ["replicate", "price"]].set_index("replicate")["price"]
                data["return_pct"] = data.apply(lambda row: 100 * (row.price / base.get(row.replicate, np.nan) - 1), axis=1)
                data = data.loc[(data.timestamp >= start) & (data.timestamp <= end)]
                mean = data.groupby("relative_min").return_pct.mean()
                ax.plot(mean.index, mean.values, color=COLORS[treatment], lw=2.4, label=LEGACY_LABELS[treatment])
            ax.axvspan(0, 4, color="#f6bd60", alpha=0.2)
            ax.axvline(0, color="#4b5563", ls="--", lw=1)
            ax.axhline(0, color="#cbd5e1", lw=1)
            ax.set_title(f"{mode.title()} - {asset}")
            ax.set_ylabel("Return from pre-shock price (%)")
            ax.legend(frameon=False, fontsize=8)
    fig.suptitle("D0/I1/A1 stock-shock transmission", y=0.99, fontsize=14)
    fig.text(0.5, 0.01, "Mean across ten paired seeds. Shading marks the four-minute idiosyncratic stock shock.", ha="center")
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_dir / "01_legacy_micro_transmission.png", dpi=220)
    plt.close(fig)


def paired_legacy_deltas(agent_df: pd.DataFrame) -> pd.DataFrame:
    legacy = agent_df.loc[(agent_df.study == "stock_future_micro") & agent_df.agent_id.isin([
        "stock_market_maker", "stock_institutional", "stock_retail_1", "stock_retail_2",
        "future_market_maker", "future_institutional", "future_retail",
    ])]
    control = legacy.loc[legacy.treatment == "d0"].set_index(["mode", "replicate", "agent_id"])
    rows = []
    for treatment in ("i1", "a1"):
        linked = legacy.loc[legacy.treatment == treatment].set_index(["mode", "replicate", "agent_id"])
        joined = linked.join(control[["roi_pct", "max_drawdown_pct"]], rsuffix="_d0", how="inner")
        for (mode, replicate, agent_id), row in joined.iterrows():
            rows.append({
                "mode": mode, "replicate": replicate, "treatment": treatment, "agent_id": agent_id,
                "role": agent_role(agent_id),
                "roi_delta_bps": 100 * (row.roi_pct - row.roi_pct_d0),
                "drawdown_delta_bps": 100 * (row.max_drawdown_pct - row.max_drawdown_pct_d0),
            })
    return pd.DataFrame(rows)


def bootstrap_mean_interval(
    values: pd.Series,
    *,
    draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Return a percentile interval for a mean using seed-level resampling."""
    clean = values.dropna().to_numpy(dtype=float)
    if not len(clean):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    sampled = rng.choice(clean, size=(draws, len(clean)), replace=True).mean(axis=1)
    low, high = np.quantile(sampled, [0.025, 0.975])
    return float(low), float(high)


def summarize_paired_values(
    values: pd.Series,
    *,
    scale: float = 1.0,
    seed_offset: int = 0,
) -> dict[str, float | int]:
    clean = values.dropna().astype(float) * scale
    low, high = bootstrap_mean_interval(clean, seed=BOOTSTRAP_SEED + seed_offset)
    return {
        "n_pairs": int(clean.size),
        "mean": float(clean.mean()),
        "std": float(clean.std(ddof=1)) if clean.size > 1 else float("nan"),
        "median": float(clean.median()),
        "q25": float(clean.quantile(0.25)),
        "q75": float(clean.quantile(0.75)),
        "ci95_low": low,
        "ci95_high": high,
        "positive_pairs": int((clean > 0).sum()),
        "negative_pairs": int((clean < 0).sum()),
        "zero_pairs": int((clean == 0).sum()),
    }


def primary_market_effects(events: pd.DataFrame) -> pd.DataFrame:
    """Summarise paired future-market effects for D0/I1/A1."""
    primary = events.loc[
        (events.study == "stock_future_micro")
        & (events.asset == "AAPL_FUT")
        & (events.event == "micro")
    ].copy()
    contrasts = (("i1", "d0"), ("a1", "i1"), ("a1", "d0"))
    metrics = (
        ("event_return_pct", "percentage_points", 1.0),
        ("event_volume", "shares", 1.0),
        ("event_trade_count", "trades", 1.0),
        ("event_realized_vol_bps", "bps", 1.0),
    )
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        mode_data = primary.loc[primary["mode"] == mode]
        for treatment, control in contrasts:
            for metric_index, (metric, unit, scale) in enumerate(metrics):
                pivot = mode_data.pivot(index="replicate", columns="treatment", values=metric)
                paired = pivot[treatment] - pivot[control]
                rows.append(
                    {
                        "mode": mode,
                        "contrast": f"{treatment.upper()}-{control.upper()}",
                        "metric": metric,
                        "unit": unit,
                        **summarize_paired_values(
                            paired,
                            scale=scale,
                            seed_offset=metric_index + 10 * len(rows),
                        ),
                    }
                )
    return pd.DataFrame(rows)


def primary_role_effects(agents: pd.DataFrame) -> pd.DataFrame:
    """Summarise paired agent outcomes after aggregating within role and seed."""
    primary = agents.loc[
        (agents.study == "stock_future_micro")
        & (agents.role.isin(["market_maker", "institutional", "retail"]))
    ].copy()
    role_seed = (
        primary.groupby(["mode", "treatment", "replicate", "role"], as_index=False)
        .agg(
            roi_pct=("roi_pct", "mean"),
            max_drawdown_pct=("max_drawdown_pct", "mean"),
            total_pnl=("total_pnl", "mean"),
            gross_exposure=("gross_exposure", "mean"),
        )
    )
    contrasts = (("i1", "d0"), ("a1", "i1"), ("a1", "d0"))
    metrics = (
        ("roi_pct", "bps", 100.0),
        ("max_drawdown_pct", "bps", 100.0),
        ("total_pnl", "currency_units", 1.0),
        ("gross_exposure", "currency_units", 1.0),
    )
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        for role in ("market_maker", "institutional", "retail"):
            subset = role_seed.loc[(role_seed["mode"] == mode) & (role_seed.role == role)]
            for treatment, control in contrasts:
                for metric_index, (metric, unit, scale) in enumerate(metrics):
                    pivot = subset.pivot(index="replicate", columns="treatment", values=metric)
                    paired = pivot[treatment] - pivot[control]
                    rows.append(
                        {
                            "mode": mode,
                            "role": role,
                            "contrast": f"{treatment.upper()}-{control.upper()}",
                            "metric": metric,
                            "unit": unit,
                            **summarize_paired_values(
                                paired,
                                scale=scale,
                                seed_offset=metric_index + 10 * len(rows),
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def secondary_market_effects(events: pd.DataFrame) -> pd.DataFrame:
    """Report descriptive C2-C0 paired ranges for the three-seed demonstration."""
    secondary = events.loc[events.study == "stock_future_bond_two_shock"].copy()
    rows: list[dict[str, Any]] = []
    for mode in SECONDARY_MODES:
        for event in ("micro", "macro"):
            for asset in ("AAPL", "AAPL_FUT", "UST10Y"):
                subset = secondary.loc[
                    (secondary["mode"] == mode)
                    & (secondary.event == event)
                    & (secondary.asset == asset)
                ]
                for metric in ("event_return_pct", "event_volume", "event_realized_vol_bps"):
                    pivot = subset.pivot(index="replicate", columns="treatment", values=metric)
                    paired = (pivot["c2"] - pivot["c0"]).dropna()
                    rows.append(
                        {
                            "mode": mode,
                            "event": event,
                            "asset": asset,
                            "contrast": "C2-C0",
                            "metric": metric,
                            "n_pairs": int(paired.size),
                            "mean": float(paired.mean()),
                            "median": float(paired.median()),
                            "minimum": float(paired.min()),
                            "maximum": float(paired.max()),
                        }
                    )
    return pd.DataFrame(rows)


def save_legacy_robustness(deltas: pd.DataFrame, output_dir: Path) -> None:
    labels = {
        "stock_market_maker": "Stock MM", "stock_institutional": "Stock institutional", "stock_retail_1": "Stock retail 1",
        "stock_retail_2": "Stock retail 2", "future_market_maker": "Future MM", "future_institutional": "Future institutional", "future_retail": "Future retail",
    }
    order = list(labels)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharey=True)
    for r, mode in enumerate(MODES):
        for c, (metric, title) in enumerate((("roi_delta_bps", "ROI change vs D0 (bps)"), ("drawdown_delta_bps", "Drawdown change vs D0 (bps)"))):
            pivot = deltas.loc[deltas["mode"] == mode].groupby(["agent_id", "treatment"])[metric].mean().unstack().reindex(index=order, columns=["i1", "a1"])
            pivot.index = [labels[value] for value in pivot.index]
            sns.heatmap(pivot, cmap="RdBu_r", center=0, annot=True, fmt=".1f", linewidths=.4, linecolor="white", cbar=r == 0, ax=axes[r,c])
            axes[r,c].set_title(f"{mode.title()} - {title}")
            axes[r,c].set_xlabel("Treatment")
            axes[r,c].set_xticklabels(["I1", "A1"])
            axes[r,c].set_ylabel("")
    fig.suptitle("Strategy robustness when a future market is linked", y=.995, fontsize=14)
    fig.tight_layout(rect=(0,0,1,.98))
    fig.savefig(output_dir / "02_legacy_robustness.png", dpi=220)
    plt.close(fig)


def save_primary_market_effects(effects: pd.DataFrame, output_dir: Path) -> None:
    metrics = (
        ("event_return_pct", "Future event return effect (percentage points)"),
        ("event_volume", "Future event volume effect (shares)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (metric, title) in zip(axes, metrics):
        data = effects.loc[effects.metric == metric].copy()
        data["label"] = data["mode"].str.title() + " | " + data["contrast"]
        data = data.sort_values(["mode", "contrast"], ascending=[False, True]).reset_index(drop=True)
        y = np.arange(len(data))
        lower = data["mean"] - data["ci95_low"]
        upper = data["ci95_high"] - data["mean"]
        for index, row in data.iterrows():
            color = {"frozen": "#168aad", "llm": "#d97706"}[row["mode"]]
            ax.errorbar(
                row["mean"],
                index,
                xerr=[[lower.iloc[index]], [upper.iloc[index]]],
                fmt="o",
                color=color,
                elinewidth=2,
                capsize=4,
                markersize=6,
            )
        ax.axvline(0, color="#475569", lw=1, ls="--")
        ax.set_yticks(y, data["label"])
        ax.set_title(title)
        ax.set_xlabel("Mean paired difference with 95% bootstrap CI")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Primary stock/future treatment effects across ten paired seeds", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "07_primary_market_effects_ci.png", dpi=240)
    plt.close(fig)


def save_primary_role_effects(effects: pd.DataFrame, output_dir: Path) -> None:
    data = effects.loc[
        (effects.contrast == "A1-D0")
        & effects.metric.isin(["roi_pct", "max_drawdown_pct"])
    ].copy()
    metrics = (
        ("roi_pct", "ROI effect (bps)"),
        ("max_drawdown_pct", "Maximum-drawdown effect (bps)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, (metric, title) in zip(axes, metrics):
        subset = data.loc[data.metric == metric].copy()
        subset["label"] = subset["mode"].str.title() + " | " + subset["role"].str.replace("_", " ").str.title()
        subset = subset.sort_values(["mode", "role"], ascending=[False, True]).reset_index(drop=True)
        y = np.arange(len(subset))
        lower = subset["mean"] - subset["ci95_low"]
        upper = subset["ci95_high"] - subset["mean"]
        for index, row in subset.iterrows():
            color = {"frozen": "#168aad", "llm": "#d97706"}[row["mode"]]
            ax.errorbar(
                row["mean"],
                index,
                xerr=[[lower.iloc[index]], [upper.iloc[index]]],
                fmt="o",
                color=color,
                elinewidth=2,
                capsize=4,
                markersize=6,
            )
        ax.axvline(0, color="#475569", lw=1, ls="--")
        ax.set_yticks(y, subset["label"])
        ax.set_title(title)
        ax.set_xlabel("Mean paired A1-D0 difference with 95% bootstrap CI")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Role-level robustness after enabling the linked market", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "08_primary_role_effects_ci.png", dpi=240)
    plt.close(fig)


def save_publication_summary(
    market_effects: pd.DataFrame,
    role_effects: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create one compact figure containing the paper's primary evidence."""
    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.2))
    panels = (
        (axes[0, 0], market_effects, "event_return_pct", "Future return", "Percentage-point difference"),
        (axes[0, 1], market_effects, "event_volume", "Future volume", "Share difference"),
    )
    mode_colors = {"frozen": "#0072B2", "llm": "#D55E00"}
    for label, (axis, frame, metric, title, xlabel) in zip(("A", "B"), panels):
        data = frame.loc[frame.metric == metric].copy()
        data["mode_label"] = data["mode"].map({"frozen": "Frozen", "llm": "LLM"})
        data["label"] = data["mode_label"] + "  " + data["contrast"]
        data["sort_mode"] = data["mode"].map({"frozen": 0, "llm": 1})
        data["sort_contrast"] = data["contrast"].map({"I1-D0": 0, "A1-I1": 1, "A1-D0": 2})
        data = data.sort_values(["sort_mode", "sort_contrast"]).reset_index(drop=True)
        positions = np.arange(len(data))
        for position, row in data.iterrows():
            axis.errorbar(
                row["mean"],
                position,
                xerr=[[row["mean"] - row["ci95_low"]], [row["ci95_high"] - row["mean"]]],
                fmt="o",
                color=mode_colors[row["mode"]],
                elinewidth=2,
                capsize=3.5,
                markersize=5.5,
            )
        axis.axvline(0, color="#64748B", lw=1, ls="--")
        axis.set_yticks(positions, data["label"])
        axis.set_xlabel(xlabel)
        axis.set_title(f"{label}  {title}", loc="left", fontweight="bold")
        axis.grid(axis="y", visible=False)

    selected_roles = role_effects.loc[
        (role_effects.contrast == "A1-D0")
        & role_effects.metric.isin(["roi_pct", "max_drawdown_pct"])
    ].copy()
    role_panels = (
        (axes[1, 0], "roi_pct", "C  Role-level ROI", "Basis-point difference"),
        (axes[1, 1], "max_drawdown_pct", "D  Role-level maximum drawdown", "Basis-point difference"),
    )
    for axis, metric, title, xlabel in role_panels:
        data = selected_roles.loc[selected_roles.metric == metric].copy()
        data["mode_label"] = data["mode"].map({"frozen": "Frozen", "llm": "LLM"})
        data["label"] = (
            data["mode_label"]
            + "  "
            + data["role"].str.replace("_", " ").str.title()
        )
        data["sort_mode"] = data["mode"].map({"frozen": 0, "llm": 1})
        data["sort_role"] = data["role"].map({"market_maker": 0, "institutional": 1, "retail": 2})
        data = data.sort_values(["sort_mode", "sort_role"]).reset_index(drop=True)
        positions = np.arange(len(data))
        for position, row in data.iterrows():
            axis.errorbar(
                row["mean"],
                position,
                xerr=[[row["mean"] - row["ci95_low"]], [row["ci95_high"] - row["mean"]]],
                fmt="o",
                color=mode_colors[row["mode"]],
                elinewidth=2,
                capsize=3.5,
                markersize=5.5,
            )
        axis.axvline(0, color="#64748B", lw=1, ls="--")
        axis.set_yticks(positions, data["label"])
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left", fontweight="bold")
        axis.grid(axis="y", visible=False)

    figure.suptitle(
        "Paired treatment effects across ten random seeds",
        y=0.985,
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.015,
        "Points are paired means; bars are 95% seed-level bootstrap intervals. Role panels compare A1 with D0.",
        ha="center",
        fontsize=9,
        color="#475569",
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95), h_pad=2.2, w_pad=2.0)
    figure.savefig(output_dir / "09_publication_ecology_summary.png", dpi=300, bbox_inches="tight")
    figure.savefig(output_dir / "09_publication_ecology_summary.pdf", bbox_inches="tight")
    plt.close(figure)


def leave_one_seed_out_market_effects(events: pd.DataFrame) -> pd.DataFrame:
    """Record sensitivity of primary paired means to removal of one seed."""
    primary = events.loc[
        (events.study == "stock_future_micro")
        & (events.asset == "AAPL_FUT")
        & (events.event == "micro")
    ].copy()
    rows: list[dict[str, Any]] = []
    for mode in MODES:
        mode_data = primary.loc[primary["mode"] == mode]
        for treatment, control in (("i1", "d0"), ("a1", "i1"), ("a1", "d0")):
            for metric in ("event_return_pct", "event_volume", "event_trade_count", "event_realized_vol_bps"):
                pivot = mode_data.pivot(index="replicate", columns="treatment", values=metric)
                paired = (pivot[treatment] - pivot[control]).dropna()
                for omitted_seed in paired.index:
                    retained = paired.drop(index=omitted_seed)
                    rows.append(
                        {
                            "mode": mode,
                            "contrast": f"{treatment.upper()}-{control.upper()}",
                            "metric": metric,
                            "omitted_seed": int(omitted_seed),
                            "n_pairs": int(retained.size),
                            "mean": float(retained.mean()),
                        }
                    )
    return pd.DataFrame(rows)


def save_three_market_paths(paths: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    events = {"micro": pd.Timestamp("2025-03-01T09:45:00+00:00"), "macro": pd.Timestamp("2025-03-01T10:00:00+00:00")}
    mode = "frozen"
    for ax, (event_name, event_time) in zip(axes, events.items()):
        subset = paths.loc[(paths.study == "stock_future_bond_two_shock") & (paths["mode"] == mode)].copy()
        for treatment in ("c0", "c2"):
            for asset in ("AAPL", "AAPL_FUT", "UST10Y"):
                data = subset.loc[(subset.treatment == treatment) & (subset.asset == asset)].copy()
                base = data.loc[data.timestamp == event_time - pd.Timedelta(seconds=30), ["replicate", "price"]].set_index("replicate")["price"]
                data["relative_min"] = (data.timestamp - event_time).dt.total_seconds() / 60
                data["return_pct"] = data.apply(lambda row: 100 * (row.price / base.get(row.replicate, np.nan) - 1), axis=1)
                data = data.loc[(data.relative_min >= -8) & (data.relative_min <= 15)]
                mean = data.groupby("relative_min").return_pct.mean()
                style = "-" if treatment == "c2" else "--"
                ax.plot(mean.index, mean.values, color=ASSET_COLORS[asset], ls=style, lw=2, label=f"{asset} {treatment.upper()}")
        ax.axvspan(0, 4, color="#f6bd60", alpha=.18)
        ax.axvline(0, color="#4b5563", ls="--", lw=1)
        ax.axhline(0, color="#cbd5e1", lw=1)
        ax.set_title(f"{event_name.title()} event")
        ax.set_xlabel("Minutes from event")
        ax.set_ylabel("Return from pre-event price (%)")
        ax.legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("Frozen C0/C2 three-market event paths: dashed control, solid linked", y=.99, fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output_dir / "03_three_market_event_paths.png", dpi=220)
    plt.close(fig)


def save_three_market_activity(events: pd.DataFrame, output_dir: Path) -> None:
    session = events.loc[(events.study == "stock_future_bond_two_shock") & (events.event == "macro")].copy()
    metrics = (("total_trade_count", "Session trade count"), ("total_volume", "Session volume"), ("session_return_pct", "Session return (%)"))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    mode = "frozen"
    for c, (metric, title) in enumerate(metrics):
        ax = axes[c]
        data = session.loc[session["mode"] == mode]
        sns.barplot(data=data, x="asset", y=metric, hue="treatment", hue_order=["c0","c2"], palette=[COLORS["c0"], COLORS["c2"]], errorbar=None, ax=ax)
        sns.stripplot(data=data, x="asset", y=metric, hue="treatment", hue_order=["c0","c2"], dodge=True, palette=["#111827", "#111827"], size=3.5, ax=ax, legend=False)
        ax.set_title(title)
        ax.set_xlabel("")
        if c:
            ax.set_ylabel("")
            legend = ax.get_legend()
            if legend:
                legend.remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, ["C0 control", "C2 linked"], loc="upper center", ncol=2, frameon=False)
    fig.suptitle("Frozen three-market activity by treatment", y=.97, fontsize=13)
    fig.tight_layout(rect=(0,0,1,.92))
    fig.savefig(output_dir / "04_three_market_activity.png", dpi=220)
    plt.close(fig)


def save_basis(decisions: pd.DataFrame, output_dir: Path) -> None:
    c2 = decisions.loc[(decisions.study == "stock_future_bond_two_shock") & (decisions.treatment == "c2")].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data = c2.loc[c2["mode"] == "frozen"].copy()
    data["relative_min"] = (data.timestamp - pd.Timestamp("2025-03-01T09:45:00+00:00")).dt.total_seconds()/60
    mean = data.groupby("relative_min").abs_observed_basis_bps.mean()
    axes[0].plot(mean.index, mean.values, lw=2.4, color="#0072B2", label="Frozen")
    axes[0].axvspan(0,4,color="#f6bd60",alpha=.18)
    axes[0].axvline(15,color="#d97706",ls="--",lw=1,label="Macro shock")
    axes[0].set_xlim(-8,30)
    axes[0].set_title("C2 stock-future basis observation")
    axes[0].set_xlabel("Minutes from micro shock")
    axes[0].set_ylabel("Absolute observed basis (bps)")
    axes[0].legend(frameon=False)
    outcomes = c2.loc[c2.execution_outcome != "observed_only"].groupby("execution_outcome").size()
    outcomes = outcomes.reindex(["fully_hedged","partial_or_unhedged","unfilled"], fill_value=0)
    bottom = 0
    palette = {"fully_hedged":"#1f8a70", "partial_or_unhedged":"#d97706", "unfilled":"#dc2626"}
    for outcome, count in outcomes.items():
        axes[1].bar(["Frozen"], [count], bottom=bottom, label=outcome.replace("_", " "), color=palette[outcome])
        bottom += count
    axes[1].set_title("C2 cross-market execution outcomes")
    axes[1].set_ylabel("Decision count across 3 seeds")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "05_c2_basis_execution.png", dpi=220)
    plt.close(fig)


def save_llm_quality(quality: pd.DataFrame, output_dir: Path) -> None:
    llm = quality.loc[quality["mode"] == "llm"].copy()
    llm["condition"] = llm.study.map({"stock_future_micro":"Primary D0/I1/A1"})
    fig, axes = plt.subplots(1, 2, figsize=(13,5))
    sns.barplot(data=llm, x="run_id", y="self_trade_prevention", hue="condition", errorbar=None, ax=axes[0])
    axes[0].tick_params(axis="x", rotation=70, labelsize=7)
    axes[0].set_title("LLM runs: prevented self-crosses")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Count")
    statuses = llm[["run_id","slow_loop_completed","slow_loop_rejected","slow_loop_failed"]].set_index("run_id")
    statuses.plot(kind="bar", stacked=True, color=["#1f8a70", "#d97706", "#dc2626"], ax=axes[1])
    axes[1].tick_params(axis="x", rotation=70, labelsize=7)
    axes[1].set_title("LLM runs: slow-loop status")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Decision count")
    axes[1].legend(["completed", "rejected", "failed"], frameon=False)
    fig.suptitle("Operational quality: LLM treatment needs execution guardrails", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_dir / "06_llm_quality_flags.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_report(
    events: pd.DataFrame,
    quality: pd.DataFrame,
    decisions: pd.DataFrame,
    market_effects: pd.DataFrame,
    role_effects: pd.DataFrame,
    secondary_effects: pd.DataFrame,
    output_dir: Path,
) -> None:
    completed_runs = int(quality.run_id.nunique())
    llm_quality = quality.loc[quality["mode"] == "llm"]
    total_llm = int(
        llm_quality[["slow_loop_completed", "slow_loop_rejected", "slow_loop_failed"]]
        .sum()
        .sum()
    )
    completed_llm = int(llm_quality.slow_loop_completed.sum())
    primary_table = market_effects.loc[
        market_effects.metric.isin(["event_return_pct", "event_volume"]),
        ["mode", "contrast", "metric", "unit", "n_pairs", "mean", "ci95_low", "ci95_high", "positive_pairs", "negative_pairs"],
    ].round(3)
    role_table = role_effects.loc[
        (role_effects.contrast == "A1-D0")
        & role_effects.metric.isin(["roi_pct", "max_drawdown_pct"]),
        ["mode", "role", "metric", "unit", "n_pairs", "mean", "ci95_low", "ci95_high", "positive_pairs", "negative_pairs"],
    ].round(3)
    c2 = decisions.loc[
        (decisions.study == "stock_future_bond_two_shock")
        & (decisions.treatment == "c2")
        & (decisions.execution_outcome != "observed_only")
    ]
    c2_outcomes = (
        c2.groupby(["mode", "execution_outcome"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["fully_hedged", "partial_or_unhedged", "unfilled"], fill_value=0)
    )
    relationship_deliveries = int(
        quality.loc[quality.treatment.isin(["i1", "a1", "c2"]), "relationship_deliveries"].sum()
    )
    text = [
        "# Financial Market Ecology: Publication Replication",
        "",
        "## Scope",
        "",
        f"The locked batch contains {completed_runs} completed one-hour simulations. The primary D0/I1/A1 experiment uses ten paired seeds in frozen and LLM-adaptive modes. The secondary C0/C2 stock-future-bond demonstration uses three paired frozen-strategy seeds and is reported descriptively.",
        "",
        "The seed is the replication unit. Confidence intervals resample paired seed-level treatment differences; agents, slow-loop decisions, orders, and fills are not treated as independent observations.",
        "",
        "## Run Quality",
        "",
        f"Across LLM cells, {completed_llm} of {total_llm} recorded slow-loop outcomes completed successfully. Relationship-enabled treatments recorded {relationship_deliveries} relationship-mediated event deliveries. Detailed warnings and execution outcomes remain in the run-quality and decision tables.",
        "",
        "## RQ1: Shock Transmission",
        "",
        "D0 to I1 isolates information delivery. I1 to A1 adds reference valuation and executable two-leg arbitrage. The table reports paired future-market effects with 95% seed-level bootstrap confidence intervals:",
        "",
        table(primary_table),
        "",
        "The mechanism claim is supported when the relationship ledger, active-event strategy records, and executable orders agree. Price, volume, and volatility effects are interpreted from the paired intervals rather than from one representative path.",
        "",
        "## RQ2: Strategy Robustness",
        "",
        "Role outcomes are first averaged within each role and seed, then contrasted between A1 and D0. ROI and maximum-drawdown effects are shown in basis points:",
        "",
        table(role_table),
        "",
        "A strategy is described as stable only when its interval is both narrow and economically small. LLM-versus-frozen differences are not interpreted as a pure model ranking when execution-quality diagnostics differ materially between modes.",
        "",
        "## Secondary Three-Market Demonstration",
        "",
        "C0/C2 retains three paired frozen-strategy seeds and demonstrates the distinction between relationship-mediated micro transmission and direct common macro exposure. It is not used for a publication-level magnitude estimate. C2 execution outcomes were:",
        "",
        table(c2_outcomes),
        "",
        "The complete descriptive C2-C0 paired ranges are available in `secondary_market_effects.csv`.",
        "",
        "## Interpretation Rule",
        "",
        "- Structural evidence establishes whether a channel delivered information or generated linked orders.",
        "- Paired confidence intervals describe the magnitude and uncertainty of primary market and role effects.",
        "- Quality diagnostics determine whether adaptive financial outcomes are interpretable or execution-confounded.",
        "- Conclusions remain internal to the configured synthetic ecology and do not estimate real AAPL, futures, or Treasury effects.",
    ]
    (output_dir / "financial_ecology_publication_report.md").write_text("\n".join(text) + "\n")


def write_claim_manifest(
    quality: pd.DataFrame,
    decisions: pd.DataFrame,
    market_effects: pd.DataFrame,
    role_effects: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write the exact computed values cited in the shared paper."""
    def effect(frame: pd.DataFrame, mode: str, contrast: str, metric: str, role: str | None = None) -> dict[str, Any]:
        selected = frame.loc[
            (frame["mode"] == mode)
            & (frame["contrast"] == contrast)
            & (frame["metric"] == metric)
        ]
        if role is not None:
            selected = selected.loc[selected["role"] == role]
        row = selected.iloc[0]
        return {
            "n_pairs": int(row["n_pairs"]),
            "mean": float(row["mean"]),
            "ci95_low": float(row["ci95_low"]),
            "ci95_high": float(row["ci95_high"]),
        }

    primary = quality.loc[quality.study == "stock_future_micro"]
    adaptive = primary.loc[primary["mode"] == "llm"]
    frozen = primary.loc[primary["mode"] == "frozen"]
    c2 = decisions.loc[
        (decisions.study == "stock_future_bond_two_shock")
        & (decisions.treatment == "c2")
        & (decisions.execution_outcome != "observed_only")
    ]
    claims = {
        "accepted_runs": int(quality.run_id.nunique()),
        "primary_runs": int(primary.run_id.nunique()),
        "primary_paired_seeds_per_cell": len(PRIMARY_REPLICATES),
        "adaptive_slow_loop": {
            "completed": int(adaptive.slow_loop_completed.sum()),
            "rejected": int(adaptive.slow_loop_rejected.sum()),
            "failed": int(adaptive.slow_loop_failed.sum()),
        },
        "primary_relationship_deliveries": int(primary.relationship_deliveries.sum()),
        "primary_self_trade_preventions": {
            "frozen": int(frozen.self_trade_prevention.sum()),
            "llm": int(adaptive.self_trade_prevention.sum()),
        },
        "market_effects": {
            "frozen_a1_minus_i1_return_percentage_points": effect(market_effects, "frozen", "A1-I1", "event_return_pct"),
            "frozen_a1_minus_i1_volume_shares": effect(market_effects, "frozen", "A1-I1", "event_volume"),
            "frozen_a1_minus_i1_trade_count": effect(market_effects, "frozen", "A1-I1", "event_trade_count"),
            "llm_i1_minus_d0_volume_shares": effect(market_effects, "llm", "I1-D0", "event_volume"),
        },
        "role_effects_a1_minus_d0": {
            "llm_market_maker_roi_bps": effect(role_effects, "llm", "A1-D0", "roi_pct", "market_maker"),
            "llm_institutional_roi_bps": effect(role_effects, "llm", "A1-D0", "roi_pct", "institutional"),
            "llm_retail_roi_bps": effect(role_effects, "llm", "A1-D0", "roi_pct", "retail"),
        },
        "secondary_c2_execution_outcomes": {
            str(outcome): int(count)
            for outcome, count in c2.execution_outcome.value_counts().items()
        },
    }
    (output_dir / "publication_claims.json").write_text(json.dumps(claims, indent=2) + "\n")


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    paths, events, quality, agents, responses, decisions = collect(args.runs_dir)
    deltas = paired_legacy_deltas(agents)
    market_effects = primary_market_effects(events)
    role_effects = primary_role_effects(agents)
    secondary_effects = secondary_market_effects(events)
    leave_one_out = leave_one_seed_out_market_effects(events)
    paths.to_csv(args.output_dir / "price_paths_30s.csv", index=False)
    events.to_csv(args.output_dir / "market_event_metrics.csv", index=False)
    quality.to_csv(args.output_dir / "run_quality.csv", index=False)
    agents.to_csv(args.output_dir / "agent_metrics.csv", index=False)
    responses.to_csv(args.output_dir / "slow_loop_responses.csv", index=False)
    decisions.to_csv(args.output_dir / "cross_market_decisions.csv", index=False)
    deltas.to_csv(args.output_dir / "legacy_paired_role_deltas.csv", index=False)
    market_effects.to_csv(args.output_dir / "primary_market_effects.csv", index=False)
    role_effects.to_csv(args.output_dir / "primary_role_effects.csv", index=False)
    secondary_effects.to_csv(args.output_dir / "secondary_market_effects.csv", index=False)
    leave_one_out.to_csv(args.output_dir / "primary_market_leave_one_seed_out.csv", index=False)
    save_legacy_paths(paths, args.output_dir)
    save_legacy_robustness(deltas, args.output_dir)
    save_three_market_paths(paths, args.output_dir)
    save_three_market_activity(events, args.output_dir)
    save_basis(decisions, args.output_dir)
    save_llm_quality(quality, args.output_dir)
    save_primary_market_effects(market_effects, args.output_dir)
    save_primary_role_effects(role_effects, args.output_dir)
    save_publication_summary(market_effects, role_effects, args.output_dir)
    write_report(
        events,
        quality,
        decisions,
        market_effects,
        role_effects,
        secondary_effects,
        args.output_dir,
    )
    write_claim_manifest(quality, decisions, market_effects, role_effects, args.output_dir)
    print(f"Wrote analysis package to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
