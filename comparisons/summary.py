"""Collect the per-study results into the Markdown tables the README quotes.

    python -m comparisons.summary

Reads ``results/headroom_*.json`` and ``results/comparison_*.json`` and prints
one table for each. The README's numbers are pasted from this output, so they
can always be traced back to a run of ``run_pipeline.py`` rather than to
whatever settings happened to be used by hand.
"""

import json
import os

from config import cnfg

STUDIES = {
    "trolley": "Trolley, linear",
    "trolley_nonlinear": "Trolley, nonlinear",
    "thermal": "Thermal, linear",
    "thermal_nonlinear": "Thermal, nonlinear",
}
COMPARISON_ARMS = ("classical PID", "best fixed PID", "LSTM scheduled",
                   "pole place/episode")


def _load(kind: str, study: str) -> dict | None:
    path = os.path.join(cnfg.METRICS_DIR, f"{kind}_{study}.json")
    if not os.path.exists(path):
        return None
    with open(path) as handle:
        return json.load(handle)


def _number(value: float) -> str:
    return f"{value:.1f}" if abs(value) >= 100 else f"{value:.2f}"


def headroom_table() -> str:
    rows, bins = [], set()
    for study, label in STUDIES.items():
        data = _load("headroom", study)
        if data is None:
            continue
        bins.add(data["n_bins"])
        cells = [
            data["global_iae_mean"],
            data["per_episode_iae_mean"],
            data["per_episode_schedule_iae_mean"],
            data["lstm_iae_mean"],
        ]
        rows.append(
            f"| {label:<20} | "
            + " | ".join(_number(c) for c in cells)
            + f" | {data['total_headroom_percent']:+.1f} %"
            + f" | {data['captured_percent']:+.1f} % |"
        )
    if not rows:
        return "(no headroom results yet)"
    n_bins = "/".join(str(b) for b in sorted(bins))
    header = (
        "| Study                | one global constant | best constant /episode "
        f"| {n_bins}-bin schedule /episode | LSTM scheduler | headroom | captured |\n"
        "|----------------------|---:|---:|---:|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows)


def comparison_table(protocol: str = "rejection", metric: str = "iae") -> str:
    rows = []
    for study, label in STUDIES.items():
        data = _load("comparison", study)
        if data is None:
            continue
        arms = data["protocols"][protocol]
        rows.append(
            f"| {label:<20} | "
            + " | ".join(_number(arms[a][metric]["mean"]) for a in COMPARISON_ARMS)
            + f" | {data['runs']} |"
        )
    if not rows:
        return "(no comparison results yet)"
    header = (
        "| Study                | Classical rule | Best fixed gains | LSTM scheduled "
        "| Pole placement /episode | episodes |\n"
        "|----------------------|---:|---:|---:|---:|---:|"
    )
    return header + "\n" + "\n".join(rows)


def main() -> None:
    print("Adaptation headroom (whole-episode IAE, lower is better)\n")
    print(headroom_table())
    print("\nFour-arm comparison (final-step IAE, disturbance rejection)\n")
    print(comparison_table())


if __name__ == "__main__":
    main()
