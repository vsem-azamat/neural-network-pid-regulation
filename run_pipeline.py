"""Run the whole study end to end, reproducibly.

    python run_pipeline.py                 # both plants, full pipeline
    python run_pipeline.py --system trolley
    python run_pipeline.py --skip analyse  # reuse existing plots

Stages, in dependency order:

    analyse   open-loop plant characterisation (step, phase, Bode, Nyquist)
    rbf       fit the plant surrogate
    lstm      train the gain scheduler on top of it
    compare   score it against the fixed-gain baselines

Everything is seeded from a single value, so a rerun reproduces the figures and
the metric tables exactly. Results land in results/ as JSON and in plots/.
"""

import argparse
import subprocess
import sys
import time

STAGES = {
    "analyse": ["-m", "simulations.analyse_plant"],
    "rbf": ["-m", "learning.train_rbf"],
    "lstm": ["-m", "learning.train_lstm_pid"],
    "compare": ["-m", "comparisons.compare"],
}
SYSTEMS = ["trolley", "thermal"]


def run(stage: str, system: str, seed: int, extra: list[str]) -> float:
    command = [sys.executable, *STAGES[stage], system, "--seed", str(seed), *extra]
    print(f"\n{'=' * 72}\n>>> {stage}: {system}\n{'=' * 72}")
    print(f"$ {' '.join(command)}\n", flush=True)
    started = time.time()
    result = subprocess.run(command)
    if result.returncode != 0:
        raise SystemExit(
            f"Stage '{stage}' failed for {system} (exit {result.returncode})"
        )
    return time.time() - started


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--system", choices=SYSTEMS, action="append",
                        help="Restrict to one plant (repeatable). Default: both.")
    parser.add_argument("--skip", choices=list(STAGES), action="append", default=[],
                        help="Skip a stage (repeatable).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=24,
                        help="Held-out episodes in the comparison stage.")
    args = parser.parse_args()

    systems = args.system or SYSTEMS
    stages = [s for s in STAGES if s not in args.skip]

    timings: dict[str, float] = {}
    for system in systems:
        for stage in stages:
            extra = ["--runs", str(args.runs)] if stage == "compare" else []
            timings[f"{stage}/{system}"] = run(stage, system, args.seed, extra)

    print(f"\n{'=' * 72}\nPipeline complete (seed {args.seed})\n{'=' * 72}")
    for key, seconds in timings.items():
        print(f"  {key:<24}{seconds:7.1f} s")
    print(f"  {'total':<24}{sum(timings.values()):7.1f} s")
    print("\nMetrics: results/*.json    Figures: plots/")


if __name__ == "__main__":
    main()
