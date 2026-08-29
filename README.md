# Adaptive PID Regulation by Neural Networks

An LSTM gain scheduler for PID control, trained through a differentiable
simulation and measured against fixed-gain baselines on two plants: a
mass–spring–damper trolley and a lumped-capacity thermal system.

Originally a bachelor's thesis; since rebuilt to make the experiment
reproducible and the comparison honest. See [What changed](#what-changed).

## Contents

1. [Result](#result)
2. [How it works](#how-it-works)
3. [Setup](#setup)
4. [Running it](#running-it)
5. [Layout](#layout)
6. [Configuration](#configuration)
7. [What changed](#what-changed)
8. [Acknowledgements](#acknowledgements)

## Result

24 held-out episodes per plant, seed 42, mean IAE over the final reference step.
Every controller runs the *same* episodes — same reference staircase, same load
disturbances, same randomised plant parameters — and the plant and controller are
reset before every run. Lower is better.

| Plant   | Protocol  | Classical rule | Best fixed gains | **LSTM scheduled** | Per-episode pole placement |
|---------|-----------|---------------:|-----------------:|-------------------:|---------------------------:|
| Trolley | tracking  |           7.41 |         **5.37** |               5.47 |                       7.67 |
| Trolley | rejection |           8.02 |         **5.60** |               5.68 |                       8.20 |
| Thermal | tracking  |          821.6 |            381.4 |          **380.1** |                      538.8 |
| Thermal | rejection |          992.9 |            495.7 |          **495.3** |                      728.7 |

**The scheduler clearly beats classical tuning** — 26 % lower IAE on the trolley,
54 % on the thermal plant — and it beats gains derived from each episode's *true*
plant parameters, which it never sees.

**It does not beat the best constant gains found by direct search.** On both
plants it lands within 2 % of that baseline, on either side of it. On the trolley
it also uses about 30 % more actuator movement to get there. Whether adaptivity
is worth anything here is therefore not settled by these experiments; on this
problem, a well-searched constant PID is already about as good.

That narrower claim is the honest one. Beating a tuning rule mostly shows that
tuning rules are conservative — they are designed to be, since they get one shot
at an unknown plant. The searched baseline is the one a gain *scheduler* has to
beat to justify itself, and reporting it is the difference between a result and
a press release.

Full tables, including overshoot, settling time, per-episode win rates and how
many runs each mean is computed from, are printed by `comparisons.compare` and
written to `results/`.

## How it works

```
        ┌──────────────┐  gains   ┌─────┐   u    ┌───────┐   y
        │ LSTM (5→48→3)│─────────▶│ PID │───────▶│ plant │────┬──▶
        └──────────────┘          └─────┘        └───────┘    │
               ▲                                              │
               └────────── normalised loop history ───────────┘
```

* **The plants are written in torch**, so the whole closed loop is
  differentiable and the tracking loss can be back-propagated through the
  simulation into the network. Gradients are exact, not estimated.
* **Truncated BPTT**: an optimizer step every `tbptt_window` samples, with the
  plant, the controller and the LSTM hidden state detached at the boundary.
* **The LSTM sees dimensionless features** — normalised error, error rate, and
  each gain as a fraction of its ceiling — so one architecture serves a plant
  measured in metres and one measured in kelvin.
* **Its head is a sigmoid into a per-gain ceiling**, so gains are bounded by
  construction and Kp, Ki and Kd can occupy the very different ranges a real
  controller needs.
* **It is warm-started at the classical gains**, so training is judged by whether
  it improves on a competent controller rather than on a random one.
* **The RBF network** is a one-step-ahead plant surrogate, fitted on trajectories
  from randomised initial conditions. Held-out one-step error is ≈1 % of the
  output's own spread on both plants. Setting `loss_target: surrogate` trains the
  controller *through* it instead of through the true plant — the setting you are
  forced into when the plant is not differentiable, and the reason the surrogate
  is in the architecture at all.

Each episode redraws the reference staircase, the load disturbances and the
plant's physical parameters, which is what makes gain scheduling a meaningful
thing to attempt: on a fixed plant tracking a constant, a constant gain is optimal.

## Setup

```sh
git clone https://github.com/vsem-azamat/neural-network-pid-regulation
cd neural-network-pid-regulation
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[dev]"
```

Python 3.11+. CPU is fine — a full pipeline run takes a few minutes.

## Running it

Everything, both plants, reproducibly:

```sh
python run_pipeline.py
```

Or stage by stage:

```sh
python -m simulations.analyse_plant trolley   # step, phase portrait, Bode, Nyquist
python -m learning.train_rbf        trolley   # fit the plant surrogate
python -m learning.train_lstm_pid   trolley   # train the gain scheduler
python -m comparisons.compare       trolley --runs 24
```

Both plants accept `trolley` or `thermal`. Add `--show` to open figures,
`--seed N` to change the seed. Metrics are written to `results/*.json`, figures
to `plots/`.

Tests:

```sh
python -m pytest          # 70 tests
ruff check .
```

## Layout

| Path                        | What it holds                                            |
|-----------------------------|----------------------------------------------------------|
| `entities/systems/`         | The plants. Add one file to add a plant.                 |
| `entities/pid.py`           | Discrete PID, five discretisations, differentiable.       |
| `models/`                   | `LSTMAdaptivePID`, `SystemRBFModel`.                      |
| `utils/run.py`              | Simulation loop and TBPTT training.                       |
| `utils/metrics.py`          | Step-response metrics.                                    |
| `utils/tuning.py`           | Classical tuning and pole placement.                      |
| `learning/scenarios.py`     | Episode generation.                                       |
| `comparisons/`              | Baselines and the comparison harness.                     |
| `config/ymls/`              | Per-plant configuration.                                  |
| `tests/`                    | 70 tests, most pinned to a specific past defect.          |

## Configuration

One YAML per plant in `config/ymls/`, validated by pydantic on load. Network
input widths are *not* configurable — they are fixed by the feature extractors,
which removes a way for the two to silently disagree.

Notable knobs:

| Key                              | Meaning                                                 |
|----------------------------------|---------------------------------------------------------|
| `control.gain_ceiling`           | Hard cap on each gain. Size it from what the plant needs.|
| `control.error_scale`            | Characteristic error, used to normalise the loss and features. |
| `scenario.randomize_plant`       | Per-episode parameter ranges.                            |
| `scenario.disturbance_scale`     | Load disturbance amplitude.                              |
| `learning.lstm.loss_target`      | `plant` (exact gradients) or `surrogate` (model-based).  |
| `learning.lstm.effort_weight`    | Penalty on actuator movement.                            |

## What changed

The original version did not train. `extract_rbf_input` rebuilt its feature row
with `torch.tensor([...])`, which copies numbers out of the autograd graph, so
the backward pass stopped at the plant boundary. The loss still had
`requires_grad=True` — the RBF's own weights were in the graph — so nothing
raised, the loss printed, the figures drew and the weights saved, while every
LSTM parameter came back with `grad=None` and `optimizer.step()` did nothing.
Measured before the fix: 6/6 parameter tensors `grad=None`, `sum|grad|` exactly
`0.0`. Every published figure came from an untrained, randomly initialised
network. `tests/test_gradient_flow.py` now fails if that recurs.

Other things that were wrong, roughly in order of how much they mattered:

* Two of the four entry points crashed on import — a deleted `SpringDamper` and a
  stray `from turtle import title`.
* Training episodes were `[torch.randn(1) * 10] * n`, which repeats one object:
  one step to a random constant, no disturbance, fixed plant.
* Load disturbances were computed in both comparison scripts and then never used;
  no call site anywhere passed one to a plant.
* Overshoot used `max(y)` and rise time compared against `0.9·|setpoint|`, both of
  which break on negative setpoints — and setpoints were drawn from (−20, 20).
* Settling time returned the *first* entry into the tolerance band instead of the
  last exit from it, certifying a permanently oscillating response as settled.
* The third comparison arm ran without a reset, starting from wherever the
  previous arm had left the plant, and each arm drew its own random setpoint.
* A metric histogram was drawn from one run and annotated with another's
  statistics.
* The LSTM's hidden state was assigned to a local and discarded, so the
  recurrence was off during every comparison.
* The thermal plant exchanged heat with 0 K rather than with ambient.
* The trolley subtracted the disturbance where the thermal plant added it.
* The RBF was fitted on independent random states including an acceleration that
  `apply_control` immediately overwrote, over a range 5× wider than the loop ever
  visits, and the thermal model was then validated in Celsius against a plant
  running in kelvin.
* `torch.autograd.set_detect_anomaly(True)` was left on in the run path.
* The two-point identification mixed coefficients from two different variants of
  the method, reporting T = 222 s for a plant whose true time constant is 100 s.
* Ziegler–Nichols returned the derivative *time* as Kd, short by a factor of Kp.
* The derivative acted on the error, so a setpoint step asked for 200 units of
  control on one sample and the trolley lurched to −8.9 before recovering.
* `requirements.txt` asked for a package named `pytorch` and omitted `pyyaml`.
* The README promised Nyquist and Bode plots that no code produced. It does now.

## Acknowledgements

Bachelor's thesis work, supervised by Ing. Cyril Oswald, Ph.D.
([ORCID](https://orcid.org/0000-0001-5268-2785)).
