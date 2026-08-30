# Adaptive PID Regulation by Neural Networks

An LSTM gain scheduler for PID control, trained through a differentiable
simulation and measured against fixed-gain baselines — including the one that
actually matters, the best constant gains found by direct search — on four
studies: a mass–spring–damper trolley and a lumped-capacity thermal system,
each in a linear and a deliberately nonlinear variant.

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

**Measure the ceiling before you measure the method.** Before asking whether a
learned scheduler beats constant gains, `comparisons.headroom` asks how much
*any* adaptive controller could win on the problem, using no network at all:
the gap between one global constant and the best constant per episode
(the value of knowing which plant you are on), plus the gap to the best
per-episode lookup-table schedule (the value of scheduling on the operating
point). Held-out episodes, whole-episode IAE, seed 42, lower is better:

| Study                | one global constant | best constant /episode | 3-bin schedule /episode | **LSTM scheduler** | headroom | captured |
|----------------------|--------------------:|-----------------------:|------------------------:|-------------------:|---------:|---------:|
| Trolley, linear      |               34.24 |                  33.65 |                   33.04 |              33.37 |   +3.5 % |  +2.5 % |
| Trolley, nonlinear   |               22.44 |                  21.35 |                   20.26 |          **19.31** |   +9.7 % | **+13.9 %** |
| Thermal, linear      |              2903.5 |                 2800.1 |                  2731.8 |             2872.0 |   +5.9 % |  +1.1 % |
| Thermal, nonlinear   |              5300.6 |                 5279.5 |                  5256.7 |         **5252.2** |   +0.8 % |  +0.9 % |

Three findings, one per row group:

* **Where there is real headroom, the scheduler takes it — and more.** On the
  nonlinear trolley (a hardening spring 14× stiffer at the end of the travel
  than at the origin, plus dry friction) the network beats every constant *and*
  the classical lookup-table schedule. It captures more than the "available"
  headroom because that number is measured with a 3-bin table — a floor for
  what scheduling can do, not a ceiling. The network schedules continuously
  and also sees actuator saturation, which no table keyed on the operating
  point can react to.
* **Where there is no headroom, it honestly wins nothing.** The nonlinear
  thermal plant loses heat 3× faster at the top of its range than at the
  bottom, yet the headroom is 0.8 %: setpoint holds are long against the time
  constant, transients are limited by heater power rather than by gains, and
  the integrator erases steady-state error whatever the plant gain is. The
  scheduler matches the table oracle to within noise — plant nonlinearity
  alone does not make gain scheduling worth having.
* **Plant identification is the hard part.** On the two linear studies the
  headroom is almost entirely *between* episodes — it can only be won by
  inferring the drawn plant parameters from loop behaviour — and the scheduler
  captures a modest fraction of it. Within one episode a linear
  time-invariant plant wants constant gains, and time-varying gains have
  nothing structural to exploit.

The four-arm comparison (`comparisons.compare`, 30 held-out episodes,
final-step IAE, disturbance-rejection protocol) tells the same story:

| Study              | Classical rule | Best fixed gains | **LSTM scheduled** | Pole placement /episode |
|--------------------|---------------:|-----------------:|-------------------:|------------------------:|
| Trolley, linear    |           7.58 |         **5.36** |               5.36 |                    7.75 |
| Trolley, nonlinear |           5.92 |             3.43 |           **3.33** |                    5.70 |
| Thermal, linear    |          777.8 |        **538.7** |              544.1 |                   745.1 |
| Thermal, nonlinear |         1514.2 |            809.3 |          **807.3** |                  1328.0 |

The scheduler's nonlinear-trolley win is not free: it uses ~24 % more total
control variation than the best constant (down from +254 % before the
gain-rate penalty existed). That trade-off is printed, not hidden.

Beating a tuning rule mostly shows that tuning rules are conservative — they
are designed to be, since they get one shot at an unknown plant. The searched
constant is the baseline a gain *scheduler* has to beat to justify itself, and
the lookup table is the baseline a *neural* scheduler has to beat to justify
the network. Full tables, including overshoot, settling time, per-episode win
rates and how many runs each mean is computed from, are printed by
`comparisons.compare` and written to `results/`.

A methodological note the numbers above earned the hard way: three times in
this project a "negative result" turned out to be an artefact of the
measurement, not a property of the problem — setpoints beyond the actuator's
reach flattened the cost surface; a warm-up asymmetry made the schedule oracle
unable to reproduce its own seed; saving the final training epoch shipped a
checkpoint worse than its own starting point. Each is now pinned by a test.
Before concluding that a method does not work, check that the measurement
could show it working.

## How it works

```
              baseline gains K0 (best constant, searched once and cached)
                                    |
   +---------------+    s(z)        v               +-----+   u    +-------+   y
   | LSTM (8-48-3) |---------> K = K0*r^(2s-1) ---->| PID |------->| plant |--+-->
   +---------------+                                +-----+        +-------+  |
           ^                                                                  |
           +------------------- normalised loop history ----------------------+
```


* **The plants are written in torch**, so the whole closed loop is
  differentiable and the tracking loss can be back-propagated through the
  simulation into the network. Gradients are exact, not estimated.
* **The head is residual.** The network emits a bounded multiplicative
  correction `K = min(baseline · range^(2σ(z)−1), ceiling)` around the best
  constant gains, with a zero-initialised output layer. The untrained
  scheduler *is* the controller it is later compared against; everything it
  learns is, by construction, the deviation from that comparison point; and no
  weight setting can leave the `[baseline/range, baseline·range]` band. (Its
  predecessor emitted absolute gains and captured −8.6 % of the nonlinear
  trolley's headroom — it lost to a single constant despite real headroom
  existing.)
* **Checkpoints are selected, not assumed.** Every few epochs the model is
  scored on validation episodes disjoint from both training and the final
  evaluation; the first candidate is the untrained network — exactly the
  baseline — so the shipped checkpoint cannot be worse than the baseline on
  validation. "Training helped" is demonstrated to a selector, not assumed.
* **Truncated BPTT**: an optimizer step every `tbptt_window` samples, with the
  plant, the controller and the LSTM hidden state detached at the boundary.
* **The LSTM sees dimensionless features** — normalised error and error rate,
  each gain as a fraction of its ceiling, the operating point and the
  commanded operating point (what a gain schedule for a nonlinear plant has
  to key on), and the control signal as a fraction of the actuator range
  (saturation is one of the few things time-varying gains can exploit even on
  a linear plant). One architecture serves a plant measured in metres and one
  measured in kelvin.
* **The loss matches the metric.** Huber tracking error (the evaluation metric
  is IAE; a squared loss overweights the transient and barely sees the tail),
  a direction-aware overshoot penalty, and a penalty on the per-second *rate*
  of gain change — cheap for a schedule that tracks the operating point,
  expensive for gain chatter that wins IAE by working the actuator.
* **The RBF network** is a one-step-ahead plant surrogate, fitted on
  trajectories from randomised initial conditions; held-out one-step error is
  ≈1 % of the output's own spread. It is optional — the plants are
  differentiable, so `loss_target: surrogate` exists to measure what training
  through a learned model *costs*, the setting you are forced into when the
  plant is not differentiable.

Each episode redraws the reference staircase, the load disturbances and the
plant's physical parameters, which is what makes gain scheduling a meaningful
thing to attempt: on a fixed plant tracking a constant, a constant gain is
optimal.

## Setup

```sh
git clone https://github.com/vsem-azamat/neural-network-pid-regulation
cd neural-network-pid-regulation
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # or: pip install -e ".[dev]"
```

Python 3.11+. CPU is fine. Training one study takes tens of minutes; the
one-off best-constant search per study is cached in `results/`.

## Running it

Everything, all studies, reproducibly:

```sh
python run_pipeline.py
```

Or stage by stage:

```sh
python -m simulations.analyse_plant trolley_nonlinear  # step, phase portrait, Bode, Nyquist
python -m learning.train_rbf        trolley_nonlinear  # fit the plant surrogate
python -m learning.train_lstm_pid   trolley_nonlinear  # train the gain scheduler
python -m comparisons.headroom      trolley_nonlinear  # how much could ANY scheduler win?
python -m comparisons.compare       trolley_nonlinear  # four-arm comparison
```

Studies: `trolley`, `trolley_nonlinear`, `thermal`, `thermal_nonlinear`.
Add `--show` to open figures, `--seed N` to change the seed. Metrics are
written to `results/*.json`, figures to `plots/`.

Tests:

```sh
python -m pytest          # 116 tests
ruff check .
```

## Layout

| Path                        | What it holds                                            |
|-----------------------------|----------------------------------------------------------|
| `entities/systems/`         | The plants. Add one file to add a plant.                 |
| `entities/pid.py`           | Discrete PID, five discretisations, differentiable.       |
| `models/`                   | `LSTMAdaptivePID` (residual head), `SystemRBFModel`.      |
| `utils/run.py`              | Simulation loop, tracking loss and TBPTT training.        |
| `utils/metrics.py`          | Step-response metrics.                                    |
| `utils/tuning.py`           | Classical tuning, relay autotuning, pole placement.       |
| `learning/scenarios.py`     | Episode generation.                                       |
| `comparisons/headroom.py`   | The adaptation-headroom diagnostic.                       |
| `comparisons/`              | Baselines, lookup-table schedules, the comparison harness.|
| `config/ymls/`              | Per-study configuration (linear and nonlinear variants).  |
| `tests/`                    | 116 tests, most pinned to a specific past defect.         |

## Configuration

One YAML per study in `config/ymls/`, validated by pydantic on load. Network
input widths are *not* configurable — they are fixed by the feature extractors,
which removes a way for the two to silently disagree.

Notable knobs:

| Key                              | Meaning                                                 |
|----------------------------------|---------------------------------------------------------|
| `control.gain_ceiling`           | Hard cap on each gain. Size it from what the plant needs.|
| `control.residual_range`         | Multiplicative half-width of the scheduler's correction band around the baseline gains. |
| `control.error_scale`            | Characteristic error, used to normalise the loss and features. |
| `scenario.randomize_plant`       | Per-episode parameter ranges.                            |
| `scenario.disturbance_scale`     | Load disturbance amplitude.                              |
| `learning.lstm.loss_target`      | `plant` (exact gradients) or `surrogate` (model-based).  |
| `learning.lstm.effort_weight`    | Penalty on actuator movement.                            |
| `learning.lstm.gain_rate_weight` | Penalty on the per-second rate of gain change.           |

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
* The LSTM's feature set contained no operating point, so on a nonlinear plant
  it was structurally incapable of scheduling, however long it trained.
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
