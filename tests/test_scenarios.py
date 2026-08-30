"""Episode generation — including a direct guard against the original defect."""

import numpy as np
import pytest

from config import load_config
from learning.scenarios import build_system, episode_stream, make_episode


@pytest.fixture(params=["trolley", "thermal"])
def config(request):
    return request.param, load_config(request.param)


def episode_steps(cfg) -> int:
    """The episode length the shipped configuration actually trains on.

    Fixing a step count here instead would test a scenario nobody runs: the
    thermal plant holds each setpoint for 80-200 s, so a short episode contains
    one segment and would pass the "reference changes" check by accident.
    """
    return int(cfg.learning.lstm.train_time / cfg.learning.dt)


def build(name, cfg, seed=0, **kwargs):
    return make_episode(
        cfg.scenario,
        episode_steps(cfg),
        cfg.learning.dt,
        np.random.default_rng(seed),
        **kwargs,
    )


def test_setpoints_actually_change_during_an_episode(config):
    """`[torch.randn(1) * 10] * n` repeats one object: a constant reference.

    On a constant reference with a fixed plant, a constant gain is optimal, so
    there is nothing for a gain scheduler to learn. This is the check that the
    training problem is a problem at all.
    """
    name, cfg = config
    episode = build(name, cfg)
    values = {float(s) for s in episode.setpoints}
    assert len(values) > 1, "the reference never changes"
    assert len(values) >= 3, f"only {len(values)} distinct setpoints in the episode"


def test_setpoints_stay_inside_the_configured_range(config):
    name, cfg = config
    low, high = cfg.scenario.setpoint.as_tuple()
    for value in build(name, cfg).setpoints:
        assert low <= float(value) <= high


def test_disturbances_are_generated_and_non_trivial(config):
    """The old comparison computed a disturbance and never used it."""
    name, cfg = config
    episode = build(name, cfg)
    values = np.array([float(d) for d in episode.disturbances])
    assert np.abs(values).max() > 0.0
    assert len(set(values.tolist())) > 1


def test_disturbances_can_be_switched_off(config):
    name, cfg = config
    episode = build(name, cfg, with_disturbance=False)
    assert all(float(d) == 0.0 for d in episode.disturbances)


def test_plant_parameters_are_redrawn_per_episode(config):
    name, cfg = config
    episodes = list(
        episode_stream(cfg.scenario, episode_steps(cfg), cfg.learning.dt, 5, seed=1)
    )
    for key in cfg.scenario.randomize_plant:
        values = {e.plant_parameters[key] for e in episodes}
        assert len(values) == len(episodes), f"{key} was not resampled"


def test_randomised_parameters_stay_inside_their_bounds(config):
    name, cfg = config
    stream = episode_stream(
        cfg.scenario, episode_steps(cfg), cfg.learning.dt, 10, seed=2
    )
    for episode in stream:
        for key, value in episode.plant_parameters.items():
            low, high = cfg.scenario.randomize_plant[key].as_tuple()
            assert low <= value <= high


def test_same_seed_reproduces_the_same_episode(config):
    name, cfg = config
    a, b = build(name, cfg, seed=7), build(name, cfg, seed=7)
    assert [float(s) for s in a.setpoints] == [float(s) for s in b.setpoints]
    assert a.plant_parameters == b.plant_parameters


def test_built_plant_uses_the_episode_parameters(config):
    name, cfg = config
    episode = build(name, cfg, seed=3)
    system = build_system(cfg, episode.plant_parameters)
    for key, value in episode.plant_parameters.items():
        assert float(getattr(system, key)) == pytest.approx(value)


def test_configured_dt_resolves_the_plant_dynamics(config):
    """The simulation loop rejects too-large steps; the shipped config must pass."""
    name, cfg = config
    stream = episode_stream(
        cfg.scenario, episode_steps(cfg), cfg.learning.dt, 20, seed=4
    )
    for episode in stream:
        system = build_system(cfg, episode.plant_parameters)
        assert cfg.learning.dt < float(system.min_dt()), (
            f"dt={cfg.learning.dt} too large for {episode.plant_parameters}"
        )
