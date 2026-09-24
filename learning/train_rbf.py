"""Train the RBF plant surrogate.

    python -m learning.train_rbf trolley
    python -m learning.train_rbf thermal

One script serves both plants: the two former scripts differed only in feature
count and constants, which is exactly the sort of duplication that lets the
thermal version drift into evaluating in Celsius while its plant runs in kelvin.
"""

import argparse

import numpy as np

from config import available_studies, load_config
from learning.rbf_dataset import (
    collect_trajectories,
    holdout_excitation,
    randomise_initial_state,
    rollout_comparison,
)
from learning.scenarios import build_system
from models.sys_rbf import SystemRBFModel
from utils import save_load
from utils.plot import plot_rbf_training_results
from utils.run import train_rbf_model
from utils.seeding import DEFAULT_SEED, seed_everything

STATE_LABELS = {"trolley": "Position (m)", "thermal": "Temperature (K)"}


def main(system_name: str, seed: int, show: bool) -> None:
    seed_everything(seed)
    config = load_config(system_name)
    rbf_config = config.learning.rbf

    print(f"Collecting {rbf_config.num_trajectories} trajectories...")
    rng = np.random.default_rng(seed)
    X, y = collect_trajectories(config, rng)
    print(f"  dataset: {tuple(X.shape)} -> {tuple(y.shape)}")
    print(f"  input range per feature: {X.min(dim=0).values.tolist()}")
    print(f"                           {X.max(dim=0).values.tolist()}")

    model = SystemRBFModel(
        input_mean=X.mean(dim=0),
        input_std=X.std(dim=0, unbiased=False),
        output_mean=y.mean(dim=0),
        output_std=y.std(dim=0, unbiased=False),
        hidden_features=rbf_config.model.hidden_size,
        input_size=X.shape[1],
        output_size=1,
    )

    history = train_rbf_model(
        model,
        X,
        y,
        num_epochs=rbf_config.num_epochs,
        batch_size=rbf_config.batch_size,
        learning_rate=rbf_config.lr,
        optimizer="adam",
        gradient_clip_value=1.0,
        validation_split=rbf_config.validation_split,
    )

    save_load.save_rbf_model(model, f"sys_rbf_{system_name}.pth")

    # Score on a held-out excitation trajectory: same distribution as training,
    # never seen during fitting.
    steps = 300
    time, controls = holdout_excitation(config, steps, seed=seed + 1_000)
    system = build_system(config)
    system.reset()
    randomise_initial_state(system, config, np.random.default_rng(seed + 2_000))
    predicted, actual = rollout_comparison(model, system, controls)

    mse = float(np.mean((np.array(predicted) - np.array(actual)) ** 2))
    scale = float(np.std(actual)) or 1.0
    print(f"\nHeld-out one-step-ahead MSE: {mse:.6f}")
    print(f"Normalised RMSE: {np.sqrt(mse) / scale:.4%} of the output's own spread")
    print(f"Final validation loss: {history['val'][-1]:.6f}")

    plot_rbf_training_results(
        time,
        predicted,
        actual,
        history,
        system_name=system_name.capitalize(),
        state_label=STATE_LABELS.get(config.plant, "Output"),
        num_epochs=rbf_config.num_epochs,
        learning_rate=rbf_config.lr,
        optimizer_name="adam",
        show=show,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=available_studies())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--show", action="store_true", help="Open plot windows.")
    args = parser.parse_args()
    main(args.system, args.seed, args.show)
