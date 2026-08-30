"""Open-loop analysis of a plant: step response, phase portrait, Bode, Nyquist.

    python -m simulations.analyse_plant trolley
    python -m simulations.analyse_plant thermal --show

The frequency-response plots are computed from the analytic transfer function,
and drawn alongside the *simulated* step response so the two descriptions can be
checked against each other. The README used to promise Bode and Nyquist plots
that no code in the project produced.

Transfer functions (output over control input):

    trolley   G(s) = 1 / (m·s² + c·s + k)
    thermal   G(s) = 1 / (C·s + h)
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import available_studies, cnfg, load_config
from entities.systems import BaseSystem, Thermal, Trolley
from learning.scenarios import build_system
from utils.seeding import DEFAULT_SEED, seed_everything


def transfer_function(system: BaseSystem, omega: np.ndarray) -> np.ndarray:
    """G(jω) for the plant, evaluated on a frequency grid."""
    s = 1j * omega
    if isinstance(system, Trolley):
        m, c, k = (float(system.mass), float(system.friction), float(system.spring))
        return 1.0 / (m * s**2 + c * s + k)
    if isinstance(system, Thermal):
        C, h = float(system.thermal_capacity), float(system.heat_transfer_coefficient)
        return 1.0 / (C * s + h)
    raise TypeError(f"No transfer function known for {type(system).__name__}")


def simulate_step(system: BaseSystem, amplitude: float, steps: int):
    time, output = system.step_response(steps=steps, final_input=amplitude)
    return time.numpy(), output.numpy()


def describe(system: BaseSystem) -> str:
    if isinstance(system, Trolley):
        omega_n = float(torch.sqrt(system.spring / system.mass))
        return (
            f"m={float(system.mass):g} kg, k={float(system.spring):g} N/m, "
            f"c={float(system.friction):g} N·s/m  →  "
            f"ω_n={omega_n:.3f} rad/s, ζ={float(system.damping_ratio):.3f}"
        )
    if isinstance(system, Thermal):
        return (
            f"C={float(system.thermal_capacity):g} J/K, "
            f"h={float(system.heat_transfer_coefficient):g} W/K  →  "
            f"τ={float(system.tau):.1f} s"
        )
    return type(system).__name__


def main(system_name: str, seed: int, show: bool) -> None:
    seed_everything(seed)
    config = load_config(system_name)
    system = build_system(config)

    amplitude = config.control.tuning_step_input
    duration = 60.0 if system_name == "trolley" else 1200.0
    steps = int(duration / config.learning.dt)

    print(f"{system_name}: {describe(system)}")

    time, response = simulate_step(system, amplitude, steps)
    omega = np.logspace(-3, 2, 800)
    G = transfer_function(system, omega)
    magnitude_db = 20 * np.log10(np.abs(G))
    phase_deg = np.degrees(np.unwrap(np.angle(G)))

    fig = plt.figure(figsize=(13, 10))
    grid = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.25)

    ax = fig.add_subplot(grid[0, :])
    ax.plot(time, response, linewidth=1.8)
    ax.axhline(response[-1], color="r", linestyle="--", linewidth=1,
               label=f"steady state {response[-1]:.3f}")
    ax.set_title(f"Step response to {amplitude:g} "
                 f"{'N' if system_name == 'trolley' else 'W'}")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position (m)" if system_name == "trolley" else "Temperature (K)")
    ax.legend()

    ax = fig.add_subplot(grid[1, 0])
    ax.plot(response[:-1], np.diff(response) / config.learning.dt, linewidth=1.4)
    ax.set_title("Phase portrait")
    ax.set_xlabel("Output")
    ax.set_ylabel("d(output)/dt")

    ax = fig.add_subplot(grid[1, 1])
    ax.plot(G.real, G.imag, linewidth=1.4)
    ax.plot(G.real, -G.imag, linewidth=1.0, linestyle=":", alpha=0.6)
    ax.plot(-1, 0, "rx", markersize=9, label="−1 point")
    ax.set_title("Nyquist")
    ax.set_xlabel("Re G(jω)")
    ax.set_ylabel("Im G(jω)")
    ax.legend()

    ax = fig.add_subplot(grid[2, 0])
    ax.semilogx(omega, magnitude_db, linewidth=1.6)
    ax.set_title("Bode magnitude")
    ax.set_xlabel("ω (rad/s)")
    ax.set_ylabel("|G| (dB)")

    ax = fig.add_subplot(grid[2, 1])
    ax.semilogx(omega, phase_deg, linewidth=1.6)
    ax.axhline(-180, color="r", linestyle="--", linewidth=1, label="−180°")
    ax.set_title("Bode phase")
    ax.set_xlabel("ω (rad/s)")
    ax.set_ylabel("∠G (deg)")
    ax.legend()

    for axis in fig.get_axes():
        axis.grid(True, alpha=0.3)

    fig.suptitle(
        f"{system_name.capitalize()} — open-loop analysis\n{describe(system)}",
        fontweight="bold",
    )

    path = os.path.join(cnfg.SYSTEMS_PLOTS, f"{system_name}_analysis.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"Plot saved to {path}")

    # The phase floor explains why relay/ultimate-gain tuning applies to one
    # plant and not the other.
    crosses = phase_deg.min() <= -180
    print(
        f"Minimum phase reached: {phase_deg.min():.1f}° — "
        f"{'crosses' if crosses else 'never reaches'} −180°, so a finite "
        f"ultimate gain {'exists' if crosses else 'does not exist'} "
        f"and relay/ultimate-gain tuning {'applies' if crosses else 'does not apply'}."
    )

    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("system", choices=available_studies())
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    main(args.system, args.seed, args.show)
