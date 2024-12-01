import torch

from classes.simulation import SimulationResults
from entities.systems import (
    SpringDamper,
    Thermal,
    Trolley,
)


def springdamper(system: SpringDamper, results: SimulationResults) -> torch.Tensor:
    inputs = [
        system.X,
        system.dXdT,
        system.d2XdT2,
        results.control_outputs[-1] if results.control_outputs else torch.tensor(0.0),
    ]
    rbf_input = torch.tensor(inputs)
    return rbf_input.unsqueeze(0)


def thermal(system: Thermal, results: SimulationResults) -> torch.Tensor:
    rbf_input = torch.tensor(
        [
            system.X,
            system.dXdT,
            results.control_outputs[-1] if results.control_outputs else 0.0,
        ]
    )
    return rbf_input.unsqueeze(0)


def trolley(system: Trolley, results: SimulationResults) -> torch.Tensor:
    inputs = [
        system.X,
        system.dXdT,
        system.d2XdT2,
        results.control_outputs[-1] if results.control_outputs else 0.0,
    ]
    rbf_input = torch.tensor(inputs)
    return rbf_input.unsqueeze(0)
