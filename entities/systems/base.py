"""Common interface for every controlled plant in the project.

The simulation loop is written against this interface only, so adding a third
plant means adding one file and nothing else.

Two conventions hold for all implementations:

* ``apply_control(u, disturbance)`` advances the plant by exactly one ``dt`` and
  returns the new measured output.  ``disturbance`` is an *additive load on the
  control channel*, in the same physical units as ``u`` (newtons for the
  trolley, watts for the thermal system), and enters with a ``+`` sign in every
  plant.  A mixed sign convention makes robustness results incomparable between
  plants.
* State is kept as autograd-connected tensors.  Nothing detaches implicitly:
  the training loop calls :meth:`detach_state` at truncation boundaries, which
  is the only place the graph is allowed to be cut.
"""

from abc import ABC, abstractmethod

import torch
from torch import Tensor

ZERO = torch.tensor(0.0)


class BaseSystem(ABC):
    dt: Tensor

    @abstractmethod
    def apply_control(self, control_output: Tensor, disturbance: Tensor = ZERO) -> Tensor:
        """Advance the plant one time step.

        Args:
            control_output: Manipulated variable produced by the controller.
            disturbance: Additive load on the control channel, same units as
                ``control_output``.

        Returns:
            The new measured output (:attr:`X`).
        """

    @abstractmethod
    def reset(self) -> None:
        """Return the plant to its initial condition."""

    @abstractmethod
    def detach_state(self) -> None:
        """Drop autograd history while keeping the numeric state."""

    @abstractmethod
    def min_dt(self, oversampling_factor: float = 10.0) -> Tensor:
        """Largest time step that still resolves the plant's own dynamics.

        Defined as a method rather than a property so ``oversampling_factor``
        can actually be supplied — as a property the argument was unreachable
        and the factor was permanently pinned at its default.
        """

    @property
    @abstractmethod
    def X(self) -> Tensor:
        """Measured output (position, temperature, ...)."""

    @property
    @abstractmethod
    def dXdT(self) -> Tensor:
        """First derivative of the measured output."""

    @property
    @abstractmethod
    def d2XdT2(self) -> Tensor:
        """Second derivative of the measured output."""

    def step_response(
        self,
        steps: int,
        final_input: float,
        initial_input: float = 0.0,
        settle_steps: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Run an open-loop step test and return ``(time, output)``.

        Index 0 of both arrays is the pre-step operating point at t=0, so the
        caller can read the baseline off the response instead of guessing it.

        Used by the classical tuning methods to identify a process model.  The
        time axis is built from the plant's own ``dt`` — deriving it from a
        separately supplied ``dt`` silently shifts every identified time
        constant whenever the two disagree.
        """
        with torch.no_grad():
            self.reset()
            for _ in range(settle_steps):
                self.apply_control(torch.tensor(float(initial_input)))

            output = [self.X.reshape(-1)[0].clone()]
            for _ in range(steps):
                self.apply_control(torch.tensor(float(final_input)))
                output.append(self.X.reshape(-1)[0].clone())

        time = torch.arange(steps + 1, dtype=torch.float32) * self.dt.reshape(-1)[0]
        return time, torch.stack(output)
