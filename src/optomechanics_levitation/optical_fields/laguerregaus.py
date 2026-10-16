"""Focused LG-like vortex modes built from the existing HG00 field."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .hermitegauss00 import HermiteGauss00


class _FocusedLaguerreGauss:
    """Share the focused HG00 parent and transverse scale between vortex modes."""

    def __init__(self, charge: int, displacement_fraction: float, **hg_options: Any) -> None:
        if displacement_fraction <= 0:
            raise ValueError("displacement_fraction must be positive")
        if "theta" in hg_options or "polarization_state" in hg_options:
            raise TypeError("Laguerre-Gauss modes have fixed internal x polarization")
        self.charge = charge
        self.displacement_fraction = float(displacement_fraction)
        self.hg = HermiteGauss00(theta=0.0, **hg_options)

    def waist_nm(self, wavelength: float) -> float:
        """Diffraction-limited transverse scale used for the HG derivatives."""
        return float(wavelength / (np.pi * self.hg.NA))

    @staticmethod
    def _positions(pos: NDArray[np.float64]) -> NDArray[np.float64]:
        positions = np.asarray(pos, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")
        return positions


class LaguerreGauss01(_FocusedLaguerreGauss):
    """Focused LG_0^±1-like field from one transverse HG00 derivative.

    In the paraxial limit, ``-w/sqrt(2) (∂x ± i∂y) HG00`` is LG_0^±1.
    The finite difference is a superposition of translated vector HG00 fields.
    """

    def __init__(self, charge: int = 1, displacement_fraction: float = 0.05,
                 **hg_options: Any) -> None:
        if abs(charge) != 1:
            raise ValueError("LG01 charge must be +1 or -1")
        super().__init__(charge, displacement_fraction, **hg_options)

    def field(self, pos: NDArray[np.float64], env_dict: Mapping[str, float | complex],
              wavelength: float, returnField: str = "E") -> NDArray[np.complex128]:
        """Return a focused charge-one vortex field at positions in nm."""
        positions = self._positions(pos)
        waist = self.waist_nm(wavelength)
        step = self.displacement_fraction * waist
        dx = np.array((step, 0.0, 0.0))
        dy = np.array((0.0, step, 0.0))
        x_derivative = (self.hg.field(positions + dx, env_dict, wavelength, returnField=returnField)
                        - self.hg.field(positions - dx, env_dict, wavelength, returnField=returnField)) / (2 * step)
        y_derivative = (self.hg.field(positions + dy, env_dict, wavelength, returnField=returnField)
                        - self.hg.field(positions - dy, env_dict, wavelength, returnField=returnField)) / (2 * step)
        return -waist / np.sqrt(2) * (x_derivative + 1j * self.charge * y_derivative)


class LaguerreGauss02(_FocusedLaguerreGauss):
    """Focused LG_0^±2-like field from two transverse HG00 derivatives.

    In the paraxial limit, ``w²/(2 sqrt(2)) (∂x ± i∂y)² HG00`` is LG_0^±2.
    Eight shifted HG00 evaluations form a central-difference approximation.
    """

    def __init__(self, charge: int = 2, displacement_fraction: float = 0.05,
                 **hg_options: Any) -> None:
        if abs(charge) != 2:
            raise ValueError("LG02 charge must be +2 or -2")
        super().__init__(charge, displacement_fraction, **hg_options)

    def field(self, pos: NDArray[np.float64], env_dict: Mapping[str, float | complex],
              wavelength: float, returnField: str = "E") -> NDArray[np.complex128]:
        """Return a focused charge-two vortex field at positions in nm."""
        positions = self._positions(pos)
        waist = self.waist_nm(wavelength)
        step = self.displacement_fraction * waist
        dx = np.array((step, 0.0, 0.0))
        dy = np.array((0.0, step, 0.0))

        def shifted(offset: NDArray[np.float64]) -> NDArray[np.complex128]:
            return self.hg.field(positions + offset, env_dict, wavelength,
                                 returnField=returnField)

        difference_xx_minus_yy = (shifted(dx) + shifted(-dx)
                                  - shifted(dy) - shifted(-dy)) / step**2
        difference_xy = (shifted(dx + dy) - shifted(dx - dy)
                         - shifted(-dx + dy) + shifted(-dx - dy)) / (4 * step**2)
        return waist**2 / (2 * np.sqrt(2)) * (
            difference_xx_minus_yy + 2j * np.sign(self.charge) * difference_xy
        )
