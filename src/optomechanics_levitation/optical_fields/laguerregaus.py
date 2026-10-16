"""Focused first-order Laguerre-Gauss-like vortex built from HG00 fields."""

from __future__ import annotations
from collections.abc import Mapping
from typing import Any
import numpy as np
from numpy.typing import NDArray
from .hermitegauss00 import HermiteGauss00

class LaguerreGauss01:
    """Charge ±1 vortex from transverse derivatives of a focused HG00 beam.

    A central difference of translated HG00 solutions approximates
    ``-w / sqrt(2) * (d/dx + i*charge*d/dy) HG00``. In the paraxial limit
    this is the normalized LG mode with radial index 0 and |charge| = 1.
    The parent field has fixed x polarization; there is no polarization
    parameter for this mode. Its vector components retain the focused HG00
    solution, including the longitudinal component.
    """

    def __init__(self, charge: int = 1, displacement_fraction: float = 0.05, **hg_options: Any) -> None:
        self.charge = charge
        self.displacement_fraction = float(displacement_fraction)
        self.hg = HermiteGauss00(theta=0.0, **hg_options)

    def waist_nm(self, wavelength: float) -> float:
        """Diffraction-limited transverse scale used for the LG derivative."""
        return float(wavelength / (np.pi * self.hg.NA))

    def field(self, pos: NDArray[np.float64], env_dict: Mapping[str, float | complex], wavelength: float, 
              returnField: str = "E") -> NDArray[np.complex128]:
        """Return a focused vortex field at positions measured in nm."""
        positions = np.asarray(pos, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("pos must have shape (N, 3)")
        waist = self.waist_nm(wavelength)
        step = self.displacement_fraction * waist
        dx = np.array((step, 0.0, 0.0))
        dy = np.array((0.0, step, 0.0))
        x_derivative = (self.hg.field(positions + dx, env_dict, wavelength, returnField=returnField)
                        - self.hg.field(positions - dx, env_dict, wavelength, returnField=returnField)) / (2 * step)
        y_derivative = (self.hg.field(positions + dy, env_dict, wavelength, returnField=returnField)
                        - self.hg.field(positions - dy, env_dict, wavelength, returnField=returnField)) / (2 * step)
        return -waist / np.sqrt(2) * (x_derivative + 1j * self.charge * y_derivative)
