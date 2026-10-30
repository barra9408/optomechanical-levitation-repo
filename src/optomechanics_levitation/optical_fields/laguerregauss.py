from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial, genlaguerre
from scipy.constants import epsilon_0, c

class LaguerreGauss:
    """Paraxial Laguerre-Gauss mode with its waist at z = 0."""
    def __init__(self, w0: float, p: int, l: int, wavelength: float, power: float, refractive_index: float = 1.0):
        self.w0 = w0
        self.p = p; self.l = l; self.m = abs(l)
        self.k = 2 * np.pi * refractive_index / wavelength
        self.amplitude = 1e9 * (2 * power / (refractive_index * epsilon_0 * c))**0.5
        self.rayleigh_length = self.k * w0**2 / 2
        self.normalization = np.sqrt(2 * factorial(p) / (np.pi * factorial(p + self.m))) / w0
        self._laguerre = genlaguerre(p, self.m)

    def _lg_profile(self, rho: NDArray[np.float64], phi: NDArray[np.float64], z: float = 0.0) -> NDArray[np.complex128]:
        waist = self.w0 * np.sqrt(1 + (z / self.rayleigh_length)**2)
        radial = np.sqrt(2) * rho / waist
        curvature = self.k * rho**2 * z / (2 * (z**2 + self.rayleigh_length**2))
        gouy = (2 * self.p + self.m + 1) * np.arctan(z / self.rayleigh_length)
        return (self.normalization * self.w0 / waist * radial**self.m * self._laguerre(radial**2)
                * np.exp(-radial**2 / 2 + 1j * (self.l * phi + curvature - gouy)))

    def _lg_field(self, position: NDArray[np.float64], amplitude: complex, polarization: tuple[complex, complex]) -> NDArray[np.complex128]:
        position = np.asarray(position, dtype=float)
        x, y, z = np.moveaxis(position, -1, 0)
        scalar_field = (amplitude * self._lg_profile(np.hypot(x, y), np.arctan2(y, x), z) * np.exp(1j * self.k * z))
        electric_field = np.zeros(position.shape, dtype=complex)
        jones = np.asarray(polarization, dtype=complex)
        electric_field[..., :2] = scalar_field[..., None] * jones / np.linalg.norm(jones)
        return electric_field

    def evaluate(self, output: str, position: NDArray[np.float64] | None = None, amplitude: complex | None = None,
                 polarization: tuple[complex, complex] = (1.0, 0.0)):
        """Return the waist profile callable or the paraxial electric field."""
        if output == "profile":
            return self._lg_profile
        elif output == "field":
            return self._lg_field(position, self.amplitude if amplitude is None else amplitude, polarization)
