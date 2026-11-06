"""Represent paraxial Laguerre-Gauss beams and their complex electric fields."""

from collections.abc import Callable
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.special import factorial, genlaguerre
from optomechanics_levitation.constants import (
    METRES_PER_NANOMETRE, SPEED_OF_LIGHT_M_PER_S, VACUUM_PERMITTIVITY_F_PER_M,
)

class LaguerreGauss:
    """Paraxial Laguerre-Gauss mode with its waist at z = 0.

    ``evaluate`` returns the pupil profile or electric field in V/m.
    """
    def __init__(self, waist_radius_nm: float, radial_index: int, azimuthal_index: int,
                 wavelength_nm: float, power_w: float, refractive_index: float = 1.0) -> None:
        """Set waist and wavelength in nm, mode indices, power in W, and index."""
        self.waist_radius_nm = waist_radius_nm
        self.radial_index = radial_index
        self.azimuthal_index = azimuthal_index
        absolute_azimuthal_index = abs(azimuthal_index)
        self.wave_number_per_nm = 2 * np.pi * refractive_index / wavelength_nm
        self.amplitude_v_per_m = (2 * power_w / (
            refractive_index * VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S))**0.5
        self.amplitude_v_per_m /= METRES_PER_NANOMETRE
        self.rayleigh_length_nm = self.wave_number_per_nm * waist_radius_nm**2 / 2
        self.normalization = np.sqrt(2 * factorial(radial_index) / (
            np.pi * factorial(radial_index + absolute_azimuthal_index))) / waist_radius_nm
        self._laguerre = genlaguerre(radial_index, absolute_azimuthal_index)
        self.absolute_azimuthal_index = absolute_azimuthal_index

    def _lg_profile(self, rho: NDArray[np.float64], phi: NDArray[np.float64],
                    z: float = 0.0) -> NDArray[np.complex128]:
        """Return the complex Laguerre-Gauss envelope at cylindrical coordinates in nm."""
        waist = self.waist_radius_nm * np.sqrt(1 + (z / self.rayleigh_length_nm)**2)
        radial = np.sqrt(2) * rho / waist
        curvature = self.wave_number_per_nm * rho**2 * z / (2 * (z**2 + self.rayleigh_length_nm**2))
        gouy = ((2 * self.radial_index + self.absolute_azimuthal_index + 1)
                * np.arctan(z / self.rayleigh_length_nm))
        return (self.normalization * self.waist_radius_nm / waist
                * radial**self.absolute_azimuthal_index
                * self._laguerre(radial**2)
                * np.exp(-radial**2 / 2 + 1j * (self.azimuthal_index * phi + curvature - gouy)))

    def _laguerre_gauss_electric_field(
        self, positions_nm: NDArray[np.float64], amplitude_v_per_m: complex,
        polarization: tuple[complex, complex],
    ) -> NDArray[np.complex128]:
        """Evaluate the polarized electric field at positions in nm."""
        positions_nm = np.asarray(positions_nm, dtype=float)
        x, y, z = np.moveaxis(positions_nm, -1, 0)
        scalar_field = (amplitude_v_per_m * self._lg_profile(np.hypot(x, y), np.arctan2(y, x), z)
                        * np.exp(1j * self.wave_number_per_nm * z))
        electric_field = np.zeros(positions_nm.shape, dtype=complex)
        jones = np.asarray(polarization, dtype=complex)
        electric_field[..., :2] = scalar_field[..., None] * jones / np.linalg.norm(jones)
        return electric_field

    def evaluate(self, output: Literal["profile", "electric_field"],
                 positions_nm: NDArray[np.float64] | None = None,
                 amplitude_v_per_m: complex | None = None,
                 polarization: tuple[complex, complex] = (1.0, 0.0)) -> (
                     Callable[..., NDArray[np.complex128]] | NDArray[np.complex128]):
        """Return a waist profile callable or a paraxial electric field in V/m.

        For ``electric_field``, ``positions_nm`` must have trailing dimension 3.
        """
        if output == "profile":
            return self._lg_profile
        elif output == "electric_field":
            amplitude = self.amplitude_v_per_m if amplitude_v_per_m is None else amplitude_v_per_m
            return self._laguerre_gauss_electric_field(positions_nm, amplitude, polarization)
