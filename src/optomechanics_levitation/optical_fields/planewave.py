"""Represent a plane wave and its pupil profile in nanometre coordinates."""

from collections.abc import Callable
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from optomechanics_levitation.constants import (
    METRES_PER_NANOMETRE, SPEED_OF_LIGHT_M_PER_S, VACUUM_PERMITTIVITY_F_PER_M,
)

class PlaneWave:
    """Plane wave along +z, with power defined through a circular pupil.

    ``evaluate`` returns either its pupil profile or electric field in V/m.
    """
    def __init__(self, wavelength_nm: float, power_w: float, pupil_radius_nm: float,
                 refractive_index: float = 1.0) -> None:
        """Set wavelength and pupil radius in nm, optical power in W, and index."""
        self.wave_number_per_nm = 2 * np.pi * refractive_index / wavelength_nm
        area_m2 = np.pi * (pupil_radius_nm * METRES_PER_NANOMETRE)**2
        self.amplitude_v_per_m = np.sqrt(2 * power_w / (
            refractive_index * VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S * area_m2))

    def _plane_wave_profile(self, rho: NDArray[np.float64], phi: NDArray[np.float64],
                            z: float = 0.0) -> NDArray[np.complex128]:
        """Return a uniform complex pupil profile for the supplied coordinates."""
        shape = np.broadcast_shapes(np.shape(rho), np.shape(phi), np.shape(z))
        return np.ones(shape, dtype=complex)

    def _plane_wave_electric_field(
        self, positions_nm: NDArray[np.float64], amplitude_v_per_m: complex,
        polarization: tuple[complex, complex],
    ) -> NDArray[np.complex128]:
        """Evaluate the electric field at positions with trailing dimension three."""
        positions_nm = np.asarray(positions_nm, dtype=float)
        phase = np.exp(1j * self.wave_number_per_nm * positions_nm[..., 2])
        electric_field = np.zeros(positions_nm.shape, dtype=complex)
        jones = np.asarray(polarization, dtype=complex)
        electric_field[..., :2] = (
            amplitude_v_per_m * phase[..., None] * jones / np.linalg.norm(jones))
        return electric_field

    def evaluate(self, output: Literal["profile", "electric_field"],
                 positions_nm: NDArray[np.float64] | None = None,
                 amplitude_v_per_m: complex | None = None,
                 polarization: tuple[complex, complex] = (1.0, 0.0)) -> (
                     Callable[..., NDArray[np.complex128]] | NDArray[np.complex128]):
        """Return a pupil profile callable or an electric field in V/m.

        For ``electric_field``, ``positions_nm`` must have trailing dimension 3.
        """
        if output == "profile":
            return self._plane_wave_profile
        elif output == "electric_field":
            amplitude = self.amplitude_v_per_m if amplitude_v_per_m is None else amplitude_v_per_m
            return self._plane_wave_electric_field(positions_nm, amplitude, polarization)
