"""Focused-beam settings shared by the optical cases."""

from collections.abc import Callable, Mapping
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray
from optomechanics_levitation import parameters as p
from optomechanics_levitation.optical_fields.focused_fields import FocusedBeam
from optomechanics_levitation.transfer_amplitudes import TransferAmplitudes

focused_beam = FocusedBeam(
    wavelength_nm=p.WAVELENGTH_NM, focal_distance_nm=p.FOCAL_DISTANCE_NM,
    n_theta=p.FOCUS_N_THETA, n_phi=p.FOCUS_N_PHI,
    numerical_aperture=p.NUMERICAL_APERTURE, power_w=p.OPTICAL_POWER_W,
    refractive_index=p.REFRACTIVE_INDEX,
)


def make_transfer_amplitudes(
    particle: dict[str, Any],
    electric_field: Callable[[NDArray[np.float64]], NDArray[np.complex128]],
) -> TransferAmplitudes:
    """Prepare one reusable scattering calculation for a case's beam."""
    def incident_field_generator(
        pos: NDArray[np.float64], env_dict: Mapping[str, float | complex],
        wavelength: float, returnField: Literal["E"] = "E",
    ) -> NDArray[np.complex128]:
        """Match the keyword signature required by pyGDM2."""
        return electric_field(pos)
    parameters = {
        "wavelength_nm": p.WAVELENGTH_NM, "environment": p.ENVIRONMENT,
        "amplitude_v_per_m": 1.0,
    }
    return TransferAmplitudes(particle, incident_field_generator, parameters)
