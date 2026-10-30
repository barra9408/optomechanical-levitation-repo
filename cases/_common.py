"""Focused-beam settings shared by the optical cases."""

from collections.abc import Callable, Mapping
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray
from optomechanics_levitation import parameters as p
from optomechanics_levitation.optical_fields.focused_fields import FocusedBeam
from optomechanics_levitation.transfer_amplitudes import TransferAmplitudes

focused_beam = FocusedBeam(wavelength=p.WAVELENGTH_NM, focal_dist=p.FOCAL_DISTANCE_NM, N_theta=p.FOCUS_N_THETA, N_phi=p.FOCUS_N_PHI, 
                           NA=p.NUMERICAL_APERTURE, power=p.OPTICAL_POWER_W, refractive_index=p.REFRACTIVE_INDEX)


def make_transfer_amplitudes(particle: dict[str, Any], field: Callable[[NDArray[np.float64]], NDArray[np.complex128]]) -> TransferAmplitudes:
    """Prepare one reusable scattering calculation for a case's beam."""
    def field_generator(pos: NDArray[np.float64], env_dict: Mapping[str, float | complex], wavelength: float, 
                        returnField: Literal["E"] = "E") -> NDArray[np.complex128]:
        return field(pos)
    parameters = {"wavelength_nm": p.WAVELENGTH_NM, "environment": p.ENVIRONMENT, "amplitude_v_per_m": 1.0}
    return TransferAmplitudes(particle, field_generator, parameters)
