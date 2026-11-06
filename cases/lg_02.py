"""Focused LG_02 case."""

from typing import Any
import numpy as np
from numpy.typing import NDArray
from optomechanics_levitation import parameters as p
from optomechanics_levitation.optical_fields.laguerregauss import LaguerreGauss
from optomechanics_levitation.transfer_amplitudes import TransferAmplitudes
from ._common import focused_beam, make_transfer_amplitudes

beam = LaguerreGauss(
    waist_radius_nm=p.ENTRANCE_WAIST_NM, radial_index=0, azimuthal_index=2,
    wavelength_nm=p.WAVELENGTH_NM, power_w=p.OPTICAL_POWER_W, refractive_index=p.REFRACTIVE_INDEX,
)
profile = beam.evaluate("profile")

def electric_field(positions_nm: NDArray[np.float64]) -> NDArray[np.complex128]:
    """Return the focused LG02 electric field in V/m at positions in nm."""
    return focused_beam.electric_field(positions_nm, profile, polarization=p.INCIDENT_POLARIZATION)

def transfer_amplitudes(particle: dict[str, Any]) -> TransferAmplitudes:
    """Prepare transfer-amplitude calculations for this beam and particle."""
    return make_transfer_amplitudes(particle, electric_field)
