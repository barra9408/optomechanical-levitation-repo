"""Focused LG_00 case."""

from typing import Any
import numpy as np
from numpy.typing import NDArray
from optomechanics_levitation import parameters as p
from optomechanics_levitation.optical_fields.laguerregauss import LaguerreGauss
from optomechanics_levitation.transfer_amplitudes import TransferAmplitudes
from ._common import focused_beam, make_transfer_amplitudes

beam = LaguerreGauss(w0=p.ENTRANCE_WAIST_NM, p=0, l=0, wavelength=p.WAVELENGTH_NM, power=p.OPTICAL_POWER_W, refractive_index=p.REFRACTIVE_INDEX)
profile = beam.evaluate("profile")

def field(position_nm: NDArray[np.float64]) -> NDArray[np.complex128]:
    return focused_beam.field(position_nm, profile, polarization=p.INCIDENT_POLARIZATION)

def transfer_amplitudes(particle: dict[str, Any]) -> TransferAmplitudes:
    return make_transfer_amplitudes(particle, field)
