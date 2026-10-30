"""Focused LG_01 case."""

from optomechanics_levitation import parameters as p
from optomechanics_levitation.optical_fields.laguerregauss import LaguerreGauss
from ._common import focused_beam

beam = LaguerreGauss(w0=p.ENTRANCE_WAIST_NM, p=0, l=1, wavelength=p.WAVELENGTH_NM, power=p.OPTICAL_POWER_W, refractive_index=p.REFRACTIVE_INDEX)
profile = beam.evaluate("profile")

def field(position_nm):
    return focused_beam.field(position_nm, profile, polarization=p.INCIDENT_POLARIZATION)
