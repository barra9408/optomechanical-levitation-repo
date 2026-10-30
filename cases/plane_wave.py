"""Focused plane-wave case."""

from optomechanics_levitation import parameters as p
from optomechanics_levitation.optical_fields.planewave import PlaneWave
from ._common import focused_beam

beam = PlaneWave(wavelength=p.WAVELENGTH_NM, power=p.OPTICAL_POWER_W, 
                 pupil_radius=p.FOCAL_DISTANCE_NM * p.NUMERICAL_APERTURE / p.REFRACTIVE_INDEX, refractive_index=p.REFRACTIVE_INDEX)
profile = beam.evaluate("profile")

def field(position_nm):
    return focused_beam.field(position_nm, profile, polarization=p.INCIDENT_POLARIZATION)
