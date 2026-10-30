"""Focused-beam settings shared by the optical cases."""

from optomechanics_levitation import parameters as p
from optomechanics_levitation.optical_fields.focused_fields import FocusedBeam

focused_beam = FocusedBeam(wavelength=p.WAVELENGTH_NM, focal_dist=p.FOCAL_DISTANCE_NM, N_theta=p.FOCUS_N_THETA, N_phi=p.FOCUS_N_PHI, 
                           NA=p.NUMERICAL_APERTURE, power=p.OPTICAL_POWER_W, refractive_index=p.REFRACTIVE_INDEX)
