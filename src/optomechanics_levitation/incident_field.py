import numpy as np
from pyGDM2 import fields

def input_beam_waist_from_filling_factor(filling_factor, numerical_aperture, focal_length_mm, refractive_index):
    return filling_factor * focal_length_mm * numerical_aperture / refractive_index

def electric_field_amplitude_from_power(power_w, profile_power_integral_m2, impedance_ohms):
    return np.sqrt(2 * power_w * impedance_ohms / profile_power_integral_m2)

def _create_incident_field(field_generator, wavelengths_nm, **beam_parameters):
    wavelengths_nm = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    return fields.efield(field_generator, wavelengths=wavelengths_nm, kwargs=beam_parameters)

def evaluate_incident_field(field_generator, positions_nm, wavelength_nm, environment, amplitude_v_per_m, field="E", **beam_parameters):
    incident_field = _create_incident_field(field_generator, wavelength_nm, **beam_parameters)
    result = []
    for component in (("E", "H") if field == "both" else (field,)):
        values = incident_field.field_generator(pos=positions_nm, env_dict=environment, wavelength=incident_field.wavelengths[0], returnField="E" if component == "E" else "B", **incident_field.kwargs_permutations[0])
        result.append(amplitude_v_per_m * np.asarray(values, dtype=complex))
    return tuple(result) if field == "both" else result[0]

