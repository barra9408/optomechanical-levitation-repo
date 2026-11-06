"""Convert beam settings and evaluate pyGDM2 incident electric or magnetic fields."""

from collections.abc import Callable, Mapping
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray
from pyGDM2 import fields

def input_beam_waist_from_filling_factor(filling_factor: float, numerical_aperture: float,
                                         focal_length_mm: float, refractive_index: float) -> float:
    """Return the entrance waist in mm from the pupil filling factor and focal length."""
    return filling_factor * focal_length_mm * numerical_aperture / refractive_index

def electric_field_amplitude_from_power(power_w: float, profile_power_integral_m2: float,
                                        impedance_ohms: float) -> float:
    """Return electric-field amplitude in V/m from power, profile integral, and impedance."""
    return np.sqrt(2 * power_w * impedance_ohms / profile_power_integral_m2)

def _create_incident_field(incident_field_generator: Callable[..., Any],
                           wavelengths_nm: NDArray[np.float64] | float,
                           **beam_parameters: Any) -> Any:
    """Wrap a pyGDM2-compatible field generator and one or more wavelengths in nm."""
    wavelengths_nm = np.atleast_1d(np.asarray(wavelengths_nm, dtype=float))
    generator = getattr(incident_field_generator, "field", incident_field_generator)
    return fields.efield(generator, wavelengths=wavelengths_nm, kwargs=beam_parameters)

def evaluate_incident_field(
    incident_field_generator: Callable[..., Any], positions_nm: NDArray[np.float64],
    wavelength_nm: float, environment: Mapping[str, float | complex],
    amplitude_v_per_m: float, component: Literal["E", "H", "both"] = "E",
    **beam_parameters: Any,
) -> NDArray[np.complex128] | tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """Evaluate incident electric and/or magnetic fields at positions in nm.

    ``positions_nm`` has shape (n, 3). ``component`` selects E, H, or both;
    the latter returns a tuple in that order. ``amplitude_v_per_m`` scales
    the normalized generator output.
    """
    incident_field = _create_incident_field(
        incident_field_generator, wavelength_nm, **beam_parameters)
    result: list[NDArray[np.complex128]] = []
    for selected_component in (("E", "H") if component == "both" else (component,)):
        values = incident_field.field_generator(
            pos=positions_nm, env_dict=environment, wavelength=incident_field.wavelengths[0],
            returnField="E" if selected_component == "E" else "B",
            **incident_field.kwargs_permutations[0],
        )
        result.append(amplitude_v_per_m * np.asarray(values, dtype=complex))
    return (result[0], result[1]) if component == "both" else result[0]
