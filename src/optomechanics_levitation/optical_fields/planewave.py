from __future__ import annotations
from collections.abc import Mapping
from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.constants import c
from .. import parameters as p

Axis = Literal["x", "y", "z"]; FieldComponent = Literal["E", "B"]

class PlaneWave:
    """Unit-amplitude plane wave with a fixed direction and polarization."""
    def __init__(self, direction: tuple[float, float, float], polarization_state: tuple[complex, complex]) -> None:
        self.direction = np.array(direction, dtype=float, copy=True)
        self.direction /= np.linalg.norm(self.direction)

        reference = np.eye(3)[np.argmin(np.abs(self.direction))]
        e1 = reference - np.dot(reference, self.direction) * self.direction
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(self.direction, e1)

        jones = np.asarray(polarization_state, dtype=complex)
        jones /= np.linalg.norm(jones)
        self.polarization = jones[0] * e1 + jones[1] * e2

    ####################################
    # INCIDENT FIELD
    ####################################
    def field(self, pos: NDArray[np.float64], env_dict: Mapping[str, float | complex], wavelength: float,
                       returnField: FieldComponent = p.DEFAULT_RETURN_FIELD) -> NDArray[np.complex128]:
        """Evaluate the plane-wave E or B field in a homogeneous vacuum."""
        phase = np.exp(2j * np.pi / wavelength * (np.asarray(pos) @ self.direction))
        electric = phase[:, None] * self.polarization
        return {"E": electric, "B": np.cross(self.direction, electric) / c}[returnField]
