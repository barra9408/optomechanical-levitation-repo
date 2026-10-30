"""Angular-momentum operators for transfer-amplitude calculations."""

from collections.abc import Callable
from typing import Literal, cast
import numpy as np
from numpy.typing import NDArray
from scipy.constants import hbar

class _AngularMomentumOperator:
    def __init__(self, component: Literal["total", "spin", "orbital"], axis: Literal["x", "y", "z"]) -> None:
        self.component = component
        self.axis = axis
        self.axis_vector = np.eye(3)["xyz".index(axis)]

    def incident(self, field_at: Callable[[NDArray[np.float64]], NDArray[np.complex128]],
                 positions_nm: NDArray[np.float64], center_nm: NDArray[np.float64],
                 wavelength_nm: float) -> NDArray[np.complex128]:
        return self._apply(field_at(positions_nm), wavelength_nm,
                           field_at=field_at, positions_nm=positions_nm, center_nm=center_nm)

    def scattered(self, field: NDArray[np.complex128], directions: NDArray[np.float64],
                  delta_phi: float, wavelength_nm: float) -> NDArray[np.complex128]:
        return self._apply(field, wavelength_nm, directions=directions, delta_phi=delta_phi)

    def _apply(self, field: NDArray[np.complex128], wavelength_nm: float, *,
               field_at: Callable[[NDArray[np.float64]], NDArray[np.complex128]] | None = None,
               positions_nm: NDArray[np.float64] | None = None,
               center_nm: NDArray[np.float64] | None = None,
               directions: NDArray[np.float64] | None = None,
               delta_phi: float | None = None) -> NDArray[np.complex128]:
        """Apply the selected operator in the incident or scattered representation."""
        if self.component == "total":
            return self._total(field, wavelength_nm, field_at, positions_nm,
                               center_nm, directions, delta_phi)
        if self.component == "spin":
            return self._spin(field, wavelength_nm, field_at, positions_nm, directions)
        return (self._total(field, wavelength_nm, field_at, positions_nm,
                            center_nm, directions, delta_phi)
                - self._spin(field, wavelength_nm, field_at, positions_nm, directions))

    def _total(self, field: NDArray[np.complex128], wavelength_nm: float, 
               field_at: Callable[[NDArray[np.float64]], NDArray[np.complex128]] | None, positions_nm: NDArray[np.float64] | None, 
               center_nm: NDArray[np.float64] | None, directions: NDArray[np.float64] | None, 
               delta_phi: float | None) -> NDArray[np.complex128]:
        if field_at is not None:
            positions_nm = cast(NDArray[np.float64], positions_nm)
            center_nm = cast(NDArray[np.float64], center_nm)
            step_nm = 1e-3 * wavelength_nm
            derivatives = [(field_at(positions_nm + step_nm * direction)
                            - field_at(positions_nm - step_nm * direction)) / (2 * step_nm)
                           for direction in np.eye(3)]
            rotation = np.cross(self.axis_vector, positions_nm - center_nm)
            orbital = sum(rotation[:, j, None] * derivatives[j] for j in range(3))
            return -1j * hbar * orbital + 1j * hbar * np.cross(self.axis_vector, field)

        directions = cast(NDArray[np.float64], directions)
        delta_phi = cast(float, delta_phi)
        derivative = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * delta_phi)
        total = -1j * hbar * derivative + 1j * hbar * np.cross(self.axis_vector, field)
        return total - directions * np.sum(directions * total, axis=-1, keepdims=True)

    def _spin(self, field: NDArray[np.complex128], wavelength_nm: float, 
              field_at: Callable[[NDArray[np.float64]], NDArray[np.complex128]] | None, positions_nm: NDArray[np.float64] | None, 
              directions: NDArray[np.float64] | None) -> NDArray[np.complex128]:
        if field_at is None:
            directions = cast(NDArray[np.float64], directions)
            projection = directions[..., "xyz".index(self.axis), None]
            return 1j * hbar * projection * np.cross(directions, field)

        positions_nm = cast(NDArray[np.float64], positions_nm)
        step_nm = 1e-3 * wavelength_nm
        mixed = []
        for direction in np.eye(3):
            field_pp = field_at(positions_nm + step_nm * (self.axis_vector + direction))
            field_pm = field_at(positions_nm + step_nm * (self.axis_vector - direction))
            field_mp = field_at(positions_nm + step_nm * (-self.axis_vector + direction))
            field_mm = field_at(positions_nm - step_nm * (self.axis_vector + direction))
            mixed.append((field_pp - field_pm - field_mp + field_mm) / (4 * step_nm**2))
        derivative_curl = np.stack((mixed[1][:, 2] - mixed[2][:, 1],
                                    mixed[2][:, 0] - mixed[0][:, 2],
                                    mixed[0][:, 1] - mixed[1][:, 0]), axis=-1)
        k_nm = 2 * np.pi / wavelength_nm
        return -1j * hbar * derivative_curl / k_nm**2

def angular_momentum_operator(axis: Literal["x", "y", "z"] = "z") -> _AngularMomentumOperator:
    """Total angular momentum about ``axis``."""
    return _AngularMomentumOperator("total", axis)


def spin_operator(axis: Literal["x", "y", "z"] = "z") -> _AngularMomentumOperator:
    """Transverse spin angular momentum about ``axis``."""
    return _AngularMomentumOperator("spin", axis)


def orbital_operator(axis: Literal["x", "y", "z"] = "z") -> _AngularMomentumOperator:
    """Orbital contribution defined as total minus transverse spin."""
    return _AngularMomentumOperator("orbital", axis)
