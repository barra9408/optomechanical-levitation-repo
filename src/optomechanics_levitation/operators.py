"""Angular-momentum operators for transfer-amplitude calculations."""

from collections.abc import Callable
from typing import Literal, cast
import numpy as np
from numpy.typing import NDArray
from .constants import REDUCED_PLANCK_CONSTANT_J_S

class _AngularMomentumOperator:
    """Apply total, spin, or orbital angular momentum to electric fields.

    ``incident`` and ``scattered`` return the operated complex field on their
    respective position or angular grids.
    """
    def __init__(
        self, component: Literal["total", "spin", "orbital"], axis: Literal["x", "y", "z"],
    ) -> None:
        """Select an angular-momentum component and Cartesian rotation axis."""
        self.component = component
        self.axis = axis
        self.axis_vector = np.eye(3)["xyz".index(axis)]

    def incident(self, electric_field_at: Callable[[NDArray[np.float64]], NDArray[np.complex128]],
                 positions_nm: NDArray[np.float64], center_nm: NDArray[np.float64],
                 wavelength_nm: float) -> NDArray[np.complex128]:
        """Operate on the incident field at (n, 3) positions in nm."""
        return self._apply(
            electric_field_at(positions_nm), wavelength_nm,
            electric_field_at=electric_field_at, positions_nm=positions_nm, center_nm=center_nm)

    def scattered(
        self, scattered_electric_field: NDArray[np.complex128], directions: NDArray[np.float64],
        delta_phi: float, wavelength_nm: float,
    ) -> NDArray[np.complex128]:
        """Operate on a scattered (n_theta, n_phi, 3) electric-field grid."""
        return self._apply(
            scattered_electric_field, wavelength_nm, directions=directions, delta_phi=delta_phi)

    def _apply(self, electric_field: NDArray[np.complex128], wavelength_nm: float, *,
               electric_field_at: Callable[
                   [NDArray[np.float64]], NDArray[np.complex128]] | None = None,
               positions_nm: NDArray[np.float64] | None = None,
               center_nm: NDArray[np.float64] | None = None,
               directions: NDArray[np.float64] | None = None,
               delta_phi: float | None = None) -> NDArray[np.complex128]:
        """Apply the selected operator in the incident or scattered representation."""
        if self.component == "total":
            return self._total(electric_field, wavelength_nm, electric_field_at, positions_nm,
                               center_nm, directions, delta_phi)
        if self.component == "spin":
            return self._spin(
                electric_field, wavelength_nm, electric_field_at, positions_nm, directions)
        return (self._total(electric_field, wavelength_nm, electric_field_at, positions_nm,
                            center_nm, directions, delta_phi)
                - self._spin(
                    electric_field, wavelength_nm, electric_field_at, positions_nm, directions))

    def _total(self, electric_field: NDArray[np.complex128], wavelength_nm: float,
               electric_field_at: Callable[[NDArray[np.float64]], NDArray[np.complex128]] | None,
               positions_nm: NDArray[np.float64] | None,
               center_nm: NDArray[np.float64] | None, directions: NDArray[np.float64] | None, 
               delta_phi: float | None) -> NDArray[np.complex128]:
        """Compute total angular momentum in the incident or scattered representation."""
        if electric_field_at is not None:
            positions_nm = cast(NDArray[np.float64], positions_nm)
            center_nm = cast(NDArray[np.float64], center_nm)
            step_nm = 1e-3 * wavelength_nm
            derivatives = [(electric_field_at(positions_nm + step_nm * direction)
                            - electric_field_at(positions_nm - step_nm * direction)) / (2 * step_nm)
                           for direction in np.eye(3)]
            rotation = np.cross(self.axis_vector, positions_nm - center_nm)
            orbital = sum(rotation[:, j, None] * derivatives[j] for j in range(3))
            return REDUCED_PLANCK_CONSTANT_J_S * (
                -1j * orbital + 1j * np.cross(self.axis_vector, electric_field))

        directions = cast(NDArray[np.float64], directions)
        delta_phi = cast(float, delta_phi)
        derivative = (
            np.roll(electric_field, -1, axis=1) - np.roll(electric_field, 1, axis=1)
        ) / (2 * delta_phi)
        total = REDUCED_PLANCK_CONSTANT_J_S * (
            -1j * derivative + 1j * np.cross(self.axis_vector, electric_field))
        return total - directions * np.sum(directions * total, axis=-1, keepdims=True)

    def _spin(self, electric_field: NDArray[np.complex128], wavelength_nm: float,
              electric_field_at: Callable[[NDArray[np.float64]], NDArray[np.complex128]] | None,
              positions_nm: NDArray[np.float64] | None,
              directions: NDArray[np.float64] | None) -> NDArray[np.complex128]:
        """Compute transverse spin using field derivatives or angular projection."""
        if electric_field_at is None:
            directions = cast(NDArray[np.float64], directions)
            projection = directions[..., "xyz".index(self.axis), None]
            return (1j * REDUCED_PLANCK_CONSTANT_J_S * projection
                    * np.cross(directions, electric_field))

        positions_nm = cast(NDArray[np.float64], positions_nm)
        step_nm = 1e-3 * wavelength_nm
        mixed = []
        for direction in np.eye(3):
            field_pp = electric_field_at(positions_nm + step_nm * (self.axis_vector + direction))
            field_pm = electric_field_at(positions_nm + step_nm * (self.axis_vector - direction))
            field_mp = electric_field_at(positions_nm + step_nm * (-self.axis_vector + direction))
            field_mm = electric_field_at(positions_nm - step_nm * (self.axis_vector + direction))
            mixed.append((field_pp - field_pm - field_mp + field_mm) / (4 * step_nm**2))
        derivative_curl = np.stack((mixed[1][:, 2] - mixed[2][:, 1],
                                    mixed[2][:, 0] - mixed[0][:, 2],
                                    mixed[0][:, 1] - mixed[1][:, 0]), axis=-1)
        k_nm = 2 * np.pi / wavelength_nm
        return -1j * REDUCED_PLANCK_CONSTANT_J_S * derivative_curl / k_nm**2

def angular_momentum_operator(axis: Literal["x", "y", "z"] = "z") -> _AngularMomentumOperator:
    """Total angular momentum about ``axis``."""
    return _AngularMomentumOperator("total", axis)


def spin_operator(axis: Literal["x", "y", "z"] = "z") -> _AngularMomentumOperator:
    """Transverse spin angular momentum about ``axis``."""
    return _AngularMomentumOperator("spin", axis)


def orbital_operator(axis: Literal["x", "y", "z"] = "z") -> _AngularMomentumOperator:
    """Orbital contribution defined as total minus transverse spin."""
    return _AngularMomentumOperator("orbital", axis)
