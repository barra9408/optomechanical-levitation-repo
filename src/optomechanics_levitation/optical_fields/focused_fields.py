"""Focus an incident pupil field with the vector Richards-Wolf integral."""

from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
from optomechanics_levitation.constants import (
    METRES_PER_NANOMETRE, SPEED_OF_LIGHT_M_PER_S, VACUUM_PERMITTIVITY_F_PER_M,
)

class FocusedBeam:
    """Focus a pupil field with the vector Richards-Wolf integral.

    ``electric_field`` returns the focused complex electric field in V/m at
    input positions with trailing dimension three, measured in nm.
    """
    def __init__(self, wavelength_nm: float, focal_distance_nm: float, n_theta: int, n_phi: int,
                 numerical_aperture: float, power_w: float, refractive_index: float = 1.0) -> None:
        """Set wavelength and focal distance in nm, grid counts, aperture, and power in W."""
        self.wave_number_per_nm = 2 * np.pi * refractive_index / wavelength_nm
        self.focal_distance_nm = focal_distance_nm
        self.power_w = power_w
        self.refractive_index = refractive_index

        self.n_theta = n_theta
        self.n_phi = n_phi
        self.theta_max = np.arcsin(numerical_aperture / refractive_index)
        self.delta_theta = self.theta_max / n_theta
        self.delta_phi = 2 * np.pi / n_phi
        self.theta = (np.arange(n_theta) + 0.5) * self.delta_theta
        self.phi = np.arange(n_phi) * self.delta_phi
        self.theta_grid, self.phi_grid = np.meshgrid(self.theta, self.phi, indexing="ij")
        self.solid_angle_weights = np.sin(self.theta_grid) * self.delta_theta * self.delta_phi

        self._directions = self._propagation_direction(
            self.theta_grid, self.phi_grid).reshape(-1, 3)
        self._rotation = self._rotation_matrix(self.theta_grid, self.phi_grid)

    def _propagating_field(
        self,
        field_profile: Callable[
            [NDArray[np.float64], NDArray[np.float64]], NDArray[np.complex128] | complex],
        polarization: tuple[complex, complex] | NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        """Scale a scalar or transverse vector pupil profile to the power through the aperture."""
        radius = self.focal_distance_nm * np.sin(self.theta_grid)
        profile = np.asarray(field_profile(radius, self.phi_grid), dtype=complex)
        if profile.shape == self.theta_grid.shape + (2,) or profile.shape == (2,):
            transverse = np.broadcast_to(profile, self.theta_grid.shape + (2,))
        else:
            transverse = profile[..., None] * np.asarray(polarization, dtype=complex)
        pupil_norm = np.sum(np.sum(np.abs(transverse)**2, axis=-1)
                            * self.focal_distance_nm**2 * np.cos(self.theta_grid)
                            * self.solid_angle_weights)
        amplitude = np.sqrt(2 * self.power_w / (
            self.refractive_index * VACUUM_PERMITTIVITY_F_PER_M
            * SPEED_OF_LIGHT_M_PER_S * pupil_norm))
        amplitude /= METRES_PER_NANOMETRE
        incident_electric_field = np.zeros(self.theta_grid.shape + (3,), dtype=complex)
        incident_electric_field[..., :2] = amplitude * transverse
        return incident_electric_field

    @staticmethod
    def _propagation_direction(theta_grid: NDArray[np.float64],
                               phi_grid: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return unit propagation vectors on the pupil angular grid."""
        return np.stack((
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid), np.cos(theta_grid)), axis=-1)

    @staticmethod
    def _rotation_matrix(
        theta_grid: NDArray[np.float64], phi_grid: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return Rz(phi) Ry(theta) Rz(-phi) at every pupil point."""
        ct, st = np.cos(theta_grid), np.sin(theta_grid)
        cp, sp = np.cos(phi_grid), np.sin(phi_grid)
        rotation = np.empty(theta_grid.shape + (3, 3), dtype=float)
        rotation[..., 0, 0] = sp**2 + ct * cp**2
        rotation[..., 0, 1] = (ct - 1) * cp * sp
        rotation[..., 0, 2] = st * cp
        rotation[..., 1, 0] = rotation[..., 0, 1]
        rotation[..., 1, 1] = cp**2 + ct * sp**2
        rotation[..., 1, 2] = st * sp
        rotation[..., 2, 0] = -st * cp
        rotation[..., 2, 1] = -st * sp
        rotation[..., 2, 2] = ct
        return rotation

    @staticmethod
    def _position_vec(positions_nm: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return observation positions measured from the geometric focus."""
        return np.asarray(positions_nm, dtype=float)

    def _richards_wolf_integral(
        self, positions_nm: NDArray[np.float64],
        field_profile: Callable[
            [NDArray[np.float64], NDArray[np.float64]], NDArray[np.complex128] | complex],
        polarization: tuple[complex, complex] | NDArray[np.complex128],
    ) -> NDArray[np.complex128]:
        """Integrate the pupil field at all requested positions in nm."""
        incident_electric_field = self._propagating_field(field_profile, polarization)
        sphere_field = np.sqrt(np.cos(self.theta_grid))[..., None] * np.einsum(
            "...ij,...j->...i", self._rotation, incident_electric_field)
        weighted_field = (sphere_field * self.solid_angle_weights[..., None]).reshape(-1, 3)
        positions_nm = self._position_vec(positions_nm)
        electric_field = np.empty((positions_nm.size // 3, 3), dtype=complex)
        factor = -1j * self.focal_distance_nm * self.wave_number_per_nm / (2 * np.pi)

        for index, point in enumerate(positions_nm.reshape(-1, 3)):
            phase = np.exp(1j * self.wave_number_per_nm * (self._directions @ point))
            electric_field[index] = factor * (phase @ weighted_field)

        return electric_field.reshape(positions_nm.shape)

    def electric_field(
        self, positions_nm: NDArray[np.float64],
        field_profile: Callable[
            [NDArray[np.float64], NDArray[np.float64]], NDArray[np.complex128] | complex],
        polarization: tuple[complex, complex] | NDArray[np.complex128] = (1.0, 0.0),
    ) -> NDArray[np.complex128]:
        """Focus a scalar or transverse vector pupil profile at the specified power."""
        return self._richards_wolf_integral(positions_nm, field_profile, polarization)
