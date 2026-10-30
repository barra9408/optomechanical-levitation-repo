"""Focus an incident pupil field with the vector Richards-Wolf integral."""

from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
from scipy.constants import epsilon_0, c

class FocusedBeam:
    """
    Evaluate the electric field near the focus of an ideal aplanatic objective.
    """
    def __init__(self, wavelength: float, focal_dist: float, N_theta: int, N_phi: int, NA: float, power : float, refractive_index: float = 1.0) -> None:
        self.k = 2 * np.pi * refractive_index / wavelength
        self.focal_dist = focal_dist
        self.power = power
        self.refractive_index = refractive_index

        self.N_theta = N_theta; self.N_phi = N_phi
        self.theta_max = np.arcsin(NA / refractive_index)
        self.delta_theta = self.theta_max / N_theta; self.delta_phi = 2 * np.pi / N_phi
        self.theta = (np.arange(N_theta) + 0.5) * self.delta_theta
        self.phi = np.arange(N_phi) * self.delta_phi
        self.Theta, self.Phi = np.meshgrid(self.theta, self.phi, indexing="ij")
        self.domega = np.sin(self.Theta) * self.delta_theta * self.delta_phi

        self._directions = self._propagation_direction(self.Theta, self.Phi).reshape(-1, 3)
        self._rotation = self._rotation_matrix(self.Theta, self.Phi)

    def _propagating_field(self, field_profile: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.complex128] | complex], 
                           polarization: tuple[complex, complex] | NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Scale a scalar or transverse vector pupil profile to the power through the aperture."""
        radius = self.focal_dist * np.sin(self.Theta)
        profile = np.asarray(field_profile(radius, self.Phi), dtype=complex)
        if profile.shape == self.Theta.shape + (2,) or profile.shape == (2,):
            transverse = np.broadcast_to(profile, self.Theta.shape + (2,))
        else:
            transverse = profile[..., None] * np.asarray(polarization, dtype=complex)
        pupil_norm = np.sum(np.sum(np.abs(transverse)**2, axis=-1)
                            * self.focal_dist**2 * np.cos(self.Theta) * self.domega)
        amplitude = 1e9 * np.sqrt(2 * self.power / (self.refractive_index * epsilon_0 * c * pupil_norm))
        incident = np.zeros(self.Theta.shape + (3,), dtype=complex)
        incident[..., :2] = amplitude * transverse
        return incident

    @staticmethod
    def _propagation_direction(Theta: NDArray[np.float64], Phi: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.stack((np.sin(Theta) * np.cos(Phi), np.sin(Theta) * np.sin(Phi), np.cos(Theta)), axis=-1)

    @staticmethod
    def _rotation_matrix(Theta: NDArray[np.float64], Phi: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return Rz(phi) Ry(theta) Rz(-phi) at every pupil point."""
        ct, st = np.cos(Theta), np.sin(Theta)
        cp, sp = np.cos(Phi), np.sin(Phi)
        rotation = np.empty(Theta.shape + (3, 3), dtype=float)
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
    def _position_vec(position: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return observation positions measured from the geometric focus."""
        positions = np.asarray(position, dtype=float)
        return positions

    def _richards_wolf_integral(self, position: NDArray[np.float64], 
                                field_profile: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.complex128] | complex],
                                polarization: tuple[complex, complex] | NDArray[np.complex128]) -> NDArray[np.complex128]:
        incident = self._propagating_field(field_profile, polarization)
        sphere_field = np.sqrt(np.cos(self.Theta))[..., None] * np.einsum("...ij,...j->...i", self._rotation, incident)
        weighted_field = (sphere_field * self.domega[..., None]).reshape(-1, 3)
        positions = self._position_vec(position)
        result = np.empty((positions.size // 3, 3), dtype=complex)
        factor = -1j * self.focal_dist * self.k / (2 * np.pi)

        for index, point in enumerate(positions.reshape(-1, 3)):
            phase = np.exp(1j * self.k * (self._directions @ point))
            result[index] = factor * (phase @ weighted_field)

        return result.reshape(positions.shape)

    def field(self, position: NDArray[np.float64], 
              field_profile: Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.complex128] | complex],
              polarization: tuple[complex, complex] | NDArray[np.complex128] = (1.0, 0.0)) -> NDArray[np.complex128]:
        """Focus a scalar or transverse vector pupil profile at the specified power."""
        return self._richards_wolf_integral(position, field_profile, polarization)
