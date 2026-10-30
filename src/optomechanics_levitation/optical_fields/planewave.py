import numpy as np
from numpy.typing import NDArray
from scipy.constants import epsilon_0, c

class PlaneWave:
    """Plane wave along +z, with power defined through a circular pupil."""
    def __init__(self, wavelength: float, power: float, pupil_radius: float, refractive_index: float = 1.0):
        self.k = 2 * np.pi * refractive_index / wavelength
        area_m2 = np.pi * (pupil_radius * 1e-9)**2
        self.amplitude = np.sqrt(2 * power / (refractive_index * epsilon_0 * c * area_m2))

    def _plane_wave_profile(self, rho: NDArray[np.float64], phi: NDArray[np.float64], z: float = 0.0) -> NDArray[np.complex128]:
        return np.ones(np.broadcast_shapes(np.shape(rho), np.shape(phi), np.shape(z)), dtype=complex)

    def _plane_wave_field(self, position: NDArray[np.float64], amplitude: complex, 
                          polarization: tuple[complex, complex]) -> NDArray[np.complex128]:
        position = np.asarray(position, dtype=float)
        phase = np.exp(1j * self.k * position[..., 2])
        electric_field = np.zeros(position.shape, dtype=complex)
        jones = np.asarray(polarization, dtype=complex)
        electric_field[..., :2] = amplitude * phase[..., None] * jones / np.linalg.norm(jones)
        return electric_field

    def evaluate(self, output: str, position: NDArray[np.float64] | None = None, amplitude: complex | None = None, 
                 polarization: tuple[complex, complex] = (1.0, 0.0)):
        """Return the constant pupil profile or the incident electric field."""
        if output == "profile":
            return self._plane_wave_profile
        elif output == "field":
            return self._plane_wave_field(position, self.amplitude if amplitude is None else amplitude, polarization)
