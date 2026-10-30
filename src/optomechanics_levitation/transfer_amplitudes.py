"""Transfer amplitudes for an operator acting on incident and scattered fields."""

from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol
import numpy as np
from numpy.typing import NDArray
from .incident_field import evaluate_incident_field
from .light_matter_interaction import LightMatterInteraction
from . import parameters as p

class TransferOperator(Protocol):
    def incident(self, field_at: Callable[[NDArray[np.float64]], NDArray[np.complex128]],
                 positions_nm: NDArray[np.float64], center_nm: NDArray[np.float64],
                 wavelength_nm: float) -> NDArray[np.complex128]: ...

    def scattered(self, field: NDArray[np.complex128], directions: NDArray[np.float64],
                  delta_phi: float, wavelength_nm: float) -> NDArray[np.complex128]: ...

class TransferAmplitudes:
    """
    Evaluate Δq = M(O E_in) - O(M E_in) for one particle and beam.
    """
    def __init__(self, particle: dict[str, Any], field_generator: Callable[..., NDArray[np.complex128]], 
                 parameters: Mapping[str, Any], n_theta: int = p.N_THETA, n_phi: int = p.N_PHI, 
                 far_field_distance_nm: float = p.FAR_FIELD_RADIUS_NM, chunk_size: int = p.CHUNK_SIZE) -> None:
        self.particle = particle
        self.field_generator = field_generator
        self.parameters = dict(parameters)
        self.wavelength_nm = self.parameters["wavelength_nm"]
        self.interaction = LightMatterInteraction(particle, self.wavelength_nm, self.parameters["environment"])
        self.positions_nm = np.asarray(particle["geometry_nm"], dtype=float)
        self.center_nm = self.positions_nm.mean(axis=0)
        self.n_theta = n_theta; self.n_phi = n_phi
        self.far_field_distance_nm = far_field_distance_nm
        self.chunk_size = chunk_size
        self.directions: NDArray[np.float64] | None = None
        self.domega: NDArray[np.float64] | None = None

    def _incident_field(self, positions_nm: NDArray[np.float64]) -> NDArray[np.complex128]:
        return evaluate_incident_field(self.field_generator, positions_nm, field="E", **self.parameters)

    def _observation_surface(self, axis: Literal["x", "y", "z"]) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        axis_index = "xyz".index(axis)
        basis = np.eye(3)[[(axis_index + 1) % 3, (axis_index + 2) % 3, axis_index]]
        theta = (np.arange(self.n_theta) + 0.5) * np.pi / self.n_theta
        phi = np.arange(self.n_phi) * 2 * np.pi / self.n_phi
        Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
        directions = np.stack((np.sin(Theta) * np.cos(Phi), np.sin(Theta) * np.sin(Phi), np.cos(Theta)), axis=-1) @ basis
        domega = np.sin(Theta) * (np.pi / self.n_theta) * (2 * np.pi / self.n_phi)
        positions_nm = self.center_nm + self.far_field_distance_nm * directions.reshape(-1, 3)
        return positions_nm, directions, domega

    def _run_simulation(self, incident_field: NDArray[np.complex128], observation_points_nm: NDArray[np.float64]) -> NDArray[np.complex128]:
        self.interaction.internal_field(incident_field)
        field = np.empty((len(observation_points_nm), 3), dtype=complex)
        for start in range(0, len(field), self.chunk_size):
            stop = min(start + self.chunk_size, len(field))
            field[start:stop] = self.interaction.far_field(observation_points_nm[start:stop], field="E")
        return field.reshape(self.n_theta, self.n_phi, 3)

    def _incident_operated_field(self, operator: TransferOperator) -> NDArray[np.complex128]:
        return operator.incident(self._incident_field, self.positions_nm, self.center_nm, self.wavelength_nm)

    def _scattered_operated_field(self, operator: TransferOperator,
                                  scattered_field: NDArray[np.complex128],
                                  directions: NDArray[np.float64]) -> NDArray[np.complex128]:
        return operator.scattered(scattered_field, directions, 2 * np.pi / self.n_phi, self.wavelength_nm)

    def transfer_amplitude(self, operator: TransferOperator) -> NDArray[np.complex128]:
        """Return the complex transfer amplitude on the observation grid.

        ``directions`` and ``domega`` hold the corresponding directions and
        solid-angle weights after this call.
        """
        axis: Literal["x", "y", "z"] = getattr(operator, "axis", "z")
        observation_points_nm, directions, self.domega = self._observation_surface(axis)
        self.directions = directions
        incident_field = self._incident_field(self.positions_nm)
        scattered_field = self._run_simulation(incident_field, observation_points_nm)
        operated_incident = self._incident_operated_field(operator)
        operated_scattered = self._scattered_operated_field(operator, scattered_field, directions)
        scattered_operated_incident = self._run_simulation(
            operated_incident, observation_points_nm)
        return scattered_operated_incident - operated_scattered
