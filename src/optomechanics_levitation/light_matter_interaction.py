"""Solve a particle's induced field and evaluate its scattered near and far fields."""

from collections.abc import Mapping
from typing import Any, Literal
import numpy as np
import scipy.linalg as la
from numpy.typing import NDArray
from pyGDM2 import propagators, structures
from scipy.spatial import cKDTree
from .constants import VACUUM_IMPEDANCE_OHMS

class LightMatterInteraction:
    """Solve the dipole interaction once and evaluate scattered E or H fields.

    Call ``internal_field`` with an (n, 3) incident electric field, then use
    ``near_field`` or ``far_field`` at observation positions in nm. The
    factorized interaction is reused for subsequent incident fields.
    """

    def __init__(self, particle: dict[str, Any], wavelength_nm: float,
                 environment: Mapping[str, float | complex]) -> None:
        """Store particle, wavelength in nm, and pyGDM2 environment settings."""
        self.particle = particle
        self.wavelength_nm = wavelength_nm
        self.environment = environment
        self.prepared_interaction: dict[str, Any] | None = None
        self.interaction_result: dict[str, Any] | None = None

    ####################################
    # INTERACTION MATRIX
    ####################################
    @staticmethod
    def _create_dyads(environment: Mapping[str, float | complex] | None) -> Any:
        """Create the Green dyads for the surrounding media."""
        env = {} if environment is None else environment
        default_eps = env.get("eps_env", 1.0)
        indices = [complex(env.get(key, default_eps))**0.5 for key in ("eps1", "eps2", "eps3")]
        return propagators.DyadsQuasistatic123(
            n1=indices[0], n2=indices[1], n3=indices[2],
            spacing=env.get("spacing_nm", env.get("spacing", 0.0)), radiative_correction=True)

    @staticmethod
    def _interaction_matrix(particle: dict[str, Any], wavelength_nm: float,
                            dyads: Any) -> tuple[NDArray[np.complex64], dict[str, Any]]:
        """Build the coupled-dipole interaction matrix and its scattering data."""
        geometry_nm = np.asarray(particle["geometry_nm"], dtype=np.float64)
        structure = structures.struct(
            particle["step_nm"], geometry_nm, particle["material"], normalization=particle["mesh"],
            auto_shift_structure=False, check_geometry_consistency=False, warn=False, verbose=False)
        self_terms = dyads.getSelfTermEE(wavelength_nm, structure)

        polarizability = dyads.getPolarizabilityTensor(wavelength_nm, structure)
        green_config = dyads.getConfigDictG(wavelength_nm, structure, None)
        matrix = np.zeros(
            (3 * len(geometry_nm), 3 * len(geometry_nm)), dtype=np.complex64, order="F")
        dyads.tsbs_EE(geometry_nm, wavelength_nm, self_terms, polarizability, green_config, matrix)

        prepared = {"structure": structure, "dyads": dyads, "wavelength_nm": wavelength_nm,
                    "polarizability": polarizability, "green_config": green_config}
        return matrix, prepared

    @staticmethod
    def _factorize_interaction_matrix(
        matrix: NDArray[np.complex64],
    ) -> tuple[NDArray[np.complex64], NDArray[np.int32]]:
        """Factorize the matrix once for all incident fields of this particle."""
        return la.lu_factor(matrix, overwrite_a=True)

    @classmethod
    def _evaluate_internal_field(
        cls, particle: dict[str, Any], incident_electric_field: NDArray[np.complex128],
        wavelength_nm: float, environment: Mapping[str, float | complex],
        prepared_interaction: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Solve the internal dipole field using the prepared interaction."""
        if prepared_interaction is None:
            dyads = cls._create_dyads(environment)
            matrix, prepared_interaction = cls._interaction_matrix(particle, wavelength_nm, dyads)
            prepared_interaction["lu"], prepared_interaction["pivots"] = (
                cls._factorize_interaction_matrix(matrix))

        incident_electric_field = np.asarray(incident_electric_field, dtype=np.complex64)
        internal_electric_field = la.lu_solve(
            (prepared_interaction["lu"], prepared_interaction["pivots"]),
            incident_electric_field.reshape(-1),
        ).reshape(-1, 3)
        return {"E_internal": internal_electric_field, "prepared_interaction": prepared_interaction}

    def solve(self, incident_electric_field: NDArray[np.complex128]) -> dict[str, Any]:
        """Solve the internal field, reusing the factorization for later beams."""
        self.interaction_result = self._evaluate_internal_field(
            self.particle, incident_electric_field, self.wavelength_nm,
            self.environment, self.prepared_interaction)
        self.prepared_interaction = self.interaction_result["prepared_interaction"]
        return self.interaction_result

    def internal_field(
        self, incident_electric_field: NDArray[np.complex128],
    ) -> NDArray[np.complex64]:
        """Return the internal electric field generated by an incident field."""
        return self.solve(incident_electric_field)["E_internal"]

    @classmethod
    def _evaluate_near_field(
        cls, interaction_result: dict[str, Any], positions_nm: NDArray[np.float64],
        component: Literal["E", "H", "both"] = "E",
    ) -> NDArray[np.complex64] | tuple[NDArray[np.complex64], NDArray[np.complex64]]:
        """Evaluate the near field from a solved interaction."""
        prepared = interaction_result["prepared_interaction"]
        structure = prepared["structure"]
        dyads = prepared["dyads"]
        positions_nm = np.asarray(positions_nm, dtype=np.float64)
        polarization = np.einsum(
            "nij,nj->ni", prepared["polarizability"], interaction_result["E_internal"])
        distances = cKDTree(structure.geometry).query(positions_nm, k=1)[0]
        outside = distances > 1.005 * structure.step
        probe_positions_nm = positions_nm[outside]

        result = []
        for selected_component in (("E", "H") if component == "both" else (component,)):
            scattered_field = np.full((len(positions_nm), 3), np.nan, dtype=np.complex64)
            green = np.zeros(
                (len(structure.geometry), len(probe_positions_nm), 3, 3), dtype=np.complex64)
            dyads.eval_G(
                structure.geometry, probe_positions_nm,
                dyads.G_EE if selected_component == "E" else dyads.G_HE,
                prepared["wavelength_nm"], prepared["green_config"], green)
            scattered_field[outside] = np.einsum("nmij,nj->mi", green, polarization)
            if selected_component == "H":
                # PyGDM's raw magnetic (B/H) response needs the SI impedance factor.
                scattered_field /= VACUUM_IMPEDANCE_OHMS
            result.append(scattered_field)
        return (result[0], result[1]) if component == "both" else result[0]

    def near_field(
        self, positions_nm: NDArray[np.float64], component: Literal["E", "H", "both"] = "E",
    ) -> NDArray[np.complex64] | tuple[NDArray[np.complex64], NDArray[np.complex64]]:
        """Return scattered E, H, or both at (n, 3) positions in nm."""
        return self._evaluate_near_field(self.interaction_result, positions_nm, component=component)

    @classmethod
    def _evaluate_far_field(
        cls, interaction_result: dict[str, Any], positions_nm: NDArray[np.float64],
        component: Literal["E", "H", "both"] = "E",
    ) -> NDArray[np.complex64] | tuple[NDArray[np.complex64], NDArray[np.complex64]]:
        """Evaluate the far field from a solved interaction."""
        prepared = interaction_result["prepared_interaction"]
        structure = prepared["structure"]
        dyads = prepared["dyads"]
        positions_nm = np.asarray(positions_nm, dtype=np.float64)
        polarization = np.einsum(
            "nij,nj->ni", prepared["polarizability"], interaction_result["E_internal"])
        eps = [complex(prepared["green_config"][key]) for key in ("eps1", "eps2", "eps3")]
        if eps[0] == eps[1] == eps[2]:
            # The interface propagator shifts grazing angles even when all layers
            # have the same permittivity. Use its free-space asymptotic limit here.
            radii_nm = np.linalg.norm(positions_nm, axis=1)
            if np.any(radii_nm < prepared["wavelength_nm"]):
                raise ValueError("Far-field observation distance must exceed the wavelength")
            directions = positions_nm / radii_nm[:, None]
            k0 = 2 * np.pi / prepared["wavelength_nm"]
            k = k0 * np.sqrt(eps[1])
            phase = np.exp(1j * k * (radii_nm[None, :] - structure.geometry @ directions.T))
            dipole_sum = (phase.T @ polarization) / radii_nm[:, None]
            scattered_electric_field = np.asarray(
                k0**2 * (
                    dipole_sum - directions * np.sum(
                        directions * dipole_sum, axis=1, keepdims=True)),
                dtype=np.complex64,
            )
        else:
            green = np.zeros((len(structure.geometry), len(positions_nm), 3, 3), dtype=np.complex64)
            dyads.eval_G(structure.geometry, positions_nm, dyads.G_EE_ff,
                         prepared["wavelength_nm"], prepared["green_config"], green)
            scattered_electric_field = np.einsum("nmij,nj->mi", green, polarization)
        if component == "E":
            return scattered_electric_field

        directions = positions_nm / np.linalg.norm(positions_nm, axis=1)[:, None]
        index = complex(prepared["green_config"]["eps2"])**0.5
        # Leading outgoing-wave H = n/Z0 * (direction cross E), as in the reference.
        scattered_magnetic_field = np.asarray(
            index / VACUUM_IMPEDANCE_OHMS * np.cross(directions, scattered_electric_field),
            dtype=np.complex64)
        if component == "both":
            return scattered_electric_field, scattered_magnetic_field
        return scattered_magnetic_field

    def far_field(
        self, positions_nm: NDArray[np.float64], component: Literal["E", "H", "both"] = "E",
    ) -> NDArray[np.complex64] | tuple[NDArray[np.complex64], NDArray[np.complex64]]:
        """Return scattered E, H, or both at (n, 3) far-field positions in nm."""
        return self._evaluate_far_field(self.interaction_result, positions_nm, component=component)

    @staticmethod
    def total_fields(incident_electric_field: NDArray[np.complex128],
                     incident_magnetic_field: NDArray[np.complex128],
                     scattered_electric_field: NDArray[np.complex128],
                     scattered_magnetic_field: NDArray[np.complex128]) -> (
                         tuple[NDArray[np.complex128], NDArray[np.complex128]]):
        """Add the incident and scattered electric and magnetic fields."""
        total_electric_field = incident_electric_field + scattered_electric_field
        total_magnetic_field = incident_magnetic_field + scattered_magnetic_field
        return total_electric_field, total_magnetic_field
