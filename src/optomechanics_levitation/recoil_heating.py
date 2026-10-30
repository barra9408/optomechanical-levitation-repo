from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Literal
from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.constants import c, epsilon_0, hbar
from .incident_field import evaluate_incident_field
from .light_matter_interaction import LightMatterInteraction
from .optical_fields.hermitegauss00 import HermiteGauss00
from .optical_fields.laguerregauss import LaguerreGauss
from .optical_fields.planewave import PlaneWave
from .particle import rotate_particle
from . import parameters as p

FloatArray = NDArray[np.float64]; ComplexArray = NDArray[np.complex128]; Particle = dict[str, Any]
Axis = Literal["x", "y", "z"]
Environment = Mapping[str, float | complex]; FieldGenerator = Callable[..., ComplexArray] | PlaneWave | HermiteGauss00 | LaguerreGauss
AngularData = dict[str, FloatArray]; Components = tuple[float, float, float, float]
Spectrum = dict[str, FloatArray | ComplexArray]

class TheoreticalTorqueRecoilHeating:
    """Point-dipole torque recoil heating of a prolate spheroid.

    With no field argument, use the original x-polarized plane-wave intensity.
    An explicit field gives the local point-dipole prediction for that beam.
    """
    def __init__(self, long_radius_nm: float, short_radius_nm: float, epsilon_r: float, wavelength_nm: float = p.WAVELENGTH_NM,
                 intensity_w_per_m2: float = p.INTENSITY_W_PER_M2) -> None:
        self.long_radius_nm = float(long_radius_nm); self.short_radius_nm = float(short_radius_nm)
        self.epsilon_r = float(epsilon_r); self.wavelength_nm = float(wavelength_nm)
        self.intensity_w_per_m2 = float(intensity_w_per_m2)

    def _polarizabilities(self) -> tuple[float, float]:
        """Return the polarizabilities parallel and perpendicular to the long axis."""
        a = self.long_radius_nm * 1e-9; b = self.short_radius_nm * 1e-9
        volume = 4 * np.pi * a * b**2 / 3; eccentricity = np.sqrt(1 - (b / a)**2)
        L_parallel = ((1 - eccentricity**2) / (2 * eccentricity**3) * (np.log((1 + eccentricity) / (1 - eccentricity)) - 2 * eccentricity))
        L_perpendicular = (1 - L_parallel) / 2
        alpha_parallel = epsilon_0 * volume * (self.epsilon_r - 1) / (1 + L_parallel * (self.epsilon_r - 1))
        alpha_perpendicular = epsilon_0 * volume * (self.epsilon_r - 1) / (1 + L_perpendicular * (self.epsilon_r - 1))
        return alpha_parallel, alpha_perpendicular

    ####################################
    # THEORETICAL TORQUE NOISE
    ####################################
    def compute(self, electric_field_v_per_m: ComplexArray | None = None, *,
                torque_axis: Axis = p.TORQUE_AXIS) -> dict[str, float]:
        """Return (Δα/α∥)² and recoil heating for a local electric field.

        The long particle axis is x. Rotating its dipole polarizability about
        ``torque_axis`` gives the induced-dipole derivative used here. Spatial
        variation of the beam across the particle is outside this model.
        """
        alpha_parallel, alpha_perpendicular = self._polarizabilities(); anisotropy_term = ((alpha_parallel - alpha_perpendicular) / alpha_parallel)**2
        if electric_field_v_per_m is None:
            electric_amplitude = np.sqrt(2 * self.intensity_w_per_m2 / (epsilon_0 * c))
            electric_field = np.array((electric_amplitude, 0, 0), dtype=complex)
        else:
            electric_field = np.asarray(electric_field_v_per_m, dtype=complex)
            if electric_field.shape != (3,):
                raise ValueError("electric_field_v_per_m must have shape (3,)")
        if torque_axis not in ("x", "y", "z"):
            raise ValueError("torque_axis must be x, y, or z")
        long_axis = np.array((1.0, 0.0, 0.0))
        axis_vector = np.eye(3)["xyz".index(torque_axis)]
        rotated_axis_derivative = np.cross(axis_vector, long_axis)
        dipole_derivative = (alpha_parallel - alpha_perpendicular) * (
            rotated_axis_derivative * electric_field[0]
            + long_axis * np.dot(rotated_axis_derivative, electric_field)
        )
        omega = 2 * np.pi * c / (self.wavelength_nm * 1e-9)
        power_derivative = omega**4 * np.vdot(dipole_derivative, dipole_derivative).real / (12 * np.pi * epsilon_0 * c**3)
        gamma_derivative = power_derivative / (hbar * omega)
        total_torque = hbar**2 * gamma_derivative / (2 * np.pi)
        return {"anisotropy_term": float(anisotropy_term), "total_torque": float(total_torque)}


class DerivativeTorqueRecoilHeating:
    """Torque recoil heating from orientation derivatives of one particle."""
    def __init__(self, particle: Particle, wavelength_nm: float = p.WAVELENGTH_NM, environment: Environment = p.ENVIRONMENT, *,
                 far_field_distance_nm: float = p.FAR_FIELD_RADIUS_NM, rotation_axes: tuple[Axis, ...] = p.DERIVATIVE_ROTATION_AXES,
                 angle_rad: float = p.DERIVATIVE_ANGLE_RAD, n_theta: int = p.DERIVATIVE_N_THETA,
                 n_phi: int = p.DERIVATIVE_N_PHI) -> None:
        self.particle = dict(particle)
        self.particle["geometry_nm"] = np.asarray(particle["geometry_nm"], dtype=float).copy()
        self.wavelength_nm = float(wavelength_nm); self.environment = dict(environment)
        self.far_field_distance_nm = float(far_field_distance_nm)
        self.rotation_axes = tuple(axis.lower() for axis in rotation_axes); self.angle_rad = float(angle_rad)
        self.n_theta = int(n_theta); self.n_phi = int(n_phi)
        self._center_nm = self.particle["geometry_nm"].mean(axis=0)
        self._observation_points_nm, self._domega = self._observation_surface()

    ####################################
    # SCATTERING FIELD
    ####################################
    def _observation_surface(self) -> tuple[FloatArray, FloatArray]:
        """Build the far-field directions and their solid-angle weights."""
        theta = (np.arange(self.n_theta) + 0.5) * np.pi / self.n_theta; phi = np.linspace(0, 2 * np.pi, self.n_phi, endpoint=False)
        Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
        directions = np.column_stack(((np.sin(Theta) * np.cos(Phi)).ravel(), (np.sin(Theta) * np.sin(Phi)).ravel(), np.cos(Theta).ravel()))
        domega = np.sin(Theta) * (np.pi / self.n_theta) * (2 * np.pi / self.n_phi)
        return self.far_field_distance_nm * directions, domega

    def _scattering_rate(self, scattered_field: ComplexArray) -> float:
        """Integrate scattered intensity to obtain Γ_sc [s⁻¹]."""
        intensity = np.sum(np.abs(scattered_field)**2, axis=1).reshape(self.n_theta, self.n_phi)
        integral = np.sum(intensity * self._domega)
        radius_m = self.far_field_distance_nm * 1e-9
        omega = 2 * np.pi * c / (self.wavelength_nm * 1e-9)
        power_sc = epsilon_0 * c * radius_m**2 * integral / 2
        return power_sc / (hbar * omega)

    def _scattered_field_at(self, field_generator: FieldGenerator, parameters: dict[str, Any],
                            axis: Axis | None = p.DERIVATIVE_REFERENCE_AXIS,
                            rotation_rad: float = p.DERIVATIVE_REFERENCE_ANGLE_RAD) -> ComplexArray:
        """Solve the particle response and evaluate its scattered far field."""
        oriented_particle = self.particle.copy()
        if axis is not None:
            angles_deg = np.zeros(3); angles_deg["xyz".index(axis)] = np.degrees(rotation_rad)
            oriented_particle["geometry_nm"] = rotate_particle(self.particle["geometry_nm"], angles_deg, origin_nm=self._center_nm)
        incident_field = evaluate_incident_field(field_generator=field_generator, positions_nm=oriented_particle["geometry_nm"],
                                                 field="E", **parameters)
        interaction = LightMatterInteraction(oriented_particle, self.wavelength_nm, self.environment)
        interaction.internal_field(incident_field)
        return interaction.far_field(self._observation_points_nm, field="E")

    ####################################
    # ORIENTATION DERIVATIVES
    ####################################
    def compute(self, field_generator: FieldGenerator, *, amplitude_v_per_m: float,
                **beam_parameters: Any) -> dict[str, dict[str, float]]:
        """Return Γ_sc, D, and S_ττ for each requested rotation axis."""
        beam_parameters.update({key: tuple(beam_parameters[key]) for key in beam_parameters.keys() & {"direction"}})
        parameters = {"wavelength_nm": self.wavelength_nm, "environment": self.environment, "amplitude_v_per_m": amplitude_v_per_m,
                      **beam_parameters}
        gamma_sc = self._scattering_rate(self._scattered_field_at(field_generator, parameters))
        D = {}; total_torque = {}
        for axis in self.rotation_axes:
            field_plus = self._scattered_field_at(field_generator, parameters, axis, +self.angle_rad)
            field_minus = self._scattered_field_at(field_generator, parameters, axis, -self.angle_rad)
            field_derivative = (field_plus - field_minus) / (2 * self.angle_rad)
            gamma_derivative = self._scattering_rate(field_derivative)
            D[axis] = gamma_derivative / gamma_sc
            total_torque[axis] = hbar**2 * gamma_derivative / (2 * np.pi)
        return {"Gamma_sc": gamma_sc, "D": D, "total_torque": total_torque}

class TorqueRecoilHeating:
    def __init__(self, torque_axis: Literal["x", "y", "z"], N_theta: int = p.SPECTRAL_N_THETA, N_phi: int = p.SPECTRAL_N_PHI):
        self.torque_axis = torque_axis
        self.N_theta = N_theta
        self.N_phi = N_phi
        self.k_nm = 2 * np.pi / p.WAVELENGTH_NM
        self.radius_m = p.RADIUS_WAVELENGTHS * p.WAVELENGTH_NM * 1e-9
        self.omega = 2 * np.pi * c / (p.WAVELENGTH_NM * 1e-9)
        self.field_to_rate = epsilon_0 * c * self.radius_m**2 / (2 * hbar * self.omega)
        self.theta = np.linspace(0, np.pi, N_theta, endpoint=False) + np.pi / (2 * N_theta)
        self.phi = np.linspace(0, 2 * np.pi, N_phi, endpoint=False)
        self.Theta, self.Phi = np.meshgrid(self.theta, self.phi, indexing="ij")
        self.axis_index = "xyz".index(torque_axis)
        basis = np.eye(3)[[(self.axis_index + 1) % 3, (self.axis_index + 2) % 3, self.axis_index]]
        self.directions = self._propagation_vector(self.Theta, self.Phi) @ basis
        self.observation_offsets_nm = self.radius_m * 1e9 * self.directions.reshape(-1, 3)
        self.delta_phi = self.phi[1] - self.phi[0]
        self.domega = np.sin(self.Theta) * (np.pi / N_theta) * self.delta_phi

    ####################
    # OPERATORS
    ####################

    def _projection_axis(self):
        """Torque axis of interest ê_µ."""
        if self.torque_axis == "x":
            return np.array([1.0, 0.0, 0.0])
        elif self.torque_axis == "y":
            return np.array([0.0, 1.0, 0.0])
        elif self.torque_axis == "z":
            return np.array([0.0, 0.0, 1.0])

    @staticmethod
    def _propagation_vector(Theta, Phi):
        return np.stack((np.sin(Theta) * np.cos(Phi), np.sin(Theta) * np.sin(Phi), np.cos(Theta)), axis=-1)

    def _angular_derivative(self, field):
        """Azimuthal derivative on the grid oriented around the torque axis."""
        partial_phi_mu_field = (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2 * self.delta_phi)
        return partial_phi_mu_field

    def _angular_momentum_operators(self, field, *, positions_nm=None, derivatives=None, derivative_curl=None):
        axis_vector = self._projection_axis()
        if positions_nm is not None:
            rotation = np.cross(axis_vector, positions_nm - np.mean(positions_nm, axis=0))
            orbital_derivative = sum(rotation[:, j, None] * derivatives[j] for j in range(3))
            J_hat = -1j * hbar * orbital_derivative + 1j * hbar * np.cross(axis_vector, field)
            S_hat_prime = -1j * hbar * derivative_curl / self.k_nm**2
        else:
            S_hat_prime = 1j * hbar * self.directions[..., self.axis_index, None] * np.cross(self.directions, field)
            J_hat = -1j * hbar * self._angular_derivative(field) + 1j * hbar * np.cross(axis_vector, field)
            J_hat -= self.directions * np.sum(self.directions * J_hat, axis=-1, keepdims=True)
        L_hat_prime = J_hat - S_hat_prime
        return S_hat_prime, L_hat_prime

    #####################
    # FIELDS
    #####################

    def _run_simulation(self, particle, in_efield):
        interaction = LightMatterInteraction(particle, p.WAVELENGTH_NM, p.ENVIRONMENT)
        interaction.internal_field(in_efield)
        center_nm = np.mean(particle["geometry_nm"], axis=0)
        positions_nm = center_nm + self.observation_offsets_nm
        
        field = np.empty((len(positions_nm), 3), dtype=complex)
        for start in range(0, len(positions_nm), p.CHUNK_SIZE):
            stop = min(start + p.CHUNK_SIZE, len(positions_nm))
            field[start:stop] = interaction.far_field(positions_nm[start:stop], field="E")

        return field.reshape(self.N_theta, self.N_phi, 3)

    def small_incident_perturbation(self, particle, field_generator, parameters):
        positions_nm = np.asarray(particle["geometry_nm"], dtype=float)

        def incident_at(points_nm):
            return evaluate_incident_field(field_generator=field_generator, positions_nm=points_nm, field="E", **parameters)

        in_field = incident_at(positions_nm)
        step_nm = p.WAVELENGTH_NM * 1e-3
        cartesian_axes = np.eye(3)
        derivatives = []
        for direction in cartesian_axes:
            field_plus = incident_at(positions_nm + step_nm * direction)
            field_minus = incident_at(positions_nm - step_nm * direction)
            derivatives.append((field_plus - field_minus) / (2 * step_nm))

        mixed_derivatives = []
        axis_vector = self._projection_axis()
        for direction in cartesian_axes:
            field_pp = incident_at(positions_nm + step_nm * (axis_vector + direction))
            field_pm = incident_at(positions_nm + step_nm * (axis_vector - direction))
            field_mp = incident_at(positions_nm + step_nm * (-axis_vector + direction))
            field_mm = incident_at(positions_nm - step_nm * (axis_vector + direction))
            mixed_derivatives.append((field_pp - field_pm - field_mp + field_mm) / (4 * step_nm**2))
        derivative_curl = np.stack((mixed_derivatives[1][:, 2] - mixed_derivatives[2][:, 1],
                                    mixed_derivatives[2][:, 0] - mixed_derivatives[0][:, 2],
                                    mixed_derivatives[0][:, 1] - mixed_derivatives[1][:, 0]), axis=-1)
        return in_field, positions_nm, derivatives, derivative_curl

    #######################
    # TORQUE RECOIL HEATING
    #######################

    def _transfer_vector(self, O_sc_efield, trans_O_in_efield):
        q_j = O_sc_efield - trans_O_in_efield
        return q_j
    
    def _torque_recoil_heating(self, particle, field_generator, parameters):
        in_field, positions_nm, derivatives, derivative_curl = self.small_incident_perturbation(particle, field_generator,          parameters)
        S_in_field, L_in_field = self._angular_momentum_operators(in_field, positions_nm=positions_nm, derivatives=derivatives, derivative_curl=derivative_curl)

        trans_S_in_field = self._run_simulation(particle, S_in_field)
        trans_L_in_field = self._run_simulation(particle, L_in_field)
        sc_field = self._run_simulation(particle, in_field)
        S_sc_field, L_sc_field = self._angular_momentum_operators(sc_field)

        q_S_vec = self._transfer_vector(S_sc_field, trans_S_in_field)
        q_L_vec = self._transfer_vector(L_sc_field, trans_L_in_field)
        q_S = self.field_to_rate * np.sum(np.sum(np.abs(q_S_vec)**2, axis=-1) * self.domega) / (2 * np.pi)
        q_L = self.field_to_rate * np.sum(np.sum(np.abs(q_L_vec)**2, axis=-1) * self.domega) / (2 * np.pi)
        q_LS = self.field_to_rate * np.sum(np.sum(np.conj(q_S_vec) * q_L_vec, axis=-1).real * self.domega) / np.pi
        S_tau_tau = q_S + q_L + q_LS

        Gamma_sc = self.field_to_rate * np.sum(np.sum(np.abs(sc_field)**2, axis=-1) * self.domega)
        S_tau_tau_photon = S_tau_tau / Gamma_sc

        return S_tau_tau, q_S, q_L, q_LS, S_tau_tau_photon, Gamma_sc

    
