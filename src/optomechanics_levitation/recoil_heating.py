from __future__ import annotations
from collections.abc import Callable, Mapping
from typing import Any
import numpy as np
from numpy.typing import NDArray
from scipy.constants import c, epsilon_0, hbar
from .incident_field import evaluate_incident_field
from .light_matter_interaction import LightMatterInteraction
from .optical_fields.hermitegauss00 import HermiteGauss00
from .optical_fields.laguerregaus import LaguerreGauss01
from .optical_fields.planewave import Axis, PlaneWave
from .particle import rotate_particle
from . import parameters as p

FloatArray = NDArray[np.float64]; ComplexArray = NDArray[np.complex128]; Particle = dict[str, Any]
Environment = Mapping[str, float | complex]; FieldGenerator = Callable[..., ComplexArray] | PlaneWave | HermiteGauss00 | LaguerreGauss01
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
        """Return D, and S_ττ for each requested rotation axis."""
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
        return {"D": D, "total_torque": total_torque}


class SpectralTorqueRecoilHeating:
    """Spectral torque noise for one particle."""
    def __init__(self, particle: Particle, wavelength_nm: float = p.WAVELENGTH_NM, environment: Environment = p.ENVIRONMENT, *,
                 torque_axis: Axis = p.TORQUE_AXIS, n_theta: int = p.SPECTRAL_N_THETA, n_phi: int = p.SPECTRAL_N_PHI,
                 radius_wavelengths: float = p.RADIUS_WAVELENGTHS, input_grid: int = p.INPUT_GRID,
                 input_span_wavelengths: float = p.INPUT_SPAN_WAVELENGTHS, chunk_size: int = p.CHUNK_SIZE) -> None:
        self.particle = dict(particle)
        self.particle["geometry_nm"] = np.asarray(particle["geometry_nm"], dtype=float).copy()
        self.wavelength_nm = float(wavelength_nm)
        self.environment = dict(environment)
        self.torque_axis = torque_axis.lower()
        self.n_theta = int(n_theta)
        self.n_phi = int(n_phi)
        self.radius_wavelengths = float(radius_wavelengths)
        self.input_grid = int(input_grid)
        self.input_span_wavelengths = float(input_span_wavelengths)
        self.chunk_size = int(chunk_size)
        self._interaction = LightMatterInteraction(self.particle, self.wavelength_nm, self.environment)
        self._prepared_interaction = None

    def compute(self, field_generator: FieldGenerator, *, amplitude_v_per_m: float, return_angular: bool = p.DEFAULT_RETURN_ANGULAR,
                incident_spectrum: Spectrum | None = p.DEFAULT_INCIDENT_SPECTRUM, **beam_parameters: Any) -> dict[str, Any]:
        """Returns total torque, spin, orbital, and correlation contribution."""        
        beam_parameters.update({key: tuple(beam_parameters[key]) for key in beam_parameters.keys() & {"direction"}})
        parameters = {"wavelength_nm": self.wavelength_nm, "environment": self.environment, "amplitude_v_per_m": amplitude_v_per_m, 
                      **beam_parameters}
        # 1. Generate incident beam
        incident_field = evaluate_incident_field(field_generator=field_generator, positions_nm=self.particle["geometry_nm"], 
                                                 field="E", **parameters)
        # 2. Calculate internal dipole interaction
        self._interaction.internal_field(incident_field)
        # 3. Generate the scatter field and calculate the spectral torque recoil heating
        self._prepared_interaction = self._interaction.prepared_interaction
        calculation = self._torque_recoil_heating_terms(self._interaction, field_generator, parameters, 
                                                        torque_axis=self.torque_axis, radius_wavelengths=self.radius_wavelengths, 
                                                        n_theta=self.n_theta, n_phi=self.n_phi, input_grid=self.input_grid, 
                                                        input_span_wavelengths=self.input_span_wavelengths, chunk_size=self.chunk_size, 
                                                        return_angular=return_angular, incident_spectrum=incident_spectrum)
        # Return each torque recoil heating term and their integrated values along every direction (if requested)
        # The latter is useful to calculate the information patterns of the torque noise
        (total, spin, orbital, correlation), angular = calculation if return_angular else (calculation, None)
        scale = 1 / (2 * np.pi) # Scale factor used in the theoretical model 
        result = {"total_torque": total * scale, "spin_term": spin * scale, "orbital_term": orbital * scale, "correlation_term": 2 * correlation * scale}
        if return_angular:
            result["angular"] = angular
        return result
    
    ####################################
    # SCATTERING FIELD
    ####################################
    @staticmethod
    def _axis_vector(axis: Axis) -> FloatArray:
        """Cartesian unit vector ê_µ"""
        return np.eye(3)[{"x": 0, "y": 1, "z": 2}[axis.lower()]]

    @staticmethod
    def _angular_surface(interaction: LightMatterInteraction, axis: Axis, radius_wavelengths: float,
                         n_theta: int, n_phi: int) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        """Build directions grid in a sphere where the scattered field will be evaluated"""
        prepared = interaction.prepared_interaction; wavelength_nm = prepared["wavelength_nm"]
        center_nm = np.mean(prepared["structure"].geometry, axis=0); radius_nm = radius_wavelengths * wavelength_nm
        axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]

        basis = np.eye(3)[[(axis_index + 1) % 3, (axis_index + 2) % 3, axis_index]]
        theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta; phi = np.arange(n_phi) * 2 * np.pi / n_phi
        Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
        
        directions = np.stack((np.sin(Theta) * np.cos(Phi), np.sin(Theta) * np.sin(Phi), np.cos(Theta)), axis=-1) @ basis
        domega = np.sin(Theta) * (np.pi / n_theta) * (2 * np.pi / n_phi)
        positions_nm = center_nm + radius_nm * directions.reshape(-1, 3)
        return positions_nm, directions, domega, radius_nm * 1e-9

    @staticmethod
    def _far_field_on_surface(interaction: LightMatterInteraction, positions_nm: FloatArray,
                              grid_shape: tuple[int, int], chunk_size: int) -> ComplexArray:
        """Evaluate the distributed electric field at all points on the sphere created by _angular_surface."""
        field = np.empty((len(positions_nm), 3), dtype=complex)
        for start in range(0, len(positions_nm), chunk_size):
            stop = min(start + chunk_size, len(positions_nm))
            field[start:stop] = interaction.far_field(positions_nm[start:stop], field="E")
        return field.reshape(*grid_shape, 3)

    ####################################
    # INCIDENT ANGULAR SPECTRUM
    ####################################
    def incident_angular_spectrum(self, field_generator: FieldGenerator, *, amplitude_v_per_m: float,
                                  center_nm: FloatArray, input_grid: int, input_span_wavelengths: float,
                                  **beam_parameters: Any) -> Spectrum:
        """Calculate an incident spectrum once for a beam in its reference frame."""
        parameters = {"wavelength_nm": self.wavelength_nm, "environment": self.environment,
                      "amplitude_v_per_m": amplitude_v_per_m, **beam_parameters}
        return self._angular_spectrum_from_field(field_generator, parameters, center_nm, self.torque_axis,
                                                 input_grid, input_span_wavelengths)

    @classmethod
    def rotate_incident_spectrum(cls, spectrum: Spectrum, frame: FloatArray, center_nm: FloatArray, axis: Axis) -> Spectrum:
        """Rotate the numerical input modes and differentiate them in the lab frame."""
        directions = spectrum["base_directions"] @ frame.T; amplitudes = spectrum["base_amplitudes"] @ frame.T
        return cls._discrete_angular_spectrum(directions, amplitudes, center_nm, axis,
                                              np.linalg.norm(spectrum["wavevectors_nm"][0]))

    @staticmethod
    def _rotate_about_axis(vector: FloatArray | ComplexArray, axis_vector: FloatArray, angle: float) -> FloatArray | ComplexArray:
        """Rotate sampled directions or amplitudes using Rodrigues' formula."""
        return (np.cos(angle) * vector + np.sin(angle) * np.cross(axis_vector, vector)
                + (1 - np.cos(angle)) * axis_vector * np.sum(vector * axis_vector, axis=-1, keepdims=True))

    @classmethod
    def _discrete_angular_spectrum(cls, directions: FloatArray, amplitudes: ComplexArray, center_nm: FloatArray,
                                   axis: Axis, k: float) -> Spectrum:
        """Differentiate each numerical angular mode by a symmetric rotation."""
        angle = p.ANGULAR_SPECTRUM_DERIVATIVE_ANGLE_RAD; axis_vector = cls._axis_vector(axis)
        directions_minus = cls._rotate_about_axis(directions, axis_vector, -angle)
        directions_plus = cls._rotate_about_axis(directions, axis_vector, +angle)
        derivative_center = np.cross(axis_vector, amplitudes)
        derivative_minus = cls._rotate_about_axis(amplitudes, axis_vector, -angle) / (2 * angle)
        derivative_plus = -cls._rotate_about_axis(amplitudes, axis_vector, +angle) / (2 * angle)
        sampled_directions = np.concatenate((directions, directions_minus, directions_plus))
        sampled_amplitudes = np.concatenate((amplitudes, np.zeros_like(amplitudes), np.zeros_like(amplitudes)))
        derivative = np.concatenate((derivative_center, derivative_minus, derivative_plus))
        return {"directions": sampled_directions, "wavevectors_nm": k * sampled_directions, "origin_nm": np.asarray(center_nm),
                "amplitudes": sampled_amplitudes, "azimuthal_derivative": derivative,
                "base_directions": directions, "base_amplitudes": amplitudes}

    @classmethod
    def _angular_spectrum_from_field(cls, field_generator: FieldGenerator, incident_parameters: dict[str, Any],
                                     center_nm: FloatArray, axis: Axis, input_grid: int,
                                     input_span_wavelengths: float) -> Spectrum:
        """
        Forward 2-D Fourier transform of E into complex, transverse angular amplitudes.

        amplitudes[j] contains a(n_j) dΩ_j. Two planes separate forward and backward waves;
        dΩ = dkx dky / (k |kz|). The planes follow the beam's reference direction.
        """
        wavelength_nm = incident_parameters["wavelength_nm"]
        size_nm = input_span_wavelengths * wavelength_nm; step_nm = size_nm / input_grid
        reference_direction = np.array(incident_parameters.get("beam_direction", getattr(field_generator, "direction", (0, 0, 1))), dtype=float)
        reference_direction /= np.linalg.norm(reference_direction)
        reference = np.eye(3)[np.argmin(np.abs(reference_direction))]
        e1 = reference - np.dot(reference, reference_direction) * reference; e1 /= np.linalg.norm(e1)
        frame = np.column_stack((e1, np.cross(reference_direction, e1), reference_direction))
        offsets = (np.arange(input_grid) - input_grid // 2) * step_nm
        X, Y = np.meshgrid(offsets, offsets, indexing="ij"); dz = wavelength_nm / 8
        local_positions = np.stack((X, Y, np.zeros_like(X)), axis=-1).reshape(-1, 3)
        sample_positions = np.asarray(center_nm) + local_positions @ frame.T
        upper_positions = sample_positions + dz * reference_direction
        plane_spectra = []
        for positions in (sample_positions, upper_positions):
            field = evaluate_incident_field(field_generator=field_generator, positions_nm=positions, field="E", **incident_parameters)
            centered_field = np.fft.ifftshift((field @ frame).reshape(input_grid, input_grid, 3), axes=(0, 1))
            plane_spectra.append(np.fft.fftshift(np.fft.fft2(centered_field, axes=(0, 1)), axes=(0, 1)) / input_grid**2)

        k = 2 * np.pi / wavelength_nm
        kxy = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(input_grid, d=step_nm))
        Kx, Ky = np.meshgrid(kxy, kxy, indexing="ij")
        Kz = np.sqrt(np.maximum(k**2 - Kx**2 - Ky**2, 0))
        propagating = Kz > 0; phase = np.exp(-1j * Kz * dz)
        denominator = np.where(propagating, 2j * np.sin(Kz * dz), 1)
        positive = (plane_spectra[1] - plane_spectra[0] * phase[..., None]) / denominator[..., None]; negative = plane_spectra[0] - positive
        positive[~propagating] = 0; negative[~propagating] = 0
        directions = []; amplitudes = []
        for sign, amplitude in ((1, positive), (-1, negative)):
            direction = np.stack((Kx / k, Ky / k, sign * Kz / k), axis=-1)
            amplitude = amplitude - direction * np.sum(direction * amplitude, axis=-1, keepdims=True)
            directions.append(direction[propagating])
            amplitudes.append(amplitude[propagating])

        directions = np.concatenate(directions); amplitudes = np.concatenate(amplitudes)
        active = np.linalg.norm(amplitudes, axis=-1) > np.max(np.linalg.norm(amplitudes, axis=-1)) * 1e-10
        return cls._discrete_angular_spectrum(directions[active] @ frame.T, amplitudes[active] @ frame.T,
                                              center_nm, axis, k)

    @staticmethod
    def _field_from_angular_spectrum(positions_nm: FloatArray, spectrum: Spectrum,
                                     amplitudes: ComplexArray, chunk_size: int) -> ComplexArray:
        """Inverse angular transform at the particle dipoles."""
        field = np.zeros((len(positions_nm), 3), dtype=complex)
        relative_positions = positions_nm - spectrum["origin_nm"]
        active = np.flatnonzero(np.any(amplitudes != 0, axis=1))
        for start in range(0, len(active), chunk_size):
            indices = active[start:start + chunk_size]
            phase = np.exp(1j * relative_positions @ spectrum["wavevectors_nm"][indices].T)
            field += phase @ amplitudes[indices]
        return field

    @classmethod
    def _apply_linear_scattering_kernel(cls, interaction: LightMatterInteraction, spectrum: Spectrum, amplitudes: ComplexArray,
                                        positions_nm: FloatArray, grid_shape: tuple[int, int], chunk_size: int) -> ComplexArray:
        """Apply K = far-field map x polarizability x inverse interaction x incident map."""
        prepared = interaction.prepared_interaction
        incident_at_particle = cls._field_from_angular_spectrum(prepared["structure"].geometry, spectrum, amplitudes, chunk_size)
        interaction.internal_field(incident_at_particle)
        return cls._far_field_on_surface(interaction, positions_nm, grid_shape, chunk_size)

    @classmethod
    def _scattering_data(cls, interaction: LightMatterInteraction, field_generator: FieldGenerator, incident_parameters: dict[str, Any],
                         axis: Axis, radius_wavelengths: float, n_theta: int, n_phi: int, input_grid: int,
                         input_span_wavelengths: float, chunk_size: int,
                         incident_spectrum: Spectrum | None = p.DEFAULT_INCIDENT_SPECTRUM
                         ) -> tuple[Spectrum, FloatArray, FloatArray, ComplexArray, Callable[[ComplexArray], ComplexArray], float]:
        positions_nm, directions_out, domega_out, radius_m = cls._angular_surface(
            interaction, axis, radius_wavelengths, n_theta, n_phi)
        prepared = interaction.prepared_interaction
        center_nm = np.mean(prepared["structure"].geometry, axis=0)
        spectrum = incident_spectrum if incident_spectrum is not None else cls._angular_spectrum_from_field(
            field_generator, incident_parameters, center_nm, axis, input_grid, input_span_wavelengths)
        omega = 2 * np.pi * c / (prepared["wavelength_nm"] * 1e-9)
        amplitude_scale = np.sqrt(epsilon_0 * c * radius_m**2 / (2 * hbar * omega))

        def scatter(amplitudes: ComplexArray) -> ComplexArray:
            return amplitude_scale * cls._apply_linear_scattering_kernel(
                interaction, spectrum, amplitudes, positions_nm, (n_theta, n_phi), chunk_size)

        outgoing = scatter(spectrum["amplitudes"])
        return spectrum, directions_out, domega_out, outgoing, scatter, omega

    @staticmethod
    def _angular_inner_product(left: ComplexArray, right: ComplexArray, domega: FloatArray) -> complex:
        return np.sum(np.sum(np.conj(left) * right, axis=-1) * domega)

    ####################################
    # OPERATORS
    ####################################
    @staticmethod
    def _spin_operator(field: ComplexArray, directions: FloatArray, axis: Axis) -> ComplexArray:
        """Transverse spin Ô_S,µ / hbar = i n̂_µ (n̂ x E)."""
        axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
        return 1j * np.asarray(directions)[..., axis_index, None] * np.cross(directions, field)

    @classmethod
    def _angular_momentum_operator(cls, field: ComplexArray, directions: FloatArray,
                                   azimuthal_derivative: ComplexArray, axis: Axis) -> ComplexArray:
        """Total angular momentum Ô_J,µ / hbar in the transverse subspace."""
        result = -1j * azimuthal_derivative + 1j * np.cross(cls._axis_vector(axis), field)
        return result - directions * np.sum(directions * result, axis=-1, keepdims=True)

    @classmethod
    def _orbital_operator(cls, field: ComplexArray, directions: FloatArray,
                          azimuthal_derivative: ComplexArray, axis: Axis) -> ComplexArray:
        """Transverse orbital operator Ô_L,µ = Ô_J,µ - Ô_S,µ."""
        return (cls._angular_momentum_operator(field, directions, azimuthal_derivative, axis) - cls._spin_operator(field, directions, axis))

    @staticmethod
    def _outgoing_azimuthal_derivative(field: ComplexArray) -> ComplexArray:
        """Scattered field derivative with respect to the azimuthal angle ϕ."""
        # Decompose the periodic azimuthal field into Fourier modes e^(i m ϕ).
        n_phi = field.shape[1]; modes = np.fft.fftfreq(n_phi, d=1 / n_phi)
        # Multiply each mode by i m to differentiate, then return to angle space.
        return np.fft.ifft(1j * modes[None, :, None] * np.fft.fft(field, axis=1), axis=1)
    
    ####################################
    # TRANSFER AMPLITUDES
    ####################################
    @classmethod
    def _spin_transfer_amplitude(cls, spectrum: Spectrum, outgoing: ComplexArray, directions_out: FloatArray,
                                 scatter: Callable[[ComplexArray], ComplexArray], axis: Axis) -> ComplexArray:
        """q_S = Ô_S,sc Ka - K(Ô_S,in a)."""
        incoming = cls._spin_operator(spectrum["amplitudes"], spectrum["directions"], axis)
        return cls._spin_operator(outgoing, directions_out, axis) - scatter(incoming)

    @classmethod
    def _orbital_transfer_amplitude(cls, spectrum: Spectrum, outgoing: ComplexArray, directions_out: FloatArray,
                                    scatter: Callable[[ComplexArray], ComplexArray], axis: Axis) -> ComplexArray:
        """q_L = Ô_L,sc Ka - K(Ô_L,in a)."""
        incoming = cls._orbital_operator(spectrum["amplitudes"], spectrum["directions"], spectrum["azimuthal_derivative"], axis)
        outgoing_derivative = cls._outgoing_azimuthal_derivative(outgoing)
        return cls._orbital_operator(outgoing, directions_out, outgoing_derivative, axis) - scatter(incoming)

    ####################################
    # SPECTRAL TORQUE NOISE
    ####################################
    @classmethod
    def _torque_recoil_heating_terms(cls, interaction: LightMatterInteraction, field_generator: FieldGenerator,
                                     incident_parameters: dict[str, Any], *, torque_axis: Axis = p.TORQUE_AXIS,
                                     radius_wavelengths: float = p.RADIUS_WAVELENGTHS, n_theta: int = p.SPECTRAL_N_THETA,
                                     n_phi: int = p.SPECTRAL_N_PHI, input_grid: int = p.INPUT_GRID,
                                     input_span_wavelengths: float = p.INPUT_SPAN_WAVELENGTHS, chunk_size: int = p.CHUNK_SIZE,
                                     return_angular: bool = p.DEFAULT_RETURN_ANGULAR, 
                                     incident_spectrum: Spectrum | None = p.DEFAULT_INCIDENT_SPECTRUM) -> Components | tuple[Components, AngularData]:
        spectrum, directions_out, domega_out, outgoing, scatter, _ = cls._scattering_data(interaction, field_generator,
                                                                                           incident_parameters, torque_axis,
                                                                                           radius_wavelengths, n_theta, n_phi, input_grid,
                                                                                           input_span_wavelengths, chunk_size,
                                                                                           incident_spectrum=incident_spectrum)

        spin_transfer = cls._spin_transfer_amplitude(spectrum, outgoing, directions_out, scatter, torque_axis)
        orbital_transfer = cls._orbital_transfer_amplitude(spectrum, outgoing, directions_out, scatter, torque_axis)
        
        spin = hbar**2 * cls._angular_inner_product(spin_transfer, spin_transfer, domega_out).real
        orbital = hbar**2 * cls._angular_inner_product(orbital_transfer, orbital_transfer, domega_out).real
        correlation = hbar**2 * cls._angular_inner_product(orbital_transfer, spin_transfer, domega_out).real
        total = spin + orbital + 2 * correlation
        components = (float(total), float(spin), float(orbital), float(correlation))
        if not return_angular:
            return components

        scale = hbar**2 / (2 * np.pi)
        spin_density = scale * np.sum(np.abs(spin_transfer)**2, axis=-1)
        orbital_density = scale * np.sum(np.abs(orbital_transfer)**2, axis=-1)
        correlated_density = 2 * scale * np.sum(np.conj(orbital_transfer) * spin_transfer, axis=-1).real
        angular = {"directions": directions_out, "domega": domega_out, "spin": spin_density, "orbital": orbital_density, 
                   "correlation": correlated_density, "total": spin_density + orbital_density + correlated_density}
        return components, angular
