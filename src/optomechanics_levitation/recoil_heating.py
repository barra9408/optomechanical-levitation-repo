"""Compute torque recoil heating and orientation information from scattered light."""

from typing import Literal
import numpy as np
from numpy.typing import NDArray
from optomechanics_levitation.constants import (
    METRES_PER_NANOMETRE, REDUCED_PLANCK_CONSTANT_J_S,
    SPEED_OF_LIGHT_M_PER_S, VACUUM_PERMITTIVITY_F_PER_M,
)
from optomechanics_levitation import parameters as p
from optomechanics_levitation.light_matter_interaction import LightMatterInteraction
from optomechanics_levitation.particle import rotate_particle
from optomechanics_levitation.transfer_amplitudes import TransferAmplitudes

class TorqueRecoilHeating:
    """Integrate spin, orbital, and correlation terms of torque recoil heating.

    ``compute`` returns the total torque noise and its three contributions.
    """
    def __init__(self, transfer_amplitudes: TransferAmplitudes) -> None:
        """Set the field-to-rate conversion for a transfer-amplitude grid."""
        self.transfer_amplitudes = transfer_amplitudes
        radius_m = transfer_amplitudes.far_field_distance_nm * METRES_PER_NANOMETRE
        angular_frequency_rad_per_s = 2 * np.pi * SPEED_OF_LIGHT_M_PER_S / (
            transfer_amplitudes.wavelength_nm * METRES_PER_NANOMETRE)
        self.field_to_rate = VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S * radius_m**2 / (
            2 * REDUCED_PLANCK_CONSTANT_J_S * angular_frequency_rad_per_s)

    def compute(self, spin_transfer_amplitude: NDArray[np.complex128],
                orbital_transfer_amplitude: NDArray[np.complex128]) -> dict[str, float]:
        """Integrate spin, orbital, and interference contributions.

        Both amplitudes have shape (n_theta, n_phi, 3). The returned mapping
        contains ``total_torque``, ``spin_term``, ``orbital_term``, and
        ``correlation_term``.
        """
        solid_angle_weights = self.transfer_amplitudes.solid_angle_weights
        spin_term = self.field_to_rate * np.sum(
            np.sum(np.abs(spin_transfer_amplitude)**2, axis=-1) * solid_angle_weights) / (2 * np.pi)
        orbital_term = self.field_to_rate * np.sum(
            np.sum(np.abs(orbital_transfer_amplitude)**2, axis=-1)
            * solid_angle_weights) / (2 * np.pi)
        correlation_term = self.field_to_rate * np.sum(
            np.sum(np.conj(spin_transfer_amplitude) * orbital_transfer_amplitude, axis=-1).real
            * solid_angle_weights) / np.pi
        total_torque = spin_term + orbital_term + correlation_term
        return {"total_torque": float(total_torque), "spin_term": float(spin_term),
                "orbital_term": float(orbital_term), "correlation_term": float(correlation_term)}

class TheoreticalTorqueRecoilHeating:
    """Estimate torque recoil heating in the point-dipole limit.

    ``compute`` returns the polarizability anisotropy term and total torque noise.
    """
    def __init__(self, long_radius_nm: float, short_radius_nm: float, relative_permittivity: float,
                 wavelength_nm: float = p.WAVELENGTH_NM) -> None:
        """Set spheroid radii in nm, relative permittivity, and wavelength in nm."""
        self.long_radius_nm = float(long_radius_nm)
        self.short_radius_nm = float(short_radius_nm)
        self.relative_permittivity = float(relative_permittivity)
        self.wavelength_nm = float(wavelength_nm)

    def _polarizabilities(self) -> tuple[float, float]:
        """Return the polarizabilities parallel and perpendicular to the long axis."""
        long_radius_m = self.long_radius_nm * METRES_PER_NANOMETRE
        short_radius_m = self.short_radius_nm * METRES_PER_NANOMETRE
        volume = 4 * np.pi * long_radius_m * short_radius_m**2 / 3
        eccentricity = np.sqrt(1 - (short_radius_m / long_radius_m)**2)
        depolarization_parallel = ((1 - eccentricity**2) / (2 * eccentricity**3)
                                   * (np.log((1 + eccentricity) / (1 - eccentricity))
                                      - 2 * eccentricity))
        depolarization_perpendicular = (1 - depolarization_parallel) / 2
        permittivity_difference = self.relative_permittivity - 1
        polarizability_factor = VACUUM_PERMITTIVITY_F_PER_M * volume * permittivity_difference
        alpha_parallel = polarizability_factor / (
            1 + depolarization_parallel * permittivity_difference)
        alpha_perpendicular = polarizability_factor / (
            1 + depolarization_perpendicular * permittivity_difference)
        return alpha_parallel, alpha_perpendicular
    
    def compute(self, *, electric_field_v_per_m: NDArray[np.complex128],
                torque_axis: Literal["x", "y", "z"] = p.TORQUE_AXIS) -> dict[str, float]:
        """Return (Δα/α∥)² and recoil heating at the particle centre.

        The long particle axis is x. Rotating its dipole polarizability about
        ``torque_axis`` gives the induced-dipole derivative used here.
        ``electric_field_v_per_m`` is a complex three-component field vector.
        """
        alpha_parallel, alpha_perpendicular = self._polarizabilities()
        anisotropy_term = ((alpha_parallel - alpha_perpendicular) / alpha_parallel)**2
        electric_field = np.asarray(electric_field_v_per_m, dtype=complex)
        long_axis = np.array((1.0, 0.0, 0.0))
        axis_vector = np.eye(3)["xyz".index(torque_axis)]
        rotated_axis_derivative = np.cross(axis_vector, long_axis)
        dipole_derivative = (alpha_parallel - alpha_perpendicular) * (
            rotated_axis_derivative * electric_field[0]
            + long_axis * np.dot(rotated_axis_derivative, electric_field))
        angular_frequency_rad_per_s = 2 * np.pi * SPEED_OF_LIGHT_M_PER_S / (
            self.wavelength_nm * METRES_PER_NANOMETRE)
        power_derivative = (
            angular_frequency_rad_per_s**4 * np.vdot(dipole_derivative, dipole_derivative).real
            / (12 * np.pi * VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S**3))
        gamma_derivative = power_derivative / (
            REDUCED_PLANCK_CONSTANT_J_S * angular_frequency_rad_per_s)
        total_torque = REDUCED_PLANCK_CONSTANT_J_S**2 * gamma_derivative / (2 * np.pi)
        return {"anisotropy_term": float(anisotropy_term), "total_torque": float(total_torque)}

class InformationFlux:
    """Orientation information carried by the scattered far field.

    A central difference of fields from particles rotated by +/- ``angle_rad``
    gives the derivative at the unrotated orientation. Rotations are about the
    particle centre, with the incident beam fixed in the laboratory frame.
    ``compute`` returns the scattering rate, normalized information, and flux.
    """

    def __init__(self, transfer_amplitudes: TransferAmplitudes, *,
                 rotation_axes: tuple[Literal["x", "y", "z"], ...] = p.DERIVATIVE_ROTATION_AXES,
                 angle_rad: float = p.DERIVATIVE_ANGLE_RAD) -> None:
        """Set rotation axes and central-difference angle in radians."""
        self.transfer_amplitudes = transfer_amplitudes
        self.rotation_axes = tuple(rotation_axes)
        self.angle_rad = float(angle_rad)
        radius_m = transfer_amplitudes.far_field_distance_nm * METRES_PER_NANOMETRE
        angular_frequency_rad_per_s = 2 * np.pi * SPEED_OF_LIGHT_M_PER_S / (
            transfer_amplitudes.wavelength_nm * METRES_PER_NANOMETRE)
        self.field_to_rate = VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S * radius_m**2 / (
            2 * REDUCED_PLANCK_CONSTANT_J_S * angular_frequency_rad_per_s)

    def _scattered_electric_field(
        self, geometry_nm: NDArray[np.float64], observation_points_nm: NDArray[np.float64],
    ) -> NDArray[np.complex128]:
        """Return the (n_theta, n_phi, 3) scattered field for one geometry in nm."""
        calculator = self.transfer_amplitudes
        particle = dict(calculator.particle)
        particle["geometry_nm"] = geometry_nm
        incident_electric_field = calculator._incident_electric_field(geometry_nm)
        interaction = LightMatterInteraction(
            particle, calculator.wavelength_nm, calculator.parameters["environment"])
        interaction.internal_field(incident_electric_field)
        scattered_electric_field = np.empty((len(observation_points_nm), 3), dtype=complex)
        for start in range(0, len(scattered_electric_field), calculator.chunk_size):
            stop = min(start + calculator.chunk_size, len(scattered_electric_field))
            scattered_electric_field[start:stop] = interaction.far_field(
                observation_points_nm[start:stop], component="E")
        return scattered_electric_field.reshape(calculator.n_theta, calculator.n_phi, 3)

    def compute(self) -> dict[str, float | dict[str, float]]:
        """Return Gamma_sc [s^-1], D_mu, and total flux [rad^-2 s^-1]."""
        calculator = self.transfer_amplitudes
        observation_points_nm, _, solid_angle_weights = calculator._observation_surface("z")
        geometry_nm = np.asarray(calculator.particle["geometry_nm"], dtype=float)
        reference_scattered_electric_field = self._scattered_electric_field(
            geometry_nm, observation_points_nm)
        reference_norm = np.sum(np.sum(np.abs(reference_scattered_electric_field)**2, axis=-1)
                                * solid_angle_weights)
        scattering_rate_per_s = self.field_to_rate * reference_norm

        d_mu: dict[str, float] = {}
        information_flux: dict[str, float] = {}
        for axis in self.rotation_axes:
            angles_deg = np.zeros(3)
            angles_deg["xyz".index(axis)] = np.degrees(self.angle_rad)
            electric_field_plus = self._scattered_electric_field(
                rotate_particle(geometry_nm, angles_deg, origin_nm=calculator.center_nm),
                observation_points_nm)
            electric_field_minus = self._scattered_electric_field(
                rotate_particle(geometry_nm, -angles_deg, origin_nm=calculator.center_nm),
                observation_points_nm)
            electric_field_derivative = (
                electric_field_plus - electric_field_minus) / (2 * self.angle_rad)
            derivative_norm = np.sum(np.sum(np.abs(electric_field_derivative)**2, axis=-1)
                                     * solid_angle_weights)
            d_mu[axis] = float(derivative_norm / reference_norm)
            information_flux[axis] = float(4 * self.field_to_rate * derivative_norm)

        return {"scattering_rate_per_s": float(scattering_rate_per_s), "d_mu": d_mu,
                "information_flux": information_flux}
