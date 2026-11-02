from typing import Literal
import numpy as np
from numpy.typing import NDArray
from scipy.constants import c, epsilon_0, hbar
from optomechanics_levitation import parameters as p
from optomechanics_levitation.light_matter_interaction import LightMatterInteraction
from optomechanics_levitation.particle import rotate_particle
from optomechanics_levitation.transfer_amplitudes import TransferAmplitudes

class TorqueRecoilHeating:
    """Torque recoil heating from spin and orbital transfer amplitudes."""
    def __init__(self, transfer_amplitudes: TransferAmplitudes) -> None:
        self.transfer_amplitudes = transfer_amplitudes
        radius_m = transfer_amplitudes.far_field_distance_nm * 1e-9
        omega = 2 * np.pi * c / (transfer_amplitudes.wavelength_nm * 1e-9)
        self.field_to_rate = epsilon_0 * c * radius_m**2 / (2 * hbar * omega)

    def compute(self, q_S_vec: NDArray[np.complex128], q_L_vec: NDArray[np.complex128]) -> dict[str, float]:
        """Integrate spin, orbital, and interference contributions."""
        domega = self.transfer_amplitudes.domega
        q_S = self.field_to_rate * np.sum(np.sum(np.abs(q_S_vec)**2, axis=-1) * domega) / (2 * np.pi)
        q_L = self.field_to_rate * np.sum(np.sum(np.abs(q_L_vec)**2, axis=-1) * domega) / (2 * np.pi)
        q_LS = self.field_to_rate * np.sum(np.sum(np.conj(q_S_vec) * q_L_vec, axis=-1).real * domega) / np.pi
        S_tau_tau = q_S + q_L + q_LS
        return {"total_torque": float(S_tau_tau), "q_S": float(q_S), "q_L": float(q_L), "q_LS": float(q_LS)}

class TheoreticalTorqueRecoilHeating:
    """Point-dipole torque recoil heating for an x-polarized plane wave.

    The plane-wave amplitude uses the optical power through the entrance pupil.
    """
    def __init__(self, long_radius_nm: float, short_radius_nm: float, epsilon_r: float, wavelength_nm: float = p.WAVELENGTH_NM) -> None:
        self.long_radius_nm = float(long_radius_nm); self.short_radius_nm = float(short_radius_nm)
        self.epsilon_r = float(epsilon_r); self.wavelength_nm = float(wavelength_nm)

    def _polarizabilities(self) -> tuple[float, float]:
        """Return the polarizabilities parallel and perpendicular to the long axis."""
        a = self.long_radius_nm * 1e-9; b = self.short_radius_nm * 1e-9
        volume = 4 * np.pi * a * b**2 / 3; eccentricity = np.sqrt(1 - (b / a)**2)
        L_parallel = ((1 - eccentricity**2) / (2 * eccentricity**3) * (np.log((1 + eccentricity) / (1 - eccentricity)) - 2 * eccentricity))
        L_perpendicular = (1 - L_parallel) / 2
        alpha_parallel = epsilon_0 * volume * (self.epsilon_r - 1) / (1 + L_parallel * (self.epsilon_r - 1))
        alpha_perpendicular = epsilon_0 * volume * (self.epsilon_r - 1) / (1 + L_perpendicular * (self.epsilon_r - 1))
        return alpha_parallel, alpha_perpendicular
    
    def compute(self, *, torque_axis: Literal["x", "y", "z"] = p.TORQUE_AXIS) -> dict[str, float]:
        """Return (Δα/α∥)² and recoil heating for the plane wave.

        The long particle axis is x. Rotating its dipole polarizability about
        ``torque_axis`` gives the induced-dipole derivative used here.
        """
        alpha_parallel, alpha_perpendicular = self._polarizabilities(); anisotropy_term = ((alpha_parallel - alpha_perpendicular) / alpha_parallel)**2
        pupil_radius_nm = p.FOCAL_DISTANCE_NM * p.NUMERICAL_APERTURE / p.REFRACTIVE_INDEX
        area_m2 = np.pi * (pupil_radius_nm * 1e-9)**2
        electric_amplitude = np.sqrt(2 * p.OPTICAL_POWER_W / (p.REFRACTIVE_INDEX * epsilon_0 * c * area_m2))
        electric_field = np.array((electric_amplitude, 0, 0), dtype=complex)
        long_axis = np.array((1.0, 0.0, 0.0))
        axis_vector = np.eye(3)["xyz".index(torque_axis)]
        rotated_axis_derivative = np.cross(axis_vector, long_axis)
        dipole_derivative = (alpha_parallel - alpha_perpendicular) * (rotated_axis_derivative * electric_field[0] + long_axis * np.dot(rotated_axis_derivative, electric_field))
        omega = 2 * np.pi * c / (self.wavelength_nm * 1e-9)
        power_derivative = omega**4 * np.vdot(dipole_derivative, dipole_derivative).real / (12 * np.pi * epsilon_0 * c**3)
        gamma_derivative = power_derivative / (hbar * omega)
        total_torque = hbar**2 * gamma_derivative / (2 * np.pi)
        return {"anisotropy_term": float(anisotropy_term), "total_torque": float(total_torque)}

class InformationFlux:
    """Orientation information carried by the scattered far field.

    A central difference of fields from particles rotated by +/- ``angle_rad``
    gives the derivative at the unrotated orientation. Rotations are about the
    particle centre, with the incident beam fixed in the laboratory frame.
    """

    def __init__(self, transfer_amplitudes: TransferAmplitudes, *, rotation_axes: tuple[Literal["x", "y", "z"], ...] = p.DERIVATIVE_ROTATION_AXES,
                 angle_rad: float = p.DERIVATIVE_ANGLE_RAD) -> None:
        self.transfer_amplitudes = transfer_amplitudes
        self.rotation_axes = tuple(rotation_axes)
        self.angle_rad = float(angle_rad)
        radius_m = transfer_amplitudes.far_field_distance_nm * 1e-9
        omega = 2 * np.pi * c / (transfer_amplitudes.wavelength_nm * 1e-9)
        self.field_to_rate = epsilon_0 * c * radius_m**2 / (2 * hbar * omega)

    def _scattered_field(self, geometry_nm: NDArray[np.float64],
                         observation_points_nm: NDArray[np.float64]) -> NDArray[np.complex128]:
        calculator = self.transfer_amplitudes
        particle = dict(calculator.particle)
        particle["geometry_nm"] = geometry_nm
        incident_field = calculator._incident_field(geometry_nm)
        interaction = LightMatterInteraction(particle, calculator.wavelength_nm, calculator.parameters["environment"])
        interaction.internal_field(incident_field)
        field = np.empty((len(observation_points_nm), 3), dtype=complex)
        for start in range(0, len(field), calculator.chunk_size):
            stop = min(start + calculator.chunk_size, len(field))
            field[start:stop] = interaction.far_field(observation_points_nm[start:stop], field="E")
        return field.reshape(calculator.n_theta, calculator.n_phi, 3)

    def compute(self) -> dict[str, float | dict[str, float]]:
        """Return Gamma_sc [s^-1], D_mu, and total flux [rad^-2 s^-1]."""
        calculator = self.transfer_amplitudes
        observation_points_nm, _, domega = calculator._observation_surface("z")
        geometry_nm = np.asarray(calculator.particle["geometry_nm"], dtype=float)
        reference_field = self._scattered_field(geometry_nm, observation_points_nm)
        reference_norm = np.sum(np.sum(np.abs(reference_field)**2, axis=-1) * domega)
        gamma_sc = self.field_to_rate * reference_norm

        D: dict[str, float] = {}
        information_flux: dict[str, float] = {}
        for axis in self.rotation_axes:
            angles_deg = np.zeros(3)
            angles_deg["xyz".index(axis)] = np.degrees(self.angle_rad)
            field_plus = self._scattered_field(rotate_particle(geometry_nm, angles_deg, origin_nm=calculator.center_nm), observation_points_nm)
            field_minus = self._scattered_field(rotate_particle(geometry_nm, -angles_deg, origin_nm=calculator.center_nm), observation_points_nm)
            derivative = (field_plus - field_minus) / (2 * self.angle_rad)
            derivative_norm = np.sum(np.sum(np.abs(derivative)**2, axis=-1) * domega)
            D[axis] = float(derivative_norm / reference_norm)
            information_flux[axis] = float(4 * self.field_to_rate * derivative_norm)

        return {"Gamma_sc": float(gamma_sc), "D": D, "information_flux": information_flux}
