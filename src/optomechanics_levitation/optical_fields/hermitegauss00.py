from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Mapping
from typing import Any, Literal
import numpy as np
from numpy.typing import NDArray
from numpy.polynomial.legendre import leggauss
from scipy import special
from .. import parameters as p

HGComponent = Literal["E", "H"]; ReturnField = Literal["E", "B", "H"]; PolarizationState = tuple[float, float, float, float]
FloatArray = NDArray[np.float64]; ComplexArray = NDArray[np.complex128]; Coefficients = dict[str, Any]

class HermiteGauss00:
    """Focused HG00 beam with reusable CPU quadrature coefficients."""
    def __init__(self, theta: float | None = p.HG_POLARIZATION_ANGLE_DEG, polarization_state: PolarizationState | None = p.HG_POLARIZATION_STATE,
                 xSpot: float = p.HG_X_SPOT_NM, ySpot: float = p.HG_Y_SPOT_NM, zSpot: float = p.HG_Z_SPOT_NM,
                 kSign: Literal[-1, 1] = p.HG_PROPAGATION_SIGN, NA: float = p.NUMERICAL_APERTURE,
                 f: float = p.HG_FOCAL_LENGTH_MM, w0: float = p.HG_ENTRANCE_WAIST_MM, phase: float = p.HG_PHASE_DEG,
                 quadrature_points: int = p.HG_QUADRATURE_POINTS, N_cpu: int = p.HG_N_CPU) -> None:
        if theta is not None:
            polarization_state = (np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta)), 0, 0)
        self.polarization_state = np.asarray(polarization_state, dtype=float)
        self.spot_nm = np.asarray((xSpot, ySpot, zSpot), dtype=float)
        self.kSign = kSign; self.NA = NA; self.f = f; self.w0 = w0; self.phase = phase
        self.quadrature_points = quadrature_points; self.N_cpu = N_cpu
        self._coefficients_cache = {}

    ####################################
    # OPTICAL COEFFICIENTS
    ####################################
    def _medium_indices(self, env_dict: Mapping[str, float | complex]) -> tuple[float, float]:
        """Return the entrance and exit refractive indices in propagation order."""
        permittivities = (np.asarray([env_dict["eps_env"]] * 3, dtype=complex) if "eps_env" in env_dict else
                          np.asarray([env_dict[key] for key in ("eps1", "eps2", "eps3")], dtype=complex))
        n1, n3 = np.sqrt(permittivities.real[[2, 0] if self.kSign == -1 else [0, 2]])
        return n1, n3

    def _quadrature(self, n1: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Prepare the Gauss–Legendre nodes and weights over the objective aperture."""
        nodes, weights = leggauss(self.quadrature_points)
        theta_max = np.arcsin(self.NA / n1)
        angles = 0.5 * theta_max * (nodes + 1); weights = 0.5 * theta_max * weights
        return np.sin(angles), np.cos(angles), weights

    def _field_coefficients(self, env_dict: Mapping[str, float | complex], wavelength: float, component: HGComponent) -> Coefficients:
        """Cache the angular weights and phase factors for E or H."""
        n1, n3 = self._medium_indices(env_dict)
        key = (float(wavelength), float(n1), float(n3), component)
        if key in self._coefficients_cache:
            return self._coefficients_cache[key]

        px, py, relative_phase, absolute_phase = self.polarization_state
        if component == "H":
            px *= self.kSign; py *= -self.kSign
        focal_length_nm = self.f * 1e6; entrance_waist_nm = self.w0 * 1e6
        vacuum_wavenumber = 2 * np.pi / wavelength
        sin_theta, cos_theta, weights = self._quadrature(n1)
        sin_theta_3 = n1 * sin_theta / n3
        cos_theta_3 = np.sqrt(1 - sin_theta_3**2 + 0j)
        cos_theta_3 *= np.sign(np.sign(np.imag(cos_theta_3)) + 0.5)

        radial_wavenumber = n1 * vacuum_wavenumber * sin_theta
        longitudinal_wavenumber_1 = n1 * vacuum_wavenumber * cos_theta
        longitudinal_wavenumber_3 = np.sqrt(n3**2 * vacuum_wavenumber**2 - radial_wavenumber**2 + 0j)
        transmission_te = 2 * longitudinal_wavenumber_1 / (longitudinal_wavenumber_1 + longitudinal_wavenumber_3)
        transmission_tm = 2 * n1 * n3 * longitudinal_wavenumber_1 / (n1**2 * longitudinal_wavenumber_3 + n3**2 * longitudinal_wavenumber_1)
        envelope = np.exp(-(focal_length_nm * sin_theta / entrance_waist_nm)**2) * np.sqrt(cos_theta) * sin_theta

        if component == "E":
            angular_0 = transmission_te + transmission_tm * cos_theta_3
            angular_1 = transmission_tm * sin_theta_3
            angular_2 = transmission_te - transmission_tm * cos_theta_3
        else:
            angular_0 = transmission_tm + transmission_te * cos_theta_3
            angular_1 = transmission_te * sin_theta_3
            angular_2 = transmission_tm - transmission_te * cos_theta_3
        common_integrand = weights * envelope

        propagation_coefficient = 1j * n3 * vacuum_wavenumber
        phase_x = np.exp(1j * absolute_phase); phase_y = np.exp(1j * (relative_phase + absolute_phase))
        incident_wavenumber = self.kSign * n1 * vacuum_wavenumber
        prefactor = 0.5j * incident_wavenumber * focal_length_nm * np.exp(-1j * incident_wavenumber * focal_length_nm) * np.exp(1j * np.deg2rad(self.phase))
        if component == "H":
            vacuum_impedance = np.sqrt((4e-7 * np.pi) / 8.85418782e-12)
            prefactor *= n3 / vacuum_impedance

        coefficients = {"radial_wavenumber": radial_wavenumber, "cos_theta_3": cos_theta_3,
                        "weighted_angular_0": common_integrand * angular_0, "weighted_angular_1": common_integrand * angular_1,
                        "weighted_angular_2": common_integrand * angular_2, "propagation_coefficient": propagation_coefficient,
                        "phase_x": phase_x, "phase_y": phase_y, "prefactor": prefactor, "px": px, "py": py}
        self._coefficients_cache[key] = coefficients
        return coefficients

    ####################################
    # FOCUSED FIELD
    ####################################
    def _evaluate_chunk(self, positions: FloatArray, component: HGComponent, coefficients: Coefficients) -> ComplexArray:
        """Evaluate the three Bessel integrals and Cartesian field components."""
        x, y, z = positions.T; rho = np.sqrt(x*x + y*y); phi = np.arctan2(y, x)
        argument = rho[:, None] * coefficients["radial_wavenumber"][None, :]
        bessel_0 = special.j0(argument); bessel_1 = special.j1(argument)

        # The series avoids cancellation in J2 = 2*J1(x)/x - J0(x) near x = 0.
        small_argument = np.abs(argument) <= 0.25; argument_squared = argument**2
        bessel_2_series = argument_squared * (1/8 + argument_squared * (-1/96 + argument_squared * (1/3072 + argument_squared * (-1/184320 + argument_squared/17694720))))
        safe_argument = np.where(small_argument, 1.0, argument)
        bessel_2 = np.where(small_argument, bessel_2_series, 2 * bessel_1 / safe_argument - bessel_0)

        propagation_phase = np.exp(coefficients["propagation_coefficient"] * z[:, None] * coefficients["cos_theta_3"][None, :] * self.kSign)
        integral_0 = np.sum(coefficients["weighted_angular_0"] * bessel_0 * propagation_phase, axis=1)
        integral_1 = np.sum(coefficients["weighted_angular_1"] * bessel_1 * propagation_phase, axis=1)
        integral_2 = np.sum(coefficients["weighted_angular_2"] * bessel_2 * propagation_phase, axis=1)
        cos_2phi = np.cos(2 * phi); sin_2phi = np.sin(2 * phi)
        px = coefficients["px"]; py = coefficients["py"]

        if component == "E":
            field_x = self.kSign * (px * (integral_0 + integral_2 * cos_2phi) + py * integral_2 * sin_2phi)
            field_y = self.kSign * (px * integral_2 * sin_2phi + py * (integral_0 - integral_2 * cos_2phi))
            field_z = -2j * integral_1 * (px * np.cos(phi) + py * np.sin(phi))
        else:
            field_x = self.kSign * (px * integral_2 * sin_2phi + py * (integral_0 + integral_2 * cos_2phi))
            field_y = self.kSign * (px * (integral_0 - integral_2 * cos_2phi) + py * integral_2 * sin_2phi)
            field_z = -2j * integral_1 * (px * np.sin(phi) + py * np.cos(phi))

        field_x *= coefficients["phase_x"]; field_y *= coefficients["phase_y"]
        return np.asarray(coefficients["prefactor"] * np.column_stack((field_x, field_y, field_z)), dtype=complex)

    def field(self, pos: FloatArray, env_dict: Mapping[str, float | complex], wavelength: float,
              returnField: ReturnField = p.DEFAULT_RETURN_FIELD) -> ComplexArray:
        """Return the focused E or H field at positions in nm, using CPU threads."""
        component = "E" if returnField.lower() == "e" else "H"
        coefficients = self._field_coefficients(env_dict, wavelength, component)
        positions = np.asarray(pos, dtype=float) - self.spot_nm
        if self.N_cpu == 1:
            return self._evaluate_chunk(positions, component, coefficients)
        with ThreadPoolExecutor(max_workers=self.N_cpu) as executor:
            return np.concatenate(list(executor.map(lambda chunk: self._evaluate_chunk(chunk, component, coefficients), np.array_split(positions, self.N_cpu))), axis=0)
