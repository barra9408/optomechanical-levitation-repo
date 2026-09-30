from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import special


def hermite_gauss_00(pos, env_dict, wavelength, theta=0, polarization_state=None, xSpot=0.0, ySpot=0.0, zSpot=0.0, kSign=-1.0, NA=0.5, f=100, w0=1, phase=0.0, returnField="E", quadrature_points=12, N_cpu=1):
    """Return complex (N, 3) focused HG00 fields using CPU quadrature.

    Positions and wavelength are in nm; f and w0 are in mm. The "B" branch
    returns H with the reference impedance factor. N_cpu > 1 uses threaded chunks.
    """
    if "eps_env" in env_dict:
        permittivities = np.asarray([env_dict["eps_env"]] * 3, dtype=complex)
    else:
        permittivities = np.asarray([env_dict[key] for key in ("eps1", "eps2", "eps3")], dtype=complex)

    n1, n3 = np.sqrt(permittivities.real[[2, 0] if kSign == -1 else [0, 2]])
    if theta is not None:
        polarization_state = (np.cos(np.deg2rad(theta)), np.sin(np.deg2rad(theta)), 0, 0)
    px, py, relative_phase, absolute_phase = np.asarray(polarization_state, dtype=float)

    focal_length_nm = f * 1e6
    entrance_waist_nm = w0 * 1e6
    vacuum_wavenumber = 2 * np.pi / wavelength
    nodes, weights = leggauss(quadrature_points)
    theta_max = np.arcsin(NA / n1)
    angles = 0.5 * theta_max * (nodes + 1)
    weights = 0.5 * theta_max * weights

    sin_theta = np.sin(angles)
    cos_theta = np.cos(angles)
    sin_theta_3 = n1 * sin_theta / n3
    cos_theta_3 = np.sqrt(1 - sin_theta_3**2 + 0j)
    cos_theta_3 *= np.sign(np.sign(np.imag(cos_theta_3)) + 0.5)

    radial_wavenumber = n1 * vacuum_wavenumber * sin_theta
    longitudinal_wavenumber_1 = n1 * vacuum_wavenumber * cos_theta
    longitudinal_wavenumber_3 = np.sqrt(n3**2 * vacuum_wavenumber**2 - radial_wavenumber**2 + 0j)
    transmission_te = 2 * longitudinal_wavenumber_1 / (longitudinal_wavenumber_1 + longitudinal_wavenumber_3)
    transmission_tm = 2 * n1 * n3 * longitudinal_wavenumber_1 / (n1**2 * longitudinal_wavenumber_3 + n3**2 * longitudinal_wavenumber_1)
    envelope = np.exp(-(focal_length_nm * sin_theta / entrance_waist_nm)**2) * np.sqrt(cos_theta) * sin_theta

    component = "E" if returnField.lower() == "e" else "H"
    if component == "H":
        px *= kSign
        py *= -kSign
    if component == "E":
        angular_0 = transmission_te + transmission_tm * cos_theta_3
        angular_1 = transmission_tm * sin_theta_3
        angular_2 = transmission_te - transmission_tm * cos_theta_3
    else:
        angular_0 = transmission_tm + transmission_te * cos_theta_3
        angular_1 = transmission_te * sin_theta_3
        angular_2 = transmission_tm - transmission_te * cos_theta_3

    common_integrand = weights * envelope
    weighted_angular_0 = common_integrand * angular_0
    weighted_angular_1 = common_integrand * angular_1
    weighted_angular_2 = common_integrand * angular_2
    propagation_coefficient = 1j * n3 * vacuum_wavenumber
    phase_x = np.exp(1j * absolute_phase)
    phase_y = np.exp(1j * (relative_phase + absolute_phase))
    incident_wavenumber = kSign * n1 * vacuum_wavenumber
    prefactor = 0.5j * incident_wavenumber * focal_length_nm * np.exp(-1j * incident_wavenumber * focal_length_nm) * np.exp(1j * np.deg2rad(phase))
    if component == "H":
        vacuum_impedance = np.sqrt((4e-7 * np.pi) / 8.85418782e-12)
        prefactor *= n3 / vacuum_impedance

    positions = np.asarray(pos, dtype=float) - np.asarray((xSpot, ySpot, zSpot), dtype=float)

    def evaluate_chunk(chunk):
        x, y, z = chunk.T
        rho = np.sqrt(x*x + y*y)
        phi = np.arctan2(y, x)
        argument = rho[:, None] * radial_wavenumber[None, :]
        bessel_0 = special.j0(argument)
        bessel_1 = special.j1(argument)

        # The series avoids cancellation in J2 = 2*J1(x)/x - J0(x) near x = 0.
        small_argument = np.abs(argument) <= 0.25
        argument_squared = argument**2
        bessel_2_series = argument_squared * (1/8 + argument_squared * (-1/96 + argument_squared * (1/3072 + argument_squared * (-1/184320 + argument_squared/17694720))))
        safe_argument = np.where(small_argument, 1.0, argument)
        bessel_2 = np.where(small_argument, bessel_2_series, 2 * bessel_1 / safe_argument - bessel_0)

        propagation_phase = np.exp(propagation_coefficient * z[:, None] * cos_theta_3[None, :] * kSign)
        integral_0 = np.sum(weighted_angular_0 * bessel_0 * propagation_phase, axis=1)
        integral_1 = np.sum(weighted_angular_1 * bessel_1 * propagation_phase, axis=1)
        integral_2 = np.sum(weighted_angular_2 * bessel_2 * propagation_phase, axis=1)
        cos_2phi = np.cos(2 * phi)
        sin_2phi = np.sin(2 * phi)

        if component == "E":
            field_x = kSign * (px * (integral_0 + integral_2 * cos_2phi) + py * integral_2 * sin_2phi)
            field_y = kSign * (px * integral_2 * sin_2phi + py * (integral_0 - integral_2 * cos_2phi))
            field_z = -2j * integral_1 * (px * np.cos(phi) + py * np.sin(phi))
        else:
            field_x = kSign * (px * integral_2 * sin_2phi + py * (integral_0 + integral_2 * cos_2phi))
            field_y = kSign * (px * (integral_0 - integral_2 * cos_2phi) + py * integral_2 * sin_2phi)
            field_z = -2j * integral_1 * (px * np.sin(phi) + py * np.cos(phi))

        field_x *= phase_x
        field_y *= phase_y
        return np.asarray(prefactor * np.column_stack((field_x, field_y, field_z)), dtype=complex)

    if N_cpu == 1:
        return evaluate_chunk(positions)
    with ThreadPoolExecutor(max_workers=N_cpu) as executor:
        return np.concatenate(list(executor.map(evaluate_chunk, np.array_split(positions, N_cpu))), axis=0)
