import numpy as np
from scipy.constants import epsilon_0, mu_0, c, hbar
from .incident_field import evaluate_incident_field
from .light_matter_interaction import evaluate_internal_field, evaluate_far_field

def maxwell_stress_tensor(electric_field, magnetic_field, permittivity=epsilon_0, permeability=mu_0):
    electric_term = permittivity * np.einsum("...i,...j->...ij", electric_field, np.conj(electric_field))
    magnetic_term = permeability * np.einsum("...i,...j->...ij", magnetic_field, np.conj(magnetic_field))
    energy_term = permittivity * np.sum(np.abs(electric_field)**2, axis=-1) + permeability * np.sum(np.abs(magnetic_field)**2, axis=-1)
    return 0.5 * np.real(electric_term + magnetic_term - 0.5 * energy_term[..., None, None] * np.eye(3))

def spherical_integration_surface(geometry_nm, step_nm, n_theta, n_phi):
    center_nm = np.mean(geometry_nm, axis=0)
    radius_nm = np.max(np.linalg.norm(geometry_nm - center_nm, axis=1)) + 4 * step_nm

    theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

    normals = np.stack((np.sin(Theta) * np.cos(Phi), np.sin(Theta) * np.sin(Phi), np.cos(Theta)), axis=-1).reshape(-1, 3)
    positions_nm = center_nm + radius_nm * normals
    area_m2 = ((radius_nm * 1e-9)**2 * np.sin(Theta) * (np.pi / n_theta) * (2 * np.pi / n_phi)).ravel()

    return positions_nm, normals, area_m2, center_nm

def evaluate_force(stress_interaction, normals, area_m2):
    traction = np.einsum("nij,nj->ni", stress_interaction, normals)
    return np.sum(traction * area_m2[:, None], axis=0)

def evaluate_torque(stress_interaction, positions_nm, normals, area_m2, center_nm):
    traction = np.einsum("nij,nj->ni", stress_interaction, normals)
    position_vectors_m = (positions_nm - center_nm) * 1e-9
    return np.sum(np.cross(position_vectors_m, traction) * area_m2[:, None], axis=0)

############################## RECOIL HEATING ########################################

def _mean_scattering_rate(dI_sc, omega, r):
    return epsilon_0 * c * r**2 * np.sum(dI_sc) / (2 * hbar * omega)

def _total_differential_fields(scattering_field, incident_field, domega_sc, domega_in):
    dI_sc = np.sum(np.abs(scattering_field)**2, axis=-1) * domega_sc
    dI_in = np.sum(np.abs(incident_field)**2, axis=-1) * domega_in
    I_in = np.sum(dI_in)
    I_tilde = np.sum(dI_sc) / I_in
    return I_tilde, dI_sc, dI_in

def _direction_axis_projection(directions, axis):
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    return np.asarray(directions)[..., axis_index]

def _angular_surface(interaction_result, axis, radius_wavelengths, n_theta, n_phi):
    prepared = interaction_result["prepared_interaction"]
    wavelength_nm = prepared["wavelength_nm"]
    center_nm = np.mean(prepared["structure"].geometry, axis=0)
    radius_nm = radius_wavelengths * wavelength_nm
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    basis = np.eye(3)[[(axis_index + 1) % 3, (axis_index + 2) % 3, axis_index]]
    theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")
    directions = np.stack((np.sin(Theta) * np.cos(Phi), np.sin(Theta) * np.sin(Phi), np.cos(Theta)), axis=-1) @ basis
    domega = np.sin(Theta) * (np.pi / n_theta) * (2 * np.pi / n_phi)
    positions_nm = center_nm + radius_nm * directions.reshape(-1, 3)
    return positions_nm, directions, domega, radius_nm * 1e-9

def _far_field_on_surface(interaction_result, positions_nm, grid_shape, chunk_size):
    field = np.empty((len(positions_nm), 3), dtype=complex)
    for start in range(0, len(positions_nm), chunk_size):
        stop = min(start + chunk_size, len(positions_nm))
        field[start:stop] = evaluate_far_field(interaction_result, positions_nm[start:stop], field="E")
    return field.reshape(*grid_shape, 3)

def _plane_wave_angular_spectrum(field_generator, incident_parameters, center_nm, axis):
    """Represent a plane wave and its angular-momentum-weighted field."""
    direction = np.asarray(incident_parameters["direction"], dtype=float)
    direction = direction / np.linalg.norm(direction)
    axis_vector = np.eye(3)[{"x": 0, "y": 1, "z": 2}[axis.lower()]]
    amplitude = evaluate_incident_field(field_generator, np.asarray(center_nm)[None, :], field="E", **incident_parameters)[0]
    spin = np.dot(direction, axis_vector) * 1j * np.cross(direction, amplitude)

    angle = 1e-3
    def rotate(vector, angle):
        return (np.cos(angle) * vector + np.sin(angle) * np.cross(axis_vector, vector)
                + (1 - np.cos(angle)) * axis_vector * np.dot(axis_vector, vector))

    directions = np.array([direction, rotate(direction, -angle), rotate(direction, angle)])
    wavevectors = 2 * np.pi / incident_parameters["wavelength_nm"] * directions
    coefficients = np.zeros((3, 3), dtype=complex)
    force_coefficients = coefficients.copy()
    spin_coefficients = coefficients.copy()
    orbital_coefficients = coefficients.copy()

    # The rotated modes act only in J_in; they carry no incident intensity.
    coefficients[0] = amplitude
    force_coefficients[0] = np.dot(direction, axis_vector) * amplitude
    spin_coefficients[0] = spin
    orbital_coefficients[0] = -spin
    orbital_coefficients[1] = rotate(amplitude, -angle) / (2j * angle)
    orbital_coefficients[2] = -rotate(amplitude, angle) / (2j * angle)

    return (directions, np.ones(3), wavevectors, np.asarray(center_nm), coefficients,
            force_coefficients, spin_coefficients, orbital_coefficients)

def _incident_angular_spectrum(field_generator, incident_parameters, center_nm, axis, input_grid, input_span_wavelengths):
    """Angular amplitudes from an explicit plane wave or a two-plane Fourier grid.

    Each Fourier coefficient includes its angular cell weight. Both propagation
    signs are retained; only propagating components enter the angular integrals.
    """
    if "direction" in incident_parameters and "helicity" in incident_parameters:
        return _plane_wave_angular_spectrum(field_generator, incident_parameters, center_nm, axis)

    wavelength_nm = incident_parameters["wavelength_nm"]
    size_nm = input_span_wavelengths * wavelength_nm
    step_nm = size_nm / input_grid
    start_xy = np.asarray(center_nm[:2]) - size_nm / 2
    offsets = np.arange(input_grid) * step_nm
    X, Y = np.meshgrid(start_xy[0] + offsets, start_xy[1] + offsets, indexing="ij")
    z0 = center_nm[2]
    dz = wavelength_nm / 8
    sample_positions = np.stack((X, Y, np.full_like(X, z0)), axis=-1).reshape(-1, 3)
    upper_positions = sample_positions.copy()
    upper_positions[:, 2] += dz
    fields = []
    for positions in (sample_positions, upper_positions):
        field = evaluate_incident_field(field_generator=field_generator, positions_nm=positions, field="E", **incident_parameters)
        fields.append(np.fft.fftshift(np.fft.fft2(field.reshape(input_grid, input_grid, 3), axes=(0, 1)), axes=(0, 1)) / input_grid**2)

    k = 2 * np.pi / wavelength_nm
    kxy = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(input_grid, d=step_nm))
    Kx, Ky = np.meshgrid(kxy, kxy, indexing="ij")
    Kz = np.sqrt(np.maximum(k**2 - Kx**2 - Ky**2, 0))
    propagating = Kz > 0.05 * k
    phase = np.exp(-1j * Kz * dz)
    denominator = np.where(propagating, 2j * np.sin(Kz * dz), 1)
    positive = (fields[1] - fields[0] * phase[..., None]) / denominator[..., None]
    negative = fields[0] - positive
    positive[~propagating] = 0
    negative[~propagating] = 0
    solid_angle = (2 * np.pi / size_nm)**2 / (k * np.where(propagating, Kz, 1))

    coefficients = []
    force_coefficients = []
    spin_coefficients = []
    orbital_coefficients = []
    wavevectors = []
    directions = []
    weights = []
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    axis_vector = np.eye(3)[axis_index]
    for sign, amplitude in ((1, positive), (-1, negative)):
        direction = np.stack((Kx / k, Ky / k, sign * Kz / k), axis=-1)
        amplitude = amplitude - direction * np.sum(direction * amplitude, axis=-1, keepdims=True)
        density = amplitude / solid_angle[..., None]
        derivative_x = np.gradient(density, kxy[1] - kxy[0], axis=0)
        derivative_y = np.gradient(density, kxy[1] - kxy[0], axis=1)
        if axis_index == 0:
            orbital = 1j * sign * Kz[..., None] * derivative_y
        elif axis_index == 1:
            orbital = -1j * sign * Kz[..., None] * derivative_x
        else:
            orbital = -1j * (Kx[..., None] * derivative_y - Ky[..., None] * derivative_x)
        orbital *= solid_angle[..., None]
        projection = _direction_axis_projection(direction, axis)
        force = projection[..., None] * amplitude
        spin = projection[..., None] * 1j * np.cross(direction, amplitude)
        orbital += 1j * np.cross(axis_vector, amplitude) - spin
        coefficients.append(amplitude[propagating])
        force_coefficients.append(force[propagating])
        spin_coefficients.append(spin[propagating])
        orbital_coefficients.append(orbital[propagating])
        wavevectors.append((k * direction)[propagating])
        directions.append(direction[propagating])
        weights.append(solid_angle[propagating])

    coefficients = np.concatenate(coefficients)
    force_coefficients = np.concatenate(force_coefficients)
    spin_coefficients = np.concatenate(spin_coefficients)
    orbital_coefficients = np.concatenate(orbital_coefficients)
    wavevectors = np.concatenate(wavevectors)
    directions = np.concatenate(directions)
    weights = np.concatenate(weights)
    active = np.maximum.reduce((np.linalg.norm(coefficients, axis=-1), np.linalg.norm(force_coefficients, axis=-1), np.linalg.norm(spin_coefficients, axis=-1), np.linalg.norm(orbital_coefficients, axis=-1)))
    active = active > np.max(active) * 1e-10
    origin_nm = np.array([*start_xy, z0])
    return directions[active], weights[active], wavevectors[active], origin_nm, coefficients[active], force_coefficients[active], spin_coefficients[active], orbital_coefficients[active]

def _field_from_spectrum(positions_nm, wavevectors, origin_nm, coefficients, chunk_size):
    field = np.zeros((len(positions_nm), 3), dtype=complex)
    relative_positions = positions_nm - origin_nm
    for start in range(0, len(wavevectors), chunk_size):
        stop = min(start + chunk_size, len(wavevectors))
        phase = np.exp(1j * relative_positions @ wavevectors[start:stop].T)
        field += phase @ coefficients[start:stop]
    return field

def _weighted_scattering(interaction_result, positions_nm, grid_shape, wavevectors, origin_nm, coefficients, chunk_size):
    prepared = interaction_result["prepared_interaction"]
    incident_at_particle = _field_from_spectrum(prepared["structure"].geometry, wavevectors, origin_nm, coefficients, chunk_size)
    weighted_interaction = evaluate_internal_field(particle=None, incident_electric_field=incident_at_particle, wavelength_nm=prepared["wavelength_nm"], environment=None, prepared_interaction=prepared)

    return _far_field_on_surface(weighted_interaction, positions_nm, grid_shape, chunk_size)

def _spectral_fields(interaction_result, field_generator, incident_parameters, axis, radius_wavelengths, n_theta, n_phi, input_grid, input_span_wavelengths, chunk_size):
    """Sample one beam and reuse its factorized interaction for weighted fields."""
    positions_nm, directions_sc, domega_sc, r = _angular_surface(interaction_result, axis, radius_wavelengths, n_theta, n_phi)
    scattering_field = _far_field_on_surface(interaction_result, positions_nm, (n_theta, n_phi), chunk_size)
    center_nm = np.mean(interaction_result["prepared_interaction"]["structure"].geometry, axis=0)
    directions_in, domega_in, wavevectors, origin_nm, incident_coefficients, force_coefficients, spin_coefficients, orbital_coefficients = _incident_angular_spectrum(field_generator, incident_parameters, center_nm, axis, input_grid, input_span_wavelengths)
    incident_field = incident_coefficients / domega_in[:, None]
    I_tilde, dI_sc, dI_in = _total_differential_fields(scattering_field, incident_field, domega_sc, domega_in)
    return positions_nm, directions_sc, domega_sc, r, scattering_field, directions_in, I_tilde, dI_sc, dI_in, wavevectors, origin_nm, force_coefficients, spin_coefficients, orbital_coefficients

def _orbital_operator(field, directions, axis):
    n_phi = field.shape[1]
    m = np.fft.fftfreq(n_phi, d=1 / n_phi)
    derivative = np.fft.ifft(m[None, :, None] * np.fft.fft(field, axis=1), axis=1)
    axis_vector = np.eye(3)[{"x": 0, "y": 1, "z": 2}[axis.lower()]]
    projection = _direction_axis_projection(directions, axis)
    spin = projection[..., None] * 1j * np.cross(directions, field)
    return derivative + 1j * np.cross(axis_vector, field) - spin

def force_spectral_density(interaction_result, field_generator, incident_parameters, force_axis="z", radius_wavelengths=100.0, n_theta=64, n_phi=128, input_grid=48, input_span_wavelengths=8.0, chunk_size=256):
    positions_nm, directions_sc, domega_sc, r, scattering_field, directions_in, I_tilde, dI_sc, dI_in, wavevectors, origin_nm, force_coefficients, _, _ = _spectral_fields(interaction_result, field_generator, incident_parameters, force_axis, radius_wavelengths, n_theta, n_phi, input_grid, input_span_wavelengths, chunk_size)
    wavelength = interaction_result["prepared_interaction"]["wavelength_nm"] * 1e-9
    omega = 2 * np.pi * c / wavelength
    gamma_sc = _mean_scattering_rate(dI_sc, omega, r)

    u_in = _direction_axis_projection(directions_in, force_axis)
    u_sc = _direction_axis_projection(directions_sc, force_axis)

    factor = (hbar * omega / c)**2 * gamma_sc / np.sum(dI_sc)

    incident_term = factor * I_tilde * np.sum(u_in**2 * dI_in)
    scattering_term = factor * np.sum(u_sc**2 * dI_sc)

    B_f_in = _weighted_scattering(interaction_result, positions_nm, (n_theta, n_phi), wavevectors, origin_nm, force_coefficients, chunk_size)
    mixed_term = -2 * factor * np.real(np.sum(np.sum(np.conj(B_f_in) * (u_sc[..., None] * scattering_field), axis=-1) * domega_sc))
    total_spectral_density = incident_term + scattering_term + mixed_term

    return total_spectral_density, incident_term, scattering_term, mixed_term

def torque_spectral_density(interaction_result, field_generator, incident_parameters, torque_axis="z", radius_wavelengths=100.0, n_theta=64, n_phi=128, input_grid=48, input_span_wavelengths=8.0, chunk_size=256):
    positions_nm, directions_sc, domega_sc, r, scattering_field, directions_in, I_tilde, dI_sc, dI_in, wavevectors, origin_nm,_, spin_coefficients, orbital_coefficients = _spectral_fields(interaction_result, field_generator, incident_parameters, torque_axis, radius_wavelengths, n_theta, n_phi, input_grid, input_span_wavelengths, chunk_size)
    wavelength = interaction_result["prepared_interaction"]["wavelength_nm"] * 1e-9
    omega = 2 * np.pi * c / wavelength
    gamma_sc = _mean_scattering_rate(dI_sc, omega, r)
    factor = hbar**2 * gamma_sc / np.sum(dI_sc)

    u_in = _direction_axis_projection(directions_in, torque_axis)
    u_sc = _direction_axis_projection(directions_sc, torque_axis)

    B_s_in = _weighted_scattering(interaction_result, positions_nm, (n_theta, n_phi), wavevectors, origin_nm, spin_coefficients, chunk_size)
    B_s_sc = u_sc[..., None] * 1j * np.cross(directions_sc, scattering_field)
    spin_term = factor * (I_tilde * np.sum(u_in**2 * dI_in) + np.sum(u_sc**2 * dI_sc) - 2 * np.real(np.sum(np.sum(np.conj(B_s_in) * B_s_sc, axis=-1) * domega_sc)))

    B_l_in = _weighted_scattering(interaction_result, positions_nm, (n_theta, n_phi), wavevectors, origin_nm, orbital_coefficients, chunk_size)
    B_l_sc = _orbital_operator(scattering_field, directions_sc, torque_axis)
    orbital_difference = B_l_in - B_l_sc
    spin_difference = B_s_in - B_s_sc

    orbital_term = factor * np.sum(np.sum(np.abs(orbital_difference)**2, axis=-1) * domega_sc)
    mixed_term = factor * np.real(np.sum(np.sum(np.conj(orbital_difference) * spin_difference, axis=-1) * domega_sc))
    total_spectral_density = spin_term + orbital_term + 2 * mixed_term

    return total_spectral_density, spin_term, orbital_term, mixed_term
