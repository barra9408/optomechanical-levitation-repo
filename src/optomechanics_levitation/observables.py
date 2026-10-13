import numpy as np
from scipy.constants import epsilon_0, mu_0, c, hbar
from pyGDM2 import fields
from .particle import rotate_particle
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

############################## RECOIL HEATING FROM ORIENTATION DERIVATIVE ########################################
def derivative_torque_spectral_density(particle, field_generator, incident_parameters, far_field_distance=1e6, rotation_axes=("y", "z"), angle_rad=0.005):
    """Return Gamma_sc, D and S_tau_tau from finite orientation derivatives.

    Distances and wavelength are in nm; angle_rad is in radians. The returned
    D and S_tau_tau are dictionaries indexed by the requested rotation axes.
    """
    wavelength = incident_parameters["wavelength_nm"]
    environment = incident_parameters["environment"]

    theta = (np.arange(64) + 0.5) * np.pi / 64
    phi = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    directions = np.column_stack(((np.sin(theta_grid) * np.cos(phi_grid)).ravel(), (np.sin(theta_grid) * np.sin(phi_grid)).ravel(), np.cos(theta_grid).ravel()))
    observation_points = far_field_distance * directions

    geometry = particle["geometry_nm"]
    rotation_origin = geometry.mean(axis=0)

    def scattering_rate(scattered_field):
        field_squared = np.sum(np.abs(scattered_field)**2, axis=1)
        field_squared = field_squared.reshape(len(theta), len(phi))

        dtheta = theta[1] - theta[0]
        dphi = phi[1] - phi[0]
        integral = np.sum(field_squared * np.sin(theta)[:, None]) * dtheta * dphi

        radius_m = far_field_distance * 1e-9
        omega = 2 * np.pi * c / (wavelength * 1e-9)
        P_sc = epsilon_0 * c * radius_m**2 * integral / 2

        return P_sc / (hbar * omega)

    def scattered_field_at(axis=None, rotation_rad=0.0):
        oriented_particle = particle.copy()

        if axis is not None:
            angles_deg = np.zeros(3)
            angles_deg["xyz".index(axis)] = np.degrees(rotation_rad)
            oriented_particle["geometry_nm"] = rotate_particle(geometry, angles_deg, origin_nm=rotation_origin)

        incident_field = evaluate_incident_field(field_generator=field_generator, positions_nm=oriented_particle["geometry_nm"],
                                                 field="E", **incident_parameters)
        interaction = evaluate_internal_field(particle=oriented_particle, incident_electric_field=incident_field,
                                              wavelength_nm=wavelength, environment=environment)

        return evaluate_far_field(interaction_result=interaction, positions_nm=observation_points, field="E")

    scattered_field = scattered_field_at()
    Gamma_sc = scattering_rate(scattered_field)

    D = {}
    S_tau_tau = {}
    for axis in rotation_axes:
        field_plus = scattered_field_at(axis, +angle_rad)
        field_minus = scattered_field_at(axis, -angle_rad)
        field_derivative = (field_plus - field_minus) / (2 * angle_rad)

        gamma_derivative = scattering_rate(field_derivative)
        D[axis] = gamma_derivative / Gamma_sc
        S_tau_tau[axis] = hbar**2 * gamma_derivative / (2 * np.pi)

    return Gamma_sc, D, S_tau_tau



############################## RECOIL HEATING FROM ANGULAR SPECTRUM ########################################

def _direction_axis_projection(directions, axis):
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    return np.asarray(directions)[..., axis_index]

def _axis_vector(axis):
    return np.eye(3)[{"x": 0, "y": 1, "z": 2}[axis.lower()]]

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

def _rotate_about_axis(vector, axis_vector, angle):
    return (np.cos(angle) * vector + np.sin(angle) * np.cross(axis_vector, vector)
            + (1 - np.cos(angle)) * axis_vector * np.sum(vector * axis_vector, axis=-1, keepdims=True))

def _plane_wave_spectrum(field_generator, incident_parameters, center_nm, axis, direction):
    """A plane wave is an angular delta; rotated deltas represent its derivative."""
    direction = np.asarray(direction, dtype=float)
    direction /= np.linalg.norm(direction)
    axis_vector = _axis_vector(axis)
    amplitude = evaluate_incident_field(field_generator, np.asarray(center_nm)[None, :], field="E", **incident_parameters)[0]
    angle = 1e-3
    directions = np.array([direction, _rotate_about_axis(direction, axis_vector, -angle),
                           _rotate_about_axis(direction, axis_vector, angle)])
    amplitudes = np.zeros((3, 3), dtype=complex)
    amplitudes[0] = amplitude

    # d/dphi = e_axis x A - d/d(angle)[R(angle) A(R(-angle) n)].
    azimuthal_derivative = np.zeros_like(amplitudes)
    azimuthal_derivative[0] = np.cross(axis_vector, amplitude)
    azimuthal_derivative[1] = _rotate_about_axis(amplitude, axis_vector, -angle) / (2 * angle)
    azimuthal_derivative[2] = -_rotate_about_axis(amplitude, axis_vector, angle) / (2 * angle)
    return {"directions": directions, "wavevectors_nm": 2 * np.pi / incident_parameters["wavelength_nm"] * directions,
            "origin_nm": np.asarray(center_nm), "amplitudes": amplitudes,
            "azimuthal_derivative": azimuthal_derivative}

def _angular_spectrum_from_field(field_generator, incident_parameters, center_nm, axis, input_grid, input_span_wavelengths):
    """Forward 2-D Fourier transform of E into complex, transverse angular amplitudes.

    amplitudes[j] contains A(n_j) dOmega_j. Two planes separate the +z and
    -z waves; dOmega = dkx dky / (k |kz|). Plane-wave deltas are kept exact.
    """
    if "direction" in incident_parameters:
        return _plane_wave_spectrum(field_generator, incident_parameters, center_nm, axis, incident_parameters["direction"])

    environment = incident_parameters["environment"]
    if field_generator is fields.plane_wave and np.allclose(
            [environment[key] for key in ("eps1", "eps2", "eps3")], environment["eps2"]):
        angle = np.deg2rad(incident_parameters.get("inc_angle", 180.0))
        direction = np.array([0.0, 0.0, np.cos(angle)])
        transverse_axis = 1 if incident_parameters.get("inc_plane", "xz").lower() in ("yz", "zy") else 0
        direction[transverse_axis] = -np.sin(angle)
        return _plane_wave_spectrum(field_generator, incident_parameters, center_nm, axis, direction)

    wavelength_nm = incident_parameters["wavelength_nm"]
    size_nm = input_span_wavelengths * wavelength_nm
    step_nm = size_nm / input_grid
    start_xy = np.asarray(center_nm[:2]) - (input_grid // 2) * step_nm
    offsets = np.arange(input_grid) * step_nm
    X, Y = np.meshgrid(start_xy[0] + offsets, start_xy[1] + offsets, indexing="ij")
    z0 = center_nm[2]
    dz = wavelength_nm / 8
    sample_positions = np.stack((X, Y, np.full_like(X, z0)), axis=-1).reshape(-1, 3)
    upper_positions = sample_positions.copy()
    upper_positions[:, 2] += dz
    plane_spectra = []
    for positions in (sample_positions, upper_positions):
        field = evaluate_incident_field(field_generator=field_generator, positions_nm=positions, field="E", **incident_parameters)
        centered_field = np.fft.ifftshift(field.reshape(input_grid, input_grid, 3), axes=(0, 1))
        plane_spectra.append(np.fft.fftshift(np.fft.fft2(centered_field, axes=(0, 1)), axes=(0, 1)) / input_grid**2)

    k = 2 * np.pi / wavelength_nm
    kxy = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(input_grid, d=step_nm))
    Kx, Ky = np.meshgrid(kxy, kxy, indexing="ij")
    Kz = np.sqrt(np.maximum(k**2 - Kx**2 - Ky**2, 0))
    propagating = Kz > 0
    phase = np.exp(-1j * Kz * dz)
    denominator = np.where(propagating, 2j * np.sin(Kz * dz), 1)
    positive = (plane_spectra[1] - plane_spectra[0] * phase[..., None]) / denominator[..., None]
    negative = plane_spectra[0] - positive
    positive[~propagating] = 0
    negative[~propagating] = 0
    solid_angle = (2 * np.pi / size_nm)**2 / (k * np.where(propagating, Kz, 1))

    directions = []
    amplitudes = []
    azimuthal_derivatives = []
    axis_index = {"x": 0, "y": 1, "z": 2}[axis.lower()]
    for sign, amplitude in ((1, positive), (-1, negative)):
        direction = np.stack((Kx / k, Ky / k, sign * Kz / k), axis=-1)
        amplitude = amplitude - direction * np.sum(direction * amplitude, axis=-1, keepdims=True)
        density = amplitude / solid_angle[..., None]
        derivative_x = np.gradient(density, kxy[1] - kxy[0], axis=0)
        derivative_y = np.gradient(density, kxy[1] - kxy[0], axis=1)
        if axis_index == 0:
            azimuthal_derivative = -sign * Kz[..., None] * derivative_y
        elif axis_index == 1:
            azimuthal_derivative = sign * Kz[..., None] * derivative_x
        else:
            azimuthal_derivative = -Ky[..., None] * derivative_x + Kx[..., None] * derivative_y
        directions.append(direction[propagating])
        amplitudes.append(amplitude[propagating])
        azimuthal_derivatives.append((azimuthal_derivative * solid_angle[..., None])[propagating])

    directions = np.concatenate(directions)
    amplitudes = np.concatenate(amplitudes)
    azimuthal_derivatives = np.concatenate(azimuthal_derivatives)
    active = np.maximum(np.linalg.norm(amplitudes, axis=-1), np.linalg.norm(azimuthal_derivatives, axis=-1))
    active = active > np.max(active) * 1e-10
    directions = directions[active]
    return {"directions": directions, "wavevectors_nm": k * directions,
            "origin_nm": np.asarray(center_nm), "amplitudes": amplitudes[active],
            "azimuthal_derivative": azimuthal_derivatives[active]}

def _field_from_angular_spectrum(positions_nm, spectrum, amplitudes, chunk_size):
    """Inverse angular transform at the particle dipoles."""
    field = np.zeros((len(positions_nm), 3), dtype=complex)
    relative_positions = positions_nm - spectrum["origin_nm"]
    for start in range(0, len(amplitudes), chunk_size):
        stop = min(start + chunk_size, len(amplitudes))
        phase = np.exp(1j * relative_positions @ spectrum["wavevectors_nm"][start:stop].T)
        field += phase @ amplitudes[start:stop]
    return field

def _apply_linear_scattering_kernel(interaction_result, spectrum, amplitudes, positions_nm, grid_shape, chunk_size):
    """Apply K = far-field map x polarizability x inverse interaction x incident map."""
    prepared = interaction_result["prepared_interaction"]
    incident_at_particle = _field_from_angular_spectrum(prepared["structure"].geometry, spectrum, amplitudes, chunk_size)
    response = evaluate_internal_field(particle=None, incident_electric_field=incident_at_particle,
                                       wavelength_nm=prepared["wavelength_nm"], environment=None,
                                       prepared_interaction=prepared)
    return _far_field_on_surface(response, positions_nm, grid_shape, chunk_size)

def _linear_momentum_operator(field, directions, axis):
    """p_mu / (hbar k) = n_mu."""
    return _direction_axis_projection(directions, axis)[..., None] * field

def _spin_operator(field, directions, axis):
    """Transverse spin S_mu / hbar = i n_mu (n cross field)."""
    return 1j * _direction_axis_projection(directions, axis)[..., None] * np.cross(directions, field)

def _angular_momentum_operator(field, directions, azimuthal_derivative, axis):
    """Total J_mu / hbar in the transverse angular-spectrum subspace."""
    result = -1j * azimuthal_derivative + 1j * np.cross(_axis_vector(axis), field)
    return result - directions * np.sum(directions * result, axis=-1, keepdims=True)

def _orbital_operator(field, directions, azimuthal_derivative, axis):
    """Transverse orbital L'_mu = J_mu - S_mu."""
    return _angular_momentum_operator(field, directions, azimuthal_derivative, axis) - _spin_operator(field, directions, axis)

def _outgoing_azimuthal_derivative(field):
    n_phi = field.shape[1]
    modes = np.fft.fftfreq(n_phi, d=1 / n_phi)
    return np.fft.ifft(1j * modes[None, :, None] * np.fft.fft(field, axis=1), axis=1)

def _force_transfer_amplitude(spectrum, outgoing, directions_out, scatter, axis):
    """q_F = n_out K A - K(n_in A)."""
    incoming = _linear_momentum_operator(spectrum["amplitudes"], spectrum["directions"], axis)
    return _linear_momentum_operator(outgoing, directions_out, axis) - scatter(incoming)

def _spin_transfer_amplitude(spectrum, outgoing, directions_out, scatter, axis):
    """q_S = S_out K A - K(S_in A)."""
    incoming = _spin_operator(spectrum["amplitudes"], spectrum["directions"], axis)
    return _spin_operator(outgoing, directions_out, axis) - scatter(incoming)

def _orbital_transfer_amplitude(spectrum, outgoing, directions_out, scatter, axis):
    """q_L = L'_out K A - K(L'_in A)."""
    incoming = _orbital_operator(spectrum["amplitudes"], spectrum["directions"],
                                 spectrum["azimuthal_derivative"], axis)
    outgoing_derivative = _outgoing_azimuthal_derivative(outgoing)
    return _orbital_operator(outgoing, directions_out, outgoing_derivative, axis) - scatter(incoming)

def _angular_inner_product(left, right, domega):
    return np.sum(np.sum(np.conj(left) * right, axis=-1) * domega)

def _scattering_data(interaction_result, field_generator, incident_parameters, axis,
                     radius_wavelengths, n_theta, n_phi, input_grid, input_span_wavelengths, chunk_size):
    positions_nm, directions_out, domega_out, radius_m = _angular_surface(
        interaction_result, axis, radius_wavelengths, n_theta, n_phi)
    prepared = interaction_result["prepared_interaction"]
    center_nm = np.mean(prepared["structure"].geometry, axis=0)
    spectrum = _angular_spectrum_from_field(field_generator, incident_parameters, center_nm,
                                            axis, input_grid, input_span_wavelengths)
    omega = 2 * np.pi * c / (prepared["wavelength_nm"] * 1e-9)
    amplitude_scale = np.sqrt(epsilon_0 * c * radius_m**2 / (2 * hbar * omega))
    def scatter(amplitudes):
        return amplitude_scale * _apply_linear_scattering_kernel(
            interaction_result, spectrum, amplitudes, positions_nm, (n_theta, n_phi), chunk_size)
    outgoing = scatter(spectrum["amplitudes"])
    return spectrum, directions_out, domega_out, outgoing, scatter, omega

def force_spectral_density(interaction_result, field_generator, incident_parameters, force_axis="z", radius_wavelengths=100.0, n_theta=64, n_phi=128, input_grid=48, input_span_wavelengths=8.0, chunk_size=256):
    """Two-sided force noise from |n_out K A - K(n_in A)| squared."""
    spectrum, directions_out, domega_out, outgoing, scatter, omega = _scattering_data(
        interaction_result, field_generator, incident_parameters, force_axis,
        radius_wavelengths, n_theta, n_phi, input_grid, input_span_wavelengths, chunk_size)
    force_transfer = _force_transfer_amplitude(spectrum, outgoing, directions_out, scatter, force_axis)
    return float((hbar * omega / c)**2 * _angular_inner_product(force_transfer, force_transfer, domega_out).real)

def torque_spectral_density(interaction_result, field_generator, incident_parameters, torque_axis="z", radius_wavelengths=100.0, n_theta=64, n_phi=128, input_grid=48, input_span_wavelengths=8.0, chunk_size=256):
    """Return total, spin, orbital and one spin-orbital cross term, before /2pi."""
    spectrum, directions_out, domega_out, outgoing, scatter, _ = _scattering_data(
        interaction_result, field_generator, incident_parameters, torque_axis,
        radius_wavelengths, n_theta, n_phi, input_grid, input_span_wavelengths, chunk_size)
    spin_transfer = _spin_transfer_amplitude(spectrum, outgoing, directions_out, scatter, torque_axis)
    orbital_transfer = _orbital_transfer_amplitude(spectrum, outgoing, directions_out, scatter, torque_axis)

    spin_term = hbar**2 * _angular_inner_product(spin_transfer, spin_transfer, domega_out).real
    orbital_term = hbar**2 * _angular_inner_product(orbital_transfer, orbital_transfer, domega_out).real
    mixed_term = hbar**2 * _angular_inner_product(orbital_transfer, spin_transfer, domega_out).real
    total = spin_term + orbital_term + 2 * mixed_term
    return float(total), float(spin_term), float(orbital_term), float(mixed_term)
