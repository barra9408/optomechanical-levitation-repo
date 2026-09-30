import numpy as np
from scipy.constants import epsilon_0, mu_0

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