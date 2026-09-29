import numpy as np
import scipy.linalg as la
from scipy.spatial import cKDTree
from pyGDM2 import propagators, structures

_Z0 = np.sqrt(4.0e-7 * np.pi / 8.85418782e-12)

def _create_dyads(environment):
    env = {} if environment is None else environment
    default_eps = env.get("eps_env", 1.0)
    indices = [complex(env.get(key, default_eps))**0.5 for key in ("eps1", "eps2", "eps3")]
    return propagators.DyadsQuasistatic123(n1=indices[0], n2=indices[1], n3=indices[2], spacing=env.get("spacing_nm", env.get("spacing", 0.0)), radiative_correction=True)

def _interaction_matrix(particle, wavelength_nm, dyads):
    geometry = np.asarray(particle["geometry_nm"], dtype=np.float64)
    structure = structures.struct(particle["step_nm"], geometry, particle["material"], normalization=particle["mesh"], auto_shift_structure=False, check_geometry_consistency=False, warn=False, verbose=False)
    self_terms = dyads.getSelfTermEE(wavelength_nm, structure)
    
    polarizability = dyads.getPolarizabilityTensor(wavelength_nm, structure)
    green_config = dyads.getConfigDictG(wavelength_nm, structure, None)
    matrix = np.zeros((3 * len(geometry), 3 * len(geometry)), dtype=np.complex64, order="F")
    dyads.tsbs_EE(geometry, wavelength_nm, self_terms, polarizability, green_config, matrix)
    
    prepared = {
        "structure": structure,
        "dyads": dyads,
        "wavelength_nm": wavelength_nm,
        "polarizability": polarizability,
        "green_config": green_config,
    }

    return matrix, prepared

def _factorize_interaction_matrix(matrix):
    return la.lu_factor(matrix, overwrite_a=True)

def evaluate_internal_field(particle, incident_electric_field, wavelength_nm, environment, prepared_interaction=None):
    if prepared_interaction is None:
        dyads = _create_dyads(environment)
        matrix, prepared_interaction = _interaction_matrix(particle, wavelength_nm, dyads)
        prepared_interaction["lu"], prepared_interaction["pivots"] = _factorize_interaction_matrix(matrix)

    incident = np.asarray(incident_electric_field, dtype=np.complex64)
    internal = la.lu_solve((prepared_interaction["lu"], prepared_interaction["pivots"]), incident.reshape(-1)).reshape(-1, 3)
    return {"E_internal": internal, "prepared_interaction": prepared_interaction}

def evaluate_near_field(interaction_result, positions_nm, field="E"):        
    prepared = interaction_result["prepared_interaction"]
    structure = prepared["structure"]
    dyads = prepared["dyads"]
    positions = np.asarray(positions_nm, dtype=np.float64)
    polarization = np.einsum("nij,nj->ni", prepared["polarizability"], interaction_result["E_internal"])
    distances = cKDTree(structure.geometry).query(positions, k=1)[0]
    outside = distances > 1.005 * structure.step
    probe_positions = positions[outside]

    result = []
    for component in (("E", "H") if field == "both" else (field,)):
        scattered = np.full((len(positions), 3), np.nan, dtype=np.complex64)
        green = np.zeros((len(structure.geometry), len(probe_positions), 3, 3), dtype=np.complex64)
        dyads.eval_G(structure.geometry, probe_positions, dyads.G_EE if component == "E" else dyads.G_HE, prepared["wavelength_nm"], prepared["green_config"], green)
        scattered[outside] = np.einsum("nmij,nj->mi", green, polarization)
        if component == "H":
            # PyGDM's raw magnetic (B/H) response needs the SI impedance factor.
            scattered /= _Z0
        result.append(scattered)
    return tuple(result) if field == "both" else result[0]

def evaluate_far_field(interaction_result, positions_nm, field="E"):
    prepared = interaction_result["prepared_interaction"]
    structure = prepared["structure"]
    dyads = prepared["dyads"]
    positions = np.asarray(positions_nm, dtype=np.float64)
    polarization = np.einsum("nij,nj->ni", prepared["polarizability"], interaction_result["E_internal"])
    green = np.zeros((len(structure.geometry), len(positions), 3, 3), dtype=np.complex64)
    dyads.eval_G(structure.geometry, positions, dyads.G_EE_ff, prepared["wavelength_nm"], prepared["green_config"], green)
    electric = np.einsum("nmij,nj->mi", green, polarization)
    if field == "E":
        return electric

    directions = positions / np.linalg.norm(positions, axis=1)[:, None]
    index = complex(prepared["green_config"]["eps2"])**0.5
    # Leading outgoing-wave H = n/Z0 * (direction cross E), as in the reference.
    magnetic = np.asarray(index / _Z0 * np.cross(directions, electric), dtype=np.complex64)
    return (electric, magnetic) if field == "both" else magnetic

def total_fields(incident_efield, incident_hfield, scattered_efield, scattered_hfield):
    total_efield = incident_efield + scattered_efield
    total_hfield = incident_hfield + scattered_hfield
    return total_efield, total_hfield
