import numpy as np
from scipy.spatial.transform import Rotation
from pyGDM2 import materials, structures

def _geometry(particle_type, step_nm, mesh, center, **shape_parameters):
    generator = getattr(structures, particle_type)
    geometry = np.array(generator(step_nm, mesh=mesh, **shape_parameters), dtype=float)
    if center:
        geometry -= geometry.mean(axis=0)
    return geometry

def _material(material_name):
    return getattr(materials, material_name)()

def particle_properties(particle_type, material_name, step_nm, mesh="hex", center=True, **shape_parameters):
    geometry = _geometry(particle_type, step_nm, mesh, center, **shape_parameters)
    material = _material(material_name)
    return {"geometry_nm": geometry, "material": material, "step_nm": step_nm, "mesh": mesh}

def translate_particle(initial_geometry, displacement_nm):
    return initial_geometry + displacement_nm

def rotate_particle(initial_geometry, angles_deg, origin_nm=(0, 0, 0)):
    rotation_matrix = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
    return (initial_geometry - origin_nm) @ rotation_matrix.T + origin_nm