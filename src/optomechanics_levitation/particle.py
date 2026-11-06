"""Build pyGDM2 particle geometry and transform its dipole positions."""

import numpy as np
from typing import Any
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from pyGDM2 import materials, structures

def _geometry(particle_type: str, step_nm: float, mesh: str, center: bool,
              **shape_parameters: float) -> NDArray[np.float64]:
    """Generate a particle's dipole coordinates in nm, optionally centred."""
    generator = getattr(structures, particle_type)
    geometry = np.array(generator(step_nm, mesh=mesh, **shape_parameters), dtype=float)
    if center:
        geometry -= geometry.mean(axis=0)
    return geometry

def _material(material_name: str) -> Any:
    """Instantiate a pyGDM2 material by name."""
    return getattr(materials, material_name)()

def particle_properties(particle_type: str, material_name: str, step_nm: float, mesh: str = "hex",
                        center: bool = True, **shape_parameters: float) -> dict[str, Any]:
    """Return geometry in nm and material data needed for scattering simulations.

    ``shape_parameters`` are forwarded to the selected pyGDM2 geometry generator.
    The mapping contains ``geometry_nm`` with shape (n, 3), ``material``,
    ``step_nm``, and ``mesh``.
    """
    geometry = _geometry(particle_type, step_nm, mesh, center, **shape_parameters)
    material = _material(material_name)
    return {"geometry_nm": geometry, "material": material, "step_nm": step_nm, "mesh": mesh}

def translate_particle(initial_geometry_nm: NDArray[np.float64],
                       displacement_nm: NDArray[np.float64]) -> NDArray[np.float64]:
    """Translate all particle coordinates by a displacement in nm."""
    return initial_geometry_nm + displacement_nm

def rotate_particle(initial_geometry_nm: NDArray[np.float64], angles_deg: NDArray[np.float64],
                    origin_nm: NDArray[np.float64] | tuple[float, float, float] = (0, 0, 0)
                    ) -> NDArray[np.float64]:
    """Rotate (n, 3) coordinates about ``origin_nm`` by xyz Euler angles in degrees."""
    rotation_matrix = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
    return (initial_geometry_nm - origin_nm) @ rotation_matrix.T + origin_nm
