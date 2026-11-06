"""Default settings used by the optical cases and retained recoil-heating scripts."""

from typing import Literal

# Particle geometry and material.
PARTICLE_TYPE: str = "spheroid"
MATERIAL: str = "sio2"
MESH: str = "cube"

# Incident beam and environment.
WAVELENGTH_NM: float = 1550.0
NUMERICAL_APERTURE: float = 0.75
ENVIRONMENT: dict[str, float] = {"eps1": 1.0, "eps2": 1.0, "eps3": 1.0, "spacing": 0.0}
OPTICAL_POWER_W: float = 0.1
REFRACTIVE_INDEX: float = 1.0
FOCAL_DISTANCE_NM: float = 3.0e6
ENTRANCE_WAIST_NM: float = 1.0e6
FOCUS_N_THETA: int = 64
FOCUS_N_PHI: int = 128
INCIDENT_POLARIZATION: tuple[float, float] = (1.0, 0.0)

# Transfer-amplitude calculation.
N_THETA: int = 64
N_PHI: int = 128
FAR_FIELD_RADIUS_NM: float = 1.0e6
CHUNK_SIZE: int = 256

# Recoil-heating calculations.
TORQUE_AXIS: Literal["x", "y", "z"] = "y"
DERIVATIVE_ROTATION_AXES: tuple[Literal["x", "y", "z"], ...] = ("y", "z")
DERIVATIVE_ANGLE_RAD: float = 0.005
SIZE_SWEEP_START_NM: float = 25.0
SIZE_SWEEP_INTERVALS: int = 40
SIZE_SWEEP_MESH_DIVISIONS: float = 35.0
SIZE_SWEEP_SPHERE_MESH_DIVISIONS: float = 22.0
SIZE_SWEEP_ASPECT_RATIOS: tuple[float, ...] = (1.0, 2.0, 4.0)
SIZE_SWEEP_TORQUE_AXES: tuple[Literal["x", "y", "z"], ...] = ("z", "y")
