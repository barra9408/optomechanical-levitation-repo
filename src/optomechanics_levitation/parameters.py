"""Default settings for transfer amplitudes and recoil-heating calculations."""

from typing import Literal

# Optical wavelength used by the theoretical recoil estimate.
WAVELENGTH_NM: float = 1550.0

# Transfer-amplitude calculation.
N_THETA: int = 64
N_PHI: int = 128
FAR_FIELD_RADIUS_NM: float = 1.0e6
CHUNK_SIZE: int = 256

# Recoil-heating calculations.
TORQUE_AXIS: Literal["x", "y", "z"] = "y"
DERIVATIVE_ROTATION_AXES: tuple[Literal["x", "y", "z"], ...] = ("y", "z")
DERIVATIVE_ANGLE_RAD: float = 0.005
