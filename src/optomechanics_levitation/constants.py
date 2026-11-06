"""Physical constants and unit conversions used throughout the package.

Physical values follow the 2022 CODATA recommendations published by NIST.
See https://physics.nist.gov/cuu/pdf/JPCRD2022CODATA.pdf.
"""

from math import pi

# Speed of light in vacuum, in metres per second (m/s); exact in SI.
SPEED_OF_LIGHT_M_PER_S: int = 299_792_458

# Vacuum electric permittivity, in farads per metre (F/m).
VACUUM_PERMITTIVITY_F_PER_M: float = 8.854_187_8188e-12

# Reduced Planck constant, in joule seconds (J s); h / (2 pi).
REDUCED_PLANCK_CONSTANT_J_S: float = 6.626_070_15e-34 / (2 * pi)

# Characteristic impedance of vacuum, in ohms (Ω); 1 / (epsilon_0 c).
VACUUM_IMPEDANCE_OHMS: float = 1 / (VACUUM_PERMITTIVITY_F_PER_M * SPEED_OF_LIGHT_M_PER_S)

# Metres in one nanometre (m/nm).
METRES_PER_NANOMETRE: float = 1e-9
