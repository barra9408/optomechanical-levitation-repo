# Optomechanical Levitation

This Python package brings together optical fields, particle geometry, and
electromagnetic scattering for studies of levitated nanoparticles. It provides
a consistent route from an incident beam to the field induced inside a particle
and the field scattered into the surrounding space. The included applications
use those fields to study angular-momentum transfer and torque recoil heating.

## Relation to pyGDM2

[pyGDM2](https://homepages.laas.fr/pwiecha/pygdm_doc/) supplies the Green
dyads, particle structures, materials, and field interfaces used by this
package. The interaction code assembles and solves the coupled-dipole problem
with those components. This repository also provides focused optical beams,
a shared workflow for evaluating scattered fields, and calculations for
specific optomechanical observables. It is a separate package that depends
on pyGDM2, rather than a replacement for its general simulation tools.

## Installation

Use Python 3.11 or newer. From a clone of this repository, run:

```bash
python -m pip install -e .
```

This installs the package and its dependencies listed in `pyproject.toml`.
Run the examples below from the repository root so that `cases/` is importable.

## Quick start

The following example creates a silica spheroid, evaluates a focused incident
field at its dipoles, solves the light–matter interaction, and samples the
scattered electric field at a distant observation point.

```python
import numpy as np
from cases import plane_wave
from optomechanics_levitation import parameters
from optomechanics_levitation.light_matter_interaction import LightMatterInteraction
from optomechanics_levitation.particle import particle_properties

particle = particle_properties(
    particle_type="spheroid", material_name="sio2", step_nm=10.0,
    mesh="cube", center=True, R1=4.8, R2=2.8, R3=2.8,
)
incident_electric_field = plane_wave.electric_field(particle["geometry_nm"])
interaction = LightMatterInteraction(
    particle, parameters.WAVELENGTH_NM, parameters.ENVIRONMENT)
internal_electric_field = interaction.internal_field(incident_electric_field)

observation_positions_nm = np.array([[0.0, 0.0, 1_000_000.0]])
scattered_electric_field = interaction.far_field(observation_positions_nm)
print(internal_electric_field.shape, scattered_electric_field.shape)
```

Positions and wavelengths are in nanometres; electric fields are in V/m.
`internal_electric_field` has one three-component vector per particle dipole,
and `scattered_electric_field` has one per observation point. See the
[simulation workflow](docs/simulation_workflow.md) for the individual steps
and field conventions.

## Applications and examples

The four modules in [`cases/`](cases/) provide a focused plane-wave input and
focused LG00, LG01, and LG02 inputs. Each exposes `electric_field(positions_nm)`
and `transfer_amplitudes(particle)`. The latter supports angular-momentum
operators; [`test/recoil_heating/`](test/recoil_heating/) contains research
scripts and a notebook that use these cases to calculate torque recoil heating
and orientation information. These scripts are computational applications,
not a general automated test suite.

## Documentation

- [Simulation workflow and field conventions](docs/simulation_workflow.md)
- [Optical fields and focusing](docs/optical_fields.md)
- [Transfer amplitudes and operators](docs/transfer_amplitudes.md)

## Research using this software

A manuscript using the torque recoil heating application is in preparation.
Its publication link will be added here when available. Future papers that use
this package can be listed in this section with their published links.

## Citation, license, and contributions

If you use this package, please cite the software using [`CITATION.cff`](CITATION.cff).
Please also cite the relevant [pyGDM papers](https://homepages.laas.fr/pwiecha/pygdm_doc/)
when its scattering methods are used. This package is licensed under
[GPL-3.0-or-later](LICENSE). pyGDM2 is distributed under its own
[GPL-3.0-or-later license](https://pypi.org/project/pyGDM2/).

Copyright (C) 2026 Sebastian Gutierrez-Bernal.

Questions, bug reports, and proposals for new applications can be opened as
[GitHub issues](https://github.com/barra9408/optomechanical-levitation-repo/issues).
Please include a short example, the expected result, and the observed result
when reporting a problem.
