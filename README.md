# Optomechanical Levitation Simulator

This Python package brings together optical fields, particle geometry, and
electromagnetic scattering for studies of levitated nanoparticles. It provides
a consistent route from an incident beam to the field induced inside a particle
and the field scattered into the surrounding space. These fields can be used
to study optomechanical observables.

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
The example below uses only the installed package.

## Quick start

The following example creates a silica spheroid, evaluates an incident plane
wave at its dipoles, solves the light–matter interaction, and samples the
scattered electric field at a distant observation point.

```python
import numpy as np
import optomechanics_levitation as ol

wavelength_nm = 1550.0
particle = ol.particle_properties(
    particle_type="spheroid", material_name="sio2", step_nm=10.0,
    mesh="cube", center=True, R1=4.8, R2=2.8, R3=2.8,
)
plane_wave = ol.PlaneWave(
    wavelength_nm=wavelength_nm, power_w=0.1, pupil_radius_nm=1_000_000.0,
)
incident_electric_field = plane_wave.evaluate(
    "electric_field", positions_nm=particle["geometry_nm"])
environment = {"eps1": 1.0, "eps2": 1.0, "eps3": 1.0, "spacing": 0.0}
interaction = ol.LightMatterInteraction(particle, wavelength_nm, environment)
internal_electric_field = interaction.internal_field(incident_electric_field)

observation_positions_nm = np.array([[0.0, 0.0, 1_000_000.0]])
scattered_electric_field = interaction.far_field(observation_positions_nm)
print(internal_electric_field.shape, scattered_electric_field.shape)
```

Positions and wavelengths are in nanometres; electric fields are in V/m.
The plane-wave power is defined over the specified circular reference pupil.
`internal_electric_field` has one three-component vector per particle dipole,
and `scattered_electric_field` has one per observation point. See the
[simulation workflow](docs/simulation_workflow.md) for the individual steps
and field conventions.

## Simulation methods

The package provides plane-wave and Laguerre–Gauss fields, vector focusing,
particle geometry, and light–matter interaction with near- and far-field
evaluation. Transfer amplitudes and angular-momentum operators support
calculations built on the scattered field. Recoil-heating classes provide
one specialized use of these methods; the simulation workflow is not limited
to that observable.

## Documentation

- [Simulation workflow and field conventions](docs/simulation_workflow.md)
- [Optical fields and focusing](docs/optical_fields.md)
- [Transfer amplitudes and operators](docs/transfer_amplitudes.md)

## Research using this software

A manuscript in preparation uses this simulator to study torque recoil heating
of levitated nanoparticles. It applies transfer amplitudes and angular-momentum
operators to recoil-related observables. The paper-specific sweeps and
results are kept separate from the reusable package while the calculations
are under review. A public citation and link will be added when available.
Future papers using this software can be listed here with their published links.

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
