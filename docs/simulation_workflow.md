# Simulation workflow

The package represents a monochromatic incident field by its complex electric
amplitude. A calculation follows four steps: define a particle, evaluate the
incident field at its dipoles, solve the induced internal field, and evaluate
the scattered field at observation positions. The same solved interaction can
be sampled at more than one set of positions.

## Particle

`particle_properties(...)` uses pyGDM2 to build the dipole geometry and
material. It returns a dictionary with `geometry_nm` (shape `(n, 3)`),
`material`, `step_nm`, and `mesh`. The geometry and discretization step are
specified in nanometres. `translate_particle(...)` and `rotate_particle(...)`
can transform the dipole coordinates before a new interaction is solved.

```python
from optomechanics_levitation.particle import particle_properties

particle = particle_properties(
    particle_type="spheroid", material_name="sio2", step_nm=10.0,
    mesh="cube", center=True, R1=4.8, R2=2.8, R3=2.8,
)
positions_nm = particle["geometry_nm"]
```

The shape arguments are passed to the selected pyGDM2 geometry generator.
They describe the shape in units of the discretization step; `step_nm` sets
its physical scale.

## Incident electric field

The modules in `cases/` combine a pupil profile with `FocusedBeam`, which
evaluates a vector Richards–Wolf integral. Each case offers an
`electric_field(positions_nm)` function returning a complex array with the
same leading shape as `positions_nm` and a final axis of length three.

```python
from cases import plane_wave

incident_electric_field = plane_wave.electric_field(positions_nm)
```

The `plane_wave` name refers to the field at the objective's entrance pupil;
the case evaluates the **focused** field at the particle. The LG00, LG01, and
LG02 cases use other pupil profiles. Their common settings are in
`parameters.py`. See [Optical fields](optical_fields.md) for the normalization,
polarization, and focusing conventions.

## Interaction and scattered field

`LightMatterInteraction` builds a coupled-dipole interaction from pyGDM2's
Green dyads, structure, and polarizability tensor. It factorizes the
interaction matrix with SciPy. Calling `internal_field(...)` solves the field
inside the particle and stores the factorization for subsequent incident
fields with the same particle, wavelength, and environment.

```python
import numpy as np
from optomechanics_levitation import parameters
from optomechanics_levitation.light_matter_interaction import LightMatterInteraction

interaction = LightMatterInteraction(
    particle, parameters.WAVELENGTH_NM, parameters.ENVIRONMENT)
internal_electric_field = interaction.internal_field(incident_electric_field)

observation_positions_nm = np.array([[0.0, 0.0, 1_000_000.0]])
scattered_electric_field = interaction.far_field(observation_positions_nm)
```

`near_field(...)` and `far_field(...)` return the **scattered** field by
default. Use `component="E"`, `"H"`, or `"both"` to request the electric
field, magnetic field, or a tuple of both. `total_fields(...)` adds incident
and scattered electric and magnetic fields when both are available. The
far-field routine requires observation points farther from the origin than
the wavelength. For near-field points inside the discretized particle, the
scattered-field array contains `NaN` values; use `internal_field(...)` for
the dipole field instead.

All spatial inputs and the wavelength use nanometres. Electric fields use
V/m, magnetic H fields use A/m, and optical power uses watts. The cases use
a homogeneous environment by default; the interaction class also accepts
the three-layer environment parameters supported by its pyGDM2 dyads.

## Application-specific calculations

The field calculation can be followed by an observable calculation.
`TransferAmplitudes` evaluates the difference between applying an operator
to the incident field before scattering and applying it to the scattered
field afterward. The included angular-momentum operators and recoil-heating
classes use this interface. This is one application of the scattering
workflow; the particle and field steps can also be used independently.
See [Transfer amplitudes](transfer_amplitudes.md) for the definition and
the assumptions used in the recoil-heating application.
