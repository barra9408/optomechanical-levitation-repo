# Transfer amplitudes

The simulator describes monochromatic scattering by a linear map
$\mathcal M_0$ from the incident electric field at the particle cells to the
scattered electric field on a far-field sphere. For a fixed particle, wavelength,
and environment,

$$
\mathbf E_{\mathrm{sc}}(\hat{\mathbf n})
=\mathcal M_0[\mathbf E_{\mathrm{in}}](\hat{\mathbf n}),
\qquad
\mathcal M_0=\mathbf G_{\mathrm{ff}}\boldsymbol{\alpha}A^{-1}.
$$

Here $A$ is the coupled-dipole interaction matrix, $\boldsymbol{\alpha}$ contains
the cell polarizabilities, and $\mathbf G_{\mathrm{ff}}$ propagates the induced
polarizations to the observation sphere. The code applies this map by solving
for the internal field and evaluating the far field. It does not assemble
$\mathcal M_0$ as a dense matrix. The factorization of $A$ is reused when a
second incident field is sent through the same particle.

## From the scattering map to a transfer amplitude

Let $\hat O$ be a linear operator associated with a quantity carried by the
optical field. The incident field and the scattered field have different
numerical representations: the former is sampled at dipole positions
$\mathbf r_i$, while the latter is sampled at outgoing directions
$\hat{\mathbf n}$. We therefore write $\hat O_{\mathrm{in}}$ and
$\hat O_{\mathrm{out}}$ for the action of the same quantity on these two
representations. Its transfer amplitude is

$$
\Delta\mathbf q_O(\hat{\mathbf n})
=\left[
\mathcal M_0\bigl(\hat O_{\mathrm{in}}\mathbf E_{\mathrm{in}}\bigr)
-\hat O_{\mathrm{out}}\bigl(\mathcal M_0\mathbf E_{\mathrm{in}}\bigr)
\right](\hat{\mathbf n}).
$$

This is the meaning of $[\mathcal M_0,\hat O]\mathbf E_{\mathrm{in}}$ when
$\mathcal M_0$ maps between two field spaces. The order fixes the convention:
the operator acts on the incident field before scattering in the first term,
and on the scattered field after scattering in the second. Both terms are
complex vector fields on the **same outgoing grid** and can be subtracted
direction by direction. Their phase must be retained when comparing or
combining operators.

For angular momentum, the paper derives this expression from a rotation of
the particle. If $\hat O_{J,\mu}$ generates rotations about
$\hat{\mathbf e}_{\mu}$, then

$$
\mathcal M_{\delta_\mu}
=\mathcal R_{\delta_\mu}\mathcal M_0\mathcal R_{-\delta_\mu},
\qquad
\mathcal R_{\delta_\mu}
=\mathbf I-\frac{i\delta_\mu}{\hbar}\hat O_{J,\mu}
+\mathcal O(\delta_\mu^2),
$$

and differentiation at the reference orientation gives

$$
\left.\partial_{\delta_\mu}\mathbf E_{\mathrm{sc}}\right|_0
=\frac{i}{\hbar}\Delta\mathbf q_{J,\mu}.
$$

The transfer-amplitude calculation itself extends to another linear operator
whenever its incident and outgoing actions are defined and return electric
fields that the scattering map can accept or compare. The derivative identity
above additionally requires that the operator generate the corresponding
transformation of the particle. A squared transfer amplitude has a recoil-noise
interpretation only under the physical assumptions appropriate to the chosen
observable.

The incident beam does not need one propagation direction. In particular,
a focused beam contains a continuum of directions. In the supplied cases,
the Richards-Wolf integral combines these directions into a callable
$\mathbf E_{\mathrm{in}}(\mathbf r)$. The simulator evaluates that field at
the required positions; an operator that needs spatial derivatives can request
it at nearby positions. The outgoing directions specify where $\mathcal M_0$
is evaluated and how $\hat O_{\mathrm{out}}$ acts. No single incident direction
is assigned to the complete beam.

## Numerical evaluation and simulator interface

For a requested axis $\mu$, `TransferAmplitudes` places a midpoint grid on a
sphere with its polar axis along $\hat{\mathbf e}_\mu$:

$$
\theta_a=\left(a+\tfrac12\right)\frac{\pi}{N_\theta},\qquad
\phi_b=b\frac{2\pi}{N_\phi},\qquad
w_{ab}=\sin\theta_a\frac{\pi}{N_\theta}\frac{2\pi}{N_\phi}.
$$

The observation points are $\mathbf r_0+R\hat{\mathbf n}_{ab}$, where
$\mathbf r_0$ is the mean particle position and $R$ is
`far_field_distance_nm`. The calculation first obtains
$\mathcal M_0\mathbf E_{\mathrm{in}}$. It then evaluates
$\hat O_{\mathrm{in}}\mathbf E_{\mathrm{in}}$ at the dipoles, sends that field
through the *same* scattering interaction, applies $\hat O_{\mathrm{out}}$ to
the original far field, and subtracts the two results. The returned array has
shape `(n_theta, n_phi, 3)`. Its entries are amplitudes, not an integrated
rate.

Each case in `cases/` supplies a focused incident field and the common
simulation settings. A prepared calculator can be used for several operators;
its interaction-matrix factorization is reused. The four available cases are
`plane_wave`, `lg_00`, `lg_01`, and `lg_02`.

```python
from cases import lg_01
from optomechanics_levitation.operators import angular_momentum_operator
from optomechanics_levitation.particle import particle_properties

particle = particle_properties(
    particle_type="spheroid",
    material_name="sio2",
    step_nm=10.0,
    mesh="cube",
    center=True,
    R1=4.8,
    R2=2.8,
    R3=2.8,
)
calculator = lg_01.transfer_amplitudes(particle)
delta_j = calculator.transfer_amplitude(angular_momentum_operator("z"))
directions = calculator.directions
weights = calculator.domega
```

The operator's `axis` selects the polar axis; if it has no `axis` attribute,
the grid uses $z$. The angular weights and directions are available after
`transfer_amplitude` returns. A later call with another axis updates both
arrays. The case fields are already normalized by `FocusedBeam` to the power
specified in `parameters.py`; their wrapper uses
`amplitude_v_per_m=1.0` to avoid rescaling them.
To supply another beam directly, `TransferAmplitudes(particle, field_generator,
parameters)` accepts the same particle dictionary, a field generator callable,
and a parameter dictionary containing `wavelength_nm`, `environment`, and
`amplitude_v_per_m`. The case modules assemble these inputs for their beams.
The following is the equivalent direct construction for the field above;
another electric-field callable can replace `lg_01.field`.

```python
from optomechanics_levitation import parameters as cfg
from optomechanics_levitation.transfer_amplitudes import TransferAmplitudes

def field_generator(pos, env_dict, wavelength, returnField="E"):
    return lg_01.field(pos)

parameters = {
    "wavelength_nm": cfg.WAVELENGTH_NM,
    "environment": cfg.ENVIRONMENT,
    "amplitude_v_per_m": 1.0,
}
calculator = TransferAmplitudes(particle, field_generator, parameters)
delta_j = calculator.transfer_amplitude(angular_momentum_operator("z"))
```

For a coherent, absorption-free scatterer in vacuum, the paper uses the
photon-flux factor

$$
\kappa=\frac{\epsilon_0 cR^2}{2\hbar\omega},
\qquad
\omega=\frac{2\pi c}{\lambda},
\qquad
\mathcal Q_O
=\frac{\kappa}{2\pi}
\int\Delta\mathbf q_O^{*}\cdot\Delta\mathbf q_O\,d\Omega
\approx\frac{\kappa}{2\pi}
\sum_{a,b}w_{ab}|\Delta\mathbf q_O(\hat{\mathbf n}_{ab})|^2.
$$

`TransferAmplitudes` returns $\Delta\mathbf q_O$ and the grid weights needed
for this integration; it does not calculate $\mathcal Q_O$ or a scattering rate.
For the angular-momentum operator, $\mathcal Q_J$ is the torque recoil spectral
density $S_{\tau\tau}^{\mu}$ under the paper's assumptions. The operator acts
on the electric field with its physical units, including the factor $\hbar$;
the returned amplitude is not divided by $\hbar$.

```python
import numpy as np
from scipy.constants import c, epsilon_0, hbar

radius_m = calculator.far_field_distance_nm * 1e-9
wavelength_m = calculator.wavelength_nm * 1e-9
omega = 2 * np.pi * c / wavelength_m
kappa = epsilon_0 * c * radius_m**2 / (2 * hbar * omega)
Q_j = kappa / (2 * np.pi) * np.sum(
    np.sum(np.abs(delta_j)**2, axis=-1) * weights
)
```

## Angular-momentum operators

The three functions in `operators.py` return objects accepted directly by
`transfer_amplitude`. Each object exposes `incident(...)` and
`scattered(...)` so that one public operator choice has the correct action
on both sides of the scattering map.

### Total angular momentum

The origin for rotations is the particle centre $\mathbf r_0$. In position
space, the component about axis $\mu$ acts on a vector field as

$$
\hat O_{J,\mu}^{\mathrm{in}}\mathbf E
=-i\hbar\left[(\mathbf r-\mathbf r_0)\times\nabla\right]_\mu\mathbf E
+i\hbar\hat{\mathbf e}_\mu\times\mathbf E.
$$

The first term differentiates the spatial profile; the second rotates the
field vector itself. On the far-field sphere, the same generator is represented
by an azimuthal derivative around the selected axis:

$$
\hat O_{J,\mu}^{\mathrm{out}}\mathbf E_{\mathrm{sc}}
=\left(\mathbf I-\hat{\mathbf n}\hat{\mathbf n}^{\mathsf T}\right)
\left[-i\hbar\,\partial_{\phi_\mu}\mathbf E_{\mathrm{sc}}
+i\hbar\hat{\mathbf e}_\mu\times\mathbf E_{\mathrm{sc}}\right].
$$

The transverse projector removes the longitudinal component introduced by
numerical differentiation. The incident spatial derivatives use centred
differences with $h=10^{-3}\lambda$; the outgoing azimuthal derivative uses
periodic neighbouring grid points. This is the operator that enters the
rotation-derivative identity above.

```python
from optomechanics_levitation.operators import angular_momentum_operator

delta_j = calculator.transfer_amplitude(angular_momentum_operator("z"))
```

### Spin

For a transverse monochromatic field, we use the spin operator of the paper,

$$
\hat O_{S,\mu}^{\mathrm{in}}\mathbf E
=-\frac{i\hbar}{k^2}
(\hat{\mathbf e}_\mu\cdot\nabla)(\nabla\times\mathbf E).
$$

The incident implementation evaluates the required mixed derivatives with a
four-point centred difference. For the scattered field, its transverse
far-field form is

$$
\hat O_{S,\mu}^{\mathrm{out}}\mathbf E_{\mathrm{sc}}
=i\hbar(\hat{\mathbf e}_\mu\cdot\hat{\mathbf n})
(\hat{\mathbf n}\times\mathbf E_{\mathrm{sc}}).
$$

Here the current code uses $k=2\pi/\lambda$ with positions and wavelength in
nanometres, as in the vacuum environment used by the cases.

```python
from optomechanics_levitation.operators import spin_operator

delta_s = calculator.transfer_amplitude(spin_operator("z"))
```

### Orbital angular momentum

We define the orbital contribution as the complement of transverse spin,
in both representations:

$$
\hat O_{L,\mu}=\hat O_{J,\mu}-\hat O_{S,\mu}.
$$

Linearity of $\mathcal M_0$ then gives the amplitude identity
$\Delta\mathbf q_{J,\mu}
=\Delta\mathbf q_{L,\mu}+\Delta\mathbf q_{S,\mu}$.
The norms do **not** obey the same additive rule, because the two amplitudes
can interfere:

$$
\mathcal Q_J=\mathcal Q_L+\mathcal Q_S+\mathcal Q_{LS},
\qquad
\mathcal Q_{LS}=\frac{\kappa}{\pi}
\mathrm{Re}\int
\Delta\mathbf q_S^{*}\cdot\Delta\mathbf q_L\,d\Omega.
$$

```python
from optomechanics_levitation.operators import orbital_operator

delta_l = calculator.transfer_amplitude(orbital_operator("z"))
# On the same grid, delta_j and delta_l + delta_s agree up to numerical error.
```

## User-defined operator

A user-defined operator needs one object with two methods. `incident` receives
the callable `field_at`, the dipole positions, the particle centre, and the
wavelength. It can call `field_at(positions_nm)` or sample nearby positions
when a derivative is needed. It must return a complex electric-field array of
shape `(N, 3)` after applying the operator. `scattered` receives the far field,
its directions, the azimuthal grid spacing, and the wavelength. It must return an array with
the **same shape** as the far field, `(n_theta, n_phi, 3)`. Both methods must use
the same physical units and sign convention. An optional `axis` attribute
chooses the outgoing polar axis.

For example, the $z$ component of linear momentum is
$\hat p_z=-i\hbar\partial_z$ in position space and
$\hat p_z=\hbar k\hat n_z$ for an outgoing plane-wave component. Spatial
coordinates enter the interface in nanometres, so the derivative below is
converted to inverse metres:

```python
import numpy as np
from scipy.constants import hbar
from cases import lg_01

class LinearMomentumZ:
    axis = "z"

    def incident(self, field_at, positions_nm, center_nm, wavelength_nm):
        step_nm = 1e-3 * wavelength_nm
        shift_nm = np.array([0.0, 0.0, step_nm])
        derivative_per_nm = (
            field_at(positions_nm + shift_nm)
            - field_at(positions_nm - shift_nm)
        ) / (2 * step_nm)
        return -1j * hbar * 1e9 * derivative_per_nm

    def scattered(self, field, directions, delta_phi, wavelength_nm):
        k_per_m = 2 * np.pi / (wavelength_nm * 1e-9)
        return hbar * k_per_m * directions[..., 2, None] * field

calculator = lg_01.transfer_amplitudes(particle)
delta_pz = calculator.transfer_amplitude(LinearMomentumZ())
```

`center_nm` and `delta_phi` are present in the interface even when a particular
operator does not need them. The field supplied to `incident` is the complete
focused beam, so the example applies to a superposition of incident
directions without assigning one direction to the whole beam. Another
observable can use the same interface by replacing these two actions; the
scattering calculation remains unchanged.
