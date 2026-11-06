# Optical fields

We describe the incident light by a complex electric-field amplitude with time
dependence $\mathbf E(\mathbf r,t)=\mathrm{Re}[\mathbf E(\mathbf r)e^{-i\omega t}]$.
In a homogeneous, lossless, nonmagnetic medium, $k=n\omega/c=2\pi n/\lambda$.
All spatial arguments supplied to the simulator are in nanometres, whereas
electric fields and optical powers are returned or specified in $\mathrm{V/m}$
and watts. The incident beam travels along $+z$; the entrance pupil is at the
beam waist $z=0$, and observation positions for the focused field are measured
from the geometric focus.

For any Cartesian component of a monochromatic field, Helmholtz's equation is
$(\nabla^2+k^2)A=0$. Writing $A=u e^{ikz}$ gives the exact envelope equation

$$
\nabla_\perp^2u+2ik\,\partial_z u+\partial_z^2u=0.
$$

The incident Laguerre-Gauss (LG) modes follow when the longitudinal second
derivative is negligible. The corresponding paraxial equation is
$\nabla_\perp^2u+2ik\,\partial_z u=0$. Focusing is treated separately with
the vector Richards-Wolf integral below; the paraxial equation is not used to
propagate the field through the objective.

## Plane wave

For a plane wave incident along $+z$, $u_{\mathrm{PW}}=1$ and

$$
\begin{gathered}
\mathbf E_{\mathrm{PW}}(\mathbf r) =E_{\mathrm{PW}}e^{ikz}\hat{\mathbf e},\\
\hat{\mathbf e}^{*} \cdot\hat{\mathbf e}=1,
\qquad \hat{\mathbf e}\cdot\hat{\mathbf z}=0.
\end{gathered}
$$

This is an exact homogeneous-medium solution of Helmholtz's equation. Its power
is specified through a *finite reference area*: an ideal plane wave extending
over the entire transverse plane has infinite power. For the circular entrance
pupil of an objective with focal distance $f$ and numerical aperture $\mathrm{NA}$,
$a=f\sin\theta_{\max}=f\,\mathrm{NA}/n$ is the pupil radius. The incident
irradiance and electric-field amplitude are

$$
I_{\mathrm{PW}}=\frac{P_{\mathrm{pupil}}}{\pi a^2},
\qquad
E_{\mathrm{PW}}=
\sqrt{\frac{2P_{\mathrm{pupil}}}{n\epsilon_0c\,\pi a^2}}.
$$

`PlaneWave.evaluate("profile")` returns the constant pupil profile, whereas
`evaluate("electric_field", positions_nm=...)` returns the unfocused electric field.
The aperture is imposed by `FocusedBeam`, not by the constant profile itself.

```python
from optomechanics_levitation.optical_fields.planewave import PlaneWave
import numpy as np

wavelength_nm = 1550.0
focal_distance_nm = 3_000_000.0
numerical_aperture = 0.75
refractive_index = 1.0
power_w = 0.1
positions_nm = np.array([[0.0, 0.0, 0.0]])
pupil_radius_nm = focal_distance_nm * numerical_aperture / refractive_index
plane_wave = PlaneWave(
    wavelength_nm=wavelength_nm,
    power_w=power_w,
    pupil_radius_nm=pupil_radius_nm,
    refractive_index=refractive_index,
)
plane_profile = plane_wave.evaluate("profile")
incident_electric_field = plane_wave.evaluate(
    "electric_field", positions_nm=positions_nm, polarization=(1.0, 0.0))
```

## Laguerre-Gauss family

In cylindrical coordinates $(\rho,\phi,z)$, the paraxial equation reads

$$
\left(\partial_\rho^2+\frac{1}{\rho}\partial_\rho
+\frac{1}{\rho^2}\partial_\phi^2+2ik\partial_z\right)u=0.
$$

Separation in $\rho$, $\phi$, and $z$ yields the regular elementary modes
$J_{|\ell|}(\kappa\rho)e^{i\ell\phi}e^{-i\kappa^2z/(2k)}$.
Superpositions of these modes, with a Gaussian transverse distribution at the
waist, produce the LG family. The implementation evaluates its closed form:

$$
\begin{aligned}
u_{p,\ell}(\rho,\phi,z)
&= \frac{C_{p,\ell}}{w(z)}
\left(\frac{\sqrt{2}\rho}{w(z)}\right)^{|\ell|}
L_p^{|\ell|}\left(\frac{2\rho^2}{w^2(z)}\right)
\exp\left[-\frac{\rho^2}{w^2(z)}\right] \\
&\quad\times\exp\left[i\ell\phi
+i\frac{k\rho^2}{2R(z)}-i\psi_{p,\ell}(z)\right],
\end{aligned}
$$

where $p=0,1,\ldots$ is the radial index, $\ell\in\mathbb Z$ is the
topological charge, $L_p^{|\ell|}$ is a generalized Laguerre polynomial, and

$$
\begin{gathered}
C_{p,\ell}=\sqrt{\frac{2p!}{\pi(p+|\ell|)!}},\qquad
z_R=\frac{kw_0^2}{2},\qquad
w(z)=w_0\sqrt{1+(z/z_R)^2},\\
\frac{1}{R(z)}=\frac{z}{z^2+z_R^2},\qquad
\psi_{p,\ell}(z)=(2p+|\ell|+1)\arctan(z/z_R).
\end{gathered}
$$

At the entrance pupil, $z=0$, so $w=w_0$ and both the curvature and Gouy
phases vanish. The mode is normalized by
$\int_0^{2\pi}d\phi\int_0^\infty \rho\,d\rho\,|u_{p,\ell}|^2=1$.
Thus $p=\ell=0$ gives the Gaussian mode, while $(p,\ell)=(0,1)$ and $(0,2)$
give the first two helical modes illustrated here. The azimuthal phase
$e^{i\ell\phi}$ carries the sign convention set by $e^{ikz-i\omega t}$.

For a unit Jones vector $\hat{\mathbf e}=(e_x,e_y,0)$, the unfocused field is

$$
\mathbf E^{\mathrm{inc}}_{p,\ell}(\mathbf r)=
\sqrt{\frac{2P_{\mathrm{LG}}}{n\epsilon_0c}}\,
u^{\mathrm{SI}}_{p,\ell}(\rho,\phi,z)e^{ikz}\hat{\mathbf e}.
$$

Here $u^{\mathrm{SI}}$ is in $\mathrm{m}^{-1}$. Since the implementation
evaluates $u$ using nanometres, the electric-field prefactor includes a factor
$10^9$: $u^{\mathrm{SI}}=10^9u^{\mathrm{nm}}$. The power $P_{\mathrm{LG}}$
refers to the complete, untruncated incident mode. Its on-axis irradiance
vanishes for $|\ell|>0$ before the objective, even when its total power is
nonzero.

```python
from optomechanics_levitation.optical_fields.laguerregauss import LaguerreGauss
import numpy as np

entrance_waist_nm = 1_000_000.0
positions_nm = np.array([[0.0, 0.0, 0.0]])
lg = LaguerreGauss(
    waist_radius_nm=entrance_waist_nm,
    radial_index=0,
    azimuthal_index=1,
    wavelength_nm=wavelength_nm,
    power_w=power_w,
    refractive_index=refractive_index,
)
lg_profile = lg.evaluate("profile")
incident_electric_field = lg.evaluate(
    "electric_field", positions_nm=positions_nm, polarization=(1.0, 0.0))
```

Both profile methods return a **callable** of $(\rho,\phi,z)$, with $z=0$ by
default. `FocusedBeam` evaluates this callable at the pupil coordinates
$(f\sin\theta,\phi)$.

## Focused fields

For an ideal aplanatic objective, define

$$
\theta_{\max}=\arcsin(\mathrm{NA}/n),\qquad
\hat{\mathbf s}(\theta,\phi)=
(\sin\theta\cos\phi,\sin\theta\sin\phi,\cos\theta),\qquad
\mathsf R(\theta,\phi)=R_z(\phi)R_y(\theta)R_z(-\phi).
$$

The rotation maps the incident transverse polarization onto a vector
perpendicular to $\hat{\mathbf s}$. The aplanatic apodization factor
$\sqrt{\cos\theta}$ accounts for the mapping of a pupil annulus onto the
reference sphere. If $\mathbf E_{\mathrm{pupil}}(\rho,\phi)$ is the incident
electric field on the entrance pupil, the field near the geometric focus is

$$
\mathbf E_{\mathrm{foc}}(\mathbf r)=
-\frac{ikf}{2\pi}
\int_0^{\theta_{\max}}\int_0^{2\pi}
\sqrt{\cos\theta}\,\mathsf R(\theta,\phi)
\mathbf E_{\mathrm{pupil}}(f\sin\theta,\phi)
e^{ik\hat{\mathbf s}(\theta,\phi)\cdot\mathbf r}
\sin\theta\,d\phi\,d\theta.
$$

The input may be a scalar profile $u(\rho,\phi)$ multiplied by a constant
Jones vector $(e_x,e_y)$, or a transverse vector profile
$\mathbf F(\rho,\phi)=(F_x,F_y)$ that specifies the polarization at each
pupil point. `FocusedBeam` normalizes either input to the optical power
transmitted through its finite pupil. In pupil coordinates,

$$
\begin{gathered}
dA=\rho\,d\rho\,d\phi
=f^2\sin\theta\cos\theta\,d\theta\,d\phi,\\
\mathcal N_A=\int_{\mathrm{pupil}} |\mathbf F|^2dA,
\qquad
\mathbf E_{\mathrm{pupil}}
=\sqrt{\frac{2P_{\mathrm{pupil}}}{n\epsilon_0c\,\mathcal N_A}}\,\mathbf F.
\end{gathered}
$$

The last expression uses SI coordinates; the numerical implementation
converts the pupil area from $\mathrm{nm}^2$ to $\mathrm{m}^2$. It includes the
Jones-vector norm in $\mathcal N_A$, so the requested pupil power is preserved
for both linear and elliptical input polarization. The objective is assumed
lossless and the medium homogeneous. The resulting field is vectorial and can
have a nonzero longitudinal component $E_z$. Each plane-wave contribution to
the integral satisfies the full Helmholtz equation in the focal region.

The following example continues with the LG profile and position defined above.

```python
from optomechanics_levitation.optical_fields.focused_fields import FocusedBeam

focus = FocusedBeam(
    wavelength_nm=wavelength_nm,
    focal_distance_nm=focal_distance_nm,
    n_theta=16,
    n_phi=32,
    numerical_aperture=numerical_aperture,
    power_w=power_w,
    refractive_index=refractive_index,
)
focused_electric_field = focus.electric_field(
    positions_nm, lg_profile, polarization=(1.0, 0.0))
```

The `PlaneWave` power parameter refers to its reference pupil, the
`LaguerreGauss` power parameter to the entire incident beam, and the
`FocusedBeam` power parameter to the power admitted by the objective.
The last value controls the focused-field amplitude. Thus a
truncated LG mode with the same numerical power in `LaguerreGauss` and
`FocusedBeam` is rescaled at the objective to keep the *admitted* power fixed.

## User-defined field

A new pupil field needs a callable accepting arrays of pupil radius (in nm)
and azimuth, and returning either a scalar array or an array with two final
components $(F_x,F_y)$. The callable describes the incident field *before*
the objective; the normalization and vector focusing are supplied by
`FocusedBeam`. For a scalar profile, the `polarization` argument sets a
uniform Jones vector. For a two-component profile, the returned components
already specify the local polarization.

```python
import numpy as np
from optomechanics_levitation.optical_fields.focused_fields import FocusedBeam

focus = FocusedBeam(
    wavelength_nm=wavelength_nm,
    focal_distance_nm=focal_distance_nm,
    n_theta=16,
    n_phi=32,
    numerical_aperture=numerical_aperture,
    power_w=power_w,
    refractive_index=refractive_index,
)
positions_nm = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])

def scalar_profile(rho_nm, phi):
    radius = rho_nm / entrance_waist_nm
    envelope = radius * np.exp(-radius**4)
    return envelope * np.exp(1j * phi)

scalar_electric_field = focus.electric_field(
    positions_nm, scalar_profile, polarization=(1.0, 1j))

def radial_profile(rho_nm, phi):
    radius = rho_nm / entrance_waist_nm
    envelope = radius * np.exp(-radius**2)
    return envelope[..., None] * np.stack((np.cos(phi), np.sin(phi)), axis=-1)

radial_electric_field = focus.electric_field(positions_nm, radial_profile)
```

The scalar and vector profiles may have any transverse shape, provided their
pupil norm is finite and nonzero. The current interface accepts fields
transverse to the incident $z$ axis; it returns the electric field at a point
or an array of points near the focus. A magnetic field or an independently
specified longitudinal component at the entrance pupil is outside this
interface.
