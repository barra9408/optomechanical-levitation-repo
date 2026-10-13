"""Compare three torque-noise methods and scan incident directions by particle size.

Run one size per SLURM_ARRAY_TASK_ID (0..79). All spectral densities saved in
the CSV files are divided by 2*pi and have units N^2 m^2/Hz.
"""

import argparse
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from pyGDM2 import fields
from scipy.constants import c, epsilon_0, hbar

from optomechanics_levitation import (
    derivative_torque_spectral_density,
    evaluate_incident_field,
    evaluate_internal_field,
    particle_properties,
    torque_spectral_density,
)


ASPECT_RATIO = 2.0
MESH_DIVISIONS = 35.0
WAVELENGTH_NM = 1550.0
NUMERICAL_APERTURE = 0.75
INTENSITY_W_PER_M2 = 1.0e9
FAR_FIELD_RADIUS_NM = 1.0e6
N_COS_THETA = 8
N_AZIMUTH = 16
HELICITIES = (+1, -1)
ENVIRONMENT = {"eps1": 1.0, "eps2": 1.0, "eps3": 1.0, "spacing": 0.0}
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "Recoule Heating Results"

SPECTRAL_COLUMNS = (
    "spin_N2_m2_per_Hz", "orbital_N2_m2_per_Hz",
    "mixed_N2_m2_per_Hz", "correlation_2mixed_N2_m2_per_Hz",
    "total_N2_m2_per_Hz",
)


def rayleigh_limit_nm():
    focal_waist_nm = WAVELENGTH_NM / (np.pi * NUMERICAL_APERTURE)
    return np.pi * focal_waist_nm**2 / WAVELENGTH_NM


def long_axis_values_nm():
    # Eight sizes through 100 nm and 72 more above it: 80 near-uniform cuts.
    return np.concatenate((
        np.linspace(25.0, 100.0, 8),
        np.linspace(100.0, rayleigh_limit_nm(), 73)[1:],
    ))


def militaru_torque_recoil_heating(long_radius_nm, short_radius_nm, epsilon_r):
    """Same dipole calculation as recoil_heating_theory_vs_numerics.ipynb."""
    a = long_radius_nm * 1e-9
    b = short_radius_nm * 1e-9
    eccentricity = np.sqrt(1 - (b / a)**2)
    L_x = ((1 - eccentricity**2) / (2 * eccentricity**3)
           * (np.log((1 + eccentricity) / (1 - eccentricity)) - 2 * eccentricity))
    L_y = (1 - L_x) / 2
    volume = 4 * np.pi * a * b**2 / 3
    alpha_x = epsilon_0 * volume * (epsilon_r - 1) / (1 + L_x * (epsilon_r - 1))
    alpha_y = epsilon_0 * volume * (epsilon_r - 1) / (1 + L_y * (epsilon_r - 1))
    anisotropy_squared = ((alpha_x - alpha_y) / alpha_x)**2

    E0 = np.sqrt(2 * INTENSITY_W_PER_M2 / (epsilon_0 * c))
    omega = 2 * np.pi * c / (WAVELENGTH_NM * 1e-9)
    power_sc = omega**4 * alpha_x**2 * E0**2 / (12 * np.pi * epsilon_0 * c**3)
    gamma_sc = power_sc / (hbar * omega)
    return {
        "militaru_alpha_x_SI": alpha_x,
        "militaru_alpha_y_SI": alpha_y,
        "militaru_delta_alpha_over_alpha_x_squared": anisotropy_squared,
        "militaru_gamma_sc_per_s": gamma_sc,
        "militaru_S_tau_tau_N2_m2_per_Hz": hbar**2 * gamma_sc * anisotropy_squared / (2 * np.pi),
    }


def directional_plane_wave(pos, env_dict, wavelength, direction, helicity, returnField="E"):
    """Circular incident beam from torque_components_distribution.ipynb."""
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    reference = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = reference - np.dot(reference, direction) * direction
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    polarization = (e1 + 1j * helicity * e2) / np.sqrt(2)
    electric = np.exp(2j * np.pi / wavelength * (np.asarray(pos) @ direction))[:, None] * polarization
    return electric if returnField.lower() == "e" else np.cross(direction, electric) / c


def make_particle(long_axis_nm):
    step_nm = long_axis_nm / MESH_DIVISIONS
    long_radius_nm = long_axis_nm / 2
    short_radius_nm = long_radius_nm / ASPECT_RATIO
    particle = particle_properties(
        particle_type="spheroid", material_name="sio2", step_nm=step_nm,
        mesh="cube", center=True,
        R1=long_radius_nm / step_nm - 0.2,
        R2=short_radius_nm / step_nm - 0.2,
        R3=short_radius_nm / step_nm - 0.2,
    )
    return particle, step_nm


def solve(particle, generator, parameters, prepared_interaction=None):
    incident_field = evaluate_incident_field(
        field_generator=generator, positions_nm=particle["geometry_nm"], field="E", **parameters)
    return evaluate_internal_field(
        particle=particle if prepared_interaction is None else None,
        incident_electric_field=incident_field,
        wavelength_nm=WAVELENGTH_NM, environment=ENVIRONMENT,
        prepared_interaction=prepared_interaction,
    )


def spectral_components(interaction, generator, parameters):
    total, spin, orbital, mixed = torque_spectral_density(
        interaction, generator, parameters, torque_axis="y", n_theta=64, n_phi=64)
    scale = 1 / (2 * np.pi)
    return dict(zip(SPECTRAL_COLUMNS, (
        spin * scale, orbital * scale, mixed * scale,
        2 * mixed * scale, total * scale,
    )))


def atomic_csv(rows, path):
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False, float_format="%.16e")
    os.replace(temporary, path)


def summary_row(case_id, long_axis_nm, step_nm, particle):
    stage_start = perf_counter()
    amplitude = np.sqrt(2 * INTENSITY_W_PER_M2 / (epsilon_0 * c))
    parameters = dict(
        wavelength_nm=WAVELENGTH_NM, environment=ENVIRONMENT,
        amplitude_v_per_m=amplitude, inc_angle=0.0, inc_plane="xz",
        E_p=1.0, E_s=0.0,
    )
    row = {
        "case_id": case_id,
        "long_axis_nm": long_axis_nm,
        "aspect_ratio": ASPECT_RATIO,
        "mesh_divisions": MESH_DIVISIONS,
        "step_nm": step_nm,
        "n_dipoles": len(particle["geometry_nm"]),
        "wavelength_nm": WAVELENGTH_NM,
        "intensity_W_per_m2": INTENSITY_W_PER_M2,
        "rayleigh_limit_nm": rayleigh_limit_nm(),
        "inc_angle_deg": 0.0,
        "torque_axis": "y",
        "militaru_applicable": long_axis_nm <= 100.0,
    }
    if long_axis_nm <= 100.0:
        print("[SUMMARY 1/3] Calculating Militaru polarizabilities, scattering rate and torque noise...", flush=True)
        row.update(militaru_torque_recoil_heating(
            long_axis_nm / 2, long_axis_nm / (2 * ASPECT_RATIO),
            particle["material"].epsilon(WAVELENGTH_NM).real))
        print(f"[SUMMARY 1/3] Done: S_tau_tau={row['militaru_S_tau_tau_N2_m2_per_Hz']:.6e} "
              f"N^2 m^2/Hz ({perf_counter() - stage_start:.1f} s)", flush=True)
    else:
        print("[SUMMARY 1/3] Militaru skipped: long axis exceeds 100 nm.", flush=True)
        row.update({key: np.nan for key in (
            "militaru_alpha_x_SI", "militaru_alpha_y_SI",
            "militaru_delta_alpha_over_alpha_x_squared",
            "militaru_gamma_sc_per_s", "militaru_S_tau_tau_N2_m2_per_Hz",
        )})

    stage_start = perf_counter()
    print("[SUMMARY 2/3] Calculating orientation derivatives about y and z...", flush=True)
    gamma_sc, D, S_tau_tau = derivative_torque_spectral_density(
        particle, fields.plane_wave, parameters,
        far_field_distance=FAR_FIELD_RADIUS_NM, rotation_axes=("y", "z"), angle_rad=0.005)
    row["derivative_gamma_sc_per_s"] = gamma_sc
    for axis in ("y", "z"):
        row[f"derivative_D_{axis}"] = D[axis]
        row[f"derivative_S_tau_tau_{axis}_N2_m2_per_Hz"] = S_tau_tau[axis]
    print(f"[SUMMARY 2/3] Done: Gamma_sc={gamma_sc:.6e} s^-1, "
          f"D_y={D['y']:.6e}, D_z={D['z']:.6e}, "
          f"S_y={S_tau_tau['y']:.6e}, S_z={S_tau_tau['z']:.6e} "
          f"({perf_counter() - stage_start:.1f} s)", flush=True)

    stage_start = perf_counter()
    print("[SUMMARY 3/3] Solving the same +z, x-polarized beam for spectral torque components...", flush=True)
    interaction = solve(particle, fields.plane_wave, parameters)
    row.update({f"spectral_{key}": value for key, value in
                spectral_components(interaction, fields.plane_wave, parameters).items()})
    row["spectral_S_tau_tau_y_N2_m2_per_Hz"] = row["spectral_total_N2_m2_per_Hz"]
    # The spectral function returns noise, not D; normalize it with the
    # scattering rate of the identical beam from the derivative method.
    row["spectral_D_y_from_derivative_gamma"] = (
        2 * np.pi * row["spectral_total_N2_m2_per_Hz"] / (hbar**2 * gamma_sc))
    print(f"[SUMMARY 3/3] Done: total={row['spectral_total_N2_m2_per_Hz']:.6e}, "
          f"spin={row['spectral_spin_N2_m2_per_Hz']:.6e}, "
          f"orbital={row['spectral_orbital_N2_m2_per_Hz']:.6e}, "
          f"correlation={row['spectral_correlation_2mixed_N2_m2_per_Hz']:.6e} "
          f"({perf_counter() - stage_start:.1f} s)", flush=True)
    return row


def angular_cases():
    cos_theta = -1 + (np.arange(N_COS_THETA) + 0.5) * 2 / N_COS_THETA
    azimuth = (np.arange(N_AZIMUTH) + 0.5) * 2 * np.pi / N_AZIMUTH
    for helicity in HELICITIES:
        for u in cos_theta:
            for phi in azimuth:
                direction = (np.sqrt(1 - u**2) * np.cos(phi),
                             np.sqrt(1 - u**2) * np.sin(phi), u)
                yield helicity, float(u), float(phi), direction


def distribution_rows(case_id, long_axis_nm, particle, checkpoint_path):
    cases = list(angular_cases())
    existing = pd.read_csv(checkpoint_path).to_dict("records") if checkpoint_path.exists() else []
    rows = {int(row["direction_index"]): row for row in existing}
    if any(
        row["case_id"] != case_id
        or not np.isclose(row["long_axis_nm"], long_axis_nm)
        or not np.isclose(row["aspect_ratio"], ASPECT_RATIO)
        or not np.isclose(row["mesh_divisions"], MESH_DIVISIONS)
        or not np.isclose(row["wavelength_nm"], WAVELENGTH_NM)
        or not np.isclose(row["intensity_W_per_m2"], INTENSITY_W_PER_M2)
        or not 0 <= int(row["direction_index"]) < len(cases)
        for row in existing
    ):
        raise ValueError(f"Checkpoint belongs to a different case: {checkpoint_path}")
    print(f"[DISTRIBUTION] Starting angular scan: {len(cases)} beams "
          f"({N_COS_THETA} cos(theta) x {N_AZIMUTH} azimuths x {len(HELICITIES)} helicities); "
          f"{len(rows)} already saved", flush=True)

    amplitude = np.sqrt(2 * INTENSITY_W_PER_M2 / (epsilon_0 * c))
    prepared_interaction = None
    for index, (helicity, cos_theta, phi, direction) in enumerate(cases):
        if index in rows:
            continue
        beam_start = perf_counter()
        print(f"[DISTRIBUTION {index + 1}/{len(cases)}] Calculating "
              f"helicity={helicity:+d}, cos(theta)={cos_theta:+.4f}, "
              f"phi={np.degrees(phi):.3f} deg...", flush=True)
        parameters = dict(
            wavelength_nm=WAVELENGTH_NM, environment=ENVIRONMENT,
            amplitude_v_per_m=amplitude, direction=direction, helicity=helicity,
        )
        interaction = solve(particle, directional_plane_wave, parameters, prepared_interaction)
        prepared_interaction = interaction["prepared_interaction"]
        rows[index] = {
            "case_id": case_id,
            "long_axis_nm": long_axis_nm,
            "aspect_ratio": ASPECT_RATIO,
            "mesh_divisions": MESH_DIVISIONS,
            "wavelength_nm": WAVELENGTH_NM,
            "intensity_W_per_m2": INTENSITY_W_PER_M2,
            "direction_index": index,
            "helicity": helicity,
            "polarization": "RCP" if helicity == 1 else "LCP",
            "cos_theta": cos_theta,
            "theta_deg": np.degrees(np.arccos(cos_theta)),
            "phi_deg": np.degrees(phi),
            "direction_x": direction[0],
            "direction_y": direction[1],
            "direction_z": direction[2],
            **spectral_components(interaction, directional_plane_wave, parameters),
        }
        atomic_csv([rows[i] for i in sorted(rows)], checkpoint_path)
        print(f"[DISTRIBUTION {index + 1}/{len(cases)}] Saved: "
              f"total={rows[index]['total_N2_m2_per_Hz']:.6e} N^2 m^2/Hz "
              f"({perf_counter() - beam_start:.1f} s)", flush=True)

    if len(rows) != len(cases):
        raise RuntimeError(f"Expected {len(cases)} directions, found {len(rows)}")
    return [rows[i] for i in range(len(cases))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=None,
                        help="Override SLURM_ARRAY_TASK_ID for a direct run")
    parser.add_argument("--list-cases", action="store_true",
                        help="Print the 80 long-axis lengths without running simulations")
    args = parser.parse_args()
    sizes = long_axis_values_nm()
    if args.list_cases:
        for case_id, size in enumerate(sizes):
            print(f"{case_id:02d}, {size:.12f} nm")
        return

    case_id = args.case_id if args.case_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= case_id < len(sizes):
        parser.error(f"case ID must be between 0 and {len(sizes) - 1}")
    start = perf_counter()
    long_axis_nm = float(sizes[case_id])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    case_stem = f"case_{case_id:02d}_L_{long_axis_nm:.6f}nm".replace(".", "p")
    summary_path = OUTPUT_DIR / f"recoil_heating_summary_{case_stem}.csv"
    distribution_path = OUTPUT_DIR / f"torque_component_distribution_{case_stem}.csv"
    checkpoint_path = distribution_path.with_name(distribution_path.stem + ".partial.csv")
    if summary_path.exists() and distribution_path.exists():
        print(f"[COMPLETE] Case {case_id:02d} already has both CSV files", flush=True)
        return

    particle, step_nm = make_particle(long_axis_nm)
    print(f"[CASE {case_id:02d}/79] long axis={long_axis_nm:.9f} nm, "
          f"mesh divisions={MESH_DIVISIONS:g}, dipoles={len(particle['geometry_nm'])}", flush=True)
    print(f"[CASE] Slurm job={os.environ.get('SLURM_JOB_ID', 'direct run')}; "
          f"step={step_nm:.6f} nm; output={OUTPUT_DIR}", flush=True)
    if not summary_path.exists():
        print("[SUMMARY] Starting three-method comparison...", flush=True)
        row = summary_row(case_id, long_axis_nm, step_nm, particle)
        atomic_csv([row], summary_path)
        print(f"[OUTPUT] {summary_path}", flush=True)
    else:
        print(f"[SUMMARY] Already saved: {summary_path}", flush=True)

    if not distribution_path.exists():
        distribution_rows(case_id, long_axis_nm, particle, checkpoint_path)
        os.replace(checkpoint_path, distribution_path)
        print(f"[OUTPUT] {distribution_path}", flush=True)
    else:
        print(f"[DISTRIBUTION] Already saved: {distribution_path}", flush=True)
    print(f"[COMPLETE] elapsed={(perf_counter() - start) / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()
