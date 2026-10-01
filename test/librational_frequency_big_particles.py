import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from optomechanics_levitation import (
    particle_properties,
    input_beam_waist_from_filling_factor,
    electric_field_amplitude_from_power,
    evaluate_incident_field,
    hermite_gauss_00,
    evaluate_internal_field,
    evaluate_near_field,
)
from optomechanics_levitation.particle import rotate_particle
from optomechanics_levitation.light_matter_interaction import total_fields
from optomechanics_levitation.observables import (
    maxwell_stress_tensor,
    spherical_integration_surface,
    evaluate_torque,
)


MODES = (("X", "Y"), ("X", "Z"), ("Z", "X"), ("Z", "Y"))
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
RESOLUTIONS = (12.5, 13.75, 15.0)
ANGLES_DEG = np.array([-1.0, 0.0, 1.0])
ACCEPTED_TOLERANCE = 0.03
WARNING_TOLERANCE = 0.05
N_THETA = 120
N_PHI = 240
PROBE_CHUNK_SIZE = 1024


def evaluate_surface_torque(
    particle, geometry_nm, surface_positions_nm, normals, area_m2, center_nm,
    surface_incident_e, surface_incident_h, stress_incident, hermitegauss_params
):
    rotated_particle = {**particle, "geometry_nm": geometry_nm}
    dipole_incident_e = evaluate_incident_field(
        field_generator=hermite_gauss_00, positions_nm=geometry_nm,
        field="E", **hermitegauss_params
    )
    interaction_result = evaluate_internal_field(
        particle=rotated_particle,
        incident_electric_field=dipole_incident_e,
        wavelength_nm=hermitegauss_params["wavelength_nm"],
        environment=hermitegauss_params["environment"]
    )

    scattered_e = np.empty_like(surface_incident_e)
    scattered_h = np.empty_like(surface_incident_h)
    for start in range(0, len(surface_positions_nm), PROBE_CHUNK_SIZE):
        stop = min(start + PROBE_CHUNK_SIZE, len(surface_positions_nm))
        scattered_e[start:stop], scattered_h[start:stop] = evaluate_near_field(
            interaction_result, surface_positions_nm[start:stop], field="both"
        )

    total_e, total_h = total_fields(
        surface_incident_e, surface_incident_h, scattered_e, scattered_h
    )
    stress_interaction = maxwell_stress_tensor(total_e, total_h) - stress_incident
    return evaluate_torque(
        stress_interaction, surface_positions_nm, normals, area_m2, center_nm
    )


def evaluate_resolution(
    long_axis_nm, aspect_ratio, transverse_resolution, density,
    equilibrium_z_nm, hermitegauss_params
):
    step_nm = long_axis_nm / (aspect_ratio * transverse_resolution)
    long_radius_nm = long_axis_nm / 2
    short_radius_nm = long_radius_nm / aspect_ratio

    particle = particle_properties(
        particle_type="spheroid", material_name="sio2", step_nm=step_nm,
        mesh="hex", center=True,
        R1=long_radius_nm / step_nm,
        R2=short_radius_nm / step_nm,
        R3=short_radius_nm / step_nm
    )

    displacement_nm = np.array([0.0, 0.0, equilibrium_z_nm])
    geometry_x = particle["geometry_nm"] + displacement_nm
    geometry_z = rotate_particle(
        particle["geometry_nm"], np.array([0.0, -90.0, 0.0])
    ) + displacement_nm
    geometries = {"X": geometry_x, "Z": geometry_z}
    n_dipoles = len(geometry_x)

    long_radius_m = long_radius_nm * 1e-9
    short_radius_m = short_radius_nm * 1e-9
    mass_kg = density * 4 * np.pi * long_radius_m * short_radius_m**2 / 3
    moment_of_inertia = mass_kg * (long_radius_m**2 + short_radius_m**2) / 5

    print(
        f"[RESOLUTION] transverse = {transverse_resolution:.2f} | "
        f"step = {step_nm:.6f} nm | dipoles = {n_dipoles}",
        flush=True
    )

    surface_positions_nm, normals, area_m2, center_nm = spherical_integration_surface(
        geometry_x, step_nm, N_THETA, N_PHI
    )
    surface_incident_e, surface_incident_h = evaluate_incident_field(
        field_generator=hermite_gauss_00, positions_nm=surface_positions_nm,
        field="both", **hermitegauss_params
    )
    stress_incident = maxwell_stress_tensor(surface_incident_e, surface_incident_h)
    print(f"[SURFACE] points = {len(surface_positions_nm)}", flush=True)

    results = {}
    for orientation in ("X", "Z"):
        base_geometry = geometries[orientation]

        start = perf_counter()
        torque_zero = evaluate_surface_torque(
            particle, base_geometry, surface_positions_nm, normals, area_m2,
            center_nm, surface_incident_e, surface_incident_h, stress_incident,
            hermitegauss_params
        )
        print(
            f"[ORIENTATION {orientation}] equilibrium torque = {torque_zero} | "
            f"elapsed = {perf_counter() - start:.1f} s",
            flush=True
        )

        rotation_axes = ("Y", "Z") if orientation == "X" else ("X", "Y")
        for rotation_axis in rotation_axes:
            torque_components = []

            for angle_deg in ANGLES_DEG:
                if angle_deg == 0.0:
                    torque = torque_zero
                else:
                    angles_rotation = np.zeros(3)
                    angles_rotation[AXIS_INDEX[rotation_axis]] = angle_deg
                    rotated_geometry = rotate_particle(base_geometry, angles_rotation)

                    start = perf_counter()
                    torque = evaluate_surface_torque(
                        particle, rotated_geometry, surface_positions_nm,
                        normals, area_m2, center_nm, surface_incident_e,
                        surface_incident_h, stress_incident, hermitegauss_params
                    )
                    print(
                        f"[ORIENTATION {orientation}, ROTATION {rotation_axis}] "
                        f"angle = {angle_deg:+.1f} deg | "
                        f"torque = {torque[AXIS_INDEX[rotation_axis]]:+.6e} N m | "
                        f"elapsed = {perf_counter() - start:.1f} s",
                        flush=True
                    )

                torque_components.append(
                    float(torque[AXIS_INDEX[rotation_axis]])
                )

            stiffness = -float(
                np.polyfit(np.deg2rad(ANGLES_DEG), torque_components, 1)[0]
            )
            frequency_kHz = float(
                np.sign(stiffness)
                * np.sqrt(np.abs(stiffness) / moment_of_inertia)
                / (2 * np.pi * 1e3)
            )
            results[(orientation, rotation_axis)] = {
                "stiffness": stiffness,
                "frequency_kHz": frequency_kHz,
                "stable": bool(np.isfinite(stiffness) and stiffness > 0),
            }
            print(
                f"[ORIENTATION {orientation}, ROTATION {rotation_axis}] "
                f"stiffness = {stiffness:+.6e} N m/rad | "
                f"frequency = {frequency_kHz:+.6f} kHz",
                flush=True
            )

    return {
        "step_nm": step_nm,
        "n_dipoles": n_dipoles,
        "results": results,
    }


def main():
    repository_dir = Path(__file__).resolve().parents[1]
    output_dir = repository_dir / "outputs" / "librational_frequency_big_particles"
    history_dir = output_dir / "convergence_history"
    output_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)

    wavelength_nm = 1550.0
    NA = 0.75
    f_mm = 3.0
    filling_factor = 1.0
    optical_power_W = 1.0
    equilibrium_z_nm = 0.0
    n_medium = 1.0
    density = 2200.0
    impedance_ohms = 376.730313668

    focal_waist_nm = wavelength_nm / (np.pi * NA)
    rayleigh_range_nm = np.pi * focal_waist_nm**2 / wavelength_nm
    long_axis_values_nm = np.linspace(100.0, rayleigh_range_nm, 20)
    cases = [
        (aspect_ratio, float(long_axis_nm))
        for aspect_ratio in (2.0, 4.0)
        for long_axis_nm in long_axis_values_nm
    ]

    case_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    aspect_ratio, long_axis_nm = cases[case_id]
    long_axis_label = f"{long_axis_nm:.6f}".replace(".", "p")
    case_stem = f"L_{long_axis_label}_r{aspect_ratio:g}"

    print(
        f"[CASE {case_id + 1}/{len(cases)}] "
        f"long axis = {long_axis_nm:.6f} nm | aspect ratio = {aspect_ratio:g}",
        flush=True
    )
    print(
        f"[CASE] Slurm job = {os.environ.get('SLURM_JOB_ID', 'direct execution')}",
        flush=True
    )
    print(
        f"[CONVERGENCE] transverse resolutions = {RESOLUTIONS} | "
        f"accepted = {100 * ACCEPTED_TOLERANCE:.1f}% | "
        f"warning = {100 * WARNING_TOLERANCE:.1f}%",
        flush=True
    )

    w0_mm = input_beam_waist_from_filling_factor(
        filling_factor=filling_factor, numerical_aperture=NA,
        focal_length_mm=f_mm, refractive_index=n_medium
    )
    profile_power_integral_m2 = np.pi * (w0_mm * 1e-3)**2 / 2
    amplitude_v_per_m = electric_field_amplitude_from_power(
        power_w=optical_power_W,
        profile_power_integral_m2=profile_power_integral_m2,
        impedance_ohms=impedance_ohms
    )
    hermitegauss_params = dict(
        wavelength_nm=wavelength_nm,
        environment={
            "eps1": n_medium**2,
            "eps2": n_medium**2,
            "eps3": n_medium**2,
        },
        amplitude_v_per_m=amplitude_v_per_m,
        theta=0, NA=NA, f=f_mm, w0=w0_mm,
        xSpot=0, ySpot=0, zSpot=0, kSign=-1, phase=0,
        quadrature_points=24, N_cpu=4
    )

    resolution_history = []
    for transverse_resolution in RESOLUTIONS:
        resolution_history.append(
            evaluate_resolution(
                long_axis_nm, aspect_ratio, transverse_resolution,
                density, equilibrium_z_nm, hermitegauss_params
            )
        )

    history_rows = []
    qualities = []
    for orientation, rotation_axis in MODES:
        stiffnesses = np.array([
            record["results"][(orientation, rotation_axis)]["stiffness"]
            for record in resolution_history
        ])
        frequencies = np.array([
            record["results"][(orientation, rotation_axis)]["frequency_kHz"]
            for record in resolution_history
        ])
        stabilities = [
            record["results"][(orientation, rotation_axis)]["stable"]
            for record in resolution_history
        ]

        stiffness_spread = (
            np.ptp(stiffnesses)
            / max(np.max(np.abs(stiffnesses)), np.finfo(float).tiny)
        )
        frequency_spread = (
            np.ptp(frequencies)
            / max(np.max(np.abs(frequencies)), np.finfo(float).tiny)
        )
        fine_stiffness_change = (
            abs(stiffnesses[-1] - stiffnesses[-2])
            / max(abs(stiffnesses[-1]), abs(stiffnesses[-2]), np.finfo(float).tiny)
        )
        fine_frequency_change = (
            abs(frequencies[-1] - frequencies[-2])
            / max(abs(frequencies[-1]), abs(frequencies[-2]), np.finfo(float).tiny)
        )
        same_stability = stabilities[0] == stabilities[1] == stabilities[2]
        finite_values = bool(
            np.all(np.isfinite(stiffnesses))
            and np.all(np.isfinite(frequencies))
        )

        if not finite_values or not same_stability:
            quality = "rejected"
        elif stiffness_spread <= ACCEPTED_TOLERANCE:
            quality = "accepted"
        elif stiffness_spread <= WARNING_TOLERANCE:
            quality = "warning"
        else:
            quality = "rejected"

        qualities.append(quality)
        print(
            f"[CONVERGENCE {orientation}/{rotation_axis}] "
            f"three-step stiffness spread = {100 * stiffness_spread:.4f}% | "
            f"fine stiffness change = {100 * fine_stiffness_change:.4f}% | "
            f"stable = {same_stability} | quality = {quality}",
            flush=True
        )

        for level, record in enumerate(resolution_history):
            result = record["results"][(orientation, rotation_axis)]
            previous = (
                resolution_history[level - 1]["results"][(orientation, rotation_axis)]
                if level else None
            )
            pairwise_stiffness_change = (
                abs(result["stiffness"] - previous["stiffness"])
                / max(
                    abs(result["stiffness"]),
                    abs(previous["stiffness"]),
                    np.finfo(float).tiny
                )
                if previous else np.nan
            )
            pairwise_frequency_change = (
                abs(result["frequency_kHz"] - previous["frequency_kHz"])
                / max(
                    abs(result["frequency_kHz"]),
                    abs(previous["frequency_kHz"]),
                    np.finfo(float).tiny
                )
                if previous else np.nan
            )
            history_rows.append({
                "orientation": orientation,
                "rotation": rotation_axis,
                "long_axis_nm": long_axis_nm,
                "step_nm": record["step_nm"],
                "n_dipoles": record["n_dipoles"],
                "aspect_ratio": aspect_ratio,
                "resolution_level": level + 1,
                "transverse_resolution": RESOLUTIONS[level],
                "stiffness_Nm_per_rad": result["stiffness"],
                "simulation_frequency_kHz": result["frequency_kHz"],
                "stable": result["stable"],
                "pairwise_stiffness_change_pct": 100 * pairwise_stiffness_change,
                "pairwise_frequency_change_pct": 100 * pairwise_frequency_change,
                "fine_stiffness_change_pct": 100 * fine_stiffness_change,
                "fine_frequency_change_pct": 100 * fine_frequency_change,
                "three_step_stiffness_spread_pct": 100 * stiffness_spread,
                "three_step_frequency_spread_pct": 100 * frequency_spread,
                "same_stability": same_stability,
                "quality": quality,
            })

    history_csv = history_dir / f"case_{case_stem}_convergence.csv"
    case_csvs = [
        output_dir / f"case_{orientation}_or_{rotation_axis}_rot_{case_stem}.csv"
        for orientation, rotation_axis in MODES
    ]

    if "rejected" in qualities:
        pd.DataFrame(history_rows).to_csv(history_csv, index=False)
        for case_csv in case_csvs:
            if case_csv.exists():
                case_csv.unlink()
        print(
            f"[REJECTED] No result CSVs saved. "
            f"Convergence history: {history_csv}",
            flush=True
        )
        return

    if history_csv.exists():
        history_csv.unlink()

    finest = resolution_history[-1]
    for (orientation, rotation_axis), case_csv in zip(MODES, case_csvs):
        result = finest["results"][(orientation, rotation_axis)]
        row = {
            "orientation": orientation,
            "rotation": rotation_axis,
            "long_axis_nm": long_axis_nm,
            "step_nm": finest["step_nm"],
            "n_dipoles": finest["n_dipoles"],
            "aspect_ratio": aspect_ratio,
            "stiffness_Nm_per_rad": result["stiffness"],
            "simulation_frequency_kHz": result["frequency_kHz"],
        }
        pd.DataFrame([row]).to_csv(case_csv, index=False)
        print(f"[OUTPUT] {case_csv}", flush=True)

    print(
        f"[COMPLETE] All four modes meet the reference criterion: "
        f"{', '.join(qualities)}",
        flush=True
    )


if __name__ == "__main__":
    main()