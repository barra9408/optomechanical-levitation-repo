import os
import sys
from pathlib import Path
from time import perf_counter
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.constants import epsilon_0
from scipy.integrate import quad

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


@lru_cache(maxsize=None)
def depolarization_factors(aspect_ratio):
    a_norm = 1.0 / aspect_ratio
    b_norm = 1.0 / aspect_ratio
    c_norm = 1.0

    def f_q(q, a, b, c):
        return np.sqrt((q + a**2) * (q + b**2) * (q + c**2))

    L_1 = a_norm * b_norm * c_norm * quad(lambda q: 1 / ((a_norm**2 + q) * f_q(q, a_norm, b_norm, c_norm)), 0, np.inf)[0] / 2
    L_3 = a_norm * b_norm * c_norm * quad(lambda q: 1 / ((c_norm**2 + q) * f_q(q, a_norm, b_norm, c_norm)), 0, np.inf)[0] / 2

    return L_1, L_1, L_3


def focused_peak_efield(hermitegauss_params, equilibrium_z_nm):
    focus_position_nm = np.array([[0.0, 0.0, equilibrium_z_nm]])
    focus_efield = evaluate_incident_field(
        field_generator=hermite_gauss_00,
        positions_nm=focus_position_nm,
        field="E",
        **hermitegauss_params
    )
    return np.linalg.norm(focus_efield[0])


def theoretical_librational_frequency(long_axis_nm, aspect_ratio, peak_efield, density, n_silica, n_medium):
    long_axis_m = long_axis_nm * 1e-9
    a = long_axis_m / (2 * aspect_ratio)
    b = a
    c = long_axis_m / 2

    mass_kg = 4 * np.pi * a * b * c * density / 3
    moment_of_inertia = mass_kg * (a**2 + c**2) / 5

    L_1, _, L_3 = depolarization_factors(aspect_ratio)
    epsilon_1 = n_silica**2
    epsilon_m = n_medium**2

    alpha_1 = 4 * np.pi * epsilon_0 * epsilon_m * a * b * c * (epsilon_1 - epsilon_m) / (3 * epsilon_m + 3 * L_1 * (epsilon_1 - epsilon_m))
    alpha_3 = 4 * np.pi * epsilon_0 * epsilon_m * a * b * c * (epsilon_1 - epsilon_m) / (3 * epsilon_m + 3 * L_3 * (epsilon_1 - epsilon_m))

    delta_alpha = np.real(alpha_3 - alpha_1)
    angular_frequency = peak_efield * np.sqrt(delta_alpha / (2 * moment_of_inertia))

    return angular_frequency / (2 * np.pi * 1e3)


def numerical_librational_frequency(long_axis_nm, aspect_ratio, step_nm, density, equilibrium_z_nm, hermitegauss_params, n_theta=24, n_phi=48):
    print("[NUMERICAL 1/4] Building the particle...", flush=True)
    long_radius_nm = long_axis_nm / 2
    short_radius_nm = long_radius_nm / aspect_ratio

    particle = particle_properties(
        particle_type="spheroid", material_name="sio2", step_nm=step_nm,
        mesh="hex", center=True,
        R1=long_radius_nm / step_nm,
        R2=short_radius_nm / step_nm,
        R3=short_radius_nm / step_nm
    )
    particle["geometry_nm"] = particle["geometry_nm"] + np.array([0.0, 0.0, equilibrium_z_nm])
    initial_geometry = particle["geometry_nm"]
    n_dipoles = len(initial_geometry)
    print(f"Number of dipoles: {n_dipoles}", flush=True)

    long_radius_m = long_radius_nm * 1e-9
    short_radius_m = short_radius_nm * 1e-9
    volume_m3 = 4 * np.pi * long_radius_m * short_radius_m**2 / 3
    mass_kg = density * volume_m3
    moment_of_inertia = mass_kg * (long_radius_m**2 + short_radius_m**2) / 5

    print("[NUMERICAL 2/4] Building the integration surface and evaluating its incident field...", flush=True)
    surface_positions_nm, normals, area_m2, center_nm = spherical_integration_surface(
        initial_geometry, step_nm, n_theta, n_phi
    )
    surface_incident_e, surface_incident_h = evaluate_incident_field(
        field_generator=hermite_gauss_00,
        positions_nm=surface_positions_nm,
        field="both",
        **hermitegauss_params
    )
    stress_incident = maxwell_stress_tensor(surface_incident_e, surface_incident_h)
    print(f"Integration points: {len(surface_positions_nm)}", flush=True)

    print("[NUMERICAL 3/4] Calculating torque at three rotation angles...", flush=True)
    angles_deg = np.array([-1.0, 0.0, 1.0])
    torque_y = []

    for angle_deg in angles_deg:
        angle_start = perf_counter()
        print(f"Angle {angle_deg:+.1f} deg: rotating particle and evaluating incident field...", flush=True)
        rotated_geometry = rotate_particle(initial_geometry, np.array([0.0, angle_deg, 0.0]))
        rotated_particle = {**particle, "geometry_nm": rotated_geometry}
        dipole_incident_e = evaluate_incident_field(
            field_generator=hermite_gauss_00,
            positions_nm=rotated_geometry,
            field="E",
            **hermitegauss_params
        )

        print(f"Angle {angle_deg:+.1f} deg: solving internal interaction...", flush=True)
        interaction_result = evaluate_internal_field(
            particle=rotated_particle,
            incident_electric_field=dipole_incident_e,
            wavelength_nm=hermitegauss_params["wavelength_nm"],
            environment=hermitegauss_params["environment"]
        )

        print(f"Angle {angle_deg:+.1f} deg: evaluating scattered fields on the surface...", flush=True)
        scattered_e, scattered_h = evaluate_near_field(
            interaction_result, surface_positions_nm, field="both"
        )
        total_e, total_h = total_fields(
            surface_incident_e, surface_incident_h, scattered_e, scattered_h
        )
        stress_interaction = maxwell_stress_tensor(total_e, total_h) - stress_incident
        torque = evaluate_torque(
            stress_interaction, surface_positions_nm, normals, area_m2, center_nm
        )
        torque_y.append(float(torque[1]))
        print(f"Angle {angle_deg:+.1f} deg: torque_y = {torque[1]:+.6e} N m | elapsed = {perf_counter() - angle_start:.1f} s", flush=True)

    print("[NUMERICAL 4/4] Fitting torque slope and calculating frequency...", flush=True)
    torque_slope = np.polyfit(np.deg2rad(angles_deg), torque_y, 1)[0]
    angular_stiffness = -torque_slope
    frequency_kHz = np.sign(angular_stiffness) * np.sqrt(np.abs(angular_stiffness) / moment_of_inertia) / (2 * np.pi * 1e3)

    print(f"Angular stiffness: {angular_stiffness:+.6e} N m/rad", flush=True)
    print(f"Numerical frequency: {frequency_kHz:.6f} kHz", flush=True)

    return frequency_kHz, n_dipoles


def main():
    repository_dir = Path(__file__).resolve().parents[1]
    output_dir = repository_dir / "outputs" / "librational_frequency_vs_numerics"
    output_dir.mkdir(parents=True, exist_ok=True)

    long_axis_values_nm = np.linspace(25.0, 100.0, 20)
    cases = [(aspect_ratio, float(long_axis_nm)) for aspect_ratio in (2.0, 4.0) for long_axis_nm in long_axis_values_nm]

    if "--combine" in sys.argv:
        print("[COMBINE] Reading the 40 individual CSV files...", flush=True)
        case_files = [output_dir / f"case_{case_id:02d}.csv" for case_id in range(len(cases))]
        results = pd.concat([pd.read_csv(path) for path in case_files], ignore_index=True)
        final_csv = output_dir / "librational_frequency_vs_numerics.csv"
        results.to_csv(final_csv, index=False)
        print(f"[COMBINE] Final CSV: {final_csv}", flush=True)
        print(f"[COMBINE] Cases combined: {len(results)}", flush=True)
        return

    start = perf_counter()
    case_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    aspect_ratio, long_axis_nm = cases[case_id]
    step_nm = long_axis_nm / (12.5 * aspect_ratio)

    print("=" * 70, flush=True)
    print(f"CASE {case_id + 1}/{len(cases)} | aspect ratio = {aspect_ratio:g} | long axis = {long_axis_nm:.6f} nm", flush=True)
    print(f"Slurm job: {os.environ.get('SLURM_JOB_ID', 'direct execution')}", flush=True)
    print(f"Discretization step: {step_nm:.6f} nm", flush=True)
    print("=" * 70, flush=True)

    density = 2200.0
    n_silica = 1.444
    wavelength_nm = 1550.0
    NA = 0.75
    f_mm = 3.0
    filling_factor = 1.0
    optical_power_W = 1.0
    equilibrium_z_nm = 0.0
    n_medium = 1.0
    impedance_ohms = 376.730313668

    print("[SETUP] Preparing optical parameters...", flush=True)
    w0_mm = input_beam_waist_from_filling_factor(
        filling_factor=filling_factor,
        numerical_aperture=NA,
        focal_length_mm=f_mm,
        refractive_index=n_medium
    )
    profile_power_integral_m2 = np.pi * (w0_mm * 1e-3)**2 / 2
    amplitude_v_per_m = electric_field_amplitude_from_power(
        power_w=optical_power_W,
        profile_power_integral_m2=profile_power_integral_m2,
        impedance_ohms=impedance_ohms
    )

    hermitegauss_params = dict(
        wavelength_nm=wavelength_nm,
        environment={"eps1": n_medium**2, "eps2": n_medium**2, "eps3": n_medium**2},
        amplitude_v_per_m=amplitude_v_per_m,
        theta=0, NA=NA, f=f_mm, w0=w0_mm,
        xSpot=0, ySpot=0, zSpot=0, kSign=-1, phase=0,
        quadrature_points=24, N_cpu=4
    )

    print("[THEORY] Evaluating the peak field and theoretical frequency...", flush=True)
    peak_efield = focused_peak_efield(hermitegauss_params, equilibrium_z_nm)
    theoretical_freq_kHz = theoretical_librational_frequency(
        long_axis_nm=long_axis_nm,
        aspect_ratio=aspect_ratio,
        peak_efield=peak_efield,
        density=density,
        n_silica=n_silica,
        n_medium=n_medium
    )
    print(f"Theoretical frequency: {theoretical_freq_kHz:.6f} kHz", flush=True)

    simulation_freq_kHz, n_dipoles = numerical_librational_frequency(
        long_axis_nm=long_axis_nm,
        aspect_ratio=aspect_ratio,
        step_nm=step_nm,
        density=density,
        equilibrium_z_nm=equilibrium_z_nm,
        hermitegauss_params=hermitegauss_params
    )
    relative_difference_pct = 100 * abs(simulation_freq_kHz - theoretical_freq_kHz) / theoretical_freq_kHz

    result = {
        "long_axis_nm": long_axis_nm,
        "step_nm": step_nm,
        "n_dipoles": n_dipoles,
        "aspect_ratio": aspect_ratio,
        "theoretical_freq_kHz": theoretical_freq_kHz,
        "simulation_freq_kHz": simulation_freq_kHz,
        "relative_difference_pct": relative_difference_pct,
    }

    case_csv = output_dir / f"case_{case_id:02d}.csv"
    pd.DataFrame([result]).to_csv(case_csv, index=False)

    print("[OUTPUT] Case complete.", flush=True)
    print(f"Relative difference: {relative_difference_pct:.4f} %", flush=True)
    print(f"Case CSV: {case_csv}", flush=True)
    print(f"Total elapsed time: {(perf_counter() - start) / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()