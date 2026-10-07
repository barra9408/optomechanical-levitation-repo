import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.constants import epsilon_0, c, hbar

from optomechanics_levitation import (
    particle_properties, input_beam_waist_from_filling_factor,
    electric_field_amplitude_from_power, evaluate_incident_field,
    hermite_gauss_00, evaluate_internal_field, evaluate_far_field,
)
from optomechanics_levitation.particle import rotate_particle


def torque_noise(interaction_result, rotation_axis="y", radius_wavelengths=100.0, n_theta=120, n_phi=240, chunk_size=256):
    prepared = interaction_result["prepared_interaction"]
    wavelength_nm = prepared["wavelength_nm"]
    center_nm = np.mean(prepared["structure"].geometry, axis=0)
    radius_nm = radius_wavelengths * wavelength_nm
    omega = 2 * np.pi * c / (wavelength_nm * 1e-9)

    # Spherical coordinates with the polar axis aligned with the rotation axis.
    axis_index = {"x": 0, "y": 1, "z": 2}[rotation_axis.lower()]
    basis = np.eye(3)[[(axis_index + 1) % 3, (axis_index + 2) % 3, axis_index]]
    theta = (np.arange(n_theta) + 0.5) * np.pi / n_theta
    phi = np.arange(n_phi) * 2 * np.pi / n_phi
    Theta, Phi = np.meshgrid(theta, phi, indexing="ij")

    normals = np.stack((np.sin(Theta) * np.cos(Phi), np.sin(Theta) * np.sin(Phi), np.cos(Theta)), axis=-1) @ basis
    positions_nm = (center_nm + radius_nm * normals).reshape(-1, 3)
    solid_angle = np.sin(Theta) * (np.pi / n_theta) * (2 * np.pi / n_phi)

    scattered_e = np.empty((len(positions_nm), 3), dtype=complex)
    for start in range(0, len(positions_nm), chunk_size):
        stop = min(start + chunk_size, len(positions_nm))
        scattered_e[start:stop] = evaluate_far_field(interaction_result, positions_nm[start:stop], field="E")
    scattered_e = scattered_e.reshape(n_theta, n_phi, 3)

    # Periodic derivative of the complex Cartesian field components.
    azimuthal_modes = np.fft.fftfreq(n_phi, d=1.0 / n_phi)
    field_modes = np.fft.fft(scattered_e, axis=1)
    derivative_phi = np.fft.ifft(1j * azimuthal_modes[None, :, None] * field_modes, axis=1)

    factor = hbar * epsilon_0 * c * (radius_nm * 1e-9)**2 / (2 * omega)
    intensity = np.sum(np.abs(scattered_e)**2, axis=-1)
    derivative_intensity = np.sum(np.abs(derivative_phi)**2, axis=-1)

    spin_channel = factor * np.sum(np.cos(Theta)**2 * intensity * solid_angle)
    orbital_channel = factor * np.sum(derivative_intensity * solid_angle)
    total_torque_noise = orbital_channel + spin_channel

    return total_torque_noise, orbital_channel, spin_channel


def librational_heating_rates(total_torque_noise, librational_freq_kHz, moment_of_inertia):
    librational_omega = 2 * np.pi * librational_freq_kHz * 1e3
    return np.pi * total_torque_noise / (moment_of_inertia * hbar * librational_omega)

def main():
    start = perf_counter()
    repository_dir = Path(__file__).resolve().parents[1]
    input_dir = repository_dir / "outputs" / "librational_frequency_big_particles"
    output_dir = repository_dir / "outputs" / "recoil_heating_big_particles"
    output_dir.mkdir(parents=True, exist_ok=True)

    wavelength_nm = 1550.0
    NA = 0.75
    f_mm = 3.0
    filling_factor = 1.0
    optical_power_W = 1.0
    equilibrium_z_nm = 0.0
    n_medium = 1.0
    density = 2200.0
    impedance_ohms = 376.730313668

    # Same sizes and aspect ratios as the librational frequency script.
    focal_waist_nm = wavelength_nm / (np.pi * NA)
    rayleigh_range_nm = np.pi * focal_waist_nm**2 / wavelength_nm
    long_axis_values_nm = np.linspace(100.0, rayleigh_range_nm, 20)
    cases = [(aspect_ratio, float(long_axis_nm)) for aspect_ratio in (2.0, 4.0) for long_axis_nm in long_axis_values_nm]

    case_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    aspect_ratio, long_axis_nm = cases[case_id]
    long_axis_label = f"{long_axis_nm:.6f}".replace(".", "p")
    case_stem = f"L_{long_axis_label}_r{aspect_ratio:g}"

    print(f"[CASE {case_id + 1}/{len(cases)}] {case_stem}", flush=True)

    w0_mm = input_beam_waist_from_filling_factor(filling_factor, NA, f_mm, n_medium)
    profile_power_integral_m2 = np.pi * (w0_mm * 1e-3)**2 / 2
    amplitude_v_per_m = electric_field_amplitude_from_power(optical_power_W, profile_power_integral_m2, impedance_ohms)

    hermitegauss_params = dict(
        wavelength_nm=wavelength_nm,
        environment={"eps1": n_medium**2, "eps2": n_medium**2, "eps3": n_medium**2},
        amplitude_v_per_m=amplitude_v_per_m,
        theta=0, NA=NA, f=f_mm, w0=w0_mm,
        xSpot=0, ySpot=0, zSpot=0, kSign=-1, phase=0,
        quadrature_points=24, N_cpu=4,
    )

    # Z/Y is excluded.
    for orientation, rotation_axes in (("X", ("Y", "Z")), ("Z", ("X",))):
        reference_csv = input_dir / f"case_{orientation}_or_{rotation_axes[0]}_rot_{case_stem}.csv"
        reference = pd.read_csv(reference_csv).iloc[0]

        particle_long_axis_nm = float(reference["long_axis_nm"])
        particle_aspect_ratio = float(reference["aspect_ratio"])
        step_nm = float(reference["step_nm"])
        long_radius_nm = particle_long_axis_nm / 2
        short_radius_nm = long_radius_nm / particle_aspect_ratio

        particle = particle_properties(
            particle_type="spheroid", material_name="sio2", step_nm=step_nm,
            mesh="hex", center=True,
            R1=long_radius_nm / step_nm,
            R2=short_radius_nm / step_nm,
            R3=short_radius_nm / step_nm,
        )

        if orientation == "Z":
            particle["geometry_nm"] = rotate_particle(particle["geometry_nm"], np.array([0.0, -90.0, 0.0]))
        particle["geometry_nm"] = particle["geometry_nm"] + np.array([0.0, 0.0, equilibrium_z_nm])

        long_radius_m = long_radius_nm * 1e-9
        short_radius_m = short_radius_nm * 1e-9
        mass_kg = density * 4 * np.pi * long_radius_m * short_radius_m**2 / 3
        moment_of_inertia = mass_kg * (long_radius_m**2 + short_radius_m**2) / 5

        print(f"[ORIENTATION {orientation}] Solving the internal field at angle = 0 deg...", flush=True)
        incident_e = evaluate_incident_field(field_generator=hermite_gauss_00, positions_nm=particle["geometry_nm"], field="E", **hermitegauss_params)
        interaction_result = evaluate_internal_field(
            particle=particle, incident_electric_field=incident_e,
            wavelength_nm=wavelength_nm, environment=hermitegauss_params["environment"],
        )

        for rotation_axis in rotation_axes:
            input_csv = input_dir / f"case_{orientation}_or_{rotation_axis}_rot_{case_stem}.csv"
            row = pd.read_csv(input_csv).iloc[0].to_dict()
            row["n_dipoles"] = int(row["n_dipoles"])
            simulation_frequency_kHz = float(row["simulation_frequency_kHz"])
            stiffness = float(row["stiffness_Nm_per_rad"])

            print(f"[MODE {orientation}/{rotation_axis}] Calculating torque noise...", flush=True)
            total_torque_noise, orbital_channel, spin_channel = torque_noise(
                interaction_result, rotation_axis=rotation_axis,
                radius_wavelengths=100.0, n_theta=120, n_phi=240, chunk_size=256,
            )

            heating_rate = librational_heating_rates(total_torque_noise, simulation_frequency_kHz, moment_of_inertia) if stiffness > 0 else np.nan

            # Preserve the original CSV columns and add noise and heating.
            row["orbital_channel_N2m2_per_Hz"] = orbital_channel
            row["spin_channel_N2m2_per_Hz"] = spin_channel
            row["total_torque_noise_N2m2_per_Hz"] = total_torque_noise
            row["librational_heating_rate_per_s"] = heating_rate

            output_csv = output_dir / input_csv.name
            pd.DataFrame([row]).to_csv(output_csv, index=False)

            print(f"[RESULT {orientation}/{rotation_axis}] heating rate = {heating_rate:.6e} s^-1", flush=True)
            print(f"[OUTPUT] {output_csv}", flush=True)

    print(f"[COMPLETE] elapsed = {(perf_counter() - start) / 60:.2f} min", flush=True)


if __name__ == "__main__":
    main()