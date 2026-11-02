"""Torque recoil heating for four beams, two axes, and three aspect ratios."""

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.constants import hbar
from cases import plane_wave, lg_00, lg_01, lg_02
from optomechanics_levitation import parameters as p
from optomechanics_levitation.operators import spin_operator, orbital_operator
from optomechanics_levitation.particle import particle_properties
from optomechanics_levitation.recoil_heating import TheoreticalTorqueRecoilHeating, TorqueRecoilHeating

BEAMS = {"plane_wave": plane_wave, "lg_00": lg_00, "lg_01": lg_01, "lg_02": lg_02}
RESULT_COLUMNS = (
    "long_axis_nm", "aspect_ratio", "size_param", "torque_axis", "beam",
    "theoretical_recoil_heating", "simulation_recoil_heating", "spin_term",
    "orbital_term", "correlation_term", "relative_difference",
    "mean_scattering_rate", "d_mu",
)
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "recoil_heating_results" / "transfer_amplitudes"
PARTS_DIR = OUTPUT_DIR / "parts"
BACKFILL_DIR = OUTPUT_DIR / "backfill"
OUTPUT_FILE = Path(__file__).resolve().parent / "size_sweep.csv"

def long_axis_values_nm() -> np.ndarray:
    """Return 40 equal intervals from 25 nm to the Rayleigh range."""
    waist_nm = p.WAVELENGTH_NM / (np.pi * p.NUMERICAL_APERTURE)
    rayleigh_range_nm = np.pi * waist_nm**2 / p.WAVELENGTH_NM
    return np.linspace(p.SIZE_SWEEP_START_NM, rayleigh_range_nm, p.SIZE_SWEEP_INTERVALS + 1)

def particle_for_size(long_axis_nm: float, aspect_ratio: float) -> dict:
    """Keep the spheroid mesh used in the previous size sweep."""
    divisions = p.SIZE_SWEEP_SPHERE_MESH_DIVISIONS if aspect_ratio == 1.0 else p.SIZE_SWEEP_MESH_DIVISIONS
    step_nm = long_axis_nm / divisions
    long_radius_nm = long_axis_nm / 2
    short_radius_nm = long_radius_nm / aspect_ratio
    return particle_properties(particle_type=p.PARTICLE_TYPE, material_name=p.MATERIAL, step_nm=step_nm, mesh=p.MESH, center=True,
        R1=long_radius_nm / step_nm - 0.2, R2=short_radius_nm / step_nm - 0.2, R3=short_radius_nm / step_nm - 0.2)

def mean_scattering_rate(calculator) -> float:
    """Integrate the unrotated scattered field to obtain Gamma_sc [s^-1]."""
    positions_nm, _, domega = calculator._observation_surface("z")
    incident_field = calculator._incident_field(calculator.positions_nm)
    scattered_field = calculator._run_simulation(incident_field, positions_nm)
    field_to_rate = TorqueRecoilHeating(calculator).field_to_rate
    return float(field_to_rate * np.sum(np.sum(np.abs(scattered_field)**2, axis=-1) * domega))

def d_mu(recoil_heating: float, gamma_sc: float) -> float:
    """Use the paper's relation between torque recoil and information per photon."""
    return float(2 * np.pi * recoil_heating / (hbar**2 * gamma_sc))

def run_case(case_id: int) -> None:
    """Calculate every beam, axis, and aspect ratio for one particle size."""
    long_axis_nm = float(long_axis_values_nm()[case_id])
    output = PARTS_DIR / f"size_{case_id:03d}.csv"
    if output.exists():
        print(f"[SIZE {case_id + 1}/{p.SIZE_SWEEP_INTERVALS + 1}] Already saved: {output}", flush=True)
        return

    started = perf_counter()
    rows = []
    print(f"[SIZE {case_id + 1}/{p.SIZE_SWEEP_INTERVALS + 1}] long axis={long_axis_nm:.6f} nm", flush=True)
    for aspect_ratio in p.SIZE_SWEEP_ASPECT_RATIOS:
        particle = particle_for_size(long_axis_nm, aspect_ratio)
        print(f"[r={aspect_ratio:g}] dipoles={len(particle['geometry_nm'])}", flush=True)
        theoretical = None
        if aspect_ratio == 2.0:
            epsilon_r = particle["material"].epsilon(p.WAVELENGTH_NM).real
            theoretical = TheoreticalTorqueRecoilHeating(long_axis_nm / 2, long_axis_nm / (2 * aspect_ratio), epsilon_r).compute(torque_axis="z")["total_torque"]

        for beam_name, beam in BEAMS.items():
            calculator = beam.transfer_amplitudes(particle)
            heating = TorqueRecoilHeating(calculator)
            gamma_sc = mean_scattering_rate(calculator)
            for axis in p.SIZE_SWEEP_TORQUE_AXES:
                print(f"[r={aspect_ratio:g}, {beam_name}, {axis}] transfer amplitudes", flush=True)
                q_S = calculator.transfer_amplitude(spin_operator(axis))
                q_L = calculator.transfer_amplitude(orbital_operator(axis))
                result = heating.compute(q_S, q_L)
                theory = theoretical if beam_name == "plane_wave" and axis == "z" else None
                simulation = result["total_torque"]
                rows.append({
                    "long_axis_nm": long_axis_nm,
                    "aspect_ratio": aspect_ratio,
                    "size_param": aspect_ratio**(2 / 3) * long_axis_nm,
                    "torque_axis": axis,
                    "beam": beam_name,
                    "theoretical_recoil_heating": theory,
                    "simulation_recoil_heating": simulation,
                    "spin_term": result["q_S"],
                    "orbital_term": result["q_L"],
                    "correlation_term": result["q_LS"],
                    "relative_difference": abs(simulation - theory) / abs(theory) if theory is not None else None,
                    "mean_scattering_rate": gamma_sc,
                    "d_mu": d_mu(simulation, gamma_sc),
                })
                print(f"[r={aspect_ratio:g}, {beam_name}, {axis}] total={simulation:.6e}", flush=True)

    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    print(f"[COMPLETE] {output} in {(perf_counter() - started) / 60:.1f} min", flush=True)

def backfill_case(case_id: int) -> None:
    """Add scattering rate and D_mu to one size without recalculating recoil."""
    output = BACKFILL_DIR / f"size_{case_id:03d}.csv"
    if output.exists():
        print(f"[SIZE {case_id + 1}] Already saved: {output}", flush=True)
        return

    long_axis_nm = float(long_axis_values_nm()[case_id])
    with OUTPUT_FILE.open(newline="") as source:
        rows = [row for row in csv.DictReader(source)
                if np.isclose(float(row["long_axis_nm"]), long_axis_nm, rtol=0, atol=1e-9)]
    expected_rows = len(p.SIZE_SWEEP_ASPECT_RATIOS) * len(p.SIZE_SWEEP_TORQUE_AXES) * len(BEAMS)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} saved rows for size {case_id}, found {len(rows)}")

    for aspect_ratio in p.SIZE_SWEEP_ASPECT_RATIOS:
        particle = particle_for_size(long_axis_nm, aspect_ratio)
        for beam_name, beam in BEAMS.items():
            gamma_sc = mean_scattering_rate(beam.transfer_amplitudes(particle))
            for row in rows:
                if float(row["aspect_ratio"]) == aspect_ratio and row["beam"] == beam_name:
                    row["mean_scattering_rate"] = gamma_sc
                    row["d_mu"] = d_mu(float(row["simulation_recoil_heating"]), gamma_sc)
            print(f"[SIZE {case_id + 1}, r={aspect_ratio:g}, {beam_name}] Gamma_sc={gamma_sc:.6e}", flush=True)

    BACKFILL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output)
    print(f"[COMPLETE] {output}", flush=True)

def combine() -> None:
    """Join the completed size jobs into one CSV."""
    temporary = OUTPUT_FILE.with_suffix(".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for case_id in range(p.SIZE_SWEEP_INTERVALS + 1):
            with (PARTS_DIR / f"size_{case_id:03d}.csv").open(newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != list(RESULT_COLUMNS):
                    raise ValueError(f"Invalid columns in size_{case_id:03d}.csv")
                rows = list(reader)
                expected_rows = len(p.SIZE_SWEEP_ASPECT_RATIOS) * len(p.SIZE_SWEEP_TORQUE_AXES) * len(BEAMS)
                if len(rows) != expected_rows:
                    raise ValueError(f"Invalid number of rows in size_{case_id:03d}.csv")
                writer.writerows(rows)
    os.replace(temporary, OUTPUT_FILE)
    print(f"[COMPLETE] {OUTPUT_FILE}", flush=True)

def combine_backfill() -> None:
    """Replace the existing CSV after all enriched size files are complete."""
    temporary = OUTPUT_FILE.with_suffix(".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        for case_id in range(p.SIZE_SWEEP_INTERVALS + 1):
            with (BACKFILL_DIR / f"size_{case_id:03d}.csv").open(newline="") as source:
                reader = csv.DictReader(source)
                if reader.fieldnames != list(RESULT_COLUMNS):
                    raise ValueError(f"Invalid columns in backfill size_{case_id:03d}.csv")
                rows = list(reader)
                expected_rows = len(p.SIZE_SWEEP_ASPECT_RATIOS) * len(p.SIZE_SWEEP_TORQUE_AXES) * len(BEAMS)
                if len(rows) != expected_rows or any(not row["mean_scattering_rate"] or not row["d_mu"] for row in rows):
                    raise ValueError(f"Incomplete backfill size_{case_id:03d}.csv")
                writer.writerows(rows)
    os.replace(temporary, OUTPUT_FILE)
    print(f"[COMPLETE] {OUTPUT_FILE}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, help="Calculate one particle size")
    parser.add_argument("--backfill-case-id", type=int, help="Add Gamma_sc and D_mu to one saved size")
    parser.add_argument("--list-cases", action="store_true", help="List the 41 particle sizes")
    parser.add_argument("--combine", action="store_true", help="Combine all size files into one CSV")
    parser.add_argument("--combine-backfill", action="store_true", help="Combine enriched size files into the existing CSV")
    args = parser.parse_args()
    if args.list_cases:
        for case_id, long_axis_nm in enumerate(long_axis_values_nm()):
            print(f"{case_id:02d}, {long_axis_nm:.12f} nm")
    elif args.combine:
        combine()
    elif args.combine_backfill:
        combine_backfill()
    elif args.backfill_case_id is not None:
        backfill_case(args.backfill_case_id)
    elif args.case_id is not None:
        run_case(args.case_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
