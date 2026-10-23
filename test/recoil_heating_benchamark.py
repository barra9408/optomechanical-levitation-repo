"""Benchmark the field-operator recoil calculation on the 44 size-sweep cases.

Each array task selects one long-axis length and evaluates aspect ratios 2 and 4,
four incident beams, and torque axes Y and Z. Results are saved after each beam.
The last completed task combines all 704 rows in outputs/Recoil Heating Benchmark.csv.
"""

import argparse
import csv
import fcntl
import math
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import optomechanics_levitation as oml
from optomechanics_levitation.light_matter_interaction import LightMatterInteraction
from optomechanics_levitation.recoil_heating import TorqueRecoilHeating

from recoil_heating_size_sweep import BEAM_NAMES, SizeSweep, long_axis_values_nm


ASPECT_RATIOS = (2.0, 4.0)
TORQUE_AXES = ("y", "z")
OUTPUT_ROOT = Path(__file__).resolve().parents[1] / "outputs"
OUTPUT_DIR = OUTPUT_ROOT / "recoil_heating_benchmark"
COMBINED_PATH = OUTPUT_ROOT / "Recoil Heating Benchmark.csv"
RESULT_COLUMNS = (
    "long_axis_nm", "aspect_ratio", "torque_axis", "beam",
    "total_torque", "spin_term", "orbital_term", "correlation_term",
    "gamma_sc", "recoil_heating_per_photon",
)
NUMERIC_COLUMNS = RESULT_COLUMNS[:2] + RESULT_COLUMNS[4:]


class BenchmarkTorqueRecoilHeating(TorqueRecoilHeating):
    """Use one interaction-matrix factorization for all inputs of a particle."""

    def __init__(self, torque_axis: str, interaction: LightMatterInteraction) -> None:
        super().__init__(torque_axis)
        self.interaction = interaction

    def _run_simulation(self, particle: dict, in_efield: np.ndarray) -> np.ndarray:
        self.interaction.internal_field(in_efield)
        center_nm = np.mean(particle["geometry_nm"], axis=0)
        positions_nm = center_nm + self.observation_offsets_nm
        field = np.empty((len(positions_nm), 3), dtype=complex)
        for start in range(0, len(positions_nm), oml.parameters.CHUNK_SIZE):
            stop = min(start + oml.parameters.CHUNK_SIZE, len(positions_nm))
            field[start:stop] = self.interaction.far_field(positions_nm[start:stop], field="E")
        return field.reshape(self.N_theta, self.N_phi, 3)


def output_path(case_id: int, aspect_ratio: float, axis: str) -> Path:
    size = float(long_axis_values_nm()[case_id])
    return OUTPUT_DIR / f"ar{aspect_ratio:g}" / f"beamwise_recoilheating_{axis}_{size:.6f}nm.csv"


def read_rows(path: Path, case_id: int, aspect_ratio: float, axis: str) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}")
        rows = list(reader)
    expected_size = float(long_axis_values_nm()[case_id])
    by_beam = {}
    for row in rows:
        beam = row["beam"]
        if (beam not in BEAM_NAMES or beam in by_beam or row["torque_axis"] != axis
                or not math.isclose(float(row["long_axis_nm"]), expected_size, rel_tol=0, abs_tol=1e-6)
                or float(row["aspect_ratio"]) != aspect_ratio
                or not all(row[column] and math.isfinite(float(row[column])) for column in NUMERIC_COLUMNS)
                or float(row["gamma_sc"]) <= 0):
            raise ValueError(f"Invalid result in {path} for beam {beam}")
        by_beam[beam] = row
    return by_beam


def write_rows(path: Path, rows: dict[str, dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows[beam] for beam in BEAM_NAMES if beam in rows)
    os.replace(temporary, path)


def run_case(case_id: int, aspect_ratio: float) -> None:
    size = float(long_axis_values_nm()[case_id])
    reference = SizeSweep(case_id, size, aspect_ratio)
    rows = {axis: read_rows(output_path(case_id, aspect_ratio, axis), case_id, aspect_ratio, axis)
            for axis in TORQUE_AXES}
    if all(len(rows[axis]) == len(BEAM_NAMES) for axis in TORQUE_AXES):
        print(f"[COMPLETE] case={case_id:02d}, AR={aspect_ratio:g}", flush=True)
        return

    interaction = LightMatterInteraction(reference.particle, oml.parameters.WAVELENGTH_NM, oml.parameters.ENVIRONMENT)
    calculators = {axis: BenchmarkTorqueRecoilHeating(axis, interaction) for axis in TORQUE_AXES}
    print(f"[CASE {case_id + 1}/44] long axis={size:.6f} nm, AR={aspect_ratio:g}, "
          f"dipoles={len(reference.particle['geometry_nm'])}", flush=True)
    for beam_name in BEAM_NAMES:
        missing_axes = [axis for axis in TORQUE_AXES if beam_name not in rows[axis]]
        if not missing_axes:
            continue
        started = perf_counter()
        beam, _, _ = reference._beam(beam_name)
        field = reference._local_field(beam)
        amplitude = reference._incident_amplitude(beam_name, beam)
        parameters = {
            "wavelength_nm": oml.parameters.WAVELENGTH_NM,
            "environment": oml.parameters.ENVIRONMENT,
            "amplitude_v_per_m": amplitude,
        }
        for axis in missing_axes:
            total, spin, orbital, correlation, per_photon, gamma_sc = calculators[axis]._torque_recoil_heating(
                reference.particle, field, parameters)
            rows[axis][beam_name] = {
                "long_axis_nm": size, "aspect_ratio": aspect_ratio,
                "torque_axis": axis, "beam": beam_name,
                "total_torque": float(total), "spin_term": float(spin),
                "orbital_term": float(orbital), "correlation_term": float(correlation),
                "gamma_sc": float(gamma_sc), "recoil_heating_per_photon": float(per_photon),
            }
            path = output_path(case_id, aspect_ratio, axis)
            write_rows(path, rows[axis])
            print(f"[{beam_name}, {axis}] S_tau_tau={total:.6e}, Gamma_sc={gamma_sc:.6e}, "
                  f"saved={path}", flush=True)
        print(f"[{beam_name}] elapsed={(perf_counter() - started) / 60:.1f} min", flush=True)


def combine() -> Path:
    all_rows = []
    for case_id in range(len(long_axis_values_nm())):
        for aspect_ratio in ASPECT_RATIOS:
            for axis in TORQUE_AXES:
                path = output_path(case_id, aspect_ratio, axis)
                rows = read_rows(path, case_id, aspect_ratio, axis)
                if len(rows) != len(BEAM_NAMES):
                    raise ValueError(f"Missing results in {path}: {len(rows)}/{len(BEAM_NAMES)} beams")
                all_rows.extend(rows[beam] for beam in BEAM_NAMES)

    temporary = COMBINED_PATH.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(temporary, COMBINED_PATH)
    return COMBINED_PATH


def combine_if_complete() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_DIR / ".combine.lock").open("w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        for case_id in range(len(long_axis_values_nm())):
            for aspect_ratio in ASPECT_RATIOS:
                for axis in TORQUE_AXES:
                    if len(read_rows(output_path(case_id, aspect_ratio, axis), case_id, aspect_ratio, axis)) != len(BEAM_NAMES):
                        return
        print(f"[COMBINED] {combine()}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=None, help="Override SLURM_ARRAY_TASK_ID")
    parser.add_argument("--list-cases", action="store_true", help="List the 44 long-axis lengths")
    parser.add_argument("--combine", action="store_true", help="Combine all complete individual CSV files")
    args = parser.parse_args()

    if args.list_cases:
        for case_id, size in enumerate(long_axis_values_nm()):
            print(f"{case_id:02d}, {size:.12f} nm")
        return
    if args.combine:
        print(combine())
        return

    case_id = args.case_id if args.case_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= case_id < len(long_axis_values_nm()):
        parser.error(f"case ID must be between 0 and {len(long_axis_values_nm()) - 1}")
    for aspect_ratio in ASPECT_RATIOS:
        run_case(case_id, aspect_ratio)
    combine_if_complete()


if __name__ == "__main__":
    main()
