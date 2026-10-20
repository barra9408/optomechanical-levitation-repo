"""Build Y and Z comparison CSVs after the 44 size tasks finish."""

import argparse
import csv
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "recoil_heating_results"
DEFAULT_Y_OUTPUT = ROOT / "outputs" / "recoil_heating_yaxis_aspectratio2.csv"
DEFAULT_Z_OUTPUT = ROOT / "outputs" / "recoil_heating_zaxis_aspectratio2.csv"
EXPECTED_SIZES = 44
BEAMS = ("plane_wave_x", "hermite_gauss_00_x", "laguerre_gauss_01", "laguerre_gauss_02")
RESULT_COLUMNS = (
    "long_axis_nm", "beam", "direction_x", "direction_y", "direction_z",
    "incident_amplitude_v_per_m", "numerical_source", "D", "anisotropy_term",
    "theoretical_total_torque", "derivative_total_torque", "spectral_total_torque",
    "spin_term", "orbital_term", "correlation_term",
    "derivative_gamma_sc", "spectral_gamma_sc",
    "derivative_recoil_heating_per_photon", "spectral_recoil_heating_per_photon",
)
TORQUE_COLUMNS = ("theoretical_total_torque", "derivative_total_torque", "spectral_total_torque")
RATE_COLUMNS = RESULT_COLUMNS[-4:]


def _size_key(row: dict[str, str]) -> float:
    return round(float(row["long_axis_nm"]), 6)


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
            raise ValueError(f"Unexpected columns in {path}")
        rows = list(reader)
    for row in rows:
        if not all(row[column] and math.isfinite(float(row[column])) for column in TORQUE_COLUMNS):
            raise ValueError(f"Missing or nonfinite torque result in {path}, {row['beam']}")
        if not all(row[column] and math.isfinite(float(row[column])) and float(row[column]) > 0
                   for column in RATE_COLUMNS):
            raise ValueError(f"Missing or invalid scattering rate in {path}, {row['beam']}")
    return rows


def _group_rows(rows: list[dict[str, str]], beams: tuple[str, ...], source: Path) -> dict[float, dict[str, dict[str, str]]]:
    grouped = {}
    for row in rows:
        size = _size_key(row)
        if row["beam"] not in beams or row["beam"] in grouped.setdefault(size, {}):
            raise ValueError(f"Unexpected or duplicate beam in {source} at {size} nm")
        grouped[size][row["beam"]] = row
    if len(grouped) != EXPECTED_SIZES or any(set(group) != set(beams) for group in grouped.values()):
        raise ValueError(f"Expected {EXPECTED_SIZES} sizes with {len(beams)} beams each in {source}")
    return grouped


def _per_size_rows(input_dir: Path, axis: str, beams: tuple[str, ...]) -> dict[float, dict[str, dict[str, str]]]:
    paths = sorted(input_dir.glob(f"beamwise_recoilheating_{axis}_*nm.csv"))
    if len(paths) != EXPECTED_SIZES:
        raise ValueError(f"Expected {EXPECTED_SIZES} {axis.upper()} files in {input_dir}; found {len(paths)}")
    rows = []
    for path in paths:
        file_rows = _read_rows(path)
        if len(file_rows) != len(beams) or len({_size_key(row) for row in file_rows}) != 1:
            raise ValueError(f"Incomplete or mixed-size rows in {path}")
        rows.extend(file_rows)
    return _group_rows(rows, beams, input_dir)


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def combine(input_dir: Path = DEFAULT_INPUT_DIR, y_output: Path = DEFAULT_Y_OUTPUT,
            z_output: Path = DEFAULT_Z_OUTPUT) -> tuple[Path, Path]:
    """Collect all four beams for each axis at the selected aspect ratio."""
    y = _per_size_rows(input_dir, "y", BEAMS)
    z = _per_size_rows(input_dir, "z", BEAMS)
    if set(y) != set(z):
        raise ValueError("The Y and Z files do not have the same 44 sizes")
    y_rows = [y[size][beam] for size in sorted(y) for beam in BEAMS]
    z_rows = [z[size][beam] for size in sorted(z) for beam in BEAMS]
    _write_rows(y_output, y_rows)
    _write_rows(z_output, z_rows)
    return y_output, z_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aspect-ratio", type=float, choices=(2.0, 4.0), default=2.0)
    args = parser.parse_args()
    ratio = int(args.aspect_ratio)
    input_dir = DEFAULT_INPUT_DIR if ratio == 2 else DEFAULT_INPUT_DIR / "ar4"
    y_output = ROOT / "outputs" / f"recoil_heating_yaxis_aspectratio{ratio}.csv"
    z_output = ROOT / "outputs" / f"recoil_heating_zaxis_aspectratio{ratio}.csv"
    for path in combine(input_dir, y_output, z_output):
        print(path)


if __name__ == "__main__":
    main()
