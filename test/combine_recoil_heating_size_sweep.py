"""Build complete Y and Z comparison CSVs after the 44 array tasks finish."""

import argparse
import csv
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "recoil_heating_results"
DEFAULT_Y_SOURCE = ROOT / "outputs" / "models_comparison.csv"
DEFAULT_Y_OUTPUT = ROOT / "outputs" / "models_comparison_y.csv"
DEFAULT_Z_OUTPUT = ROOT / "outputs" / "models_comparison_z.csv"
EXPECTED_SIZES = 44
OLD_BEAMS = ("plane_wave_x", "hermite_gauss_00_x", "laguerre_gauss_01")
BEAMS = (*OLD_BEAMS, "laguerre_gauss_02")
RESULT_COLUMNS = (
    "long_axis_nm", "beam", "direction_x", "direction_y", "direction_z",
    "incident_amplitude_v_per_m", "numerical_source", "D", "anisotropy_term",
    "theoretical_total_torque", "derivative_total_torque", "spectral_total_torque",
    "spin_term", "orbital_term", "correlation_term",
)
TORQUE_COLUMNS = ("theoretical_total_torque", "derivative_total_torque", "spectral_total_torque")


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


def combine(input_dir: Path = DEFAULT_INPUT_DIR, y_source: Path = DEFAULT_Y_SOURCE,
            y_output: Path = DEFAULT_Y_OUTPUT, z_output: Path = DEFAULT_Z_OUTPUT) -> tuple[Path, Path]:
    """Keep the existing Y data and add LG02; collect all four Z beams."""
    existing_y = _group_rows(_read_rows(y_source), OLD_BEAMS, y_source)
    new_y = _per_size_rows(input_dir, "y", ("laguerre_gauss_02",))
    new_z = _per_size_rows(input_dir, "z", BEAMS)
    if set(existing_y) != set(new_y) or set(existing_y) != set(new_z):
        raise ValueError("The Y source and new Y/Z files do not have the same 44 sizes")
    y_rows = []
    for size in sorted(existing_y):
        for beam in BEAMS:
            source = new_y[size] if beam == "laguerre_gauss_02" else existing_y[size]
            y_rows.append(source[beam])
    z_rows = [new_z[size][beam] for size in sorted(new_z) for beam in BEAMS]
    _write_rows(y_output, y_rows)
    _write_rows(z_output, z_rows)
    return y_output, z_output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--y-source", type=Path, default=DEFAULT_Y_SOURCE)
    parser.add_argument("--y-output", type=Path, default=DEFAULT_Y_OUTPUT)
    parser.add_argument("--z-output", type=Path, default=DEFAULT_Z_OUTPUT)
    args = parser.parse_args()
    for path in combine(args.input_dir, args.y_source, args.y_output, args.z_output):
        print(path)


if __name__ == "__main__":
    main()
