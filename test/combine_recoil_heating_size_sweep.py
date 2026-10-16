"""Combine the 44 size-sweep files into one CSV for local plotting."""

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = ROOT / "outputs" / "recoil_heating_results"
DEFAULT_OUTPUT = ROOT / "outputs" / "three_beam_swept_comparison.csv"
EXPECTED_SIZES = 44
BEAMS = ("plane_wave_x", "hermite_gauss_00_x", "laguerre_gauss_01")
REQUIRED_COLUMNS = ("long_axis_nm", "beam", "theoretical_total_torque",
                    "derivative_total_torque", "spectral_total_torque")


def combine(input_dir: Path, output: Path) -> Path:
    """Require all nine series to be complete before writing the merged CSV."""
    paths = sorted(input_dir.glob("beamwise_recoilheating_*nm.csv"))
    if len(paths) != EXPECTED_SIZES:
        raise ValueError(f"Expected {EXPECTED_SIZES} size files in {input_dir}; found {len(paths)}")
    all_rows = []
    columns = None
    seen_sizes = set()
    for path in paths:
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            if columns is None:
                columns = tuple(reader.fieldnames or ())
            elif tuple(reader.fieldnames or ()) != columns:
                raise ValueError(f"Column mismatch in {path}")
            if not set(REQUIRED_COLUMNS).issubset(columns):
                raise ValueError(f"Missing required columns in {path}")
            rows = list(reader)
        if len(rows) != len(BEAMS) or {row["beam"] for row in rows} != set(BEAMS):
            raise ValueError(f"Expected one row per beam in {path}")
        sizes = {round(float(row["long_axis_nm"]), 6) for row in rows}
        if len(sizes) != 1 or next(iter(sizes)) in seen_sizes:
            raise ValueError(f"Inconsistent or duplicate size in {path}")
        seen_sizes.update(sizes)
        for row in rows:
            for key in REQUIRED_COLUMNS[2:]:
                if not row[key]:
                    raise ValueError(f"Missing {key} in {path}, {row['beam']}")
            all_rows.append(row)
    all_rows.sort(key=lambda row: (float(row["long_axis_nm"]), BEAMS.index(row["beam"])))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(all_rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(combine(args.input_dir, args.output))


if __name__ == "__main__":
    main()
