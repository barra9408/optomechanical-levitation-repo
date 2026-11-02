"""Scattered torque information patterns for focused plane-wave and LG02 beams."""

import argparse
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
from cases import plane_wave, lg_02
from optomechanics_levitation import parameters as p
from optomechanics_levitation.operators import spin_operator, orbital_operator
from optomechanics_levitation.particle import particle_properties
from optomechanics_levitation.recoil_heating import TorqueRecoilHeating

LONG_AXES_NM = (100.0, 877.0)
ASPECT_RATIO = 4.0
BEAMS = {"plane_wave": plane_wave, "lg_02": lg_02}
CASES = tuple((beam_name, beam, long_axis_nm) for beam_name, beam in BEAMS.items()
              for long_axis_nm in LONG_AXES_NM)
CSV_COLUMNS = ("surface", "direction_x", "direction_y", "direction_z", "spin", "orbital", "correlation", "total")
OUTPUT_DIR = Path(__file__).resolve().parent


def run_case(case_id: int) -> None:
    """Save one scattered angular pattern for a beam and particle size."""
    beam_name, beam, long_axis_nm = CASES[case_id]
    output = OUTPUT_DIR / f"{beam_name}_info_patterns_{long_axis_nm:g}nm_ar{ASPECT_RATIO:g}.csv"
    if output.exists():
        print(f"[{beam_name}, {long_axis_nm:g} nm] Already saved: {output}", flush=True)
        return

    started = perf_counter()
    step_nm = long_axis_nm / p.SIZE_SWEEP_MESH_DIVISIONS
    long_radius_nm = long_axis_nm / 2
    short_radius_nm = long_radius_nm / ASPECT_RATIO
    particle = particle_properties(particle_type=p.PARTICLE_TYPE, material_name=p.MATERIAL,
        step_nm=step_nm, mesh=p.MESH, center=True,
        R1=long_radius_nm / step_nm - 0.2,
        R2=short_radius_nm / step_nm - 0.2, R3=short_radius_nm / step_nm - 0.2)
    print(f"[{beam_name}, {long_axis_nm:g} nm] dipoles={len(particle['geometry_nm'])}, torque=z", flush=True)

    calculator = beam.transfer_amplitudes(particle)
    q_S = calculator.transfer_amplitude(spin_operator("z"))
    print(f"[{beam_name}, {long_axis_nm:g} nm] spin complete", flush=True)
    q_L = calculator.transfer_amplitude(orbital_operator("z"))
    print(f"[{beam_name}, {long_axis_nm:g} nm] orbital complete", flush=True)

    heating = TorqueRecoilHeating(calculator)
    scale = heating.field_to_rate / (2 * np.pi)
    spin = scale * np.sum(np.abs(q_S)**2, axis=-1)
    orbital = scale * np.sum(np.abs(q_L)**2, axis=-1)
    correlation = 2 * scale * np.sum(np.conj(q_S) * q_L, axis=-1).real
    total = spin + orbital + correlation

    temporary = output.with_suffix(".tmp")
    with temporary.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for index in np.ndindex(total.shape):
            direction = calculator.directions[index]
            writer.writerow({"surface": "scattered", "direction_x": direction[0],
                "direction_y": direction[1], "direction_z": direction[2],
                "spin": spin[index], "orbital": orbital[index],
                "correlation": correlation[index], "total": total[index]})
    temporary.replace(output)
    print(f"[{beam_name}, {long_axis_nm:g} nm] {total.size} directions saved in "
          f"{(perf_counter() - started) / 60:.1f} min: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, help="Calculate one beam and particle size")
    parser.add_argument("--list-cases", action="store_true", help="List the four angular patterns")
    args = parser.parse_args()
    if args.list_cases:
        for case_id, (beam_name, _, long_axis_nm) in enumerate(CASES):
            print(f"{case_id:02d}, {beam_name}, {long_axis_nm:g} nm")
    elif args.case_id is not None:
        run_case(args.case_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
