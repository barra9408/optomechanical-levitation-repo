"""Compare recoil-heating methods and angular contribution shares across particle sizes."""

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.constants import c, epsilon_0
import optomechanics_levitation as oml

ASPECT_RATIO = 2.0; MESH_DIVISIONS = 35.0
N_COS_THETA = 12; N_AZIMUTH = 24; THEORY_LIMIT_NM = 157.0
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "recoil_heating_results"
CSV_COLUMNS = ("helicity", "direction_x", "direction_y", "direction_z", "spin", "orbital", "correlation")
RESULT_COLUMNS = ("long_axis_nm", "helicity", "D", "anisotropy_term", "theoretical_torque_total_torque", "derivative_total_torque",
                  "spectral_total_torque", "spin_term", "orbital_term", "correlation_term", "spin_percent", "orbital_percent",
                  "reduction_percent", "increase_percent", "spin_variance", "orbital_variance", "reduction_variance", "increase_variance")

def rayleigh_limit_nm() -> float:
    """Return the Rayleigh range of the configured optical focus in nanometres."""
    waist_nm = oml.parameters.WAVELENGTH_NM / (np.pi * oml.parameters.NUMERICAL_APERTURE)
    return np.pi * waist_nm**2 / oml.parameters.WAVELENGTH_NM

def long_axis_values_nm() -> np.ndarray:
    """Use 15 sizes through 157 nm and 29 new sizes up to the Rayleigh range."""
    return np.r_[np.linspace(25.0, THEORY_LIMIT_NM, 15), np.linspace(THEORY_LIMIT_NM, rayleigh_limit_nm(), 30)[1:]]

class SizeSweep:
    """Calculate one particle size, both fixed helicities, and their angular contributions."""
    def __init__(self, case_id: int, long_axis_nm: float) -> None:
        self.case_id = case_id; self.long_axis_nm = long_axis_nm
        self.step_nm = long_axis_nm / MESH_DIVISIONS; long_radius_nm = long_axis_nm / 2
        short_radius_nm = long_radius_nm / ASPECT_RATIO
        self.particle = oml.particle_properties(particle_type=oml.parameters.PARTICLE_TYPE, material_name=oml.parameters.MATERIAL,
                                               step_nm=self.step_nm, mesh=oml.parameters.MESH, center=True,
                                               R1=long_radius_nm / self.step_nm - 0.2, R2=short_radius_nm / self.step_nm - 0.2,
                                               R3=short_radius_nm / self.step_nm - 0.2)
        self.center_nm = np.mean(self.particle["geometry_nm"], axis=0)
        self.amplitude = np.sqrt(2 * oml.parameters.INTENSITY_W_PER_M2 / (epsilon_0 * c))
        self.beams = {helicity: oml.PlaneWave((0.0, 0.0, 1.0), (1.0, 1j * helicity)).field
                      for helicity in oml.parameters.HELICITIES}
        self.spectra = {}
        self.derivative = oml.DerivativeTorqueRecoilHeating(self.particle, wavelength_nm=oml.parameters.WAVELENGTH_NM,
                                                           environment=oml.parameters.ENVIRONMENT, rotation_axes=(oml.parameters.TORQUE_AXIS,))
        self.spectral = oml.SpectralTorqueRecoilHeating(self.particle, wavelength_nm=oml.parameters.WAVELENGTH_NM,
                                                       environment=oml.parameters.ENVIRONMENT, torque_axis=oml.parameters.TORQUE_AXIS,
                                                       n_theta=oml.parameters.SPECTRAL_N_THETA, n_phi=oml.parameters.SPECTRAL_N_PHI,
                                                       radius_wavelengths=oml.parameters.FAR_FIELD_RADIUS_NM / oml.parameters.WAVELENGTH_NM)
        self.cos_theta = -1 + (np.arange(N_COS_THETA) + 0.5) * 2 / N_COS_THETA
        self.azimuth = (np.arange(N_AZIMUTH) + 0.5) * 2 * np.pi / N_AZIMUTH
        self.output_path = OUTPUT_DIR / f"planewave_recoilheating_{long_axis_nm:.6f}nm.csv"
        self.partial_path = OUTPUT_DIR / f"planewave_recoilheating_{long_axis_nm:.6f}nm.partial.csv"

    ####################################
    # INCIDENT PLANE WAVES
    ####################################
    @staticmethod
    def _beam_frame(direction: np.ndarray) -> np.ndarray:
        """Return the same local frame used by the angular-scan experiment."""
        direction = np.array(direction, dtype=float, copy=True); direction /= np.linalg.norm(direction)
        reference = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = reference - np.dot(reference, direction) * direction; x_axis /= np.linalg.norm(x_axis)
        return np.column_stack((x_axis, np.cross(direction, x_axis), direction))

    def _rotated_field(self, pos: np.ndarray, env_dict: dict, wavelength: float, beam_helicity: int,
                       beam_direction: tuple, beam_origin_nm: tuple, returnField: str = "E") -> np.ndarray:
        """Rotate a circular plane wave from its local frame into the laboratory frame."""
        frame = self._beam_frame(beam_direction)
        local_positions = (np.asarray(pos) - np.asarray(beam_origin_nm)) @ frame
        return self.beams[beam_helicity](local_positions, env_dict, wavelength, returnField=returnField) @ frame.T

    def _spectrum(self, helicity: int) -> dict:
        """Sample each helicity's numerical angular spectrum once per particle size."""
        if helicity not in self.spectra:
            print(f"[SPECTRUM] Sampling numerical modes for σ={helicity:+d}", flush=True)
            self.spectra[helicity] = self.spectral.incident_angular_spectrum(
                self.beams[helicity], amplitude_v_per_m=self.amplitude, center_nm=np.zeros(3),
                input_grid=oml.parameters.INPUT_GRID, input_span_wavelengths=oml.parameters.INPUT_SPAN_WAVELENGTHS)
        return self.spectra[helicity]

    def _spectral_result(self, helicity: int, direction: np.ndarray) -> dict:
        """Calculate the four integrated torque channels for one incident direction."""
        frame = self._beam_frame(direction)
        spectrum = self.spectral.rotate_incident_spectrum(self._spectrum(helicity), frame, self.center_nm, oml.parameters.TORQUE_AXIS)
        return self.spectral.compute(self._rotated_field, amplitude_v_per_m=self.amplitude, incident_spectrum=spectrum,
                                     beam_helicity=helicity, beam_direction=tuple(direction),
                                     beam_origin_nm=tuple(self.center_nm))

    ####################################
    # THREE RECOIL-HEATING METHODS
    ####################################
    def _theoretical_result(self) -> dict | None:
        """Apply the point-dipole comparison only through 157 nm, inclusive."""
        if self.long_axis_nm > THEORY_LIMIT_NM:
            print(f"[THEORY] Skipped above {THEORY_LIMIT_NM:g} nm", flush=True)
            return None
        long_radius_nm = self.long_axis_nm / 2; short_radius_nm = long_radius_nm / ASPECT_RATIO
        epsilon_r = self.particle["material"].epsilon(oml.parameters.WAVELENGTH_NM).real
        result = oml.TheoreticalTorqueRecoilHeating(long_radius_nm, short_radius_nm, epsilon_r).compute()
        print(f"[THEORY] anisotropy={result['anisotropy_term']:.6e}, torque={result['total_torque']:.6e}", flush=True)
        return result

    def _fixed_incidence(self, helicity: int, theory: dict | None) -> dict:
        """Compare derivative and spectral torque for one +z incident helicity."""
        started = perf_counter(); direction = np.array((0.0, 0.0, 1.0))
        print(f"[FIXED σ={helicity:+d}] Derivative model, axis={oml.parameters.TORQUE_AXIS}", flush=True)
        derivative = self.derivative.compute(self._rotated_field, amplitude_v_per_m=self.amplitude, beam_helicity=helicity,
                                             beam_direction=tuple(direction), beam_origin_nm=tuple(self.center_nm))
        print(f"[FIXED σ={helicity:+d}] Spectral model", flush=True)
        spectral = self._spectral_result(helicity, direction)
        axis = oml.parameters.TORQUE_AXIS
        row = {"long_axis_nm": self.long_axis_nm, "helicity": helicity, "D": derivative["D"][axis],
               "anisotropy_term": theory["anisotropy_term"] if theory is not None else None,
               "theoretical_torque_total_torque": theory["total_torque"] if theory is not None else None,
               "derivative_total_torque": derivative["total_torque"][axis],
               "spectral_total_torque": spectral["total_torque"], "spin_term": spectral["spin_term"],
               "orbital_term": spectral["orbital_term"], "correlation_term": spectral["correlation_term"]}
        print(f"[FIXED σ={helicity:+d}] D={row['D']:.6e}, derivative={row['derivative_total_torque']:.6e}, "
              f"spectral={row['spectral_total_torque']:.6e} ({(perf_counter() - started) / 60:.1f} min)", flush=True)
        return row

    ####################################
    # ANGULAR DISTRIBUTION
    ####################################
    @staticmethod
    def _row(helicity: int, direction: np.ndarray, result: dict) -> dict:
        """Keep the incident direction and the channels."""
        return {"helicity": helicity, "direction_x": float(direction[0]), "direction_y": float(direction[1]),
                "direction_z": float(direction[2]), "spin": result["spin_term"], "orbital": result["orbital_term"],
                "correlation": result["correlation_term"]}

    @staticmethod
    def _row_key(row: dict) -> tuple:
        """Identify one direction when resuming an interrupted size calculation."""
        return (int(row["helicity"]), *(round(float(row[key]), 12) for key in ("direction_x", "direction_y", "direction_z")))

    def _partial_rows(self) -> list[dict]:
        """Read any completed angular samples from this size's temporary CSV."""
        if not self.partial_path.exists():
            return []
        with self.partial_path.open(newline="") as file:
            return [{key: int(row[key]) if key == "helicity" else float(row[key]) for key in CSV_COLUMNS}
                    for row in csv.DictReader(file)]

    @staticmethod
    def _contribution_statistics(rows: list[dict]) -> dict[str, float]:
        """Return ratios of means and sample variances of per-direction percentages."""
        spin = np.array([row["spin"] for row in rows]); orbital = np.array([row["orbital"] for row in rows])
        correlation = np.array([row["correlation"] for row in rows]); q0 = spin + orbital
        terms = {"spin": spin, "orbital": orbital, "reduction": np.maximum(-correlation, 0),
                 "increase": np.maximum(correlation, 0)}
        statistics = {}
        for name, values in terms.items():
            statistics[f"{name}_percent"] = 100 * np.mean(values) / np.mean(q0)
            statistics[f"{name}_variance"] = np.var(100 * values[q0 > 0] / q0[q0 > 0], ddof=1)
        return statistics

    def _contribution_scan(self, existing: list[dict]) -> dict[int, dict[str, float]]:
        """Scan both helicities and summarize their spin, orbital, and correlation shares."""
        completed = {self._row_key(row) for row in existing}
        total = len(oml.parameters.HELICITIES) * N_COS_THETA * N_AZIMUTH; done = len(completed)
        started = perf_counter()
        print(f"[CONTRIBUTIONS] incident grid={N_COS_THETA} x {N_AZIMUTH} x {len(oml.parameters.HELICITIES)} helicities; "
              f"{done}/{total} complete", flush=True)

        with self.partial_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            if file.tell() == 0:
                writer.writeheader(); file.flush()
            for helicity in oml.parameters.HELICITIES:
                for i in range(N_COS_THETA):
                    for j in range(N_AZIMUTH):
                        u = self.cos_theta[i]; phi = self.azimuth[j]
                        direction = np.array((np.sqrt(1 - u**2) * np.cos(phi), np.sqrt(1 - u**2) * np.sin(phi), u))
                        key = (helicity, *(round(float(value), 12) for value in direction))
                        if key in completed:
                            continue
                        row = self._row(helicity, direction, self._spectral_result(helicity, direction))
                        writer.writerow(row); file.flush(); completed.add(key); done += 1
                        print(f"[CONTRIBUTIONS {done}/{total}] remaining={total - done}, σ={helicity:+d}, "
                              f"i={i + 1}/{N_COS_THETA}, j={j + 1}/{N_AZIMUTH}, "
                              f"elapsed={(perf_counter() - started) / 60:.1f} min", flush=True)

        rows = self._partial_rows()
        if len(rows) != total:
            raise RuntimeError("Incomplete contribution scan; the partial CSV was retained for resumption")
        return {helicity: self._contribution_statistics([row for row in rows if row["helicity"] == helicity])
                for helicity in oml.parameters.HELICITIES}

    ####################################
    # SIZE-SWEEP OUTPUT
    ####################################
    def run(self) -> Path:
        """Write one CSV for this size with two helicity rows and their shares."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            with self.output_path.open(newline="") as file:
                if tuple(csv.DictReader(file).fieldnames) == RESULT_COLUMNS:
                    print(f"[COMPLETE] Already saved: {self.output_path}", flush=True)
                    return self.output_path
        started = perf_counter(); existing = self._partial_rows()
        print(f"[CASE {self.case_id + 1}/{len(long_axis_values_nm())}] long axis={self.long_axis_nm:.6f} nm, "
              f"dipoles={len(self.particle['geometry_nm'])}, existing angular samples={len(existing)}", flush=True)
        theory = self._theoretical_result()
        positive = self._fixed_incidence(+1, theory); negative = self._fixed_incidence(-1, theory)
        statistics = self._contribution_scan(existing)
        positive.update(statistics[+1]); negative.update(statistics[-1])

        temporary = self.output_path.with_suffix(".tmp")
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS); writer.writeheader(); writer.writerows((positive, negative))
        os.replace(temporary, self.output_path); self.partial_path.unlink()
        print(f"[COMPLETE] {self.output_path} ({(perf_counter() - started) / 60:.1f} min)", flush=True)
        return self.output_path


def main() -> None:
    """Run one independent particle size from a Slurm array or a direct call."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=None, help="Override SLURM_ARRAY_TASK_ID")
    parser.add_argument("--list-cases", action="store_true", help="List all particle lengths without running simulations")
    args = parser.parse_args(); sizes = long_axis_values_nm()
    if args.list_cases:
        for case_id, size in enumerate(sizes):
            print(f"{case_id:02d}, {size:.12f} nm")
        return
    case_id = args.case_id if args.case_id is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
    if not 0 <= case_id < len(sizes):
        parser.error(f"case ID must be between 0 and {len(sizes) - 1}")
    SizeSweep(case_id, float(sizes[case_id])).run()

if __name__ == "__main__":
    main()
