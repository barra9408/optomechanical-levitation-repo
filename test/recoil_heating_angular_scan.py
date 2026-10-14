"""Incident and scattered information patterns for plane-wave and HG00 beams."""

import argparse
import csv
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.constants import c, epsilon_0
import optomechanics_levitation as oml

LONG_AXIS_NM = 100.0; ASPECT_RATIO = 2.0; MESH_DIVISIONS = 35.0
N_COS_THETA = 12; N_AZIMUTH = 24
HG_SPECTRUM_GRID = 96; HG_SPECTRUM_SPAN_WAVELENGTHS = 16.0
HELICITY = oml.parameters.HELICITIES[0]
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
CSV_COLUMNS = ("surface", "direction_x", "direction_y", "direction_z", "spin", "orbital", "correlation", "total")

class AngularScan:
    """Calculate both information-pattern surfaces for one incident beam family."""
    def __init__(self, particle: dict, beam: str) -> None:
        self.particle = particle; self.beam = beam
        self.center_nm = np.mean(particle["geometry_nm"], axis=0)
        self.amplitude = np.sqrt(2 * oml.parameters.INTENSITY_W_PER_M2 / (epsilon_0 * c))
        self.cos_theta = -1 + (np.arange(N_COS_THETA) + 0.5) * 2 / N_COS_THETA
        self.azimuth = (np.arange(N_AZIMUTH) + 0.5) * 2 * np.pi / N_AZIMUTH
        self.spectral = oml.SpectralTorqueRecoilHeating(particle, wavelength_nm=oml.parameters.WAVELENGTH_NM,
                                                       environment=oml.parameters.ENVIRONMENT, torque_axis=oml.parameters.TORQUE_AXIS,
                                                       n_theta=oml.parameters.SPECTRAL_N_THETA, n_phi=oml.parameters.SPECTRAL_N_PHI,
                                                       radius_wavelengths=oml.parameters.FAR_FIELD_RADIUS_NM / oml.parameters.WAVELENGTH_NM)
        filename = {"plane-wave": "plane_wave_info_patterns.csv", "hermite-gauss": "hermite_gauss_info_patterns.csv"}[beam]
        self.output_path = OUTPUT_DIR / filename
        self.partial_path = OUTPUT_DIR / filename.replace(".csv", ".sigma_plus.partial.csv")

    ####################################
    # INCIDENT BEAMS
    ####################################
    @staticmethod
    def _beam_frame(direction: np.ndarray) -> np.ndarray:
        """Return the local (x, y, propagation) axes in laboratory coordinates."""
        direction = np.array(direction, dtype=float, copy=True); direction /= np.linalg.norm(direction)
        reference = np.array([1.0, 0.0, 0.0]) if abs(direction[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x_axis = reference - np.dot(reference, direction) * direction; x_axis /= np.linalg.norm(x_axis)
        return np.column_stack((x_axis, np.cross(direction, x_axis), direction))

    def _hg_local_field(self, pos: np.ndarray, env_dict: dict, wavelength: float, returnField: str = "E") -> np.ndarray:
        """Combine two linearly polarized HG00 fields into circular polarization."""
        x_field = self.hg_x.field(pos, env_dict, wavelength, returnField=returnField)
        y_field = self.hg_y.field(pos, env_dict, wavelength, returnField=returnField)
        return (x_field + 1j * HELICITY * y_field) / np.sqrt(2)

    def _rotated_field(self, pos: np.ndarray, env_dict: dict, wavelength: float, beam_direction: tuple, beam_origin_nm: tuple, 
                       returnField: str = "E") -> np.ndarray:
        """Evaluate either local beam and rotate its field into the lab."""
        frame = self._beam_frame(beam_direction)
        local_positions = (np.asarray(pos) - np.asarray(beam_origin_nm)) @ frame
        return self.local_field(local_positions, env_dict, wavelength, returnField=returnField) @ frame.T

    def _prepare_hg(self) -> None:
        """Prepare the two local HG fields for circular polarization."""
        waist_mm = oml.parameters.HG_FOCAL_LENGTH_MM * oml.parameters.NUMERICAL_APERTURE
        options = dict(kSign=+1, NA=oml.parameters.NUMERICAL_APERTURE, f=oml.parameters.HG_FOCAL_LENGTH_MM, w0=waist_mm,
                       quadrature_points=oml.parameters.HG_QUADRATURE_POINTS, N_cpu=oml.parameters.HG_N_CPU)
        self.hg_x = oml.HermiteGauss00(theta=0.0, **options); self.hg_y = oml.HermiteGauss00(theta=90.0, **options)

    def _prepare_beam(self) -> None:
        """Sample the numerical angular spectrum of either beam in its local frame."""
        if self.beam == "plane-wave":
            self.local_field = oml.PlaneWave((0.0, 0.0, 1.0), (1.0, 1j * HELICITY)).field
            grid = oml.parameters.INPUT_GRID; span = oml.parameters.INPUT_SPAN_WAVELENGTHS
        else:
            self._prepare_hg(); self.local_field = self._hg_local_field
            grid = HG_SPECTRUM_GRID; span = HG_SPECTRUM_SPAN_WAVELENGTHS
        focus = self.local_field(np.zeros((1, 3)), oml.parameters.ENVIRONMENT, oml.parameters.WAVELENGTH_NM)[0]
        self.incident_amplitude = self.amplitude / np.linalg.norm(focus)
        print(f"[{self.beam}] Preparing numerical spectrum for helicity={HELICITY:+d}", flush=True)
        self.incident_spectrum = self.spectral.incident_angular_spectrum(
            self.local_field, amplitude_v_per_m=self.incident_amplitude, center_nm=np.zeros(3),
            input_grid=grid, input_span_wavelengths=span)

    def _calculate(self, direction: np.ndarray, return_angular: bool = False) -> dict:
        """Calculate the integrated terms and, when requested, their outgoing densities."""
        parameters = {"beam_direction": tuple(direction), "beam_origin_nm": tuple(self.center_nm)}
        spectrum = self.spectral.rotate_incident_spectrum(self.incident_spectrum, self._beam_frame(direction), self.center_nm,
                                                          self.spectral.torque_axis)
        return self.spectral.compute(self._rotated_field, amplitude_v_per_m=self.incident_amplitude,
                                     incident_spectrum=spectrum, return_angular=return_angular, **parameters)

    ####################################
    # CSV DATA
    ####################################
    @staticmethod
    def _row(surface: str, direction: np.ndarray, spin: float, orbital: float, correlation: float, total: float) -> dict:
        """Keep only the surface, its direction, and the four plotted channels."""
        return {"surface": surface, "direction_x": direction[0], "direction_y": direction[1], "direction_z": direction[2],
                "spin": spin, "orbital": orbital, "correlation": correlation, "total": total}

    def _incident_row(self, direction: np.ndarray, result: dict) -> dict:
        """Build an integrated-noise row for one incident direction."""
        return self._row("incident", direction, result["spin_term"], result["orbital_term"],
                         result["correlation_term"], result["total_torque"])

    def _scattered_rows(self, angular: dict) -> list[dict]:
        """Build outgoing per-solid-angle rows for fixed +z incidence."""
        rows = []
        for i, j in np.ndindex(angular["total"].shape):
            rows.append(self._row("scattered", angular["directions"][i, j],
                                  angular["spin"][i, j], angular["orbital"][i, j], angular["correlation"][i, j],
                                  angular["total"][i, j]))
        return rows

    @staticmethod
    def _row_key(surface: str, direction: np.ndarray) -> tuple[str, float, float, float]:
        """Identify a sampled direction without adding indices to the CSV."""
        return (surface, *(round(float(value), 12) for value in direction))

    def _completed(self) -> set[tuple[str, float, float, float]]:
        """Recover completed samples from an interrupted job's partial CSV."""
        if not self.partial_path.exists():
            return set()
        with self.partial_path.open(newline="") as file:
            reader = csv.DictReader(file); rows = [{column: row[column] for column in CSV_COLUMNS} for row in reader]
            old_format = tuple(reader.fieldnames) != CSV_COLUMNS
        if old_format:
            with self.partial_path.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS); writer.writeheader(); writer.writerows(rows)
        return {self._row_key(row["surface"], (row["direction_x"], row["direction_y"], row["direction_z"])) for row in rows}

    @staticmethod
    def _progress(done: int, total: int, label: str, started: float) -> None:
        """Print an iteration counter and elapsed time to the Slurm output log."""
        elapsed = perf_counter() - started
        print(f"[{done:3d}/{total}] {100 * done / total:5.1f}% | {label} | elapsed {elapsed / 60:.1f} min", flush=True)

    ####################################
    # ANGULAR SCAN
    ####################################
    def run(self) -> Path:
        """Scan positive helicity and save one complete CSV for this beam family."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self._prepare_beam()
        completed = self._completed(); total = N_COS_THETA * N_AZIMUTH + 1
        done = sum(key[0] == "incident" for key in completed)
        scattered_complete = sum(key[0] == "scattered" for key in completed) == oml.parameters.SPECTRAL_N_THETA * oml.parameters.SPECTRAL_N_PHI
        done += scattered_complete
        started = perf_counter()
        print(f"[{self.beam}] Helicity: {HELICITY:+d}; incident grid: {N_COS_THETA} x {N_AZIMUTH}; scattered grid: "
              f"{oml.parameters.SPECTRAL_N_THETA} x {oml.parameters.SPECTRAL_N_PHI}; particle long axis: {LONG_AXIS_NM:g} nm", flush=True)
        print(f"[{self.beam}] {done}/{total} cases already in checkpoint; output: {self.output_path}", flush=True)

        with self.partial_path.open("a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            if file.tell() == 0:
                writer.writeheader(); file.flush()
            for i in range(N_COS_THETA):
                for j in range(N_AZIMUTH):
                    u = self.cos_theta[i]; phi = self.azimuth[j]
                    direction = np.array((np.sqrt(1 - u**2) * np.cos(phi), np.sqrt(1 - u**2) * np.sin(phi), u))
                    key = self._row_key("incident", direction)
                    if key in completed:
                        continue
                    result = self._calculate(direction)
                    writer.writerow(self._incident_row(direction, result)); file.flush()
                    completed.add(key); done += 1
                    self._progress(done, total, f"sigma={HELICITY:+d}, incident i={i + 1}/{N_COS_THETA}, j={j + 1}/{N_AZIMUTH}", started)

            if not scattered_complete:
                print(f"[{self.beam}] Calculating scattered surface for sigma={HELICITY:+d}", flush=True)
                result = self._calculate(np.array((0.0, 0.0, 1.0)), return_angular=True)
                for row in self._scattered_rows(result["angular"]):
                    key = self._row_key(row["surface"], (row["direction_x"], row["direction_y"], row["direction_z"]))
                    if key not in completed:
                        writer.writerow(row); completed.add(key)
                file.flush(); done += 1
                self._progress(done, total, f"sigma={HELICITY:+d}, scattered surface", started)

        self.partial_path.replace(self.output_path)
        print(f"[{self.beam}] Complete: {self.output_path}", flush=True)
        return self.output_path


def main() -> None:
    """Build the shared particle configuration and run the selected Slurm task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam", required=True, choices=("plane-wave", "hermite-gauss"))
    args = parser.parse_args()

    step_nm = LONG_AXIS_NM / MESH_DIVISIONS; long_radius_nm = LONG_AXIS_NM / 2
    particle = oml.particle_properties(particle_type=oml.parameters.PARTICLE_TYPE, material_name=oml.parameters.MATERIAL,
                                       step_nm=step_nm, mesh=oml.parameters.MESH, center=True, R1=long_radius_nm / step_nm - 0.2, 
                                       R2=long_radius_nm / ASPECT_RATIO / step_nm - 0.2, R3=long_radius_nm / ASPECT_RATIO / step_nm - 0.2)
    AngularScan(particle, args.beam).run()

if __name__ == "__main__":
    main()
