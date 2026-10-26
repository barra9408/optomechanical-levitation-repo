"""Scattered torque information patterns for x-polarized plane waves and LG02 beams."""

import argparse
import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.constants import c, epsilon_0, hbar

import optomechanics_levitation as oml
from optomechanics_levitation.light_matter_interaction import LightMatterInteraction
from optomechanics_levitation.recoil_heating import TorqueRecoilHeating


LONG_AXES_NM = (100.0, 877.0)
ASPECT_RATIO = 4.0
MESH_DIVISIONS = 35.0
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
CSV_COLUMNS = ("surface", "direction_x", "direction_y", "direction_z", "spin", "orbital", "correlation", "total")


class ScatteredTorqueRecoilHeating(TorqueRecoilHeating):
    """Expose the angular densities of the existing torque-transfer calculation."""

    def __init__(self, particle: dict) -> None:
        super().__init__("z", N_theta=oml.parameters.SPECTRAL_N_THETA, N_phi=oml.parameters.SPECTRAL_N_PHI)
        self.interaction = LightMatterInteraction(particle, oml.parameters.WAVELENGTH_NM, oml.parameters.ENVIRONMENT)
        # Retain the original scan's far-field observation distance.
        self.radius_m = oml.parameters.FAR_FIELD_RADIUS_NM * 1e-9
        self.observation_offsets_nm = oml.parameters.FAR_FIELD_RADIUS_NM * self.directions.reshape(-1, 3)
        self.field_to_rate = epsilon_0 * c * self.radius_m**2 / (2 * hbar * self.omega)

    def _run_simulation(self, particle: dict, in_efield: np.ndarray) -> np.ndarray:
        """Reuse the interaction matrix for all three fields."""
        self.interaction.internal_field(in_efield)
        center_nm = np.mean(particle["geometry_nm"], axis=0)
        positions_nm = center_nm + self.observation_offsets_nm
        field = np.empty((len(positions_nm), 3), dtype=complex)
        for start in range(0, len(positions_nm), oml.parameters.CHUNK_SIZE):
            stop = min(start + oml.parameters.CHUNK_SIZE, len(positions_nm))
            field[start:stop] = self.interaction.far_field(positions_nm[start:stop], field="E")
        return field.reshape(self.N_theta, self.N_phi, 3)

    def scattered_information_pattern(self, particle: dict, field_generator, parameters: dict) -> dict:
        """Compute torque densities per unit solid angle on the outgoing grid."""
        in_field, positions_nm, derivatives, derivative_curl = self.small_incident_perturbation(
            particle, field_generator, parameters)
        spin_in, orbital_in = self._angular_momentum_operators(
            in_field, positions_nm=positions_nm, derivatives=derivatives, derivative_curl=derivative_curl)
        transferred_spin = self._run_simulation(particle, spin_in)
        transferred_orbital = self._run_simulation(particle, orbital_in)
        scattered_field = self._run_simulation(particle, in_field)
        spin_sc, orbital_sc = self._angular_momentum_operators(scattered_field)
        q_spin = self._transfer_vector(spin_sc, transferred_spin)
        q_orbital = self._transfer_vector(orbital_sc, transferred_orbital)

        scale = self.field_to_rate / (2 * np.pi)
        spin = scale * np.sum(np.abs(q_spin)**2, axis=-1)
        orbital = scale * np.sum(np.abs(q_orbital)**2, axis=-1)
        correlation = 2 * scale * np.sum(np.conj(q_spin) * q_orbital, axis=-1).real
        return {"directions": self.directions, "spin": spin, "orbital": orbital,
                "correlation": correlation, "total": spin + orbital + correlation}


class AngularScan:
    """Calculate one scattered information pattern for a beam and particle size."""

    def __init__(self, particle: dict, beam: str, long_axis_nm: float) -> None:
        self.particle = particle
        self.beam = beam
        self.long_axis_nm = long_axis_nm
        self.center_nm = np.mean(particle["geometry_nm"], axis=0)
        self.amplitude = np.sqrt(2 * oml.parameters.INTENSITY_W_PER_M2 / (epsilon_0 * c))
        self.calculator = ScatteredTorqueRecoilHeating(particle)
        stem = f"{beam.replace('-', '_')}_info_patterns_{long_axis_nm:g}nm_ar{ASPECT_RATIO:g}"
        self.output_path = OUTPUT_DIR / f"{stem}.csv"
        self.partial_path = OUTPUT_DIR / f"{stem}.partial.csv"

    def _prepare_beam(self):
        """Build the +z beam and normalize its focus or LG bright ring."""
        if self.beam == "plane-wave":
            beam = oml.PlaneWave((0.0, 0.0, 1.0), (1.0, 0.0))
            reference_positions = np.zeros((1, 3))
        else:
            options = dict(kSign=+1, NA=oml.parameters.NUMERICAL_APERTURE,
                           f=oml.parameters.HG_FOCAL_LENGTH_MM,
                           w0=oml.parameters.HG_FOCAL_LENGTH_MM * oml.parameters.NUMERICAL_APERTURE,
                           quadrature_points=oml.parameters.HG_QUADRATURE_POINTS,
                           N_cpu=oml.parameters.HG_N_CPU)
            beam = oml.LaguerreGauss02(charge=+2, **options)
            radius = beam.waist_nm(oml.parameters.WAVELENGTH_NM) * np.sqrt(abs(beam.charge) / 2)
            reference_positions = np.array(((radius, 0, 0), (-radius, 0, 0),
                                            (0, radius, 0), (0, -radius, 0)))
        reference_field = beam.field(reference_positions, oml.parameters.ENVIRONMENT,
                                     oml.parameters.WAVELENGTH_NM)
        reference_norm = np.max(np.linalg.norm(reference_field, axis=1))
        if reference_norm <= 0:
            raise ValueError(f"{self.beam} has zero reference electric field")

        def field(pos, env_dict, wavelength, returnField="E"):
            return beam.field(np.asarray(pos) - self.center_nm, env_dict, wavelength,
                              returnField=returnField)

        return field, self.amplitude / reference_norm

    @staticmethod
    def _row(direction: np.ndarray, angular: dict, index: tuple[int, int]) -> dict:
        i, j = index
        return {"surface": "scattered", "direction_x": direction[0], "direction_y": direction[1],
                "direction_z": direction[2], "spin": angular["spin"][i, j],
                "orbital": angular["orbital"][i, j], "correlation": angular["correlation"][i, j],
                "total": angular["total"][i, j]}

    @staticmethod
    def _row_key(direction) -> tuple[float, float, float]:
        return tuple(round(float(value), 12) for value in direction)

    def _completed(self) -> set[tuple[float, float, float]]:
        """Recover finished directions from an interrupted job."""
        if not self.partial_path.exists():
            return set()
        with self.partial_path.open(newline="") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
                raise ValueError(f"Unexpected columns in {self.partial_path}")
            rows = list(reader)
        if any(row["surface"] != "scattered" for row in rows):
            raise ValueError(f"Unexpected incident data in {self.partial_path}")
        return {self._row_key((row["direction_x"], row["direction_y"], row["direction_z"])) for row in rows}

    def run(self) -> Path:
        """Save the scattered information pattern on the original angular grid."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            print(f"[{self.beam}, {self.long_axis_nm:g} nm] Already complete: {self.output_path}", flush=True)
            return self.output_path
        completed = self._completed()
        total = self.calculator.N_theta * self.calculator.N_phi
        print(f"[{self.beam}, {self.long_axis_nm:g} nm] torque=z, AR={ASPECT_RATIO:g}, "
              f"scattered grid={self.calculator.N_theta} x {self.calculator.N_phi}, "
              f"checkpoint={len(completed)}/{total}", flush=True)
        if len(completed) < total:
            field, amplitude = self._prepare_beam()
            parameters = {"wavelength_nm": oml.parameters.WAVELENGTH_NM,
                          "environment": oml.parameters.ENVIRONMENT,
                          "amplitude_v_per_m": amplitude}
            started = perf_counter()
            angular = self.calculator.scattered_information_pattern(self.particle, field, parameters)
            with self.partial_path.open("a", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
                if file.tell() == 0:
                    writer.writeheader()
                for index in np.ndindex(angular["total"].shape):
                    direction = angular["directions"][index]
                    key = self._row_key(direction)
                    if key not in completed:
                        writer.writerow(self._row(direction, angular, index))
                        completed.add(key)
            print(f"[{self.beam}, {self.long_axis_nm:g} nm] {len(completed)}/{total} directions "
                  f"in {(perf_counter() - started) / 60:.1f} min", flush=True)
        if len(completed) != total:
            raise RuntimeError(f"Incomplete scattered grid in {self.partial_path}")
        self.partial_path.replace(self.output_path)
        print(f"[{self.beam}, {self.long_axis_nm:g} nm] Complete: {self.output_path}", flush=True)
        return self.output_path


def main() -> None:
    """Build the selected aspect-ratio-four particle and run one Slurm array task."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beam", required=True, choices=("plane-wave", "laguerre-gauss-02"))
    parser.add_argument("--long-axis-nm", required=True, type=float, choices=LONG_AXES_NM)
    args = parser.parse_args()

    step_nm = args.long_axis_nm / MESH_DIVISIONS
    long_radius_nm = args.long_axis_nm / 2
    particle = oml.particle_properties(
        particle_type=oml.parameters.PARTICLE_TYPE, material_name=oml.parameters.MATERIAL,
        step_nm=step_nm, mesh=oml.parameters.MESH, center=True,
        R1=long_radius_nm / step_nm - 0.2,
        R2=long_radius_nm / ASPECT_RATIO / step_nm - 0.2,
        R3=long_radius_nm / ASPECT_RATIO / step_nm - 0.2,
    )
    AngularScan(particle, args.beam, args.long_axis_nm).run()


if __name__ == "__main__":
    main()
