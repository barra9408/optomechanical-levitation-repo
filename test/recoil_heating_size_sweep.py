"""Add LG02 torque noise about Y and calculate all four beams about Z.

The theoretical method uses the local electric field of each beam in the
point-dipole model. Derivative and spectral methods use the full beam fields.
The particle's long axis remains X and every beam propagates along +Z.
"""

import argparse
import csv
import os
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.constants import c, epsilon_0
import optomechanics_levitation as oml


ASPECT_RATIO = 2.0
MESH_DIVISIONS = 35.0
SIZE_KNOT_NM = 157.0  # Retain the original size grid
INCIDENT_DIRECTION = (0.0, 0.0, 1.0)
BEAM_NAMES = ("plane_wave_x", "hermite_gauss_00_x", "laguerre_gauss_01", "laguerre_gauss_02")
FOCUSED_SPECTRUM_GRID = 96
FOCUSED_SPECTRUM_SPAN_WAVELENGTHS = 16.0
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs" / "recoil_heating_results"
AXIS_BEAMS = {"y": ("laguerre_gauss_02",), "z": BEAM_NAMES}
RESULT_COLUMNS = (
    "long_axis_nm", "beam", "direction_x", "direction_y", "direction_z",
    "incident_amplitude_v_per_m", "numerical_source", "D", "anisotropy_term",
    "theoretical_total_torque", "derivative_total_torque", "spectral_total_torque",
    "spin_term", "orbital_term", "correlation_term",
)


def rayleigh_limit_nm() -> float:
    """Return the Rayleigh range of the configured optical focus in nm."""
    waist_nm = oml.parameters.WAVELENGTH_NM / (np.pi * oml.parameters.NUMERICAL_APERTURE)
    return np.pi * waist_nm**2 / oml.parameters.WAVELENGTH_NM

def long_axis_values_nm() -> np.ndarray:
    """Keep the original 44 sizes from 25 nm to the Rayleigh range."""
    return np.r_[np.linspace(25.0, SIZE_KNOT_NM, 15),
                 np.linspace(SIZE_KNOT_NM, rayleigh_limit_nm(), 30)[1:]]

class SizeSweep:
    """Calculate the missing Y row and all Z rows for one particle size."""
    def __init__(self, case_id: int, long_axis_nm: float) -> None:
        self.case_id = case_id
        self.long_axis_nm = long_axis_nm
        step_nm = long_axis_nm / MESH_DIVISIONS
        long_radius_nm = long_axis_nm / 2
        short_radius_nm = long_radius_nm / ASPECT_RATIO
        self.particle = oml.particle_properties(
            particle_type=oml.parameters.PARTICLE_TYPE, material_name=oml.parameters.MATERIAL,
            step_nm=step_nm, mesh=oml.parameters.MESH, center=True,
            R1=long_radius_nm / step_nm - 0.2,
            R2=short_radius_nm / step_nm - 0.2,
            R3=short_radius_nm / step_nm - 0.2,
        )
        self.center_nm = np.mean(self.particle["geometry_nm"], axis=0)
        self.electric_amplitude = np.sqrt(2 * oml.parameters.INTENSITY_W_PER_M2 / (epsilon_0 * c))
        self.spectral = {}
        self.output_paths = {
            axis: OUTPUT_DIR / f"beamwise_recoilheating_{axis}_{long_axis_nm:.6f}nm.csv"
            for axis in AXIS_BEAMS
        }

    def _spectral_model(self, axis: str) -> oml.SpectralTorqueRecoilHeating:
        if axis not in self.spectral:
            self.spectral[axis] = oml.SpectralTorqueRecoilHeating(
                self.particle, wavelength_nm=oml.parameters.WAVELENGTH_NM,
                environment=oml.parameters.ENVIRONMENT, torque_axis=axis,
                n_theta=oml.parameters.SPECTRAL_N_THETA, n_phi=oml.parameters.SPECTRAL_N_PHI,
                radius_wavelengths=oml.parameters.FAR_FIELD_RADIUS_NM / oml.parameters.WAVELENGTH_NM,
            )
        return self.spectral[axis]

    @staticmethod
    def _focused_options() -> dict:
        """Use the same focused-beam setup as the angular scan."""
        return dict(
            kSign=+1, NA=oml.parameters.NUMERICAL_APERTURE,
            f=oml.parameters.HG_FOCAL_LENGTH_MM,
            w0=oml.parameters.HG_FOCAL_LENGTH_MM * oml.parameters.NUMERICAL_APERTURE,
            quadrature_points=oml.parameters.HG_QUADRATURE_POINTS,
            N_cpu=oml.parameters.HG_N_CPU,
        )

    def _beam(self, name: str):
        if name == "plane_wave_x":
            return oml.PlaneWave(INCIDENT_DIRECTION, (1.0, 0.0)), oml.parameters.INPUT_GRID, oml.parameters.INPUT_SPAN_WAVELENGTHS
        options = self._focused_options()
        if name == "hermite_gauss_00_x":
            return oml.HermiteGauss00(theta=0.0, **options), FOCUSED_SPECTRUM_GRID, FOCUSED_SPECTRUM_SPAN_WAVELENGTHS
        if name == "laguerre_gauss_01":
            return oml.LaguerreGauss01(charge=+1, **options), FOCUSED_SPECTRUM_GRID, FOCUSED_SPECTRUM_SPAN_WAVELENGTHS
        if name == "laguerre_gauss_02":
            return oml.LaguerreGauss02(charge=+2, **options), FOCUSED_SPECTRUM_GRID, FOCUSED_SPECTRUM_SPAN_WAVELENGTHS
        raise ValueError(f"Unknown beam: {name}")

    def _local_field(self, beam):
        """Place the beam focus at the particle centre."""
        def field(pos, env_dict, wavelength, returnField="E"):
            return beam.field(np.asarray(pos) - self.center_nm, env_dict, wavelength, returnField=returnField)
        return field

    def _incident_amplitude(self, name: str, beam) -> float:
        """Match the same E-field scale at focus or at the LG bright ring."""
        if name.startswith("laguerre_gauss_"):
            radius = beam.waist_nm(oml.parameters.WAVELENGTH_NM) * np.sqrt(abs(beam.charge) / 2)
            sample = np.array(((radius, 0, 0), (-radius, 0, 0),
                               (0, radius, 0), (0, -radius, 0)))
        else:
            sample = np.zeros((1, 3))
        field = beam.field(sample, oml.parameters.ENVIRONMENT, oml.parameters.WAVELENGTH_NM)
        reference_norm = np.max(np.linalg.norm(field, axis=1))
        if reference_norm <= 0:
            raise ValueError(f"{name} has zero reference electric field")
        return float(self.electric_amplitude / reference_norm)

    def _theoretical_model(self) -> oml.TheoreticalTorqueRecoilHeating:
        """Prepare the point-dipole model for this particle size."""
        long_radius_nm = self.long_axis_nm / 2
        short_radius_nm = long_radius_nm / ASPECT_RATIO
        epsilon_r = self.particle["material"].epsilon(oml.parameters.WAVELENGTH_NM).real
        return oml.TheoreticalTorqueRecoilHeating(long_radius_nm, short_radius_nm, epsilon_r)

    def _calculate_beam(self, name: str, axes: tuple[str, ...],
                        theoretical_model: oml.TheoreticalTorqueRecoilHeating) -> dict[str, dict]:
        """Share one derivative solve and incident spectrum across requested axes."""
        started = perf_counter()
        beam, input_grid, input_span = self._beam(name)
        field = self._local_field(beam)
        amplitude = self._incident_amplitude(name, beam)
        electric_at_center = amplitude * field(
            self.center_nm[None, :], oml.parameters.ENVIRONMENT, oml.parameters.WAVELENGTH_NM,
        )[0]
        print(f"[{name}] derivative, axes={axes}", flush=True)
        derivative = oml.DerivativeTorqueRecoilHeating(
            self.particle, wavelength_nm=oml.parameters.WAVELENGTH_NM,
            environment=oml.parameters.ENVIRONMENT, rotation_axes=axes,
        ).compute(field, amplitude_v_per_m=amplitude)
        sample_axis = axes[0]
        print(f"[{name}] incident spectrum ({input_grid} x {input_grid})", flush=True)
        base_spectrum = self._spectral_model(sample_axis).incident_angular_spectrum(
            field, amplitude_v_per_m=amplitude, center_nm=self.center_nm,
            input_grid=input_grid, input_span_wavelengths=input_span,
        )
        rows = {}
        for axis in axes:
            spectrum = (base_spectrum if axis == sample_axis else
                        oml.SpectralTorqueRecoilHeating.rotate_incident_spectrum(
                            base_spectrum, np.eye(3), self.center_nm, axis))
            theory = theoretical_model.compute(
                electric_field_v_per_m=electric_at_center, torque_axis=axis,
            )
            print(f"[{name}, {axis}] spectral", flush=True)
            spectral = self._spectral_model(axis).compute(
                field, amplitude_v_per_m=amplitude, incident_spectrum=spectrum,
            )
            rows[axis] = dict(
                long_axis_nm=self.long_axis_nm, beam=name,
                direction_x=INCIDENT_DIRECTION[0], direction_y=INCIDENT_DIRECTION[1],
                direction_z=INCIDENT_DIRECTION[2], incident_amplitude_v_per_m=amplitude,
                numerical_source="computed", D=derivative["D"][axis],
                anisotropy_term=theory["anisotropy_term"],
                theoretical_total_torque=theory["total_torque"],
                derivative_total_torque=derivative["total_torque"][axis],
                spectral_total_torque=spectral["total_torque"],
                spin_term=spectral["spin_term"], orbital_term=spectral["orbital_term"],
                correlation_term=spectral["correlation_term"],
            )
            print(f"[{name}, {axis}] theoretical={theory['total_torque']:.6e}, "
                  f"derivative={rows[axis]['derivative_total_torque']:.6e}, "
                  f"spectral={rows[axis]['spectral_total_torque']:.6e}", flush=True)
        print(f"[{name}] complete in {(perf_counter() - started) / 60:.1f} min", flush=True)
        return rows

    def _saved_rows(self, axis: str) -> dict[str, dict]:
        path = self.output_paths[axis]
        if not path.exists():
            return {}
        with path.open(newline="") as file:
            reader = csv.DictReader(file)
            if tuple(reader.fieldnames or ()) != RESULT_COLUMNS:
                raise ValueError(f"Unexpected columns in {path}")
            rows = list(reader)
        if any(row["beam"] not in AXIS_BEAMS[axis] for row in rows) or len({row["beam"] for row in rows}) != len(rows):
            raise ValueError(f"Invalid beam rows in {path}")
        return {row["beam"]: row for row in rows}

    def _save_rows(self, axis: str, rows: dict[str, dict]) -> None:
        path = self.output_paths[axis]
        temporary = path.with_suffix(".tmp")
        with temporary.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS)
            writer.writeheader()
            writer.writerows(rows[name] for name in AXIS_BEAMS[axis] if name in rows)
        os.replace(temporary, path)

    def run(self) -> dict[str, Path]:
        """Write each Y/Z result as it completes so a size can be resumed."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        rows = {axis: self._saved_rows(axis) for axis in AXIS_BEAMS}
        if all(len(rows[axis]) == len(AXIS_BEAMS[axis]) for axis in AXIS_BEAMS):
            print(f"[COMPLETE] Already saved: {self.output_paths}", flush=True)
            return self.output_paths
        print(f"[CASE {self.case_id + 1}/{len(long_axis_values_nm())}] "
              f"long axis={self.long_axis_nm:.6f} nm, dipoles={len(self.particle['geometry_nm'])}, "
              f"saved Y={len(rows['y'])}/1, Z={len(rows['z'])}/4", flush=True)
        theoretical_model = self._theoretical_model()
        for name in BEAM_NAMES:
            missing_axes = tuple(axis for axis in AXIS_BEAMS
                                 if name in AXIS_BEAMS[axis] and name not in rows[axis])
            if missing_axes:
                calculated = self._calculate_beam(name, missing_axes, theoretical_model)
                for axis in missing_axes:
                    rows[axis][name] = calculated[axis]
                    self._save_rows(axis, rows[axis])
        print(f"[COMPLETE] {self.output_paths}", flush=True)
        return self.output_paths

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", type=int, default=None, help="Override SLURM_ARRAY_TASK_ID")
    parser.add_argument("--list-cases", action="store_true", help="List all particle lengths without running simulations")
    args = parser.parse_args()
    sizes = long_axis_values_nm()
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
