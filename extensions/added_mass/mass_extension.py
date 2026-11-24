"""Theoretical added-mass study of the South Wing three-storey rig.

Run with Python 3.10+ after installing the packages in requirements.txt.
The calculations use SI units. Floors are numbered upwards from the base.
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from reportlab.lib.colors import HexColor
from reportlab.pdfgen.canvas import Canvas


FLOOR_MASS = 1.83  # kg per unloaded floor
STOREY_HEIGHT = 0.20  # m
STRIP_WIDTH = 0.08  # m
STRIP_THICKNESS = 0.001  # m
YOUNG_MODULUS = 210e9  # Pa
SECOND_MOMENT = STRIP_WIDTH * STRIP_THICKNESS**3 / 12
STOREY_STIFFNESS = 24 * YOUNG_MODULUS * SECOND_MOMENT / STOREY_HEIGHT**3
DRIFT = np.array([[1.0, 0, 0], [-1.0, 1, 0], [0, -1.0, 1]])
STIFFNESS = STOREY_STIFFNESS * DRIFT.T @ DRIFT
BASE_DAMPING = 0.02  # illustrative reference ratio for baseline mode 1
REFERENCE_FRACTION = 0.5
MASS_FRACTIONS = np.linspace(0, 1, 41)
DRIVE_FREQUENCIES = np.linspace(0, 30, 6001)  # Hz, 0.005 Hz steps
PROFILES = {
    "Floor 1": np.array([1.0, 0, 0]),
    "Floor 2": np.array([0, 1.0, 0]),
    "Floor 3": np.array([0, 0, 1.0]),
    "Even split": np.full(3, 1 / 3),
}
COLORS = {
    "Floor 1": ("#13796D", []),
    "Floor 2": ("#A05B04", [5, 2]),
    "Floor 3": ("#6D49A8", [2, 2]),
    "Even split": ("#626F7D", [6, 2, 1, 2]),
    "Baseline": ("#23384D", [8, 3]),
}


@dataclass(frozen=True)
class Modes:
    mass_kg: np.ndarray
    omega: np.ndarray
    shapes: np.ndarray  # columns satisfy shapes.T @ M @ shapes = I

    @property
    def frequency_hz(self):
        return self.omega / (2 * np.pi)


def solve_modes(mass_kg):
    """Solve K φ = ω² M φ through a symmetric, mass-scaled eigenproblem."""
    mass_kg = np.asarray(mass_kg, dtype=float)
    if mass_kg.shape != (3,) or not np.isfinite(mass_kg).all() or np.any(mass_kg <= 0):
        raise ValueError("Supply three finite, positive floor masses in kg.")
    inverse_sqrt_mass = 1 / np.sqrt(mass_kg)
    scaled_stiffness = STIFFNESS * np.outer(inverse_sqrt_mass, inverse_sqrt_mass)
    eigenvalues, scaled_shapes = np.linalg.eigh(scaled_stiffness)
    shapes = inverse_sqrt_mass[:, None] * scaled_shapes
    shapes *= np.where(shapes[-1, :] < 0, -1.0, 1.0)  # common roof sign
    return Modes(mass_kg, np.sqrt(eigenvalues), shapes)


def damping_ratios(modes, beta):
    """Modal ratios for fixed storey dashpots, C = beta K."""
    return beta * modes.omega / 2


def harmonic_drifts(modes, ratios, frequency_hz):
    """Return complex storey displacement differences per unit base acceleration.

    Each row is a drive frequency and each column is a storey. The relative
    displacement equation has harmonic forcing -M 1 per unit acceleration.
    """
    drive = (2 * np.pi * np.atleast_1d(frequency_hz))[:, None]
    participation = modes.shapes.T @ modes.mass_kg
    denominator = modes.omega**2 - drive**2 + 2j * drive * ratios * modes.omega
    modal_coordinates = -participation / denominator
    return modal_coordinates @ (DRIFT @ modes.shapes).T


def peak_drift(drifts, frequency_hz):
    """Peak in mm per m/s², its frequency, and its one-based storey number."""
    amplitude = np.abs(drifts)
    point, storey = np.unravel_index(np.argmax(amplitude), amplitude.shape)
    return 1000 * amplitude[point, storey], frequency_hz[point], storey + 1


def verify_model(baseline, cases, beta):
    """Check the analytic uniform chain and independent matrix response solves."""
    analytic = 2 * np.sqrt(STOREY_STIFFNESS / FLOOR_MASS) * np.sin(
        (2 * np.arange(1, 4) - 1) * np.pi / 14
    )
    np.testing.assert_allclose(baseline.omega, analytic, rtol=1e-12)
    drive_hz = np.array([0.0, 2.0, 5.0, 11.0, 20.0])
    for name, series in cases.items():
        frequencies = np.array([modes.frequency_hz for _, modes in series])
        if np.any(np.diff(frequencies, axis=0) > 1e-10):
            raise AssertionError(f"Frequency increased at fixed stiffness for {name}.")
        for fraction, modes in series:
            mass = np.diag(modes.mass_kg)
            np.testing.assert_allclose(modes.mass_kg.sum(), FLOOR_MASS * (3 + fraction))
            np.testing.assert_allclose(modes.shapes.T @ mass @ modes.shapes, np.eye(3), atol=1e-12)
            np.testing.assert_allclose(
                STIFFNESS @ modes.shapes, (mass @ modes.shapes) * modes.omega**2,
                rtol=1e-12, atol=1e-10,
            )
            if name == "Even split":
                np.testing.assert_allclose(modes.omega, analytic / np.sqrt(1 + fraction / 3))
            matrices = [STIFFNESS - w**2 * mass + 1j * w * beta * STIFFNESS
                        for w in 2 * np.pi * drive_hz]
            direct = np.array([DRIFT @ np.linalg.solve(matrix, -modes.mass_kg) for matrix in matrices])
            np.testing.assert_allclose(harmonic_drifts(modes, damping_ratios(modes, beta), drive_hz),
                                       direct, rtol=1e-11, atol=1e-12)


def write_csv(path, headings, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(headings)
        writer.writerows(rows)


def report_cases(output, baseline, cases, beta):
    """Write the full modal sweep and the harmonic response for the reference cases."""
    modal_rows = []
    for name, series in cases.items():
        for fraction, modes in series[1:]:
            modal_rows.append((name, fraction, modes))
    modal_rows.insert(0, ("Baseline", 0.0, baseline))
    write_csv(
        output / "modal_results.csv",
        ["placement", "added_mass_fraction_of_one_floor", "total_added_mass_kg",
         "floor_1_mass_kg", "floor_2_mass_kg", "floor_3_mass_kg", "mode", "frequency_hz",
         "frequency_change_pct", "fixed_dashpot_damping_pct", "uniform_modal_damping_pct",
         "floor_1_mass_normalised_shape", "floor_2_mass_normalised_shape", "floor_3_mass_normalised_shape"],
        ([name, fraction, fraction * FLOOR_MASS, *modes.mass_kg, i + 1, frequency,
          100 * (frequency / baseline.frequency_hz[i] - 1), 100 * damping_ratios(modes, beta)[i],
          100 * BASE_DAMPING, *modes.shapes[:, i]]
         for name, fraction, modes in modal_rows for i, frequency in enumerate(modes.frequency_hz)),
    )

    selected = {"Baseline": baseline}
    for name in cases:
        selected[name] = next(modes for fraction, modes in cases[name] if np.isclose(fraction, REFERENCE_FRACTION))

    peak_rows = []
    envelopes = {}
    print(f"Storey stiffness: {STOREY_STIFFNESS:.1f} N/m  |  beta: {beta:.6f} s")
    print(f"Reference added mass: {REFERENCE_FRACTION * FLOOR_MASS:.3f} kg")
    print("Placement     | Natural frequencies (Hz) | Fixed-dashpot damping (%) | Peak drift (mm per m/s^2)")
    for name, modes in selected.items():
        fixed_ratios = damping_ratios(modes, beta)
        for law, ratios in (("fixed_dashpots", fixed_ratios), ("uniform_modal", np.full(3, BASE_DAMPING))):
            response = harmonic_drifts(modes, ratios, DRIVE_FREQUENCIES)
            peak, peak_hz, storey = peak_drift(response, DRIVE_FREQUENCIES)
            refined = np.linspace(0, 30, 12001)
            refined_peak = peak_drift(harmonic_drifts(modes, ratios, refined), refined)[0]
            np.testing.assert_allclose(peak, refined_peak, rtol=0.002)
            peak_rows.append([name, law, peak, peak_hz, storey])
            if law == "fixed_dashpots":
                envelopes[name] = 1000 * np.max(np.abs(response), axis=1)
                freq_text = " / ".join(f"{f:5.2f}" for f in modes.frequency_hz)
                ratio_text = " / ".join(f"{100 * z:4.2f}" for z in ratios)
                print(f"{name:13} | {freq_text}     | {ratio_text}         | {peak:6.2f} at {peak_hz:.3f} Hz (storey {storey})")

    write_csv(output / "drift_peaks.csv", ["placement", "damping_model", "peak_mm_per_m_s2",
              "drive_frequency_hz", "peak_storey"], peak_rows)
    write_csv(output / "drift_curves.csv", ["drive_frequency_hz", *envelopes],
              zip(DRIVE_FREQUENCIES, *envelopes.values()))
    return envelopes


def draw_axes(canvas, box, limits, x_ticks, y_ticks, x_label, y_label):
    left, bottom, width, height = box
    x0, x1, y0, y1 = limits

    def point(x, y):
        return left + width * (x - x0) / (x1 - x0), bottom + height * (y - y0) / (y1 - y0)

    canvas.setFont("Helvetica", 10)
    for y in y_ticks:
        _, y_px = point(x0, y)
        canvas.setStrokeColor(HexColor("#DFE5EA"))
        canvas.setLineWidth(0.55)
        canvas.line(left, y_px, left + width, y_px)
        canvas.setFillColor(HexColor("#334454"))
        canvas.drawRightString(left - 6, y_px - 3, f"{y:g}")
    canvas.setStrokeColor(HexColor("#6A7884"))
    canvas.line(left, bottom, left + width, bottom)
    canvas.line(left, bottom, left, bottom + height)
    for x in x_ticks:
        x_px, _ = point(x, y0)
        canvas.line(x_px, bottom, x_px, bottom - 3)
        canvas.drawCentredString(x_px, bottom - 16, f"{x:g}")
    canvas.setFont("Helvetica-Bold", 11)
    if x_label:
        canvas.drawCentredString(left + width / 2, bottom - 33, x_label)
    if y_label:
        canvas.saveState()
        canvas.translate(13, bottom + height / 2)
        canvas.rotate(90)
        canvas.drawCentredString(0, 0, y_label)
        canvas.restoreState()
    return point


def plot_line(canvas, transform, x_data, y_data, name):
    color, dash = COLORS[name]
    canvas.setStrokeColor(HexColor(color))
    canvas.setLineWidth(2.1)
    canvas.setDash(dash)
    path = canvas.beginPath()
    points = iter(zip(x_data, y_data))
    path.moveTo(*transform(*next(points)))
    for point in points:
        path.lineTo(*transform(*point))
    canvas.drawPath(path)
    canvas.setDash([])


def legend(canvas, names, y, spacing):
    for index, name in enumerate(names):
        x = 52 + index * spacing
        plot_line(canvas, lambda a, b: (a, b), [x, x + 21], [y, y], name)
        canvas.setFont("Helvetica", 10)
        canvas.setFillColor(HexColor("#334454"))
        canvas.drawString(x + 26, y - 3, name)


def make_figures(output, baseline, cases, envelopes):
    """Draw vector plots that can be inserted into a report or presentation."""
    canvas = Canvas(str(output / "frequency_shift.pdf"), pagesize=(600, 250))
    canvas.setTitle("Natural frequency change with additional floor mass")
    shifts = {name: np.array([100 * (modes.frequency_hz / baseline.frequency_hz - 1)
                             for _, modes in series]) for name, series in cases.items()}
    for mode in range(3):
        left = 49 + 184 * mode
        point = draw_axes(canvas, (left, 61, 157, 134), (0, 100, -30, 0),
                          [0, 25, 50, 75, 100], [-30, -20, -10, 0], "",
                          "Frequency change (%)" if mode == 0 else "")
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawCentredString(left + 78.5, 207, f"Mode {mode + 1}")
        for name in reversed(cases):
            plot_line(canvas, point, 100 * MASS_FRACTIONS, shifts[name][:, mode], name)
    canvas.setFillColor(HexColor("#334454"))
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawCentredString(300, 24, "Total added mass (% of one original floor mass)")
    legend(canvas, cases, 235, 116)
    canvas.save()

    canvas = Canvas(str(output / "harmonic_drift.pdf"), pagesize=(600, 248))
    canvas.setTitle("Theoretical harmonic base excitation and maximum storey drift")
    upper = 10 * np.ceil(max(np.max(values) for values in envelopes.values()) / 10)
    for index, (left, f_min, f_max, ticks, caption) in enumerate([
        (54, 0, 30, np.arange(0, 31, 5), "Full sweep"),
        (326, 2.7, 3.6, [2.7, 3.0, 3.3, 3.6], "First resonance"),
    ]):
        point = draw_axes(canvas, (left, 55, 245, 130), (f_min, f_max, 0, upper),
                          ticks, np.linspace(0, upper, 5), "",
                          "Max. storey drift (mm per m/s²)" if index == 0 else "")
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawCentredString(left + 122.5, 197, caption)
        in_band = (DRIVE_FREQUENCIES >= f_min) & (DRIVE_FREQUENCIES <= f_max)
        for name in reversed(envelopes):
            plot_line(canvas, point, DRIVE_FREQUENCIES[in_band], envelopes[name][in_band], name)
    canvas.setFillColor(HexColor("#334454"))
    canvas.setFont("Helvetica-Bold", 11)
    canvas.drawCentredString(300, 24, "Base excitation frequency (Hz)")
    legend(canvas, envelopes, 230, 103)
    canvas.save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results",
                        help="Directory for CSV and vector PDF results.")
    output = parser.parse_args().output
    output.mkdir(parents=True, exist_ok=True)
    baseline = solve_modes(np.full(3, FLOOR_MASS))
    beta = 2 * BASE_DAMPING / baseline.omega[0]
    cases = {name: [(fraction, solve_modes(FLOOR_MASS * (1 + fraction * profile)))
                    for fraction in MASS_FRACTIONS] for name, profile in PROFILES.items()}
    verify_model(baseline, cases, beta)
    envelopes = report_cases(output, baseline, cases, beta)
    make_figures(output, baseline, cases, envelopes)
    print("Analytic frequency, eigenpair, mass scaling, direct response and grid checks passed.")
    print(f"Results saved to {output.resolve()}")


if __name__ == "__main__":
    main()
