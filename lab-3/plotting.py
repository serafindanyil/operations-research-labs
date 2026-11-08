"""Plots of the graphical method: all strategy lines, the envelope and the optimal point."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graphical_method import LOWER_ENVELOPE

PLOT_POINT_COUNT = 401


def plot_graphical_solution(solution, line_labels, axis_label, title, file_path):
    """Draw every line, highlight the envelope and mark the optimal point, then save the figure."""
    probabilities = np.linspace(0.0, 1.0, PLOT_POINT_COUNT)
    intercepts = solution.intercepts.astype(float)
    slopes = solution.slopes.astype(float)
    line_values = intercepts[:, np.newaxis] + slopes[:, np.newaxis] * probabilities[np.newaxis, :]

    is_lower_envelope = solution.envelope_kind == LOWER_ENVELOPE
    envelope_values = line_values.min(axis=0) if is_lower_envelope else line_values.max(axis=0)
    envelope_label = "Нижня обвідна" if is_lower_envelope else "Верхня обвідна"

    figure, axes = plt.subplots(figsize=(10, 5.5))
    for line_index, line_label in enumerate(line_labels):
        axes.plot(probabilities, line_values[line_index], linewidth=1.3, label=line_label)
    axes.plot(probabilities, envelope_values, color="black", linewidth=3.2, alpha=0.45,
              label=envelope_label)

    optimal_probability = float(solution.optimal_probability)
    game_value = float(solution.game_value)
    axes.scatter([optimal_probability], [game_value], color="red", s=60, zorder=5,
                 label=f"Оптимальна точка ({solution.optimal_probability}; {solution.game_value})")
    axes.vlines(optimal_probability, line_values.min(), game_value, colors="red",
                linestyles="dashed", linewidth=1)
    axes.axvline(0.0, color="gray", linewidth=0.8)
    axes.axvline(1.0, color="gray", linewidth=0.8)

    axes.set_xlim(-0.03, 1.03)
    axes.set_xlabel(axis_label)
    axes.set_ylabel("Виграш гравця A")
    axes.set_title(title)
    axes.grid(True, linestyle=":", linewidth=0.7)
    axes.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=9)
    figure.tight_layout()
    figure.savefig(file_path, dpi=180)
    plt.close(figure)
