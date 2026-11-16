"""Convergence plot of the Brown–Robinson method: vᴺ, αᴺ and βᴺ against the iteration number."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

APPROXIMATE_VALUE_COLOR = "#2a78d6"
UPPER_ESTIMATE_COLOR = "#eb6834"
LOWER_ESTIMATE_COLOR = "#1baf7a"
EXACT_VALUE_COLOR = "#6b6a64"


def plot_convergence(records, exact_game_value, file_path):
    """Draw vᴺ with the bounds αᴺ and βᴺ and the exact value v as a dashed line, then save the figure."""
    iteration_numbers = [record.iteration_number for record in records]
    approximate_values = [float(record.approximate_game_value) for record in records]
    upper_estimates = [float(record.upper_value_estimate) for record in records]
    lower_estimates = [float(record.lower_value_estimate) for record in records]

    figure, axes = plt.subplots(figsize=(10, 5.5))
    axes.plot(iteration_numbers, upper_estimates, color=UPPER_ESTIMATE_COLOR, linewidth=1.5,
              marker="^", markersize=7, label="αᴺ — верхня оцінка")
    axes.plot(iteration_numbers, lower_estimates, color=LOWER_ESTIMATE_COLOR, linewidth=1.5,
              marker="v", markersize=7, label="βᴺ — нижня оцінка")
    axes.plot(iteration_numbers, approximate_values, color=APPROXIMATE_VALUE_COLOR, linewidth=2.2,
              marker="o", markersize=8, label="vᴺ = (αᴺ + βᴺ)/2")
    axes.axhline(float(exact_game_value), color=EXACT_VALUE_COLOR, linestyle="--", linewidth=1.3,
                 label=f"Точна ціна гри v = {exact_game_value}")

    # Label only the last approximation so the final value can be read off directly.
    final_record = records[-1]
    axes.annotate(f"v¹⁰ = {final_record.approximate_game_value}",
                  xy=(final_record.iteration_number, float(final_record.approximate_game_value)),
                  xytext=(-10, 12), textcoords="offset points", fontsize=10)

    axes.set_xticks(iteration_numbers)
    axes.set_xlim(0.5, iteration_numbers[-1] + 0.5)
    axes.set_xlabel("Номер ітерації N")
    axes.set_ylabel("Значення оцінки ціни гри")
    axes.set_title("Збіжність методу Брауна–Робінсон")
    axes.grid(True, linestyle=":", linewidth=0.7, color="#c9c8c2")
    for spine_name in ("top", "right"):
        axes.spines[spine_name].set_visible(False)
    axes.legend(loc="lower right", fontsize=10, frameon=False)
    figure.tight_layout()
    figure.savefig(file_path, dpi=180)
    plt.close(figure)
