"""Lab 3: graphical method for (2×n) and (m×2) matrix games, variant 23.

Variant 23: n = 5, m = 4, C1 = 8, C2 = 22, so the games are (2×5) and (4×2).
Every matrix has at least three pairwise different elements and contains both C1 and C2.
"""

from pathlib import Path

import numpy as np

from graphical_method import solve_m_by_two_game, solve_two_by_n_game, to_fraction_matrix
from plotting import plot_graphical_solution
from pure_strategies import (
    find_column_maximums,
    find_lower_game_value,
    find_row_minimums,
    find_saddle_points,
    find_upper_game_value,
)

VARIANT_NUMBER = 23
COLUMN_COUNT_N = 5
ROW_COUNT_M = 4
REQUIRED_NUMBER_C1 = 8
REQUIRED_NUMBER_C2 = 22
MINIMUM_DISTINCT_ELEMENTS = 3

FIGURES_DIRECTORY = Path(__file__).resolve().parent / "figures"
SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# Each game: name, matrix, short description used in titles.
GAMES = [
    ("A", np.array([[12, 14, 16, 18, 22],
                    [8, 10, 13, 15, 20]]), "(2×5) із сідловою точкою"),
    ("B", np.array([[8, 20, 18, 15, 22],
                    [22, 10, 16, 21, 12]]), "(2×5) без сідлової точки"),
    ("C", np.array([[12, 16],
                    [10, 22],
                    [8, 14],
                    [11, 13]]), "(4×2) із сідловою точкою"),
    ("D", np.array([[8, 22],
                    [20, 10],
                    [15, 14],
                    [12, 13]]), "(4×2) без сідлової точки"),
]


def format_number(value):
    """Write an exact number as an integer or as a fraction like 5/12."""
    return str(value)


def format_vector(values):
    """Write a vector in the report style: (5/12; 7/12; 0)."""
    return "(" + "; ".join(format_number(value) for value in values) + ")"


def format_line(intercept, slope, variable_name):
    """Write a line as 'b ± k·t', e.g. '22 − 14x', '14 + y' or '12'."""
    if slope == 0:
        return format_number(intercept)
    sign = "+" if slope > 0 else "−"
    coefficient = "" if abs(slope) == 1 else format_number(abs(slope))
    return f"{format_number(intercept)} {sign} {coefficient}{variable_name}"


def check_variant_constraints(payoff_matrix, expected_shape):
    """Check the matrix size, the number of distinct elements and the presence of C1 and C2."""
    distinct_elements = np.unique(payoff_matrix)
    return {
        "розмір": payoff_matrix.shape == expected_shape,
        f"≥ {MINIMUM_DISTINCT_ELEMENTS} різних елементів": len(distinct_elements) >= MINIMUM_DISTINCT_ELEMENTS,
        f"містить C1 = {REQUIRED_NUMBER_C1}": REQUIRED_NUMBER_C1 in distinct_elements,
        f"містить C2 = {REQUIRED_NUMBER_C2}": REQUIRED_NUMBER_C2 in distinct_elements,
    }


def check_optimality(payoff_matrix, first_player_strategy, second_player_strategy, game_value):
    """Check exactly that X*, Y* are distributions, X*·A ≥ v, A·Y* ≤ v and X*·A·Y* = v."""
    exact_matrix = to_fraction_matrix(payoff_matrix)
    payoffs_against_columns = first_player_strategy @ exact_matrix
    payoffs_of_rows = exact_matrix @ second_player_strategy
    return {
        "ΣX* = 1": sum(first_player_strategy) == 1 and all(first_player_strategy >= 0),
        "ΣY* = 1": sum(second_player_strategy) == 1 and all(second_player_strategy >= 0),
        "X*·A ≥ v": all(payoffs_against_columns >= game_value),
        "A·Y* ≤ v": all(payoffs_of_rows <= game_value),
        "X*·A·Y* = v": first_player_strategy @ exact_matrix @ second_player_strategy == game_value,
    }


def print_checks(title, checks):
    """Print a group of named boolean checks."""
    status = ", ".join(f"{name}: {'так' if passed else 'НІ'}" for name, passed in checks.items())
    print(f"  {title}: {status}")


def analyze_game(game_name, payoff_matrix, description):
    """Solve one game in pure strategies and graphically, print everything, save the plot."""
    row_count, column_count = payoff_matrix.shape
    is_two_by_n = row_count == 2
    print(f"\n=== Гра {game_name} {description} ===")
    print("Матриця:")
    for row_index in range(row_count):
        print(f"  A{row_index + 1}: {payoff_matrix[row_index].tolist()}")
    expected_shape = (2, COLUMN_COUNT_N) if is_two_by_n else (ROW_COUNT_M, 2)
    print_checks("Обмеження варіанта", check_variant_constraints(payoff_matrix, expected_shape))

    lower_game_value, maximin_strategies = find_lower_game_value(payoff_matrix)
    upper_game_value, minimax_strategies = find_upper_game_value(payoff_matrix)
    saddle_points = find_saddle_points(payoff_matrix)
    print(f"Мінімуми рядків: {format_vector(find_row_minimums(payoff_matrix))}")
    print(f"Максимуми стовпців: {format_vector(find_column_maximums(payoff_matrix))}")
    print(f"α = {lower_game_value} (максимінні: {', '.join(f'A{index + 1}' for index in maximin_strategies)})")
    print(f"β = {upper_game_value} (мінімаксні: {', '.join(f'B{index + 1}' for index in minimax_strategies)})")
    if saddle_points:
        saddle_text = ", ".join(f"(A{row + 1}; B{column + 1})" for row, column in saddle_points)
        print(f"Сідлові точки: {saddle_text}, ціна гри в чистих стратегіях v = {lower_game_value}")
    else:
        print("Сідлової точки немає (α < β), розв'язок шукаємо у змішаних стратегіях")

    if is_two_by_n:
        solution = solve_two_by_n_game(payoff_matrix)
        variable_name, line_name, strategy_name = "x", "f", "B"
        axis_label = "x — імовірність стратегії A1"
        envelope_text = "максимум нижньої обвідної"
    else:
        solution = solve_m_by_two_game(payoff_matrix)
        variable_name, line_name, strategy_name = "y", "g", "A"
        axis_label = "y — імовірність стратегії B1"
        envelope_text = "мінімум верхньої обвідної"

    line_labels = []
    print("Прямі:")
    for line_index, (intercept, slope) in enumerate(zip(solution.intercepts, solution.slopes)):
        line_text = format_line(intercept, slope, variable_name)
        subscript = str(line_index + 1).translate(SUBSCRIPT_DIGITS)
        line_labels.append(f"{line_name}{subscript}({variable_name}) = {line_text}  ({strategy_name}{line_index + 1})")
        print(f"  {line_labels[-1]}")
    active_text = ", ".join(f"{strategy_name}{index + 1}" for index in solution.active_lines)
    print(f"Оптимум ({envelope_text}): {variable_name}* = {solution.optimal_probability}, "
          f"v = {solution.game_value}, активні стратегії: {active_text}")
    print(f"X* = {format_vector(solution.first_player_strategy)}")
    print(f"Y* = {format_vector(solution.second_player_strategy)}")
    print_checks("Перевірка", check_optimality(
        payoff_matrix, solution.first_player_strategy, solution.second_player_strategy, solution.game_value
    ))
    if saddle_points:
        agreement = "збігаються" if solution.game_value == lower_game_value == upper_game_value else "НЕ збігаються"
        print(f"Порівняння: α = β = {lower_game_value}, графічна v = {solution.game_value} — {agreement}")

    figure_path = FIGURES_DIRECTORY / f"game_{game_name}.png"
    plot_graphical_solution(solution, line_labels, axis_label,
                            f"Гра {game_name} {description}", figure_path)
    print(f"Графік: {figure_path.relative_to(FIGURES_DIRECTORY.parent.parent)}")
    return saddle_points, solution


def main():
    """Solve the four games of variant 23 and print the comparison table."""
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    print(f"Варіант {VARIANT_NUMBER}: n = {COLUMN_COUNT_N}, m = {ROW_COUNT_M}, "
          f"C1 = {REQUIRED_NUMBER_C1}, C2 = {REQUIRED_NUMBER_C2}")

    summary_rows = []
    for game_name, payoff_matrix, description in GAMES:
        saddle_points, solution = analyze_game(game_name, payoff_matrix, description)
        saddle_text = ", ".join(f"(A{row + 1}; B{column + 1})" for row, column in saddle_points) or "немає"
        size_text = f"{payoff_matrix.shape[0]}×{payoff_matrix.shape[1]}"
        summary_rows.append((game_name, size_text, saddle_text,
                             format_vector(solution.first_player_strategy),
                             format_vector(solution.second_player_strategy),
                             format_number(solution.game_value)))

    print("\n=== Порівняння результатів ===")
    header = ("Матриця", "Розмір", "Сідлова точка", "X*", "Y*", "v")
    column_widths = [max(len(row[index]) for row in [header, *summary_rows]) for index in range(len(header))]
    for row in [header, *summary_rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, column_widths)))


if __name__ == "__main__":
    main()
