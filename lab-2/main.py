"""Lab 2: Brown–Robinson method (fictitious play) for a matrix game, variant 23.

Variant 23: n = 5, m = 4, C1 = 5, C2 = 11, so the game is (4×5): four strategies of player A
(rows) and five strategies of player B (columns), at least four pairwise different elements in [5; 11].
"""

from pathlib import Path

import numpy as np

from brown_robinson import (
    check_mixed_solution,
    embed_strategy,
    run_brown_robinson,
    solve_two_by_two_game,
)
from plotting import plot_convergence
from pure_strategies import solve_in_pure_strategies

VARIANT_NUMBER = 23
COLUMN_COUNT_N = 5
ROW_COUNT_M = 4
LOWER_BOUND_C1 = 5
UPPER_BOUND_C2 = 11
MINIMUM_DISTINCT_ELEMENTS = 4
ITERATION_COUNT = 10

FIGURES_DIRECTORY = Path(__file__).resolve().parent / "figures"

PAYOFF_MATRIX = np.array([
    [5, 10, 8, 11, 6],
    [9, 6, 11, 7, 8],
    [7, 11, 5, 9, 10],
    [10, 7, 9, 6, 5],
])

# Round 1 is played with A1 and B1 (indices 0 and 0).
FIRST_ROW_INDEX = 0
FIRST_COLUMN_INDEX = 0

# The exact solution is taken from the 2×2 subgame A2, A3 × B1, B4 = [[9, 7], [7, 9]]
# and then checked on the full matrix with the verification criterion X*·A ≥ v ≥ A·Y*.
# The value v and Y* are unique; X* is one of the optimal strategies of player A.
ACTIVE_ROW_INDICES = [1, 2]
ACTIVE_COLUMN_INDICES = [0, 3]


def format_number(value):
    """Write an exact number as an integer or as a fraction like 31/4."""
    return str(value)


def format_vector(values):
    """Write a vector in the report style: (1/10; 1/5; 1/2; 1/5)."""
    return "(" + "; ".join(format_number(value) for value in values) + ")"


def format_strategies(prefix, indices):
    """Write strategy indices as names, e.g. [0, 4] → 'B1, B5'."""
    return ", ".join(f"{prefix}{index + 1}" for index in indices)


def check_variant_constraints(payoff_matrix):
    """Check the matrix size, the value range [C1; C2] and the number of distinct elements."""
    distinct_elements = np.unique(payoff_matrix)
    return {
        f"розмір {ROW_COUNT_M}×{COLUMN_COUNT_N}": payoff_matrix.shape == (ROW_COUNT_M, COLUMN_COUNT_N),
        f"усі елементи в [{LOWER_BOUND_C1}; {UPPER_BOUND_C2}]": bool(
            payoff_matrix.min() >= LOWER_BOUND_C1 and payoff_matrix.max() <= UPPER_BOUND_C2
        ),
        f"≥ {MINIMUM_DISTINCT_ELEMENTS} різних елементів ({len(distinct_elements)})":
            len(distinct_elements) >= MINIMUM_DISTINCT_ELEMENTS,
    }


def print_checks(title, checks):
    """Print a group of named boolean checks."""
    status = ", ".join(f"{name}: {'так' if passed else 'НІ'}" for name, passed in checks.items())
    print(f"{title}: {status}")


def print_pure_strategy_analysis(payoff_matrix):
    """Print row minima, column maxima, α, β and whether a saddle point exists."""
    analysis = solve_in_pure_strategies(payoff_matrix)
    print(f"Мінімуми рядків: {format_vector(analysis['row_minimums'])}")
    print(f"Максимуми стовпців: {format_vector(analysis['column_maximums'])}")
    print(f"α = {analysis['lower_game_value']} "
          f"(максимінна: {format_strategies('A', analysis['maximin_strategies'])})")
    print(f"β = {analysis['upper_game_value']} "
          f"(мінімаксні: {format_strategies('B', analysis['minimax_strategies'])})")
    if analysis["saddle_points"]:
        print(f"Сідлові точки: {analysis['saddle_points']}")
    else:
        print("Сідлової точки немає (α < β), гру розв'язуємо у змішаних стратегіях")


def print_iteration_table(records):
    """Print the full trace: played strategies, accumulated payoffs and the three estimates."""
    header = ("N", "i", "j", "A·yᴺ", "xᴺ·A", "αᴺ", "βᴺ", "vᴺ", "vᴺ ≈")
    rows = []
    for record in records:
        rows.append((
            str(record.iteration_number),
            f"A{record.first_player_strategy + 1}",
            f"B{record.second_player_strategy + 1}",
            format_vector(record.accumulated_row_payoffs),
            format_vector(record.accumulated_column_payoffs),
            format_number(record.upper_value_estimate),
            format_number(record.lower_value_estimate),
            format_number(record.approximate_game_value),
            f"{float(record.approximate_game_value):.3f}",
        ))
    column_widths = [max(len(row[index]) for row in [header, *rows]) for index in range(len(header))]
    for row in [header, *rows]:
        print("  ".join(cell.ljust(width) for cell, width in zip(row, column_widths)))


def find_exact_solution(payoff_matrix):
    """Solve the active 2×2 subgame and extend X*, Y* with zeros to the full game."""
    sub_matrix = payoff_matrix[np.ix_(ACTIVE_ROW_INDICES, ACTIVE_COLUMN_INDICES)]
    sub_first_strategy, sub_second_strategy, game_value = solve_two_by_two_game(sub_matrix)
    row_count, column_count = payoff_matrix.shape
    first_player_strategy = embed_strategy(sub_first_strategy, ACTIVE_ROW_INDICES, row_count)
    second_player_strategy = embed_strategy(sub_second_strategy, ACTIVE_COLUMN_INDICES, column_count)
    return sub_matrix, first_player_strategy, second_player_strategy, game_value


def main():
    """Run 10 iterations of Brown–Robinson for variant 23, compare with the exact solution, plot vᴺ."""
    FIGURES_DIRECTORY.mkdir(exist_ok=True)
    print(f"Варіант {VARIANT_NUMBER}: n = {COLUMN_COUNT_N}, m = {ROW_COUNT_M}, "
          f"C1 = {LOWER_BOUND_C1}, C2 = {UPPER_BOUND_C2}")
    print("Матриця A:")
    for row_index, row in enumerate(PAYOFF_MATRIX):
        print(f"  A{row_index + 1}: {row.tolist()}")
    print_checks("Обмеження варіанта", check_variant_constraints(PAYOFF_MATRIX))

    print("\n=== Аналіз у чистих стратегіях ===")
    print_pure_strategy_analysis(PAYOFF_MATRIX)

    print(f"\n=== Метод Брауна–Робінсон: {ITERATION_COUNT} ітерацій ===")
    print(f"Перша партія: A{FIRST_ROW_INDEX + 1}, B{FIRST_COLUMN_INDEX + 1}; "
          "за рівності обирається стратегія з меншим номером")
    records = run_brown_robinson(PAYOFF_MATRIX, ITERATION_COUNT, FIRST_ROW_INDEX, FIRST_COLUMN_INDEX)
    print_iteration_table(records)

    final_record = records[-1]
    print(f"\nКількості після {ITERATION_COUNT} партій: x = {format_vector(final_record.first_player_counts)}, "
          f"y = {format_vector(final_record.second_player_counts)}")
    print(f"p¹⁰ = {format_vector(final_record.first_player_frequencies)}")
    print(f"q¹⁰ = {format_vector(final_record.second_player_frequencies)}")
    print(f"α¹⁰ = {final_record.upper_value_estimate}, β¹⁰ = {final_record.lower_value_estimate}, "
          f"v¹⁰ = {final_record.approximate_game_value}, "
          f"α¹⁰ − β¹⁰ = {final_record.upper_value_estimate - final_record.lower_value_estimate}")

    print("\n=== Точний розв'язок для порівняння ===")
    sub_matrix, first_player_strategy, second_player_strategy, game_value = find_exact_solution(PAYOFF_MATRIX)
    print(f"Підгра {format_strategies('A', ACTIVE_ROW_INDICES)} × "
          f"{format_strategies('B', ACTIVE_COLUMN_INDICES)}: {sub_matrix.tolist()}")
    print(f"X* = {format_vector(first_player_strategy)} (одна з оптимальних стратегій A)")
    print(f"Y* = {format_vector(second_player_strategy)}")
    print(f"v = {game_value}")
    print(f"X*·A = {format_vector(first_player_strategy @ PAYOFF_MATRIX.astype(object))}, "
          f"A·Y* = {format_vector(PAYOFF_MATRIX.astype(object) @ second_player_strategy)}")
    print_checks("Перевірка", check_mixed_solution(
        PAYOFF_MATRIX, first_player_strategy, second_player_strategy, game_value
    ))
    approximation_error = abs(final_record.approximate_game_value - game_value)
    print(f"Похибка |v¹⁰ − v| = {approximation_error} = {float(approximation_error):.3f}; "
          f"β¹⁰ ≤ v ≤ α¹⁰: {'так' if final_record.lower_value_estimate <= game_value <= final_record.upper_value_estimate else 'НІ'}")

    figure_path = FIGURES_DIRECTORY / "brown_robinson_convergence.png"
    plot_convergence(records, game_value, figure_path)
    print(f"\nГрафік: {figure_path.relative_to(FIGURES_DIRECTORY.parent.parent)}")


if __name__ == "__main__":
    main()
