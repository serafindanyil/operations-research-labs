"""Lab 1: conflict situations and matrix games, variant 23.

Variant 23: n = 5, m = 4, C1 = 5, C2 = 11. Player A has m = 4 pure strategies (rows),
player B has n = 5 pure strategies (columns). Every matrix must contain at least four
pairwise different elements, all within the segment [C1, C2].
"""

import numpy as np

from pure_strategies import solve_in_pure_strategies

VARIANT_NUMBER = 23
ROW_COUNT_M = 4
COLUMN_COUNT_N = 5
LOWER_BOUND_C1 = 5
UPPER_BOUND_C2 = 11
MINIMUM_DISTINCT_ELEMENTS = 4

# Conflict game: two coffee shops on one street share about 16 thousand visits a month.
# The payoff is the number of visits (in thousands) that go to coffee shop A;
# the remaining visits go to coffee shop B, so the game is constant-sum (antagonistic).
COFFEE_SHOP_A_STRATEGIES = [
    "A1: discounts",
    "A2: loyalty card",
    "A3: social media advertising",
    "A4: wider menu",
]
COFFEE_SHOP_B_STRATEGIES = [
    "B1: discounts",
    "B2: loyalty card",
    "B3: delivery",
    "B4: advertising",
    "B5: no changes",
]
CONFLICT_GAME = np.array([
    [7, 9, 6, 8, 10],
    [9, 8, 7, 10, 11],
    [6, 10, 5, 7, 9],
    [8, 7, 6, 9, 8],
])

# Example matrices for task 3, grouped by the required number of saddle points.
EXAMPLE_GROUPS = [
    ("More than one saddle point", "many", [
        ("S2a", np.array([[7, 8, 7, 9, 10],
                          [5, 6, 5, 8, 6],
                          [7, 9, 7, 10, 11],
                          [6, 5, 6, 7, 8]])),
        ("S2b", np.array([[9, 9, 10, 11, 9],
                          [6, 8, 7, 5, 6],
                          [9, 10, 9, 11, 10],
                          [7, 6, 8, 5, 7]])),
    ]),
    ("Exactly one saddle point", "one", [
        ("S1a", np.array([[6, 7, 5, 8, 9],
                          [8, 9, 9, 10, 11],
                          [5, 6, 7, 9, 6],
                          [7, 5, 6, 8, 10]])),
        ("S1b", np.array([[10, 8, 9, 11, 7],
                          [11, 9, 10, 10, 8],
                          [6, 5, 7, 9, 8],
                          [9, 7, 11, 10, 5]])),
    ]),
    ("No saddle point", "none", [
        ("N1", np.array([[5, 11, 8, 6, 9],
                         [10, 6, 7, 9, 5],
                         [8, 7, 11, 5, 10],
                         [6, 9, 5, 10, 7]])),
        ("N2", np.array([[11, 5, 9, 7, 6],
                         [6, 10, 5, 8, 11],
                         [9, 7, 10, 5, 8],
                         [5, 8, 6, 11, 7]])),
    ]),
]


def check_variant_constraints(payoff_matrix):
    """Check the matrix size, the value range [C1, C2] and the number of distinct elements."""
    distinct_element_count = len(np.unique(payoff_matrix))
    return {
        f"size {ROW_COUNT_M}x{COLUMN_COUNT_N}": payoff_matrix.shape == (ROW_COUNT_M, COLUMN_COUNT_N),
        f"values in [{LOWER_BOUND_C1}, {UPPER_BOUND_C2}]": bool(
            payoff_matrix.min() >= LOWER_BOUND_C1 and payoff_matrix.max() <= UPPER_BOUND_C2),
        f">= {MINIMUM_DISTINCT_ELEMENTS} distinct elements ({distinct_element_count})":
            distinct_element_count >= MINIMUM_DISTINCT_ELEMENTS,
    }


def check_saddle_point_count(saddle_points, required_category):
    """Check that the number of saddle points fits the category: 'many', 'one' or 'none'."""
    saddle_point_count = len(saddle_points)
    if required_category == "many":
        return saddle_point_count > 1
    if required_category == "one":
        return saddle_point_count == 1
    return saddle_point_count == 0


def format_strategies(prefix, strategy_indices):
    """Write zero-based strategy indices as 'A1, A3' or 'B2'."""
    return ", ".join(f"{prefix}{strategy_index + 1}" for strategy_index in strategy_indices)


def print_payoff_table(payoff_matrix, solution):
    """Print the payoff matrix with a column of row minimums and a row of column maximums."""
    column_count = payoff_matrix.shape[1]
    header = "      " + "".join(f"{'B' + str(column_index + 1):>5}" for column_index in range(column_count))
    print(header + "    min")
    for row_index, row in enumerate(payoff_matrix):
        cells = "".join(f"{value:>5}" for value in row)
        print(f"  A{row_index + 1}  {cells}  | {solution['row_minimums'][row_index]:>3}")
    maximum_cells = "".join(f"{value:>5}" for value in solution["column_maximums"])
    print(f"  max {maximum_cells}")


def print_game_summary(name, payoff_matrix):
    """Print the pure-strategy analysis of one game and return its solution dictionary."""
    solution = solve_in_pure_strategies(payoff_matrix)
    print(f"\nMatrix {name}")
    print_payoff_table(payoff_matrix, solution)
    print(f"  Row minimums:    {solution['row_minimums'].tolist()}")
    print(f"  Lower value (maximin) alpha = {solution['lower_game_value']}, "
          f"maximin strategies of A: {format_strategies('A', solution['maximin_strategies'])}")
    print(f"  Column maximums: {solution['column_maximums'].tolist()}")
    print(f"  Upper value (minimax) beta  = {solution['upper_game_value']}, "
          f"minimax strategies of B: {format_strategies('B', solution['minimax_strategies'])}")
    if solution["game_value"] is not None:
        saddle_points_text = ", ".join(
            f"(A{row_index + 1}; B{column_index + 1})" for row_index, column_index in solution["saddle_points"])
        print(f"  alpha = beta: saddle points {saddle_points_text}, game value v = {solution['game_value']}")
    else:
        print(f"  alpha < beta: no saddle point, no pure solution; "
              f"{solution['lower_game_value']} <= v <= {solution['upper_game_value']}")
    for constraint_name, is_satisfied in check_variant_constraints(payoff_matrix).items():
        print(f"  [{'OK' if is_satisfied else 'FAIL'}] {constraint_name}")
    return solution


def main():
    """Solve the conflict game and all example matrices of variant 23 in pure strategies."""
    print(f"Variant {VARIANT_NUMBER}: m = {ROW_COUNT_M} (rows, player A), n = {COLUMN_COUNT_N} "
          f"(columns, player B), C1 = {LOWER_BOUND_C1}, C2 = {UPPER_BOUND_C2}")

    print("\n" + "=" * 70 + "\nTasks 1-2: conflict game of two coffee shops (thousands of visits to A)\n" + "=" * 70)
    print("Strategies of coffee shop A: " + "; ".join(COFFEE_SHOP_A_STRATEGIES))
    print("Strategies of coffee shop B: " + "; ".join(COFFEE_SHOP_B_STRATEGIES))
    print_game_summary("G", CONFLICT_GAME)

    for group_title, required_category, matrices in EXAMPLE_GROUPS:
        print("\n" + "=" * 70 + f"\nTask 3: {group_title}\n" + "=" * 70)
        for name, payoff_matrix in matrices:
            solution = print_game_summary(name, payoff_matrix)
            is_in_category = check_saddle_point_count(solution["saddle_points"], required_category)
            print(f"  [{'OK' if is_in_category else 'FAIL'}] category: {group_title.lower()}")


if __name__ == "__main__":
    main()
