"""Lab 5: reduction of matrix games to linear programming problems, variant 23.

The program checks the game for a saddle point, shifts the payoff matrix so that the game
value is positive, builds the pair of dual linear programs, solves both with PuLP (CBC),
repeats the solution with an exact tableau simplex method, restores the optimal mixed
strategies and the game value and checks them against every pure strategy.
"""

from fractions import Fraction

import numpy as np

from linear_programs import make_matrix_positive, solve_first_player_problem, solve_second_player_problem
from pure_strategies import (
    find_column_maximums,
    find_lower_game_value,
    find_row_minimums,
    find_saddle_points,
    find_upper_game_value,
)
from simplex_method import name_variable, simplex_maximize

VARIANT_NUMBER = 23
LOWER_BOUND, UPPER_BOUND = 5, 11
PAYOFF_MATRIX = np.array([
    [8, 5, 10, 7, 11],
    [6, 11, 7, 9, 5],
    [10, 7, 5, 11, 8],
    [7, 9, 11, 6, 10],
])


def format_number(value):
    """Write an exact number as an integer or a fraction like 173/21."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def format_vector(values):
    """Write a vector in the report style: (0; 1/21; 3/7; 11/21)."""
    return "(" + "; ".join(format_number(value) for value in values) + ")"


def format_strategies(indices, letter):
    """Write strategy indices as names, e.g. [0, 2] -> A1, A3."""
    return ", ".join(f"{letter}{index + 1}" for index in indices)


def print_matrix(title, matrix):
    """Print a matrix with B1..Bn column headers and A1..Am row headers."""
    print(title)
    print("      " + "".join(f"{f'B{column_index + 1}':>5}" for column_index in range(matrix.shape[1])))
    for row_index, row in enumerate(matrix):
        print(f"  A{row_index + 1}  " + "".join(f"{int(value):>5}" for value in row))


def print_variant_check(payoff_matrix):
    """Print the variant constraints: size, range [C1, C2] and the number of distinct elements."""
    distinct_values = np.unique(payoff_matrix)
    in_range = bool(((payoff_matrix >= LOWER_BOUND) & (payoff_matrix <= UPPER_BOUND)).all())
    print(f"Variant {VARIANT_NUMBER}: m = {payoff_matrix.shape[0]}, n = {payoff_matrix.shape[1]}, "
          f"[C1; C2] = [{LOWER_BOUND}; {UPPER_BOUND}]")
    print(f"  Distinct elements: {len(distinct_values)} {distinct_values.tolist()}, all in range: {in_range}")


def analyze_pure_strategies(payoff_matrix):
    """Step 1: row minima, column maxima, lower and upper game values, saddle points."""
    print("\n1. Pure strategies")
    lower_game_value, maximin_strategies = find_lower_game_value(payoff_matrix)
    upper_game_value, minimax_strategies = find_upper_game_value(payoff_matrix)
    print(f"  Row minimums:    {format_vector(find_row_minimums(payoff_matrix))}")
    print(f"  Column maximums: {format_vector(find_column_maximums(payoff_matrix))}")
    print(f"  Lower game value α = {lower_game_value} ({format_strategies(maximin_strategies, 'A')})")
    print(f"  Upper game value β = {upper_game_value} ({format_strategies(minimax_strategies, 'B')})")
    saddle_points = find_saddle_points(payoff_matrix)
    print(f"  Saddle points: {saddle_points if saddle_points else 'none, α < β -> mixed strategies'}")


def shift_matrix(payoff_matrix):
    """Step 2: make all elements positive so that the game value V' > 0."""
    print("\n2. Positive shift")
    positive_matrix, shift_constant = make_matrix_positive(payoff_matrix)
    print(f"  Smallest element: {payoff_matrix.min()}, shift K = {shift_constant}")
    print_matrix("  Matrix A' = A + K:", positive_matrix)
    return positive_matrix, shift_constant


def solve_with_pulp(positive_matrix):
    """Step 3: solve the pair of dual LPs with PuLP (CBC)."""
    print("\n3. Pair of dual LPs solved with PuLP (CBC)")
    first_player_plan, first_player_optimum = solve_first_player_problem(positive_matrix)
    second_player_plan, second_player_optimum = solve_second_player_problem(positive_matrix)
    print(f"  Player A: Z = Σx -> min, A'^T x >= 1: x* = {format_vector(first_player_plan)}, "
          f"Z* = {format_number(first_player_optimum)}")
    print(f"  Player B: F = Σy -> max, A' y <= 1:   y* = {format_vector(second_player_plan)}, "
          f"F* = {format_number(second_player_optimum)}")
    print(f"  Duality theorem Z* = F*: {first_player_optimum == second_player_optimum}")
    return first_player_plan, second_player_plan, second_player_optimum


def solve_with_simplex(positive_matrix):
    """Step 4: the same LP of player B with the exact tableau simplex method."""
    print("\n4. Tableau simplex method for the LP of player B (exact fractions)")
    row_count, column_count = positive_matrix.shape
    result = simplex_maximize(positive_matrix.tolist(), [1] * row_count, [1] * column_count)
    print(f"  Iteration 0: basis s1..s{row_count}, F = 0")
    for iteration_number, (entering, leaving, objective) in enumerate(result["iterations"], start=1):
        print(f"  Iteration {iteration_number}: {name_variable(entering, column_count)} enters, "
              f"{name_variable(leaving, column_count)} leaves, F = {format_number(objective)}")
    print(f"  y* = {format_vector(result['primal_plan'])}, F* = {format_number(result['optimum'])}")
    print(f"  Dual plan from the final tableau: x* = {format_vector(result['dual_plan'])}")
    return result


def expected_payoffs(payoff_matrix, row_strategy, column_strategy):
    """Return p·A (payoff against each pure column), A·q (payoff of each pure row) and p·A·q."""
    exact_matrix = payoff_matrix.astype(object)
    payoffs_by_column = row_strategy @ exact_matrix
    payoffs_by_row = exact_matrix @ column_strategy
    return payoffs_by_column, payoffs_by_row, row_strategy @ exact_matrix @ column_strategy


def restore_and_check_solution(payoff_matrix, first_player_plan, second_player_plan, optimum, shift_constant):
    """Step 5: p = x·V', q = y·V', V = V' − K, then check p·A >= V and A·q <= V."""
    print("\n5. Optimal strategies and game value")
    shifted_game_value = 1 / optimum
    game_value = shifted_game_value - shift_constant
    row_strategy = first_player_plan * shifted_game_value
    column_strategy = second_player_plan * shifted_game_value
    print(f"  V' = 1 / Z* = {format_number(shifted_game_value)}, "
          f"V = V' − K = {format_number(game_value)} ≈ {float(game_value):.3f}".replace(".", ","))
    print(f"  Player A p* = {format_vector(row_strategy)}, Σp = {format_number(sum(row_strategy))}")
    print(f"  Player B q* = {format_vector(column_strategy)}, Σq = {format_number(sum(column_strategy))}")

    payoffs_by_column, payoffs_by_row, expected_value = expected_payoffs(
        payoff_matrix, row_strategy, column_strategy)
    print(f"  p*·A (vs B1..B5): {format_vector(payoffs_by_column)}, all >= V: "
          f"{all(payoff >= game_value for payoff in payoffs_by_column)}")
    print(f"  A·q* (A1..A4):    {format_vector(payoffs_by_row)}, all <= V: "
          f"{all(payoff <= game_value for payoff in payoffs_by_row)}")
    print(f"  p*·A·q* = {format_number(expected_value)}")


def main():
    print_variant_check(PAYOFF_MATRIX)
    print_matrix("Payoff matrix A:", PAYOFF_MATRIX)
    analyze_pure_strategies(PAYOFF_MATRIX)
    positive_matrix, shift_constant = shift_matrix(PAYOFF_MATRIX)
    first_player_plan, second_player_plan, optimum = solve_with_pulp(positive_matrix)
    simplex_result = solve_with_simplex(positive_matrix)
    same_result = (list(simplex_result["primal_plan"]) == list(second_player_plan)
                   and list(simplex_result["dual_plan"]) == list(first_player_plan)
                   and simplex_result["optimum"] == optimum)
    print(f"  PuLP and the simplex method give the same plans: {same_result}")
    restore_and_check_solution(PAYOFF_MATRIX, first_player_plan, second_player_plan, optimum, shift_constant)


if __name__ == "__main__":
    main()
