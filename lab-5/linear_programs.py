"""Reduction of a matrix game to a pair of dual linear programs solved with PuLP (CBC solver)."""

from fractions import Fraction

import numpy as np
import pulp

# Largest denominator used to turn the floating-point CBC answer back into an exact fraction.
MAXIMUM_DENOMINATOR = 10000


def make_matrix_positive(payoff_matrix):
    """Shift the matrix by a constant K so that every element becomes strictly positive.

    The shift does not change the optimal strategies and increases the game value by K,
    so V > 0 is guaranteed. If all elements are already positive, K = 0.
    Return the shifted matrix and K.
    """
    smallest_element = int(payoff_matrix.min())
    shift_constant = 1 - smallest_element if smallest_element <= 0 else 0
    return payoff_matrix + shift_constant, shift_constant


def to_fraction(value):
    """Convert a solver value to an exact Fraction (the LP optimum is rational)."""
    return Fraction(value).limit_denominator(MAXIMUM_DENOMINATOR)


def solve_and_check(problem):
    """Solve the problem with CBC and stop if no optimal solution is found."""
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"The LP has no optimal solution: {pulp.LpStatus[status]}")


def solve_first_player_problem(positive_matrix):
    """Solve the LP of player A (rows).

        Z = x_1 + ... + x_m -> min
        subject to  a_1j x_1 + ... + a_mj x_m >= 1,  j = 1, ..., n,
                    x_i >= 0,

    where x_i = p_i / V. Return the plan x and the optimum Z* = 1 / V as Fractions.
    """
    row_count, column_count = positive_matrix.shape
    problem = pulp.LpProblem("first_player", pulp.LpMinimize)
    variables = [pulp.LpVariable(f"x_{row_index + 1}", lowBound=0) for row_index in range(row_count)]

    problem += pulp.lpSum(variables)
    for column_index in range(column_count):
        problem += pulp.lpSum(int(positive_matrix[row_index, column_index]) * variables[row_index]
                              for row_index in range(row_count)) >= 1

    solve_and_check(problem)
    plan = np.array([to_fraction(variable.value()) for variable in variables], dtype=object)
    return plan, to_fraction(pulp.value(problem.objective))


def solve_second_player_problem(positive_matrix):
    """Solve the LP of player B (columns), the dual of the problem of player A.

        F = y_1 + ... + y_n -> max
        subject to  a_i1 y_1 + ... + a_in y_n <= 1,  i = 1, ..., m,
                    y_j >= 0,

    where y_j = q_j / V. Return the plan y and the optimum F* = 1 / V as Fractions.
    """
    row_count, column_count = positive_matrix.shape
    problem = pulp.LpProblem("second_player", pulp.LpMaximize)
    variables = [pulp.LpVariable(f"y_{column_index + 1}", lowBound=0) for column_index in range(column_count)]

    problem += pulp.lpSum(variables)
    for row_index in range(row_count):
        problem += pulp.lpSum(int(positive_matrix[row_index, column_index]) * variables[column_index]
                              for column_index in range(column_count)) <= 1

    solve_and_check(problem)
    plan = np.array([to_fraction(variable.value()) for variable in variables], dtype=object)
    return plan, to_fraction(pulp.value(problem.objective))
