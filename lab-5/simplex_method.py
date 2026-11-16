"""Tableau simplex method in exact arithmetic for the LP of player B (max F, A·y <= 1, y >= 0)."""

from fractions import Fraction


def build_initial_tableau(constraint_matrix, right_hand_sides, objective_coefficients):
    """Build the starting simplex tableau with one slack variable per constraint.

    Rows 0..m-1 hold the constraints [A | E | b], the last row holds the objective [-c | 0 | 0].
    The slack variables s_1, ..., s_m form the initial basis.
    """
    row_count = len(constraint_matrix)
    tableau = []
    for row_index, (row, right_hand_side) in enumerate(zip(constraint_matrix, right_hand_sides)):
        if right_hand_side < 0:
            raise ValueError("The slack basis needs nonnegative right-hand sides.")
        slack_columns = [Fraction(int(slack_index == row_index)) for slack_index in range(row_count)]
        tableau.append([Fraction(int(value)) for value in row] + slack_columns + [Fraction(right_hand_side)])
    objective_row = ([-Fraction(coefficient) for coefficient in objective_coefficients]
                     + [Fraction(0)] * row_count + [Fraction(0)])
    tableau.append(objective_row)
    basis = [len(objective_coefficients) + row_index for row_index in range(row_count)]
    return tableau, basis


def choose_entering_column(objective_row):
    """Return the first column with a negative reduced cost, or None if the plan is optimal."""
    for column_index, reduced_cost in enumerate(objective_row[:-1]):
        if reduced_cost < 0:
            return column_index
    return None


def choose_leaving_row(tableau, basis, entering_column):
    """Minimum ratio test b_i / a_ik over a_ik > 0; ties go to the lowest basic variable index."""
    candidates = [(tableau[row_index][-1] / tableau[row_index][entering_column], basis[row_index], row_index)
                  for row_index in range(len(basis)) if tableau[row_index][entering_column] > 0]
    if not candidates:
        raise ValueError("The LP is unbounded.")
    return min(candidates)[2]


def pivot(tableau, pivot_row, pivot_column):
    """Divide the pivot row by the pivot element and eliminate the pivot column from all other rows."""
    pivot_element = tableau[pivot_row][pivot_column]
    tableau[pivot_row] = [value / pivot_element for value in tableau[pivot_row]]
    for row_index, row in enumerate(tableau):
        multiplier = row[pivot_column]
        if row_index != pivot_row and multiplier != 0:
            tableau[row_index] = [value - multiplier * pivot_value
                                  for value, pivot_value in zip(row, tableau[pivot_row])]


def simplex_maximize(constraint_matrix, right_hand_sides, objective_coefficients):
    """Solve max c·y subject to A·y <= b, y >= 0 with the tableau simplex method.

    Return a dictionary with
      primal_plan  - optimal y,
      dual_plan    - optimal plan of the dual problem (reduced costs of the slack columns),
      optimum      - optimal value of the objective,
      iterations   - list of (entering variable index, leaving variable index, objective after the pivot).
    Variable indices 0..n-1 are y_1..y_n, indices n..n+m-1 are the slack variables s_1..s_m.
    """
    variable_count = len(objective_coefficients)
    row_count = len(constraint_matrix)
    tableau, basis = build_initial_tableau(constraint_matrix, right_hand_sides, objective_coefficients)
    iterations = []

    while True:
        entering_column = choose_entering_column(tableau[-1])
        if entering_column is None:
            break
        leaving_row = choose_leaving_row(tableau, basis, entering_column)
        leaving_variable = basis[leaving_row]
        pivot(tableau, leaving_row, entering_column)
        basis[leaving_row] = entering_column
        iterations.append((entering_column, leaving_variable, tableau[-1][-1]))

    all_values = [Fraction(0)] * (variable_count + row_count)
    for row_index, basic_variable in enumerate(basis):
        all_values[basic_variable] = tableau[row_index][-1]
    dual_plan = [tableau[-1][variable_count + row_index] for row_index in range(row_count)]
    return {
        "primal_plan": all_values[:variable_count],
        "dual_plan": dual_plan,
        "optimum": tableau[-1][-1],
        "iterations": iterations,
    }


def name_variable(variable_index, variable_count):
    """Return the variable name for a tableau column: y1..yn, then s1..sm."""
    if variable_index < variable_count:
        return f"y{variable_index + 1}"
    return f"s{variable_index - variable_count + 1}"
