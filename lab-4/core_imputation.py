"""Search for one C-core imputation of a game in 0-1 reduced form as a linear program (PuLP + CBC)."""

from fractions import Fraction

import pulp

from cooperative_game import list_proper_coalitions


def find_core_imputation(normalized_function, players, target_shares):
    """Find the C-core imputation X' closest to the target shares.

    Linear program in the 0-1 reduced form:
        minimize   d
        subject to x'_i − p_i ≤ d,  p_i − x'_i ≤ d      (distance to the target p),
                   x'(S) ≥ V'(S) for every proper coalition S (coalition rationality),
                   x'_1 + ... + x'_n = 1                    (collective rationality),
                   x'_i ≥ 0                                 (individual rationality).

    Return the imputation as exact Fractions and the optimal maximum deviation d.
    """
    ordered_players = sorted(players)
    problem = pulp.LpProblem("core_imputation", pulp.LpMinimize)
    shares = {player: pulp.LpVariable(f"share_{player}", lowBound=0) for player in ordered_players}
    maximum_deviation = pulp.LpVariable("maximum_deviation", lowBound=0)

    problem += maximum_deviation
    for player in ordered_players:
        problem += shares[player] - float(target_shares[player]) <= maximum_deviation
        problem += float(target_shares[player]) - shares[player] <= maximum_deviation
    for coalition in list_proper_coalitions(players):
        problem += (pulp.lpSum(shares[player] for player in coalition)
                    >= float(normalized_function[coalition]))
    problem += pulp.lpSum(shares.values()) == 1

    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"The LP has no optimal solution: {pulp.LpStatus[status]}")

    imputation = {player: Fraction(shares[player].value()).limit_denominator(1000)
                  for player in ordered_players}
    deviation = Fraction(maximum_deviation.value()).limit_denominator(1000)
    return imputation, deviation
