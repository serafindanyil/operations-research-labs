"""Analysis of a cooperative game given by its characteristic function.

A characteristic function is a dict that maps every coalition (a frozenset of
player numbers) to its guaranteed total payoff V(S). All values are exact Fractions.
"""

from fractions import Fraction
from itertools import combinations
from math import factorial

import numpy as np


def list_coalitions(players):
    """Return every coalition of the players, from the empty one up to the grand coalition."""
    ordered_players = sorted(players)
    coalitions = []
    for coalition_size in range(len(ordered_players) + 1):
        for members in combinations(ordered_players, coalition_size):
            coalitions.append(frozenset(members))
    return coalitions


def list_proper_coalitions(players):
    """Return every coalition except the empty one and the grand coalition."""
    grand_coalition = frozenset(players)
    return [coalition for coalition in list_coalitions(players)
            if coalition and coalition != grand_coalition]


def check_superadditivity(characteristic_function, players):
    """Check V(A ∪ B) ≥ V(A) + V(B) for every pair of non-empty disjoint coalitions.

    Return a list of (coalition_a, coalition_b, union_value, separate_sum, holds).
    """
    coalitions = [coalition for coalition in list_coalitions(players) if coalition]
    checks = []
    for first_index, coalition_a in enumerate(coalitions):
        for coalition_b in coalitions[first_index + 1:]:
            if coalition_a.isdisjoint(coalition_b):
                union_value = characteristic_function[coalition_a | coalition_b]
                separate_sum = characteristic_function[coalition_a] + characteristic_function[coalition_b]
                checks.append((coalition_a, coalition_b, union_value, separate_sum,
                               union_value >= separate_sum))
    return checks


def sum_of_individual_values(characteristic_function, coalition):
    """Return Σ V(i) over the players of the coalition."""
    return sum((characteristic_function[frozenset({player})] for player in coalition), Fraction(0))


def compute_cooperation_surplus(characteristic_function, players):
    """Return V(N) − Σ V(i): the extra payoff the grand coalition creates."""
    grand_coalition = frozenset(players)
    return characteristic_function[grand_coalition] - sum_of_individual_values(
        characteristic_function, grand_coalition)


def is_essential(characteristic_function, players):
    """A game is essential when Σ V(i) < V(N), i.e. the cooperation surplus is positive."""
    return compute_cooperation_surplus(characteristic_function, players) > 0


def normalize_to_zero_one_form(characteristic_function, players):
    """Return the 0-1 reduced form V'(S) = [V(S) − Σ_{i∈S} V(i)] / [V(N) − Σ_{i∈N} V(i)]."""
    cooperation_surplus = compute_cooperation_surplus(characteristic_function, players)
    if cooperation_surplus <= 0:
        raise ValueError("The 0-1 reduced form exists only for an essential game.")
    return {
        coalition: (characteristic_function[coalition]
                    - sum_of_individual_values(characteristic_function, coalition)) / cooperation_surplus
        for coalition in list_coalitions(players)
    }


def check_nonempty_core_condition(normalized_function, players):
    """Check Theorem 9: V'(S) ≤ 1 / (n − |S| + 1) for every non-empty coalition S.

    Return a list of (coalition, normalized_value, upper_bound, holds).
    """
    player_count = len(players)
    checks = []
    for coalition in list_coalitions(players):
        if not coalition:
            continue
        upper_bound = Fraction(1, player_count - len(coalition) + 1)
        normalized_value = normalized_function[coalition]
        checks.append((coalition, normalized_value, upper_bound, normalized_value <= upper_bound))
    return checks


def compute_coalition_payment(allocation, coalition):
    """Return x(S) = Σ_{i∈S} x_i for an allocation indexed by player number."""
    return sum((allocation[player] for player in coalition), Fraction(0))


def belongs_to_core(allocation, characteristic_function, players):
    """Theorem 6: X is in the C-core iff Σ x_i = V(N) and x(S) ≥ V(S) for every coalition S."""
    grand_coalition = frozenset(players)
    if compute_coalition_payment(allocation, grand_coalition) != characteristic_function[grand_coalition]:
        return False
    return all(compute_coalition_payment(allocation, coalition) >= characteristic_function[coalition]
               for coalition in list_coalitions(players))


def restore_from_zero_one_form(normalized_allocation, characteristic_function, players):
    """Convert a 0-1 form imputation back: x_i = [V(N) − Σ V(j)] · x'_i + V(i)."""
    cooperation_surplus = compute_cooperation_surplus(characteristic_function, players)
    return {
        player: cooperation_surplus * normalized_allocation[player]
        + characteristic_function[frozenset({player})]
        for player in sorted(players)
    }


def find_core_vertices(characteristic_function, players):
    """Find the vertices of the C-core of a game.

    Every vertex is a point where Σ x_i = V(N) and n − 1 coalition constraints
    x(S) = V(S) are active. Each such linear system is solved with NumPy and the
    solution is kept when it belongs to the core.
    """
    ordered_players = sorted(players)
    grand_coalition = frozenset(players)
    proper_coalitions = list_proper_coalitions(players)
    vertices = set()
    for active_coalitions in combinations(proper_coalitions, len(ordered_players) - 1):
        coefficient_rows = [[1.0] * len(ordered_players)]
        right_hand_side = [float(characteristic_function[grand_coalition])]
        for coalition in active_coalitions:
            coefficient_rows.append([1.0 if player in coalition else 0.0 for player in ordered_players])
            right_hand_side.append(float(characteristic_function[coalition]))
        coefficient_matrix = np.array(coefficient_rows)
        if abs(np.linalg.det(coefficient_matrix)) < 1e-9:
            continue
        solution = np.linalg.solve(coefficient_matrix, np.array(right_hand_side))
        allocation = {player: Fraction(value).limit_denominator(1000)
                      for player, value in zip(ordered_players, solution)}
        if belongs_to_core(allocation, characteristic_function, players):
            vertices.add(tuple(allocation[player] for player in ordered_players))
    return sort_vertices_around_center(vertices)


def sort_vertices_around_center(vertices):
    """Order polygon vertices counter-clockwise in the (x_1, x_2) plane so they draw as a polygon."""
    if not vertices:
        return []
    vertex_array = np.array([[float(value) for value in vertex] for vertex in vertices])
    center = vertex_array.mean(axis=0)
    return sorted(vertices, key=lambda vertex: float(np.arctan2(
        float(vertex[1]) - center[1], float(vertex[0]) - center[0])))


def compute_shapley_value(characteristic_function, players):
    """Theorem 10: φ_i = Σ_{S∋i} (|S| − 1)!(n − |S|)! / n! · [V(S) − V(S \\ {i})]."""
    player_count = len(players)
    shapley_vector = {}
    for player in sorted(players):
        component = Fraction(0)
        for coalition in list_coalitions(players):
            if player not in coalition:
                continue
            coalition_size = len(coalition)
            weight = Fraction(factorial(coalition_size - 1) * factorial(player_count - coalition_size),
                              factorial(player_count))
            marginal_contribution = (characteristic_function[coalition]
                                     - characteristic_function[coalition - {player}])
            component += weight * marginal_contribution
        shapley_vector[player] = component
    return shapley_vector
