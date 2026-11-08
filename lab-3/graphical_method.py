"""Graphical (envelope) method for (2×n) and (m×2) matrix games with exact arithmetic.

Every pure strategy of the player with many strategies becomes a straight line
    line(t) = intercept + slope · t,   0 ≤ t ≤ 1,
where t is the probability of the first strategy of the player with two strategies.
"""

from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations

import numpy as np

LOWER_ENVELOPE = "lower"
UPPER_ENVELOPE = "upper"


@dataclass
class GraphicalSolution:
    """Result of the graphical method for one game."""

    intercepts: np.ndarray
    slopes: np.ndarray
    envelope_kind: str
    optimal_probability: Fraction
    game_value: Fraction
    active_lines: list
    first_player_strategy: np.ndarray
    second_player_strategy: np.ndarray


def to_fraction_matrix(payoff_matrix):
    """Return a copy of the matrix as a NumPy object array of exact Fractions."""
    return np.vectorize(lambda value: Fraction(int(value)), otypes=[object])(payoff_matrix)


def build_lines_two_by_n(payoff_matrix):
    """Lines of a (2×n) game: fⱼ(x) = a₁ⱼ·x + a₂ⱼ·(1 − x) = a₂ⱼ + (a₁ⱼ − a₂ⱼ)·x, one per column."""
    exact_matrix = to_fraction_matrix(payoff_matrix)
    intercepts = exact_matrix[1, :]
    slopes = exact_matrix[0, :] - exact_matrix[1, :]
    return intercepts, slopes


def build_lines_m_by_two(payoff_matrix):
    """Lines of an (m×2) game: gᵢ(y) = aᵢ₁·y + aᵢ₂·(1 − y) = aᵢ₂ + (aᵢ₁ − aᵢ₂)·y, one per row."""
    exact_matrix = to_fraction_matrix(payoff_matrix)
    intercepts = exact_matrix[:, 1]
    slopes = exact_matrix[:, 0] - exact_matrix[:, 1]
    return intercepts, slopes


def evaluate_lines(intercepts, slopes, probability):
    """Return the value of every line at the given probability."""
    return intercepts + slopes * probability


def evaluate_envelope(intercepts, slopes, probability, envelope_kind):
    """Return the lower (minimum of the lines) or the upper (maximum of the lines) envelope value."""
    line_values = evaluate_lines(intercepts, slopes, probability)
    return line_values.min() if envelope_kind == LOWER_ENVELOPE else line_values.max()


def find_candidate_points(intercepts, slopes):
    """Return the sorted points where the envelope can change: 0, 1 and all line intersections inside [0, 1]."""
    candidate_points = {Fraction(0), Fraction(1)}
    for first_line, second_line in combinations(range(len(slopes)), 2):
        slope_difference = slopes[first_line] - slopes[second_line]
        if slope_difference == 0:
            continue
        intersection = (intercepts[second_line] - intercepts[first_line]) / slope_difference
        if 0 <= intersection <= 1:
            candidate_points.add(intersection)
    return sorted(candidate_points)


def find_envelope_optimum(intercepts, slopes, envelope_kind):
    """Find the maximum of the lower envelope or the minimum of the upper envelope.

    A piecewise linear envelope reaches its extremum at one of the candidate points,
    so it is enough to compare the envelope values there. Ties go to the smallest point.
    Returns the optimal probability, the envelope value there and the active line indices.
    """
    candidate_points = find_candidate_points(intercepts, slopes)
    envelope_values = [
        evaluate_envelope(intercepts, slopes, point, envelope_kind) for point in candidate_points
    ]
    best_value = max(envelope_values) if envelope_kind == LOWER_ENVELOPE else min(envelope_values)
    optimal_probability = candidate_points[envelope_values.index(best_value)]
    line_values = evaluate_lines(intercepts, slopes, optimal_probability)
    active_lines = [int(line_index) for line_index in np.flatnonzero(line_values == best_value)]
    return optimal_probability, best_value, active_lines


def solve_opponent_strategy(slopes, active_lines, optimal_probability, envelope_kind):
    """Return the opponent's probabilities over the lines (strategies) at the optimal point.

    The opponent mixes active lines so that the mixed line never crosses the game value:
    - inside (0, 1) the mixed line must be horizontal, so one rising and one falling
      active line are combined with weights solving  w·k₊ + (1 − w)·k₋ = 0;
    - at an end point a single active line with a suitable slope sign is played purely.
    """
    opponent_strategy = np.array([Fraction(0)] * len(slopes), dtype=object)
    horizontal_lines = [line_index for line_index in active_lines if slopes[line_index] == 0]
    if horizontal_lines:
        opponent_strategy[horizontal_lines[0]] = Fraction(1)
        return opponent_strategy

    if optimal_probability in (0, 1):
        # Lower envelope max at t = 1 (or upper envelope min at t = 0) needs a rising line, otherwise a falling one.
        needs_rising_line = (optimal_probability == 1) == (envelope_kind == LOWER_ENVELOPE)
        slope_sign = 1 if needs_rising_line else -1
        chosen_line = max(active_lines, key=lambda line_index: slope_sign * slopes[line_index])
        opponent_strategy[chosen_line] = Fraction(1)
        return opponent_strategy

    rising_line = next(line_index for line_index in active_lines if slopes[line_index] > 0)
    falling_line = next(line_index for line_index in active_lines if slopes[line_index] < 0)
    rising_weight = slopes[falling_line] / (slopes[falling_line] - slopes[rising_line])
    opponent_strategy[rising_line] = rising_weight
    opponent_strategy[falling_line] = 1 - rising_weight
    return opponent_strategy


def solve_two_by_n_game(payoff_matrix):
    """Solve a (2×n) game: player A maximizes the lower envelope over x = P(A1)."""
    intercepts, slopes = build_lines_two_by_n(payoff_matrix)
    optimal_probability, game_value, active_lines = find_envelope_optimum(
        intercepts, slopes, LOWER_ENVELOPE
    )
    first_player_strategy = np.array([optimal_probability, 1 - optimal_probability], dtype=object)
    second_player_strategy = solve_opponent_strategy(
        slopes, active_lines, optimal_probability, LOWER_ENVELOPE
    )
    return GraphicalSolution(
        intercepts, slopes, LOWER_ENVELOPE, optimal_probability, game_value,
        active_lines, first_player_strategy, second_player_strategy,
    )


def solve_m_by_two_game(payoff_matrix):
    """Solve an (m×2) game: player B minimizes the upper envelope over y = P(B1)."""
    intercepts, slopes = build_lines_m_by_two(payoff_matrix)
    optimal_probability, game_value, active_lines = find_envelope_optimum(
        intercepts, slopes, UPPER_ENVELOPE
    )
    second_player_strategy = np.array([optimal_probability, 1 - optimal_probability], dtype=object)
    first_player_strategy = solve_opponent_strategy(
        slopes, active_lines, optimal_probability, UPPER_ENVELOPE
    )
    return GraphicalSolution(
        intercepts, slopes, UPPER_ENVELOPE, optimal_probability, game_value,
        active_lines, first_player_strategy, second_player_strategy,
    )
