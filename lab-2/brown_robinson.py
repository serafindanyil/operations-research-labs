"""Brown–Robinson iterative method (fictitious play) and exact checks of a mixed-strategy solution."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass
class IterationRecord:
    """State of the fictitious play right after round number `iteration_number`.

    `first_player_strategy` and `second_player_strategy` are the pure strategies played in this round.
    The accumulated payoffs are A·yᴺ (one value per row) and xᴺ·A (one value per column),
    where xᴺ and yᴺ count how many times each strategy has been played so far.
    """

    iteration_number: int
    first_player_strategy: int
    second_player_strategy: int
    first_player_counts: np.ndarray
    second_player_counts: np.ndarray
    accumulated_row_payoffs: np.ndarray
    accumulated_column_payoffs: np.ndarray
    upper_value_estimate: Fraction
    lower_value_estimate: Fraction

    @property
    def approximate_game_value(self):
        """Return vᴺ = (αᴺ + βᴺ) / 2."""
        return (self.upper_value_estimate + self.lower_value_estimate) / 2

    @property
    def first_player_frequencies(self):
        """Return the empirical mixed strategy pᴺ = xᴺ / N of player A."""
        return [Fraction(int(count), self.iteration_number) for count in self.first_player_counts]

    @property
    def second_player_frequencies(self):
        """Return the empirical mixed strategy qᴺ = yᴺ / N of player B."""
        return [Fraction(int(count), self.iteration_number) for count in self.second_player_counts]


def run_brown_robinson(payoff_matrix, iteration_count, first_row_index=0, first_column_index=0):
    """Play `iteration_count` fictitious rounds and return one IterationRecord per round.

    Round 1 uses the given pure strategies. In every next round player A takes the row with the
    largest accumulated payoff A·yᴺ and player B takes the column with the smallest accumulated
    payoff xᴺ·A. np.argmax / np.argmin return the first index, so ties go to the lowest number.
    αᴺ = max(A·yᴺ) / N is the upper estimate and βᴺ = min(xᴺ·A) / N is the lower estimate of v.
    """
    row_count, column_count = payoff_matrix.shape
    first_player_counts = np.zeros(row_count, dtype=int)
    second_player_counts = np.zeros(column_count, dtype=int)
    accumulated_row_payoffs = np.zeros(row_count, dtype=int)
    accumulated_column_payoffs = np.zeros(column_count, dtype=int)

    row_index, column_index = first_row_index, first_column_index
    records = []
    for iteration_number in range(1, iteration_count + 1):
        first_player_counts[row_index] += 1
        second_player_counts[column_index] += 1
        # Adding one column (row) keeps each iteration O(m + n) instead of recomputing A·y and x·A.
        accumulated_row_payoffs += payoff_matrix[:, column_index]
        accumulated_column_payoffs += payoff_matrix[row_index, :]

        records.append(IterationRecord(
            iteration_number=iteration_number,
            first_player_strategy=row_index,
            second_player_strategy=column_index,
            first_player_counts=first_player_counts.copy(),
            second_player_counts=second_player_counts.copy(),
            accumulated_row_payoffs=accumulated_row_payoffs.copy(),
            accumulated_column_payoffs=accumulated_column_payoffs.copy(),
            upper_value_estimate=Fraction(int(accumulated_row_payoffs.max()), iteration_number),
            lower_value_estimate=Fraction(int(accumulated_column_payoffs.min()), iteration_number),
        ))

        row_index = int(np.argmax(accumulated_row_payoffs))
        column_index = int(np.argmin(accumulated_column_payoffs))
    return records


def to_fraction_matrix(payoff_matrix):
    """Convert an integer matrix to a NumPy object array of Fractions for exact arithmetic."""
    return np.array([[Fraction(int(value)) for value in row] for row in payoff_matrix], dtype=object)


def solve_two_by_two_game(sub_matrix):
    """Solve a 2×2 game without a saddle point by the closed-form formulas.

    For [[a, b], [c, d]]: p₁ = (d − c) / D, q₁ = (d − b) / D, v = (ad − bc) / D, where D = a − b − c + d.
    """
    (a, b), (c, d) = to_fraction_matrix(sub_matrix)
    denominator = a - b - c + d
    first_row_probability = (d - c) / denominator
    first_column_probability = (d - b) / denominator
    game_value = (a * d - b * c) / denominator
    return (
        [first_row_probability, 1 - first_row_probability],
        [first_column_probability, 1 - first_column_probability],
        game_value,
    )


def embed_strategy(sub_strategy, active_indices, strategy_count):
    """Place the probabilities of the active strategies into a full-length vector of zeros."""
    full_strategy = np.array([Fraction(0)] * strategy_count, dtype=object)
    for probability, strategy_index in zip(sub_strategy, active_indices):
        full_strategy[strategy_index] = probability
    return full_strategy


def check_mixed_solution(payoff_matrix, first_player_strategy, second_player_strategy, game_value):
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
