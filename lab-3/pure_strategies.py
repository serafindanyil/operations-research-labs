"""Solution of a matrix game in pure strategies (maximin, minimax, saddle points)."""

import numpy as np


def find_row_minimums(payoff_matrix):
    """Return the minimum of every row: the worst payoff of each strategy of player A."""
    return payoff_matrix.min(axis=1)


def find_column_maximums(payoff_matrix):
    """Return the maximum of every column: the worst loss of each strategy of player B."""
    return payoff_matrix.max(axis=0)


def find_lower_game_value(payoff_matrix):
    """Return the lower game value (maximin) and the indices of the maximin strategies of A."""
    row_minimums = find_row_minimums(payoff_matrix)
    lower_game_value = row_minimums.max()
    maximin_strategies = np.flatnonzero(row_minimums == lower_game_value)
    return lower_game_value, maximin_strategies


def find_upper_game_value(payoff_matrix):
    """Return the upper game value (minimax) and the indices of the minimax strategies of B."""
    column_maximums = find_column_maximums(payoff_matrix)
    upper_game_value = column_maximums.min()
    minimax_strategies = np.flatnonzero(column_maximums == upper_game_value)
    return upper_game_value, minimax_strategies


def find_saddle_points(payoff_matrix):
    """Return (row_index, column_index) pairs of elements that are a row minimum and a column maximum."""
    is_row_minimum = payoff_matrix == find_row_minimums(payoff_matrix)[:, np.newaxis]
    is_column_maximum = payoff_matrix == find_column_maximums(payoff_matrix)[np.newaxis, :]
    saddle_positions = np.argwhere(is_row_minimum & is_column_maximum)
    return [(int(row_index), int(column_index)) for row_index, column_index in saddle_positions]
