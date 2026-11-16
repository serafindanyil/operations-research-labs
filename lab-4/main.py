"""Lab 4: elements of cooperative game theory, variant 23.

A cooperative game of three players is given by its characteristic function V(S).
The program checks superadditivity and essentiality, builds the 0-1 reduced form,
checks the non-empty C-core condition (Theorem 9), finds one C-core imputation with
a PuLP linear program, computes the Shapley vector and checks that it lies in the C-core.
"""

from fractions import Fraction
from pathlib import Path

from cooperative_game import (
    belongs_to_core,
    check_nonempty_core_condition,
    check_superadditivity,
    compute_coalition_payment,
    compute_cooperation_surplus,
    compute_shapley_value,
    find_core_vertices,
    is_essential,
    list_coalitions,
    normalize_to_zero_one_form,
    restore_from_zero_one_form,
    sum_of_individual_values,
)
from core_imputation import find_core_imputation
from plotting import plot_core

VARIANT_NUMBER = 23
PLAYERS = frozenset({1, 2, 3})
CHARACTERISTIC_FUNCTION = {
    frozenset(): Fraction(0),
    frozenset({1}): Fraction(2400),
    frozenset({2}): Fraction(1800),
    frozenset({3}): Fraction(1800),
    frozenset({1, 2}): Fraction(4600),
    frozenset({1, 3}): Fraction(4800),
    frozenset({2, 3}): Fraction(4200),
    frozenset({1, 2, 3}): Fraction(8000),
}

FIGURES_DIRECTORY = Path(__file__).resolve().parent / "figures"


def format_number(value):
    """Write an exact number as an integer, a fraction like 9100/3, or a decimal with a comma."""
    value = Fraction(value)
    if value.denominator == 1:
        return str(value.numerator)
    if value.denominator in (2, 5, 10):
        return f"{float(value):g}".replace(".", ",")
    return f"{value.numerator}/{value.denominator}"


def format_approximately(value):
    """Write a value rounded to two decimals with a decimal comma, e.g. 3033,33."""
    return f"{float(value):.2f}".replace(".", ",")


def format_vector(values):
    """Write a vector in the report style: (3200; 2400; 2400)."""
    return "(" + "; ".join(format_number(value) for value in values) + ")"


def format_coalition(coalition):
    """Write a coalition as {1;2}, or ∅ for the empty one."""
    if not coalition:
        return "∅"
    return "{" + ";".join(str(player) for player in sorted(coalition)) + "}"


def print_characteristic_function():
    """Print the input data of the variant."""
    print(f"Variant {VARIANT_NUMBER}: characteristic function")
    for coalition in list_coalitions(PLAYERS):
        print(f"  V({format_coalition(coalition)}) = {format_number(CHARACTERISTIC_FUNCTION[coalition])}")


def print_superadditivity_and_essentiality():
    """Task 1: superadditivity table and the essentiality check."""
    print("\n1. Superadditivity: V(A ∪ B) ≥ V(A) + V(B)")
    checks = check_superadditivity(CHARACTERISTIC_FUNCTION, PLAYERS)
    for coalition_a, coalition_b, union_value, separate_sum, holds in checks:
        print(f"  {format_coalition(coalition_a)} and {format_coalition(coalition_b)}: "
              f"{format_number(union_value)} ≥ {format_number(separate_sum)} -> {'holds' if holds else 'FAILS'}")
    print(f"  Superadditive: {all(check[-1] for check in checks)}")

    individual_total = sum_of_individual_values(CHARACTERISTIC_FUNCTION, PLAYERS)
    grand_value = CHARACTERISTIC_FUNCTION[PLAYERS]
    print(f"  Σ V(i) = {format_number(individual_total)}, V(N) = {format_number(grand_value)}, "
          f"essential: {is_essential(CHARACTERISTIC_FUNCTION, PLAYERS)}")
    print(f"  Cooperation surplus V(N) − Σ V(i) = "
          f"{format_number(compute_cooperation_surplus(CHARACTERISTIC_FUNCTION, PLAYERS))}")


def print_zero_one_form(normalized_function):
    """Task 2: the characteristic function in 0-1 reduced form."""
    print("\n2. 0-1 reduced form V'(S)")
    for coalition in list_coalitions(PLAYERS):
        print(f"  V'({format_coalition(coalition)}) = {format_number(normalized_function[coalition])}")


def print_core_condition(normalized_function):
    """Task 3, part 1: Theorem 9, V'(S) ≤ 1 / (n − |S| + 1)."""
    print("\n3. Non-empty C-core condition (Theorem 9)")
    checks = check_nonempty_core_condition(normalized_function, PLAYERS)
    for coalition, normalized_value, upper_bound, holds in checks:
        print(f"  {format_coalition(coalition)}: {format_number(normalized_value)} ≤ "
              f"{format_number(upper_bound)} -> {'holds' if holds else 'FAILS'}")
    print(f"  C-core is non-empty by Theorem 9: {all(check[-1] for check in checks)}")


def print_core_check(title, allocation):
    """Print x(S) against V(S) for every non-empty coalition and the C-core verdict."""
    print(f"  {title}: check x(S) ≥ V(S)")
    for coalition in list_coalitions(PLAYERS):
        if not coalition:
            continue
        payment = compute_coalition_payment(allocation, coalition)
        relation = "=" if payment == CHARACTERISTIC_FUNCTION[coalition] else "≥"
        print(f"    {format_coalition(coalition)}: {format_number(payment)} {relation} "
              f"{format_number(CHARACTERISTIC_FUNCTION[coalition])}")
    print(f"  Belongs to the C-core: {belongs_to_core(allocation, CHARACTERISTIC_FUNCTION, PLAYERS)}")


def find_and_print_core_imputation(normalized_function):
    """Task 3, part 2: the C-core vertices and one imputation found by the LP."""
    core_vertices = find_core_vertices(CHARACTERISTIC_FUNCTION, PLAYERS)
    print("  C-core vertices (x1; x2; x3):")
    for vertex in core_vertices:
        print(f"    {format_vector(vertex)}")

    # Target: share the surplus in proportion to what each player earns alone, p_i = V(i) / Σ V(j).
    individual_total = sum_of_individual_values(CHARACTERISTIC_FUNCTION, PLAYERS)
    target_shares = {player: CHARACTERISTIC_FUNCTION[frozenset({player})] / individual_total
                     for player in sorted(PLAYERS)}
    normalized_imputation, deviation = find_core_imputation(normalized_function, PLAYERS, target_shares)
    imputation = restore_from_zero_one_form(normalized_imputation, CHARACTERISTIC_FUNCTION, PLAYERS)

    print(f"  LP target p = V(i) / Σ V(j) = {format_vector(target_shares.values())}")
    print(f"  LP solution X' = {format_vector(normalized_imputation.values())}, max deviation d = "
          f"{format_number(deviation)}")
    print(f"  X = 2000·X' + V(i) = {format_vector(imputation.values())}, "
          f"Σ = {format_number(sum(imputation.values()))}")
    print_core_check("X", imputation)
    return core_vertices, normalized_imputation, imputation


def compute_and_print_shapley_value():
    """Task 4: the Shapley vector and its C-core membership."""
    print("\n4. Shapley vector")
    shapley_vector = compute_shapley_value(CHARACTERISTIC_FUNCTION, PLAYERS)
    for player, component in shapley_vector.items():
        print(f"  φ{player} = {format_number(component)} ≈ {format_approximately(component)}")
    print(f"  Φ = {format_vector(shapley_vector.values())}, Σ = {format_number(sum(shapley_vector.values()))}")
    print_core_check("Φ", shapley_vector)
    return shapley_vector


def to_zero_one_form(allocation):
    """Convert an imputation to 0-1 form: x'_i = [x_i − V(i)] / [V(N) − Σ V(j)]."""
    cooperation_surplus = compute_cooperation_surplus(CHARACTERISTIC_FUNCTION, PLAYERS)
    return [(allocation[player] - CHARACTERISTIC_FUNCTION[frozenset({player})]) / cooperation_surplus
            for player in sorted(PLAYERS)]


def save_core_figure(normalized_function, core_vertices, normalized_imputation, shapley_vector):
    """Draw the C-core with the LP imputation X and the Shapley vector Φ."""
    normalized_vertices = [to_zero_one_form(dict(zip(sorted(PLAYERS), vertex))) for vertex in core_vertices]
    pair_constraints = [
        (2, normalized_function[frozenset({1, 2})], "x₁ + x₂ = V(1,2)"),
        (1, normalized_function[frozenset({1, 3})], "x₁ + x₃ = V(1,3)"),
        (0, normalized_function[frozenset({2, 3})], "x₂ + x₃ = V(2,3)"),
    ]
    marked_points = [
        (list(normalized_imputation.values()), "Поділ X (ЛП)", "o", "#ff7f0e"),
        (to_zero_one_form(shapley_vector), "Вектор Шеплі Φ", "*", "#ffd700"),
    ]
    surplus = compute_cooperation_surplus(CHARACTERISTIC_FUNCTION, PLAYERS)
    corner_labels = []
    for corner_player in sorted(PLAYERS):
        corner_allocation = [CHARACTERISTIC_FUNCTION[frozenset({player})]
                             + (surplus if player == corner_player else 0) for player in sorted(PLAYERS)]
        corner_labels.append(f"Гравець {corner_player}\n{format_vector(corner_allocation)}")
    output_path = FIGURES_DIRECTORY / "core.png"
    plot_core(normalized_vertices, pair_constraints, marked_points, corner_labels, output_path)
    print(f"\nFigure saved: {output_path}")


def main():
    print_characteristic_function()
    print_superadditivity_and_essentiality()
    normalized_function = normalize_to_zero_one_form(CHARACTERISTIC_FUNCTION, PLAYERS)
    print_zero_one_form(normalized_function)
    print_core_condition(normalized_function)
    core_vertices, normalized_imputation, _ = find_and_print_core_imputation(normalized_function)
    shapley_vector = compute_and_print_shapley_value()
    save_core_figure(normalized_function, core_vertices, normalized_imputation, shapley_vector)


if __name__ == "__main__":
    main()
