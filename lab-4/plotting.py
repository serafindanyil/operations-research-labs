"""Picture of the C-core of a three-player game on the triangle of imputations."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Corners of the imputation triangle on the plane: the point where player k takes the whole surplus.
TRIANGLE_CORNERS = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]])
CORNER_LABEL_OFFSETS = [(-0.03, -0.07), (0.03, -0.07), (0.0, 0.04)]
CORNER_LABEL_ALIGNMENTS = ["right", "left", "center"]


def to_plane(normalized_shares):
    """Map barycentric coordinates (x'_1, x'_2, x'_3) with Σ = 1 to a point on the plane."""
    return np.array([float(share) for share in normalized_shares]) @ TRIANGLE_CORNERS


def compute_pair_line_endpoints(outside_player_index, normalized_pair_value):
    """Return the ends of the line x'_i + x'_j = V'(i, j), i.e. x'_k = 1 − V'(i, j), inside the triangle."""
    outside_share = 1 - float(normalized_pair_value)
    endpoints = []
    for zero_player_index in range(3):
        if zero_player_index == outside_player_index:
            continue
        shares = np.zeros(3)
        shares[outside_player_index] = outside_share
        remaining_player_index = 3 - outside_player_index - zero_player_index
        shares[remaining_player_index] = 1 - outside_share
        endpoints.append(shares @ TRIANGLE_CORNERS)
    return np.array(endpoints)


def plot_core(normalized_core_vertices, pair_constraints, marked_points, corner_labels, output_path):
    """Draw the imputation triangle, the coalition lines, the shaded C-core and the marked imputations.

    normalized_core_vertices: core vertices in 0-1 form, ordered around the polygon.
    pair_constraints: list of (outside_player_index, V'(pair), label).
    marked_points: list of (normalized_shares, label, marker, color).
    corner_labels: text for each triangle corner.
    """
    figure, axes = plt.subplots(figsize=(8, 6.4))

    triangle_outline = np.vstack([TRIANGLE_CORNERS, TRIANGLE_CORNERS[:1]])
    axes.plot(triangle_outline[:, 0], triangle_outline[:, 1], color="black", linewidth=1.2,
              label="Множина поділів")

    core_points = np.array([to_plane(vertex) for vertex in normalized_core_vertices])
    axes.fill(core_points[:, 0], core_points[:, 1], color="#9ecae1", alpha=0.6,
              edgecolor="#08519c", linewidth=1.5, label="C-ядро")

    line_colors = ["#d62728", "#2ca02c", "#9467bd"]
    for (outside_player_index, normalized_pair_value, label), color in zip(pair_constraints, line_colors):
        endpoints = compute_pair_line_endpoints(outside_player_index, normalized_pair_value)
        axes.plot(endpoints[:, 0], endpoints[:, 1], linestyle="--", color=color, linewidth=2,
                  zorder=3, label=label)

    for normalized_shares, label, marker, color in marked_points:
        point = to_plane(normalized_shares)
        axes.scatter(point[0], point[1], marker=marker, s=110, color=color, edgecolor="black",
                     zorder=5, label=label)

    for corner, text, offset, alignment in zip(TRIANGLE_CORNERS, corner_labels,
                                               CORNER_LABEL_OFFSETS, CORNER_LABEL_ALIGNMENTS):
        axes.text(corner[0] + offset[0], corner[1] + offset[1], text, ha=alignment, fontsize=10)

    axes.set_title("C-ядро кооперативної гри на трикутнику поділів")
    axes.set_aspect("equal")
    axes.set_xlim(-0.35, 1.35)
    axes.set_ylim(-0.14, 0.98)
    axes.axis("off")
    axes.legend(loc="upper right", fontsize=9)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
