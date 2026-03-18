#!/usr/bin/env python3
"""
Fragment layout heatmaps for mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

Each cell shows "T{lane}:i{element_index}" so you can trace exactly which
register slot holds each matrix position.

Generates:
  figures/frag_a_heatmap.png  (16x16)
  figures/frag_b_heatmap.png  (16x8)
  figures/frag_c_heatmap.png  (16x8)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap


# ---------------------------------------------------------------------------
# Fragment mapping functions (mirrors the CUDA get_row / get_col)
# ---------------------------------------------------------------------------

def get_a_maps():
    """Return (thread_map, elem_map) for Fragment A [16x16]."""
    thread_map = np.full((16, 16), -1, dtype=int)
    elem_map = np.full((16, 16), -1, dtype=int)

    for lane in range(32):
        group_id = lane >> 2
        tid_in_group = lane & 3
        for i in range(8):
            row = group_id + 8 * ((i >> 1) & 1)
            col = tid_in_group * 2 + (i & 1) + 8 * (i >> 2)
            thread_map[row, col] = lane
            elem_map[row, col] = i

    return thread_map, elem_map


def get_b_maps():
    """Return (thread_map, elem_map) for Fragment B [16x8]."""
    thread_map = np.full((16, 8), -1, dtype=int)
    elem_map = np.full((16, 8), -1, dtype=int)

    for lane in range(32):
        for i in range(4):
            row = (lane & 3) * 2 + (i & 1) + 8 * (i >> 1)
            col = lane >> 2
            thread_map[row, col] = lane
            elem_map[row, col] = i

    return thread_map, elem_map


def get_c_maps():
    """Return (thread_map, elem_map) for Fragment C/D [16x8]."""
    thread_map = np.full((16, 8), -1, dtype=int)
    elem_map = np.full((16, 8), -1, dtype=int)

    for lane in range(32):
        for i in range(4):
            row = (lane >> 2) + 8 * (i >> 1)
            col = (lane & 3) * 2 + (i & 1)
            thread_map[row, col] = lane
            elem_map[row, col] = i

    return thread_map, elem_map


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_fragment(thread_map, elem_map, title, out_path):
    rows, cols = thread_map.shape

    # Scale figure so cells are roughly square and readable
    cell_size = 0.72
    fig_w = cols * cell_size + 2.5
    fig_h = rows * cell_size + 2.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # 32 distinct colours (one per thread)
    tab20 = plt.colormaps["tab20"]
    tab20b = plt.colormaps["tab20b"]
    colors_32 = [tab20(i / 20) for i in range(20)] + \
                [tab20b(i / 20) for i in range(12)]
    cmap = ListedColormap(colors_32, N=32)
    boundaries = np.arange(-0.5, 32.5, 1)
    norm = BoundaryNorm(boundaries, cmap.N)

    im = ax.imshow(thread_map, cmap=cmap, norm=norm, aspect="equal")

    # Annotate each cell: T{lane}:i{elem}
    for r in range(rows):
        for c in range(cols):
            lane = thread_map[r, c]
            idx = elem_map[r, c]
            label = f"T{lane}:i{idx}"
            # rough luminance check for text contrast
            rgba = cmap(norm(lane))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            color = "white" if lum < 0.45 else "black"
            ax.text(c, r, label, ha="center", va="center",
                    fontsize=7, fontweight="medium", color=color)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.set_title(title, fontsize=12, pad=10)

    cbar = plt.colorbar(im, ax=ax, ticks=range(0, 32, 4), shrink=0.8)
    cbar.set_label("Thread (lane) ID")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, os.pardir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    a_thread, a_elem = get_a_maps()
    b_thread, b_elem = get_b_maps()
    c_thread, c_elem = get_c_maps()

    plot_fragment(a_thread, a_elem,
                  "Fragment A (16x16): mma.m16n8k16 Thread Ownership",
                  os.path.join(fig_dir, "frag_a_heatmap.png"))

    plot_fragment(b_thread, b_elem,
                  "Fragment B (16x8): mma.m16n8k16 Thread Ownership",
                  os.path.join(fig_dir, "frag_b_heatmap.png"))

    plot_fragment(c_thread, c_elem,
                  "Fragment C/D (16x8): mma.m16n8k16 Thread Ownership",
                  os.path.join(fig_dir, "frag_c_heatmap.png"))


if __name__ == "__main__":
    main()
