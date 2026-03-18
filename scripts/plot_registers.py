#!/usr/bin/env python3
"""
Register packing diagram for mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16

Each 32-bit register holds two packed FP16 values.  This script draws a table
for each fragment showing which matrix coordinates land in which register,
with the lo/hi half labelled.

Generates:
  figures/register_packing.png
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# ---------------------------------------------------------------------------
# Layout data
# ---------------------------------------------------------------------------

# For a single thread (parameterised by lane):
#   r = group_id = lane >> 2      (Fragment A row base)
#   c = tid_in_group*2 = (lane&3)*2   (Fragment A col base)
#
# Fragment A  (4 regs, 8 elements)
#   reg[0]: i=0 -> (r,   c)     | i=1 -> (r,   c+1)
#   reg[1]: i=2 -> (r+8, c)     | i=3 -> (r+8, c+1)
#   reg[2]: i=4 -> (r,   c+8)   | i=5 -> (r,   c+9)
#   reg[3]: i=6 -> (r+8, c+8)   | i=7 -> (r+8, c+9)
#
# Fragment B  (2 regs, 4 elements)
#   kr = (lane&3)*2,  kc = lane>>2
#   reg[0]: i=0 -> (kr,   kc)   | i=1 -> (kr+1, kc)
#   reg[1]: i=2 -> (kr+8, kc)   | i=3 -> (kr+9, kc)
#
# Fragment C  (2 regs, 4 elements)
#   cr = lane>>2,  cc = (lane&3)*2
#   reg[0]: i=0 -> (cr,   cc)   | i=1 -> (cr,   cc+1)
#   reg[1]: i=2 -> (cr+8, cc)   | i=3 -> (cr+8, cc+1)


def draw_reg_table(ax, title, subtitle, reg_labels):
    """
    reg_labels: list of (reg_name, lo_label, hi_label)
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(-len(reg_labels) - 0.5, 1.5)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(5, 1.2, title, ha="center", va="center",
            fontsize=14, fontweight="bold")
    ax.text(5, 0.6, subtitle, ha="center", va="center",
            fontsize=10, fontstyle="italic", color="0.35")

    header_y = -0.1
    ax.text(1.5, header_y, "Register", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(4.5, header_y, "lo 16 bits (i even)", ha="center", va="center",
            fontsize=10, fontweight="bold")
    ax.text(7.5, header_y, "hi 16 bits (i odd)", ha="center", va="center",
            fontsize=10, fontweight="bold")

    colors = ["#e8f4fd", "#ffffff"]
    for idx, (reg_name, lo, hi) in enumerate(reg_labels):
        y = -idx - 0.8
        bg = colors[idx % 2]

        # Row background
        rect = patches.FancyBboxPatch((0.2, y - 0.35), 9.6, 0.7,
                                       boxstyle="round,pad=0.05",
                                       facecolor=bg, edgecolor="0.7",
                                       linewidth=0.8)
        ax.add_patch(rect)

        ax.text(1.5, y, reg_name, ha="center", va="center",
                fontsize=11, fontweight="bold", family="monospace")
        ax.text(4.5, y, lo, ha="center", va="center",
                fontsize=10, family="monospace")
        ax.text(7.5, y, hi, ha="center", va="center",
                fontsize=10, family="monospace")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    fig_dir = os.path.join(script_dir, os.pardir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Fragment A ---
    draw_reg_table(axes[0],
                   "Fragment A  (16x16)",
                   "r = lane>>2,  c = (lane&3)*2",
                   [
                       ("reg[0]", "i0: A[r,   c  ]", "i1: A[r,   c+1]"),
                       ("reg[1]", "i2: A[r+8, c  ]", "i3: A[r+8, c+1]"),
                       ("reg[2]", "i4: A[r,   c+8]", "i5: A[r,   c+9]"),
                       ("reg[3]", "i6: A[r+8, c+8]", "i7: A[r+8, c+9]"),
                   ])

    # --- Fragment B ---
    draw_reg_table(axes[1],
                   "Fragment B  (16x8)",
                   "kr = (lane&3)*2,  kc = lane>>2",
                   [
                       ("reg[0]", "i0: B[kr,   kc]", "i1: B[kr+1, kc]"),
                       ("reg[1]", "i2: B[kr+8, kc]", "i3: B[kr+9, kc]"),
                   ])

    # --- Fragment C/D ---
    draw_reg_table(axes[2],
                   "Fragment C/D  (16x8)",
                   "cr = lane>>2,  cc = (lane&3)*2",
                   [
                       ("reg[0]", "i0: C[cr,   cc  ]", "i1: C[cr,   cc+1]"),
                       ("reg[1]", "i2: C[cr+8, cc  ]", "i3: C[cr+8, cc+1]"),
                   ])

    plt.tight_layout()
    out_path = os.path.join(fig_dir, "register_packing.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
