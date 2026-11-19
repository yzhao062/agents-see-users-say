#!/usr/bin/env python3
"""Plot scaling and reasoning effort for open-weight agents.

Reads data.csv beside this file and generates fig.pdf and fig.png (300 dpi).
"""
from pathlib import Path
import csv
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Style configuration
plt.rcParams.update({
    "font.size": 7.0,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # explicit prose face
    "pdf.fonttype": 42,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 7.0,
    "ytick.labelsize": 7.0,
})

# Acts filled navy with solid lines, Knows open dark gray with dashed lines, labels in the series ink
# (teal and orange are reserved for what the tool shows and what the user says).
TEAL = "#1F2A44"          # Acts marker & line (navy)
DARK_TEAL = "#1F2A44"     # Acts value text
ORANGE = "#555555"        # Knows marker & line (dark gray)
DARK_ORANGE = "#555555"   # Knows value text
LIGHT_GRAY = "#E6E6E6"    # Background gridlines
INK = "#333333"           # Spines and text


def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "data.csv"

    # Load data
    with open(csv_path, mode="r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    first_acts = [int(float(r["first, noted"].split("/")[0])) for r in rows]
    first_knows = [int(float(r["first, noted"].split("/")[1])) for r in rows]
    page_acts = [int(float(r["page with count"].split("/")[0])) for r in rows]
    page_knows = [int(float(r["page with count"].split("/")[1])) for r in rows]
    send_listed = [int(float(r["send, listed"])) for r in rows]

    # Coordinates: 10 settings grouped into 3 model families
    xs = np.array([0.0, 1.05, 2.10, 3.15, 4.60, 5.65, 6.70, 8.20, 9.25, 10.30])
    labels = ["9B", "27B, no think", "27B", "122B", "low", "med", "high", "low", "med", "high"]

    # Full page width 6.75 in, height 2.7 in (sufficient vertical margin for group names)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.62, 2.7), sharey=True)  # tight bbox stays under \textwidth
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.38, top=0.91, wspace=0.14)

    def setup_ax(ax, title):
        ax.grid(axis="y", color=LIGHT_GRAY, linewidth=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(INK)
        ax.spines["left"].set_linewidth(0.6)
        ax.spines["bottom"].set_color(INK)
        ax.spines["bottom"].set_linewidth(0.6)

        # Subtle vertical separator lines between model groups
        ax.axvline(3.875, color="#EDEDED", linewidth=0.8, linestyle=":", zorder=0)
        ax.axvline(7.45, color="#EDEDED", linewidth=0.8, linestyle=":", zorder=0)

        # Y-axis limits and ticks
        ax.set_ylim(-14, 126)  # room for two label tiers below the panel titles
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])

        # X-axis limits and ticks
        ax.set_xlim(-0.7, 11.1)
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=7.0)
        ax.tick_params(axis="x", pad=2.0, length=2.5, width=0.5)
        ax.tick_params(axis="y", pad=1.5, length=2.5, width=0.5)

        # Panel title
        ax.set_title(title, loc="left", fontsize=7.5, fontweight="bold", pad=6)

        # Centered model family names below tick labels
        trans = ax.get_xaxis_transform()
        y_text = -0.44
        ax.text(1.575, y_text, "Qwen3.5", transform=trans, ha="center", va="top",
                fontsize=7.5, color="#222222", fontweight="bold")
        ax.text(5.65, y_text, "gpt-oss\n20b", transform=trans, ha="center", va="top",
                fontsize=7.5, color="#222222", fontweight="bold")
        ax.text(9.25, y_text, "gpt-oss\n120b", transform=trans, ha="center", va="top",
                fontsize=7.5, color="#222222", fontweight="bold")

    setup_ax(ax1, "(a) First match, noted")
    setup_ax(ax2, "(b) Page with count")
    setup_ax(ax3, "(c) Send, listed")

    # ==========================================
    # --- PANEL 1: First match, noted ---
    # ==========================================
    idx_20b = slice(4, 7)
    ax1.plot(xs[idx_20b], first_acts[idx_20b], color=TEAL, linewidth=1.2, zorder=2)
    ax1.plot(xs[idx_20b], first_knows[idx_20b], color=ORANGE, linewidth=1.2, linestyle="--", zorder=2)

    idx_120b = slice(7, 10)
    # Beeswarm at i=9: acts offset left, knows offset right
    x_act_120b = [xs[7], xs[8], xs[9] - 0.16]
    x_kno_120b = [xs[7], xs[8], xs[9] + 0.16]
    ax1.plot(x_act_120b, first_acts[idx_120b], color=TEAL, linewidth=1.2, zorder=2)
    ax1.plot(x_kno_120b, first_knows[idx_120b], color=ORANGE, linewidth=1.2, linestyle="--", zorder=2)

    for i in range(10):
        xa = xs[i] - 0.16 if i == 9 else xs[i]
        xk = xs[i] + 0.16 if i == 9 else xs[i]
        ax1.scatter(xa, first_acts[i], s=22, color=TEAL, marker="o", edgecolors=TEAL,
                    zorder=4, label="Acts" if i == 0 else None)
        ax1.scatter(xk, first_knows[i], s=24, facecolors="none", edgecolors=ORANGE,
                    linewidth=1.2, marker="o", zorder=4, label="Knows" if i == 0 else None)

    # Acts in Panel 1: ALL 100s at ONE consistent vertical offset y = 106
    acts_p1_pos = [
        (xs[0], 102, "bottom", "center"),
        (xs[1], 104, "bottom", "center"),        # alternate tiers 104 / 114 so adjacent labels never touch
        (xs[2], 114, "bottom", "center"),
        (xs[3], 104, "bottom", "center"),
        (xs[4], 114, "bottom", "center"),
        (xs[5], 100, "bottom", "center"),
        (xs[6], 97, "bottom", "center"),
        (xs[7], 104, "bottom", "center"),
        (xs[8], 114, "bottom", "center"),
        (xs[9] - 0.16, 90, "bottom", "center"),
    ]
    for i, (lx, ly, va, ha) in enumerate(acts_p1_pos):
        ax1.text(lx, ly, str(first_acts[i]), color=DARK_TEAL, fontsize=7.0,
                 fontweight="normal", va=va, ha=ha, zorder=5)

    # Knows in Panel 1 (clear of dashed line and clear of 84 marker)
    knows_p1_pos = [
        (xs[0], 72, "bottom", "center"),
        (xs[1], 77, "bottom", "center"),
        (xs[2], 80, "top", "center"),
        (xs[3], 83, "top", "center"),
        (xs[4], 7, "top", "center"),
        (xs[5], 82, "bottom", "center"),
        (xs[6], 74, "top", "center"),
        (xs[7], -6, "top", "center"),
        (xs[8] - 0.28, 55, "center", "right"),  # away from dashed line!
        (xs[9] + 0.16, 73, "top", "center"),    # below knows marker at 81, clear of 84 marker!
    ]
    for i, (lx, ly, va, ha) in enumerate(knows_p1_pos):
        ax1.text(lx, ly, str(first_knows[i]), color=DARK_ORANGE, fontsize=7.0,
                 fontweight="normal", va=va, ha=ha, zorder=5)

    # ==========================================
    # --- PANEL 2: Page with count ---
    # ==========================================
    # Beeswarm at i=0 and i=7: acts offset left, knows offset right
    x_act_p2_20b = xs[idx_20b]
    x_kno_p2_20b = xs[idx_20b]
    ax2.plot(x_act_p2_20b, page_acts[idx_20b], color=TEAL, linewidth=1.2, zorder=2)
    ax2.plot(x_kno_p2_20b, page_knows[idx_20b], color=ORANGE, linewidth=1.2, linestyle="--", zorder=2)

    x_act_p2_120b = [xs[7] - 0.16, xs[8], xs[9]]
    x_kno_p2_120b = [xs[7] + 0.16, xs[8], xs[9]]
    ax2.plot(x_act_p2_120b, page_acts[idx_120b], color=TEAL, linewidth=1.2, zorder=2)
    ax2.plot(x_kno_p2_120b, page_knows[idx_120b], color=ORANGE, linewidth=1.2, linestyle="--", zorder=2)

    for i in range(10):
        xa = xs[i] - 0.16 if i in (0, 7) else xs[i]
        xk = xs[i] + 0.16 if i in (0, 7) else xs[i]
        ax2.scatter(xa, page_acts[i], s=22, color=TEAL, marker="o", edgecolors=TEAL, zorder=4)
        ax2.scatter(xk, page_knows[i], s=24, facecolors="none", edgecolors=ORANGE,
                    linewidth=1.2, marker="o", zorder=4)

    # Knows value annotations panel 2 (staggered tier 1 = 104, tier 2 = 114)
    knows_p2_pos = [
        (xs[0] + 0.16, 97, "bottom", "center"),
        (xs[1], 104, "bottom", "center"),
        (xs[2], 114, "bottom", "center"),
        (xs[3], 104, "bottom", "center"),
        (xs[4], 104, "bottom", "center"),
        (xs[5], 114, "bottom", "center"),
        (xs[6], 104, "bottom", "center"),
        (xs[7] + 0.16, 104, "bottom", "center"),
        (xs[8], 114, "bottom", "center"),
        (xs[9], 104, "bottom", "center"),
    ]
    for i, (lx, ly, va, ha) in enumerate(knows_p2_pos):
        ax2.text(lx, ly, str(page_knows[i]), color=DARK_ORANGE, fontsize=7.0,
                 fontweight="normal", va=va, ha=ha, zorder=5)

    # Acts value annotations panel 2 (clear of markers and steep lines)
    acts_p2_pos = [
        (xs[0] - 0.16, 78, "top", "center"),
        (xs[1], 54, "bottom", "center"),
        (xs[2], 34, "top", "center"),
        (xs[3], 52, "top", "center"),
        (xs[4], 64, "top", "center"),
        (xs[5], 44, "top", "center"),
        (xs[6], 19, "top", "center"),
        (xs[7] - 0.40, 95, "center", "right"),  # fully clear of marker!
        (xs[8] + 0.25, 17, "bottom", "left"),   # right of marker, away from steep line!
        (xs[9] + 0.25, 3, "center", "left"),    # right of marker!
    ]
    for i, (lx, ly, va, ha) in enumerate(acts_p2_pos):
        ax2.text(lx, ly, str(page_acts[i]), color=DARK_TEAL, fontsize=7.0,
                 fontweight="normal", va=va, ha=ha, zorder=5)

    # ==========================================
    # --- PANEL 3: Send, listed ---
    # ==========================================
    ax3.plot(xs[idx_20b], send_listed[idx_20b], color=TEAL, linewidth=1.2, zorder=2)
    ax3.plot(xs[idx_120b], send_listed[idx_120b], color=TEAL, linewidth=1.2, zorder=2)

    for i in range(10):
        ax3.scatter(xs[i], send_listed[i], s=22, color=TEAL, marker="o", edgecolors=TEAL, zorder=4)

    send_p3_pos = [
        (xs[0], 102, "bottom", "center"),
        (xs[1], 60, "bottom", "center"),
        (xs[2], 74, "bottom", "center"),
        (xs[3], 90, "bottom", "center"),
        (xs[4], 89, "bottom", "center"),
        (xs[5] + 0.25, 57, "bottom", "left"),  # right and above, clear of lines and marker!
        (xs[6] + 0.25, 16, "bottom", "left"),  # right of marker, clear of steep line!
        (xs[7], 81, "bottom", "center"),
        (xs[8] + 0.25, 10, "bottom", "left"),  # right of marker, clear of steep line!
        (xs[9], -6, "top", "center"),           # below marker, completely separated from 6!
    ]
    for i, (lx, ly, va, ha) in enumerate(send_p3_pos):
        ax3.text(lx, ly, str(send_listed[i]), color=DARK_TEAL, fontsize=7.0,
                 fontweight="normal", va=va, ha=ha, zorder=5)

    ax1.set_ylabel("% of rollouts", fontsize=7.5, labelpad=2)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.02, 0.03), frameon=True, framealpha=0.9,
               edgecolor="#E0E0E0", fancybox=False, fontsize=7.0, handlelength=1.4,
               handletextpad=0.4, borderaxespad=0.2)

    # Save outputs with tight bounding box to prevent any edge clipping
    pdf_path = base_dir / "fig.pdf"
    png_path = base_dir / "fig.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path.name} and {png_path.name} (300 dpi)")


if __name__ == "__main__":
    main()
