#!/usr/bin/env python3
"""Plot appendix beeswarm figure for the tau-bench retail experiment:
Per-agent silent wrong-account writes, total wrong-account writes, and reward
across 13 agents on 87 conversations.
- True beeswarm spreading coincident dots horizontally by at least their rendered diameter
- All text at least 7.0 pt (axis labels and titles 7.5 pt)
- Figure height 2.05 in, width 6.75 in
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style tokens
TEAL = "#2F7F93"       # Highlight for twinall (complete lookup)
CORAL = "#ED8D5A"      # Other twin conditions with errors
GRAY = "#9A9A9A"       # Base condition
INK = "#1F2A44"        # Median tick line (navy ink)

plt.rcParams.update({
    "font.size": 7.0,
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # explicit prose face
    "pdf.fonttype": 42
})
# Labels keep the condition codes used in the paper's tau-bench table and caption; prompt line and note
# variants of the released lookup get their own marker shapes.
LABELS = {"base": "1 customer\n(base)", "twin": "Released\n(twin)", "twinfid": "+ prompt\n(twinfid)",
          "twinnote": "+ note\n(twinnote)", "twinall": "Complete\n(twinall)"}
MARKER = {"base": "o", "twin": "o", "twinfid": "^", "twinnote": "s", "twinall": "o"}


def compute_beeswarm(y_vals, x_center, ax, fig, d_pt=4.4):
    """Compute x coordinates in data units such that all points in the condition
    have at least d_pt display points distance from each other (true beeswarm)."""
    disp_pts = [ax.transData.transform((x_center, y)) * 72.0 / fig.dpi for y in y_vals]
    x_c = disp_pts[0][0]
    indices = sorted(range(len(y_vals)), key=lambda i: (y_vals[i], i))
    placed = []
    result_x = [0.0] * len(y_vals)

    for idx in indices:
        y_pt = disp_pts[idx][1]
        best_x = None
        for r in np.arange(0, 50.0, 0.15):
            for sign in [1, -1] if r > 0 else [1]:
                cand_x = x_c + sign * r
                collision = False
                for px, py in placed:
                    dist_sq = (cand_x - px)**2 + (y_pt - py)**2
                    if dist_sq < (d_pt * 0.98)**2:
                        collision = True
                        break
                if not collision:
                    best_x = cand_x
                    break
            if best_x is not None:
                break
        placed.append((best_x, y_pt))
        data_coords = ax.transData.inverted().transform((best_x * fig.dpi / 72.0, y_pt * fig.dpi / 72.0))
        result_x[idx] = data_coords[0]

    return result_x


def main():
    script_dir = Path(__file__).resolve().parent
    csv_path = script_dir / "data.csv"
    df = pd.read_csv(csv_path)

    # Exclude aggregate sum row
    agents = df[~df["Agent"].str.startswith("Sum")].copy()
    assert len(agents) == 13, f"Expected 13 agents, found {len(agents)}"

    # Width: full page width 6.75 in, height: 2.05 in
    fig, axes = plt.subplots(
        1, 3,
        figsize=(6.75, 2.05),
        gridspec_kw={"width_ratios": [1, 1, 1.28]},
        layout="constrained"
    )

    conds_ab = ["twin", "twinfid", "twinnote", "twinall"]
    labels_ab = [LABELS[c] for c in conds_ab]

    conds_c = ["base", "twin", "twinfid", "twinnote", "twinall"]
    labels_c = [LABELS[c] for c in conds_c]

    # Pre-set limits so transData is accurate for display point calculations
    axes[0].set_xlim(-0.5, 3.5)
    axes[0].set_ylim(-1.2, 26)
    axes[1].set_xlim(-0.5, 3.5)
    axes[1].set_ylim(-1.2, 26)
    axes[2].set_xlim(-0.5, 4.5)
    axes[2].set_ylim(-2, 82)
    for ax_, labs in ((axes[0], labels_ab), (axes[1], labels_ab), (axes[2], labels_c)):
        ax_.set_xticks(range(len(labs)))
        ax_.set_xticklabels(labs, fontsize=7.0, linespacing=0.88)

    fig.canvas.draw()

    # (a) Silent wrong-account writes
    ax = axes[0]
    for i, c in enumerate(conds_ab):
        y = agents[f"{c}_silent"].tolist()
        assert len(y) == 13
        x = compute_beeswarm(y, i, ax, fig, d_pt=4.4)
        color = TEAL if c == "twinall" else CORAL
        med = np.median(y)
        pts_at_med = [xi for xi, yi in zip(x, y) if yi == med]
        w = max(0.20, (max(pts_at_med) - min(pts_at_med)) / 2 + 0.06) if pts_at_med else 0.20
        ax.hlines(med, i - w, i + w, color=INK, linewidth=1.2, zorder=2)
        ax.scatter(x, y, s=10, color=color, marker=MARKER[c], edgecolor="white", linewidth=0.35, zorder=3)

    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_ab, fontsize=7.0, linespacing=0.88)
    ax.set_ylabel("conversations (of 87)", fontsize=7.5)
    ax.set_title("(a) Silent wrong-account writes", fontsize=7.5, loc="left", pad=3)
    ax.grid(color="#E6E6E6", linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.0, pad=1.5)

    # (b) Total wrong-account writes
    ax = axes[1]
    for i, c in enumerate(conds_ab):
        y = agents[f"{c}_total"].tolist()
        assert len(y) == 13
        x = compute_beeswarm(y, i, ax, fig, d_pt=4.4)
        color = TEAL if c == "twinall" else CORAL
        med = np.median(y)
        pts_at_med = [xi for xi, yi in zip(x, y) if yi == med]
        w = max(0.20, (max(pts_at_med) - min(pts_at_med)) / 2 + 0.06) if pts_at_med else 0.20
        ax.hlines(med, i - w, i + w, color=INK, linewidth=1.2, zorder=2)
        ax.scatter(x, y, s=10, color=color, marker=MARKER[c], edgecolor="white", linewidth=0.35, zorder=3)

    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.set_xticks(range(4))
    ax.set_xticklabels(labels_ab, fontsize=7.0, linespacing=0.88)
    ax.set_ylabel("conversations (of 87)", fontsize=7.5)
    ax.set_title("(b) Total wrong-account writes", fontsize=7.5, loc="left", pad=3)
    ax.grid(color="#E6E6E6", linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.0, pad=1.5)

    # (c) Reward (tasks completed)
    ax = axes[2]
    for i, c in enumerate(conds_c):
        y = agents[f"{c}_reward"].tolist()
        assert len(y) == 13
        x = compute_beeswarm(y, i, ax, fig, d_pt=4.4)
        color = TEAL if c == "twinall" else (GRAY if c == "base" else CORAL)
        med = np.median(y)
        pts_at_med = [xi for xi, yi in zip(x, y) if yi == med]
        w = max(0.20, (max(pts_at_med) - min(pts_at_med)) / 2 + 0.06) if pts_at_med else 0.20
        ax.hlines(med, i - w, i + w, color=INK, linewidth=1.2, zorder=2)
        ax.scatter(x, y, s=10, color=color, marker=MARKER[c], edgecolor="white", linewidth=0.35, zorder=3)

    ax.set_yticks([0, 20, 40, 60, 80])
    ax.set_xticks(range(5))
    ax.set_xticklabels(labels_c, fontsize=7.0, linespacing=0.88)
    ax.set_ylabel("conversations (of 87)", fontsize=7.5)
    ax.set_title("(c) Reward (tasks completed)", fontsize=7.5, loc="left", pad=3)
    ax.grid(color="#E6E6E6", linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7.0, pad=1.5)

    pdf_out = script_dir / "fig.pdf"
    png_out = script_dir / "fig.png"
    fig.savefig(pdf_out)
    fig.savefig(png_out, dpi=300)
    print(f"Successfully generated {pdf_out.name} and {png_out.name} (300 dpi)")


if __name__ == "__main__":
    main()
