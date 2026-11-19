"""Generate appendix stacked bar chart of the API truncation survey.

Horizontal 100% stacked bar chart showing tools that report whether more
records exist vs. those that give no sign by route (page, limit, neither),
with font sizes >= 7.0 pt, brackets showing total no-sign tools,
and outside leader lines for narrow segments.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

def main():
    base_dir = Path(__file__).parent
    csv_path = base_dir / "data.csv"
    pdf_path = base_dir / "fig.pdf"
    png_path = base_dir / "fig.png"

    # Style settings (all text >= 7.0 pt)
    plt.rcParams.update({
        "font.size": 7.0,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],  # explicit prose face
        "hatch.linewidth": 0.5,
        "pdf.fonttype": 42,
    })

    # Read data
    df = pd.read_csv(csv_path)
    for col in ["Capped", "all", "page", "limit", "neither"]:
        df[col] = pd.to_numeric(df[col].replace("--", 0))

    # Difference: tools reporting whether more exist = Capped - all
    df["reports_more"] = df["Capped"] - df["all"]

    # Color tokens from paper palette & colorblind-safe coral family
    COLOR_MINT = "#5FAF8F"      # paper mint token (reports whether more exist)
    COLOR_PAGE = "#C2592B"      # dark terracotta / rust (page, cursor, offset)
    COLOR_LIMIT = "#ED8D5A"     # paper coral token (limit parameter only)
    COLOR_NEITHER = "#F6C4A6"   # light peach (neither)

    colors = [COLOR_MINT, COLOR_PAGE, COLOR_LIMIT, COLOR_NEITHER]
    seg_names = ["reports_more", "page", "limit", "neither"]
    hatches = [None, "////", "\\\\\\", "...."]  # the three no-sign categories also differ in grayscale

    # Figure dimensions: single column 3.25 in wide, height 2.65 in to fit 7pt text comfortably
    fig = plt.figure(figsize=(3.25, 2.65))
    # [left, bottom, width, height]
    ax = fig.add_axes([0.44, 0.16, 0.50, 0.66])  # taller axes, legend closer to the bars

    y_pos = np.array([2.15, 1.15, 0.0])
    bar_h = 0.36
    y_labels = [f"{row.Source} ({int(row.Capped)})" for _, row in df.iterrows()]

    lefts = np.zeros(len(df))
    for seg_idx, (col, color) in enumerate(zip(seg_names, colors)):
        widths = (df[col] / df["Capped"] * 100).to_numpy()
        counts = df[col].to_numpy()
        ax.barh(y_pos, widths, left=lefts, height=bar_h, color=color, edgecolor="white", linewidth=0.7,
                hatch=hatches[seg_idx], zorder=2)
        lefts += widths

    # Segment counts inside where they fit at 7 pt (segments >= 14% width):
    # Row 0: Upstream APIs: 18 inside mint segment (width 100%)
    ax.text(50.0, 2.15, "18", ha="center", va="center", fontsize=7.0, color="#1A1A1A", zorder=3)

    # Row 1: MCP servers, toolkits:
    # 126 inside mint segment (width 59.7%)
    ax.text(59.7156 / 2.0, 1.15, "126", ha="center", va="center", fontsize=7.0, color="#1A1A1A", zorder=3)
    # 38 inside limit segment (width 18.0%)
    ax.text(59.7156 + 8.0569 + 18.0095 / 2.0, 1.15, "38", ha="center", va="center", fontsize=7.0, color="#1A1A1A", zorder=3,
            bbox=dict(fc=COLOR_LIMIT, ec="none", pad=0.4))  # keep the count clear of the hatch
    # 30 inside neither segment (width 14.2%)
    ax.text(59.7156 + 8.0569 + 18.0095 + 14.2180 / 2.0, 1.15, "30", ha="center", va="center", fontsize=7.0, color="#1A1A1A", zorder=3,
            bbox=dict(fc=COLOR_NEITHER, ec="none", pad=0.4))

    # 17 outside with leader (page segment width 8.1% = 9.4 pt < 12 pt text):
    x_17 = 59.7156 + 8.0569 / 2.0
    y_bot_mcp = 1.15 - bar_h / 2.0
    ax.plot([x_17, x_17], [y_bot_mcp, y_bot_mcp - 0.13], color="#333333", linewidth=0.6, zorder=3)
    ax.text(x_17, y_bot_mcp - 0.16, "17", ha="center", va="top", fontsize=7.0, color="#1A1A1A", zorder=3)

    # Row 2: Agent benchmarks:
    # 70 inside page segment (width 85.4%)
    ax.text(85.3659 / 2.0, 0.0, "70", ha="center", va="center", fontsize=7.0, color="white", zorder=3,
            bbox=dict(fc=COLOR_PAGE, ec="none", pad=0.6))

    # 6 and 6 outside with leaders (limit & neither segments width 7.3% each):
    y_bot_bm = 0.0 - bar_h / 2.0
    x_6a = 85.3659 + 7.3171 / 2.0
    x_6b = 85.3659 + 7.3171 + 7.3171 / 2.0
    ax.plot([x_6a, x_6a], [y_bot_bm, y_bot_bm - 0.13], color="#333333", linewidth=0.6, zorder=3)
    ax.text(x_6a, y_bot_bm - 0.16, "6", ha="center", va="top", fontsize=7.0, color="#1A1A1A", zorder=3)

    ax.plot([x_6b, x_6b], [y_bot_bm, y_bot_bm - 0.13], color="#333333", linewidth=0.6, zorder=3)
    ax.text(x_6b, y_bot_bm - 0.16, "6", ha="center", va="top", fontsize=7.0, color="#1A1A1A", zorder=3)

    # Brackets showing total no-sign tools:
    # MCP: over [59.72%, 100%] -> "no sign: 85 (40%)"
    br_y_mcp = 1.15 + bar_h / 2.0 + 0.11
    ax.plot([59.7156, 59.7156, 100.0, 100.0],
            [br_y_mcp - 0.04, br_y_mcp, br_y_mcp, br_y_mcp - 0.04],
            color="#333333", linewidth=0.6, clip_on=False, zorder=4)
    ax.text((59.7156 + 100.0) / 2.0, br_y_mcp + 0.03, "no sign: 85 (40%)",
            ha="center", va="bottom", fontsize=7.0, color="#1A1A1A", clip_on=False, zorder=4)

    # Agent benchmarks: over [0%, 100%] -> "no sign: 82 (100%)"
    br_y_bm = 0.0 + bar_h / 2.0 + 0.11
    ax.plot([0.0, 0.0, 100.0, 100.0],
            [br_y_bm - 0.04, br_y_bm, br_y_bm, br_y_bm - 0.04],
            color="#333333", linewidth=0.6, clip_on=False, zorder=4)
    ax.text(50.0, br_y_bm + 0.03, "no sign: 82 (100%)",
            ha="center", va="bottom", fontsize=7.0, color="#1A1A1A", clip_on=False, zorder=4)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=7.0)
    ax.tick_params(axis="y", length=0, pad=3)

    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{int(v)}%"))
    ax.tick_params(axis="x", labelsize=7.0, pad=2, length=2.5)

    ax.set_ylim(-0.62, 2.42)  # clearance between the two "6" leader counts and the x axis

    # Grid and spines
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5, zorder=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#666666")
    ax.spines["bottom"].set_linewidth(0.5)

    ax.set_xlabel("Share of capped tools", fontsize=7.5, labelpad=2)

    # Legend: 2 rows x 2 cols placed above plot
    legend_labels = ["Reports more exist", "Page, cursor, offset", "Limit parameter only", "Neither"]
    patches = [Patch(facecolor=c, edgecolor="white", hatch=h, label=l) for c, h, l in zip(colors, hatches, legend_labels)]
    # Row-major order: Row 1 = [Reports more exist, Page, cursor, offset], Row 2 = [Limit parameter only, Neither]
    h_reordered = [patches[0], patches[2], patches[1], patches[3]]
    fig.legend(handles=h_reordered, loc="upper center", bbox_to_anchor=(0.53, 0.99),
               ncol=2, frameon=False, fontsize=7.0, handlelength=0.9,
               handleheight=0.6, handletextpad=0.3, columnspacing=1.0, labelspacing=0.25)

    # Save PDF and 300 dpi PNG
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Generated {pdf_path} and {png_path} (300 dpi).")

if __name__ == "__main__":
    main()
