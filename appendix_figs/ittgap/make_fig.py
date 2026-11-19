"""Generate appendix forest plot of the ITT gap.

Full-width figure (6.75 in x 3.95 in) with exact table values aligned in text columns:
- Left text columns: Condition (shortened), Agents, Counted, Listed or shown
- Center: Forest plot of the gap with agent-cluster 95% CI (thick bar) and crossed 95% CI (thin line)
- Right text columns: Gap, Agent-cluster 95% CI, Crossed 95% CI
- Teal for referent ("Which record"), orange for stated absence ("Whether to act")
- Real minus signs for all negative values. All fonts >= 7.0 pt.
- Generous spacing between cohort blocks so group headings have clear empty space above them.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

HERE = Path(__file__).resolve().parent

# Color palette tokens
TEAL = "#2F7F93"       # referent ("Which record"); family colors and shapes match the main-text figures
ORANGE = "#D9692F"     # stated absence ("Whether to act")
INK = "#222222"        # primary text (high contrast dark)
DARK_TEAL = "#1A5360"  # series dark
DARK_ORANGE = "#9E4316"
GRAY = "#666666"       # neutral elements / zero line
GRID = "#EAEAEA"       # subtle grid
RULE = "#333333"       # table rule lines

# Figure dimensions (height grown to 3.95 in for clear cohort heading slots)
W_IN, H_IN = 6.75, 3.95
DPI = 300
W_PT = W_IN * 72.0     # 486.0 pt
H_PT = H_IN * 72.0     # 284.4 pt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],  # explicit prose face
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 7.0,
})

def minus(s):
    """Replace ASCII hyphens with real unicode minus signs."""
    return str(s).replace("-", "−")

def load_data():
    with open(HERE / "data.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def main():
    rows = load_data()
    assert len(rows) == 15, f"Expected 15 rows, found {len(rows)}"

    fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)

    # Full-canvas overlay axes in exact points (0 to W_PT, 0 to H_PT)
    ax_bg = fig.add_axes([0, 0, 1, 1])
    ax_bg.set_xlim(0, W_PT)
    ax_bg.set_ylim(0, H_PT)
    ax_bg.axis("off")

    # Geometry for rows from bottom to top
    y_bottom_row = 38.0
    dy_row = 10.0
    gap_struct = 3.0
    dy_head_to_row = 11.5
    gap_block_to_head = 18.5

    # Rows 10..14 (Cohort 3: all 31 agents)
    y_r14 = y_bottom_row
    y_r13 = y_r14 + dy_row
    y_r12 = y_r13 + dy_row + gap_struct
    y_r11 = y_r12 + dy_row
    y_r10 = y_r11 + dy_row
    y_h3  = y_r10 + dy_head_to_row

    # Rows 5..9 (Cohort 2: Common cohort: clear-acting in both structures)
    y_r9 = y_h3 + gap_block_to_head
    y_r8 = y_r9 + dy_row
    y_r7 = y_r8 + dy_row + gap_struct
    y_r6 = y_r7 + dy_row
    y_r5 = y_r6 + dy_row
    y_h2 = y_r5 + dy_head_to_row

    # Rows 0..4 (Cohort 1: Clear-acting agents of each structure)
    y_r4 = y_h2 + gap_block_to_head
    y_r3 = y_r4 + dy_row
    y_r2 = y_r3 + dy_row + gap_struct
    y_r1 = y_r2 + dy_row
    y_r0 = y_r1 + dy_row
    y_h1 = y_r0 + dy_head_to_row

    row_y = [
        y_r0, y_r1, y_r2, y_r3, y_r4,
        y_r5, y_r6, y_r7, y_r8, y_r9,
        y_r10, y_r11, y_r12, y_r13, y_r14
    ]
    cohort_y = [y_h1, y_h2, y_h3]
    cohort_names = [
        "Clear-acting agents of each task family",
        "Common cohort: clear-acting in both task families",
        "All 31 agents"
    ]

    # Forest plot axes
    plot_x0 = 226.0
    plot_w = 118.0
    plot_x1 = plot_x0 + plot_w
    plot_y0 = 29.0
    plot_y1 = y_r0 + 6.0

    ax_plot = fig.add_axes([plot_x0 / W_PT, plot_y0 / H_PT, plot_w / W_PT, (plot_y1 - plot_y0) / H_PT])
    ax_plot.set_ylim(plot_y0, plot_y1)
    ax_plot.set_xlim(-16, 66)

    # Forest plot styling
    ax_plot.grid(axis="x", color=GRID, linewidth=0.5, linestyle="-", zorder=0)
    ax_plot.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax_plot.spines[s].set_visible(False)
    ax_plot.spines["bottom"].set_linewidth(0.5)
    ax_plot.spines["bottom"].set_color(RULE)

    # Ticks & labels on x-axis
    ticks = [-10, 0, 10, 20, 30, 40, 50, 60]
    ax_plot.set_xticks(ticks)
    ax_plot.set_xticklabels([minus(str(t)) for t in ticks], fontsize=7.0)
    ax_plot.tick_params(axis="x", labelsize=7.0, pad=2.0, length=2.5, width=0.5)
    ax_plot.set_yticks([])

    # Dashed zero line
    ax_plot.axvline(0, color=GRAY, linestyle="--", linewidth=0.7, zorder=1)

    # X-axis title
    ax_plot.set_xlabel("gap over all rollouts (points)", fontsize=7.5, labelpad=2.5)

    # Column coordinate centers
    X_COND = 6.0
    X_AGENTS = 136.0
    X_COUNTED = 168.5
    X_SHOWN = 205.0

    X_GAP = 361.0
    X_CLUSTER = 407.0
    X_CROSSED = 454.0

    # Table Header layout
    rule_top = y_h1 + 9.0
    header_y = rule_top + 4.5

    ax_bg.text(X_COND, header_y, "Condition", fontsize=7.5, fontweight="bold", va="bottom", ha="left", color=INK)
    ax_bg.text(X_AGENTS, header_y, "Agents", fontsize=7.0, fontweight="bold", va="bottom", ha="center", color=INK)
    ax_bg.text(X_COUNTED, header_y, "Counted", fontsize=7.0, fontweight="bold", va="bottom", ha="center", color=INK)

    # Two-line header for Listed or shown
    ax_bg.text(X_SHOWN, header_y + 4.5, "Listed or", fontsize=7.0, fontweight="bold", va="bottom", ha="center", color=INK)
    ax_bg.text(X_SHOWN, header_y - 4.5, "shown", fontsize=7.0, fontweight="bold", va="bottom", ha="center", color=INK)

    # Right headers: 95% Confidence Interval in regular weight
    ax_bg.text(X_GAP, header_y, "Gap", fontsize=7.0, fontweight="bold", va="bottom", ha="center", color=INK)
    ax_bg.text((X_CLUSTER + X_CROSSED) / 2, header_y + 5.0, "95% Confidence Interval", fontsize=7.0, fontweight="normal", va="bottom", ha="center", color=INK)
    ax_bg.text(X_CLUSTER, header_y - 4.5, "Agent-cluster", fontsize=7.0, fontweight="bold", va="bottom", ha="center", color=INK)
    ax_bg.text(X_CROSSED, header_y - 4.5, "Crossed", fontsize=7.0, fontweight="bold", va="bottom", ha="center", color=INK)

    # Top and bottom table rules
    ax_bg.plot([X_COND, W_PT - 8.0], [rule_top, rule_top], color=RULE, linewidth=0.6)
    ax_bg.plot([X_COND, W_PT - 8.0], [plot_y0, plot_y0], color=RULE, linewidth=0.6)

    # Cohort section titles
    for c_idx in range(3):
        cy = cohort_y[c_idx]
        cname = cohort_names[c_idx]
        ax_bg.text(X_COND, cy, cname, fontsize=7.5, fontweight="bold", fontstyle="italic", va="center", ha="left", color=INK)

    # Data rows
    for i, r in enumerate(rows):
        y = row_y[i]
        cond_raw = r["Structure and condition"]
        is_referent = "Which record" in cond_raw

        # Shorten condition name by dropping prefix
        if ": " in cond_raw:
            cond_short = cond_raw.split(": ", 1)[1]
        else:
            cond_short = cond_raw

        color = TEAL if is_referent else ORANGE
        text_color = INK
        marker = "o" if is_referent else "s"

        # Colored bullet pip to visually reinforce structure category
        ax_bg.plot(X_COND + 2.0, y, marker=marker, markersize=3.2, markerfacecolor=(color if is_referent else "white"),
                   markeredgecolor=color, markeredgewidth=(0 if is_referent else 0.8))

        # Left text columns
        ax_bg.text(X_COND + 7.5, y, cond_short, fontsize=7.0, va="center", ha="left", color=text_color)
        ax_bg.text(X_AGENTS, y, r["Agents"], fontsize=7.0, va="center", ha="center", color=text_color)
        ax_bg.text(X_COUNTED, y, r["Counted"], fontsize=7.0, va="center", ha="center", color=text_color)
        ax_bg.text(X_SHOWN, y, r["Listed or shown"], fontsize=7.0, va="center", ha="center", color=text_color)

        # Numeric values for forest plot
        gap_val = float(r["Gap"])
        cl_low = float(r["Agent-cluster 95% CI low"])
        cl_high = float(r["Agent-cluster 95% CI high"])
        cr_low = float(r["Crossed 95% CI low"])
        cr_high = float(r["Crossed 95% CI high"])

        # 1. Crossed interval: thin longer line with end caps
        ax_plot.plot([cr_low, cr_high], [y, y], color=color, linewidth=1.1, solid_capstyle="round", zorder=2)
        ax_plot.plot([cr_low, cr_high], [y, y], ls="", marker="|", markersize=3.0, markeredgecolor=color, markeredgewidth=0.8, zorder=2)

        # 2. Agent-cluster interval: thick short bar
        ax_plot.plot([cl_low, cl_high], [y, y], color=color, linewidth=3.4, solid_capstyle="round", zorder=3)

        # 3. Point estimate: marker
        if is_referent:  # filled navy circle with a white rim so it stays visible on the navy bar
            ax_plot.plot(gap_val, y, marker="o", markersize=4.2, markerfacecolor=color, markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        else:
            ax_plot.plot(gap_val, y, marker="s", markersize=4.0, markerfacecolor="white", markeredgecolor=color, markeredgewidth=1.1, zorder=4)

        # Right text columns: exact table values with real minus sign
        gap_str = minus(r["Gap"])
        cluster_str = minus(r["Agent-cluster 95% CI"])
        crossed_str = minus(r["Crossed 95% CI"])

        ax_bg.text(X_GAP, y, gap_str, fontsize=7.0, va="center", ha="center", color=text_color)
        ax_bg.text(X_CLUSTER, y, cluster_str, fontsize=7.0, va="center", ha="center", color=text_color)
        ax_bg.text(X_CROSSED, y, crossed_str, fontsize=7.0, va="center", ha="center", color=text_color)

    # Legend at top of figure (centered horizontally)
    leg_y = header_y + 17.0
    leg_x = 45.0

    # 1. Which record (referent)
    ax_bg.plot(leg_x, leg_y, marker="o", markersize=4.2, markerfacecolor=TEAL, markeredgecolor="white", markeredgewidth=0.6)
    ax_bg.text(leg_x + 6.0, leg_y, "Which-record tasks", fontsize=7.0, va="center", ha="left", color=INK)
    leg_x += 114.0

    # 2. Whether to act (stated absence)
    ax_bg.plot(leg_x, leg_y, marker="s", markersize=4.0, markerfacecolor="white", markeredgecolor=ORANGE, markeredgewidth=1.1)
    ax_bg.text(leg_x + 6.0, leg_y, "Empty-claim tasks", fontsize=7.0, va="center", ha="left", color=INK)
    leg_x += 136.0

    # 3. Agent-cluster 95% CI
    ax_bg.plot([leg_x, leg_x + 12.0], [leg_y, leg_y], color=GRAY, linewidth=3.4, solid_capstyle="round")
    ax_bg.text(leg_x + 16.0, leg_y, "Agent-cluster 95% CI", fontsize=7.0, va="center", ha="left", color=INK)
    leg_x += 105.0

    # 4. Crossed 95% CI
    ax_bg.plot([leg_x, leg_x + 12.0], [leg_y, leg_y], color=GRAY, linewidth=1.1, solid_capstyle="round")
    ax_bg.plot([leg_x, leg_x + 12.0], [leg_y, leg_y], ls="", marker="|", markersize=3.0, markeredgecolor=GRAY, markeredgewidth=0.8)
    ax_bg.text(leg_x + 16.0, leg_y, "Crossed 95% CI", fontsize=7.0, va="center", ha="left", color=INK)

    # Save deliverables
    fig.savefig(HERE / "fig.pdf")
    fig.savefig(HERE / "fig.png", dpi=DPI)
    plt.close(fig)
    print("Wrote fig.pdf and fig.png successfully.")

if __name__ == "__main__":
    main()
