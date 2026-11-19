"""Generate dumbbell (Cleveland dot) plot of W1 acting rates per template.

Reads data.csv in this directory and writes fig.pdf and fig.png (300 dpi).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd


def main():
    base_dir = Path(__file__).resolve().parent
    csv_path = base_dir / "data.csv"
    pdf_path = base_dir / "fig.pdf"
    png_path = base_dir / "fig.png"

    # Style configuration
    # Every text element at least 7.0 pt
    plt.rcParams.update({
        "font.size": 7.0,
        "font.family": "serif",
        "font.serif": ["Times New Roman"],  # explicit prose face
        "pdf.fonttype": 42,
    })

    # Palette tokens
    # No-fetch counted filled navy, Shown open dark gray; mint negative-gap rows stay.
    TEAL = "#1F2A44"       # no-fetch counted (navy)
    DARK_TEAL = "#1F2A44"  # its value text
    GRAY = "#555555"       # shown series marker (dark gray)
    DARK_GRAY = "#555555"  # dark gray for high-contrast value text and n column
    MINT = "#5FAF8F"       # mint (negative gap highlight)
    INK = "#333333"        # ink text / axis ticks
    LINE_GRAY = "#D0D0D0"  # connecting line for normal rows

    df = pd.read_csv(csv_path)
    # Compute gap: no-fetch counted minus shown
    df["gap"] = df["No-Fetch Counted"] - df["Shown"]
    # Sort: largest gap at top to smallest at bottom
    # Ascending sort so largest is at highest y-index (top of plot)
    df = df.sort_values(by=["gap", "No-Fetch Counted"], ascending=[True, True]).reset_index(drop=True)

    # 3.25 in wide native column width, height expanded to 3.8 in for 16 rows with 7 pt text
    fig = plt.figure(figsize=(3.25, 3.8))
    # Margins: left=0.28 (0.91 in for template labels), bottom=0.09, width=0.56, height=0.85
    ax = fig.add_axes([0.28, 0.09, 0.56, 0.81])  # top band reserved for the legend

    # Light grid behind data
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#C9C9C9")
    ax.spines["bottom"].set_color("#C9C9C9")

    n_rows = len(df)
    dx = 1.8  # offset in data units between marker center and value text

    for i, row in df.iterrows():
        c = row["No-Fetch Counted"]
        s = row["Shown"]
        gap = row["gap"]
        n = row["No-Fetch n"]

        is_neg = (gap < 0)

        if is_neg:
            # Negative-gap rows (account, dns_record): mint connector line
            ax.plot([c, s], [i, i], color=MINT, linewidth=1.6, zorder=2)
            # Smaller markers on negative gap rows so the mint connector stays visible
            ms_c = 2.4
            ms_s = 2.4
        else:
            # Normal rows: thin gray connector line
            ax.plot([s, c], [i, i], color=LINE_GRAY, linewidth=0.9, zorder=1)
            ms_c = 3.5
            ms_s = 3.5

        # Shown marker: gray open diamond
        ax.plot(s, i, marker="D", markersize=ms_s, markerfacecolor="white",
                markeredgecolor=GRAY, markeredgewidth=0.8, zorder=3)
        # No-fetch counted marker: teal filled circle
        ax.plot(c, i, marker="o", markersize=ms_c, markerfacecolor=TEAL,
                markeredgecolor=TEAL, markeredgewidth=0.5, zorder=4)

        # Print both values on every row outside their markers
        if c > s:
            # Normal rows: shown on left, counted on right
            ax.text(s - dx, i, f"{s}", va="center", ha="right", fontsize=7.0, color=DARK_GRAY)
            ax.text(c + dx, i, f"{c}", va="center", ha="left", fontsize=7.0, color=DARK_TEAL)
        else:
            # Negative gap rows: counted on left, shown on right
            # Opposite sides of the pair keep both labels readable
            ax.text(c - dx, i, f"{c}", va="center", ha="right", fontsize=7.0, color=DARK_TEAL)
            ax.text(s + dx, i, f"{s}", va="center", ha="left", fontsize=7.0, color=DARK_GRAY)
            # Mint gap label beside the pair (e.g. -1, -2)
            ax.text(s + dx + 7.0, i, f"−{abs(gap)}", va="center", ha="left", fontsize=7.0, color=MINT, fontweight="bold")

        # n column: pure right-aligned number in dark gray #555555
        ax.text(1.09, i, f"{n}", transform=ax.get_yaxis_transform(),
                va="center", ha="right", fontsize=7.0, color=DARK_GRAY)

    # Column header for n
    ax.text(1.09, n_rows - 0.25, "n", transform=ax.get_yaxis_transform(),
            va="bottom", ha="right", fontsize=7.5, fontstyle="italic", color=DARK_GRAY)

    # Y-axis ticks and labels
    ax.set_yticks(range(n_rows))
    ax.tick_params(axis="both", which="both", length=2.5, pad=2.0, colors=INK)

    # Template names: mint for negative gap rows, ink for normal rows
    labels = ax.set_yticklabels(df["Template"], fontsize=7.0)
    for i, label in enumerate(labels):
        if df.loc[i, "gap"] < 0:
            label.set_color(MINT)
        else:
            label.set_color(INK)

    ax.set_ylim(-0.7, n_rows - 0.3)

    # X-axis: 0% to 80% (with data margin to 88 for value labels)
    ax.set_xlim(0, 88)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xticklabels(["0%", "20%", "40%", "60%", "80%"], fontsize=7.0)
    ax.set_xlabel("Acting when the claim is false (% of rollouts)", fontsize=7.5, labelpad=2)  # no 5 pt subscript

    # Legend inside the axes where it does not cover data, without heavy frame (frameon=False)
    legend_handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=TEAL,
               markeredgecolor=TEAL, markersize=3.5, label="No-fetch counted"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="white",
               markeredgecolor=GRAY, markeredgewidth=0.8, markersize=3.5, label="Shown"),
        Line2D([0], [0], color=MINT, lw=1.8, label="shown above\nno-fetch")
    ]
    legend_handles[2] = Line2D([0], [0], color=MINT, lw=1.8, label="shown above no-fetch")
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.56, 0.995), ncol=3,
               frameon=False, fontsize=7.0, handlelength=1.1, handletextpad=0.3, columnspacing=0.9,
               borderaxespad=0.0)  # legend in its own band, off the volume / customer / branch rows

    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    print(f"Generated {pdf_path} and {png_path}")


if __name__ == "__main__":
    main()
