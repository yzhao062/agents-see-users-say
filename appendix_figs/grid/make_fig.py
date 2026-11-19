"""Appendix heatmap of the 14-agent x 8-condition grid (percent of rollouts acting).

Reads data.csv from this directory and writes fig.pdf and fig.png (300 dpi) beside it.
    python make_fig.py

Every text is FS = 7.5 pt (cell values, row and column labels, header, key, colorbar), at the printed size
because the PDF is included at its native 6.75 in width.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
TEAL, CORAL, INK, GRIDGRAY = "#2F7F93", "#ED8D5A", "#333333", "#E6E6E6"
VAL_INK = "#1A1A1A"  # darker ink for cell values on light fills (higher contrast than the #333333 token)
CORAL_ROWS = {"bare (first match only)", "documented"}  # caption: coral rows hide the second match
K1_ROW = "one match (K = 1), listed"

FS = 7.5  # point size of every text in the figure
GAP = 0.3  # extra vertical space (in row heights) before the K = 1 row
LEFT, CELL_W, BOTTOM, TOP, ROW_H = 1.68, 0.325, 0.47, 0.57, 0.19  # layout in inches
W = 6.75  # full page width in inches
H = BOTTOM + TOP + (8 + GAP) * ROW_H  # about 2.6 in
ROT = 30  # column label angle

CMAP = LinearSegmentedColormap.from_list("white_teal", ["#FFFFFF", TEAL])
NORM = Normalize(0, 100)


def lum(rgb):
    c = np.array(rgb, dtype=float)
    c = np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def text_color(rgb):
    """White on dark cells, dark ink on light cells: whichever gives the higher WCAG contrast."""
    l = lum(rgb)
    c_white = 1.05 / (l + 0.05)
    c_ink = (l + 0.05) / (lum(to_rgb(VAL_INK)) + 0.05)
    return "white" if c_white > c_ink else VAL_INK


def load():
    df = pd.read_csv(HERE / "data.csv", index_col=0)
    assert df.shape == (8, 14), df.shape
    return df


def build(df=None):
    if df is None:
        df = load()
    plt.rcParams.update({"font.size": FS, "font.family": "serif", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([LEFT / W, BOTTOM / H, 14 * CELL_W / W, (8 + GAP) * ROW_H / H])
    ax.set_xlim(0, 14)
    ax.set_ylim(8 + GAP, 0)
    ax.axis("off")

    cells = {}  # (row name, column name) -> annotation Text, for the value check
    for i, (rname, row) in enumerate(df.iterrows()):
        y = i + (GAP if rname == K1_ROW else 0)
        for j, (cname, v) in enumerate(row.items()):
            v = int(v)
            rgb = CMAP(NORM(v))[:3]
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=rgb, edgecolor=GRIDGRAY, linewidth=0.4, zorder=2))
            cells[(rname, cname)] = ax.text(j + 0.5, y + 0.5, f"{v}", ha="center", va="center", fontsize=FS,
                                            color=text_color(rgb), zorder=3)
        ax.text(-0.3, y + 0.5, rname, ha="right", va="center", fontsize=FS, color=INK)
        if rname in CORAL_ROWS:
            ax.add_patch(Rectangle((-0.2, y), 0.12, 1, facecolor=CORAL, edgecolor="none", zorder=2, clip_on=False))

    # light rule in the gap that sets the K = 1 row apart
    ax.plot([-0.3, 14], [7 + GAP / 2] * 2, color=GRIDGRAY, linewidth=0.5, zorder=1, clip_on=False)

    for j, cname in enumerate(df.columns):
        ax.text(j + 0.5, -0.18, cname, rotation=ROT, rotation_mode="anchor", ha="left", va="bottom",
                fontsize=FS, color=INK)
    ax.text(-0.3, -0.25, df.index.name or "What the agent sees (K = 2)", ha="right", va="bottom", fontsize=FS,
            color=INK, style="italic")

    # key for the coral marks and colorbar, in the bottom strip
    sw_h = 0.09  # swatch height in inches
    mid_y = 0.30  # vertical center of the strip line, in inches from the bottom
    fig.add_artist(Rectangle((0.22 / W, (mid_y - sw_h / 2) / H), 0.12 / W, sw_h / H, facecolor=CORAL,
                             edgecolor="none", transform=fig.transFigure))
    fig.text(0.40 / W, mid_y / H, "second match hidden", ha="left", va="center", fontsize=FS, color=INK)
    cb_w, cb_h = 1.30, 0.09
    grid_right = LEFT + 14 * CELL_W
    cax = fig.add_axes([(grid_right - cb_w - 0.07) / W, (mid_y - cb_h / 2) / H, cb_w / W, cb_h / H])
    cb = fig.colorbar(ScalarMappable(norm=NORM, cmap=CMAP), cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    cb.outline.set_edgecolor("#9A9A9A")
    cb.outline.set_linewidth(0.4)
    cax.tick_params(labelsize=FS, length=1.5, width=0.4, pad=1, colors=INK)
    fig.text((grid_right - cb_w - 0.07 - 0.08) / W, mid_y / H, "acting (% of rollouts)", ha="right",
             va="center", fontsize=FS, color=INK)
    return fig, ax, cells


def main():
    fig, ax, cells = build()
    fig.savefig(HERE / "fig.pdf")
    fig.savefig(HERE / "fig.png", dpi=300)
    print("wrote fig.pdf and fig.png,", len(cells), "annotated cells")


if __name__ == "__main__":
    main()
