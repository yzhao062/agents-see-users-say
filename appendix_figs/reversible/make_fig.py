"""Appendix figure for the reversible-verb comparison: acting at K = 2, irreversible verb -> reversible verb, three contracts.

Reads data.csv beside this file and writes fig.pdf and fig.png (300 dpi) beside it.
  python make_fig.py

Every text is at least 7 pt at final size (the figure is included at its native 6.75 in width); value labels are
dark shades of the dot colors; two dots whose values are within about one marker diameter are dodged vertically so both
stay visible; the printed labels sit outside the dot pair (lower value to the left, higher value to the right).
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

plt.rcParams.update({"font.size": 7, "font.family": "serif", "pdf.fonttype": 42})

TEAL, ORANGE, PURPLE, GRAY, MINT, CORAL, INK, LGRAY = (
    "#2F7F93", "#D9692F", "#8A6FB0", "#9A9A9A", "#5FAF8F", "#ED8D5A", "#333333", "#C9C9C9")
GRAY_TXT, PURPLE_TXT = "#555555", "#6B46B5"  # value labels; the reversible labels are saturated purple and bold
# so a reader binds each value to its own marker when the dots are close

FS = 7.0        # value labels, names, told column, tick labels, legend (the minimum for every text)
FS_HEAD = 8.0   # panel titles
FS_AX = 7.5     # axis labels and the "told" headings

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE / "data.csv", dtype=str, keep_default_na=False)
n = len(df)
assert n == 13

# ---- layout (inches) ----
W = 6.75
AX_L = 1.32                      # left edge of the first panel (room for the group label and the agent names)
TOLD_GAP, TOLD_W = 0.05, 0.27    # told column: gap after the plot, and its width
GAP_AB, GAP_BC = 0.10, 0.24      # space between the panels (a)-(b) after the told column, and (b)-(c)
RIGHT = 0.02
AW = (W - AX_L - RIGHT - 2 * (TOLD_GAP + TOLD_W) - GAP_AB - GAP_BC) / 3   # plot width of one panel
LEFTS = [AX_L,
         AX_L + AW + TOLD_GAP + TOLD_W + GAP_AB,
         AX_L + 2 * AW + TOLD_GAP + TOLD_W + GAP_AB + GAP_BC]
XLIM = (-10, 124)
PT_PER_U = AW * 72 / (XLIM[1] - XLIM[0])  # points per data unit along x

ROW_IN = 0.158                   # row pitch in inches
BOT, TOP = 0.53, 0.25

# row positions: a small gap between the open-weight and API-hosted groups
GAP = 0.6
ys, g_prev, y = [], None, -1.0
for g in df["Group"]:
    y += 1.0 + (GAP if (g_prev is not None and g != g_prev) else 0.0)
    ys.append(y)
    g_prev = g
YLIM = (ys[-1] + 0.6, ys[0] - 0.6)  # inverted: first table row on top
AH = ROW_IN * (ys[-1] - ys[0] + 1.2)
H = BOT + AH + TOP

PANELS = [  # title, irreversible column, reversible column, told column (None if the table has none)
    ("(a) Page", "Page_irreversible", "Page_reversible", "Told_page"),
    ("(b) Compound", "Compound_irreversible", "Compound_reversible", None),
    ("(c) Documented", "Documented_irreversible", "Documented_reversible", "Told_documented"),
]

# ---- marker geometry (points) ----
MS_G, MS_P = 4.2, 3.2            # gray circle diameter; purple diamond side (half-diagonal = 0.707 * side)
R_G, R_P = MS_G / 2, MS_P * 0.7071
SEP = 4.7                        # vertical separation (pt) of the two dots when they are dodged; > R_G + R_P
DODGE_BELOW = 6.4                # dodge when the dots' centers are closer than this many points horizontally
LABEL_OFF = 4.4                  # distance from a dot center to its value label (pt)
DY = (SEP / 2) / 72 / ROW_IN     # half of the dodge in row units

fig = plt.figure(figsize=(W, H))

# background axes in inch coordinates: coral row bands behind every panel and the names, group labels
bg = fig.add_axes([0, BOT / H, 1, AH / H])
bg.set_xlim(0, W)
bg.set_ylim(*YLIM)
bg.axis("off")
for yy, c in zip(ys, df["Coral"]):
    if c == "True":
        bg.add_patch(Rectangle((0.2, yy - 0.5), W - 0.2 - 0.03, 1.0, facecolor=CORAL, alpha=0.25,
                               edgecolor="none", zorder=0))
group_texts = []
for g in ("open-weight", "Bedrock"):
    yy = [v for v, gg in zip(ys, df["Group"]) if gg == g]
    group_texts.append(bg.text(0.095, (yy[0] + yy[-1]) / 2, {"Bedrock": "API-hosted"}.get(g, g), rotation=90, ha="center", va="center", fontsize=FS,
                               fontstyle="italic", color=GRAY_TXT))


def val_label(ax, x, yy, text, side, color):
    """Value label, placed LABEL_OFF points beyond the dot center on the left ('l') or right ('r')."""
    t = ax.annotate(text, xy=(x, yy), xytext=(LABEL_OFF if side == "r" else -LABEL_OFF, 0),
                    textcoords="offset points", ha="left" if side == "r" else "right", va="center",
                    fontsize=FS, color=color, fontweight="bold" if color == PURPLE_TXT else "normal", zorder=5,
                    annotation_clip=False)
    t.set_gid("val")
    return t


axes = []
for k, (title, ci, cr, ct) in enumerate(PANELS):
    left = LEFTS[k]
    ax = fig.add_axes([left / W, BOT / H, AW / W, AH / H])
    ax.patch.set_alpha(0)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_xticks([0, 50, 100])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="x", labelsize=FS, pad=1.5, length=2, width=0.5)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("% of rollouts acting", labelpad=1, fontsize=FS_AX)
    ax.set_title(title, fontsize=FS_HEAD, loc="left", pad=2, x=-0.01)
    if k == 0:
        ax.set_yticks(ys)
        ax.set_yticklabels(df["Agent"], fontsize=FS)
        for lab, c in zip(ax.get_yticklabels(), df["Coral"]):
            if c == "True":
                lab.set_fontweight("bold")
    else:
        ax.set_yticks([])

    for yy, i, r in zip(ys, df[ci].astype(float), df[cr].astype(float)):
        close = abs(i - r) * PT_PER_U < DODGE_BELOW
        yg, yp = (yy - DY, yy + DY) if close else (yy, yy)  # gray (irreversible) above, purple (reversible) below
        # ax.lines per row, in this order: segment, gray dot, purple dot
        ax.plot([i, r], [yg, yp], color=LGRAY, linewidth=1.3, solid_capstyle="round", zorder=2)
        ax.plot([i], [yg], marker="o", ms=MS_G, mfc=GRAY, mec="white", mew=0.3, ls="", zorder=3, clip_on=False)
        ax.plot([r], [yp], marker="D", ms=MS_P, mfc=PURPLE, mec="white", mew=0.3, ls="", zorder=4, clip_on=False)
        if i == r:  # one label for the equal pair, to the right of the stacked dots
            val_label(ax, r, yy, f"{int(r)}", "r", INK)
        else:
            lo, hi = (i, r) if i < r else (r, i)
            lo_c, hi_c = (GRAY_TXT, PURPLE_TXT) if i < r else (PURPLE_TXT, GRAY_TXT)
            val_label(ax, lo, yy, f"{int(lo)}", "l", lo_c)
            val_label(ax, hi, yy, f"{int(hi)}", "r", hi_c)

    if ct is not None:  # told column, right of the panel it belongs to
        xf = (AW + TOLD_GAP + TOLD_W / 2) / AW
        h = ax.annotate("told", xy=(xf, 1.0), xycoords="axes fraction", xytext=(0, 2), textcoords="offset points",
                        ha="center", va="baseline", fontsize=FS_AX, fontstyle="italic", color=INK,
                        annotation_clip=False)
        h.set_gid("told_head")
        tr = blended_transform_factory(ax.transAxes, ax.transData)  # x in axes fraction, y in data units
        for yy, t in zip(ys, df[ct]):
            tt = ax.text(xf, yy, t, transform=tr, ha="center", va="center", fontsize=FS,
                         color=INK if t != "--" else GRAY_TXT, clip_on=False)
            tt.set_gid("told")
    axes.append(ax)

handles = [
    Line2D([], [], ls="", marker="o", ms=MS_G, mfc=GRAY, mec="white", mew=0.3),
    Line2D([], [], ls="", marker="D", ms=MS_P, mfc=PURPLE, mec="white", mew=0.3),
    Patch(facecolor=CORAL, alpha=0.25, edgecolor="none"),
]
labels = ["irreversible verb", "reversible verb", "reversible verb gained 20+ points under compound or documented"]
leg = fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False, fontsize=FS,
                 handletextpad=0.3, columnspacing=1.2, borderaxespad=0.1, handlelength=1.2)

if __name__ == "__main__":
    fig.savefig(HERE / "fig.pdf")
    fig.savefig(HERE / "fig.png", dpi=300)
    print("wrote fig.pdf and fig.png", fig.get_size_inches())
