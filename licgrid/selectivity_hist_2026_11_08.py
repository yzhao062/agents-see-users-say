"""Figure 5: horizontal dot histograms of each Bedrock agent's separation (acting under the
count when the claim is true minus when it is false, in points), one row without the claim (gray) and one with it.

  python selectivity_cells_2026_11_08.py selectivity-cells.csv
  python selectivity_hist_2026_11_08.py --csv selectivity-cells.csv \
      --out figures/selectivity-hist.pdf
The script asserts the published facts (31 agents, 29 wider, means 15.2 and 60.2) before drawing.
"""
import argparse
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--csv", required=True)
_ap.add_argument("--out", required=True)
_a = _ap.parse_args()
CSV = _a.csv
OUT_PDF = _a.out
OUT_PNG = os.path.splitext(_a.out)[0] + ".png" if os.environ.get("PNG") else None

# ---------------------------------------------------------------- palette
INK = "#1f2a44"
MID = "#9A9A9A"
LIGHT = "#C9C9C9"
ORANGE = "#D9692F"   # the user's claim
TEAL = "#2f7f93"     # what the tool shows (Figure 1)
RED = "#B8322A"      # only the two agents that do not separate more sharply

# ---------------------------------------------------------------- data
rows = []
with open(CSV, newline="") as f:
    for r in csv.DictReader(f):
        rows.append(
            dict(
                agent=r["agent"],
                tn=float(r["counted_true_noclaim"]),
                fn=float(r["counted_false_noclaim"]),
                tc=float(r["counted_true_claim"]),
                fc=float(r["counted_false_claim"]),
                s0=float(r["sep_noclaim"]),
                s1=float(r["sep_claim"]),
            )
        )

# ---------------------------------------------------------------- assertions
checks = []


def check(name, cond):
    assert cond, "FAILED: " + name
    checks.append(name)


n = len(rows)
check("31 agents", n == 31)
check(
    "sep_noclaim = true - false (no claim) for every agent",
    all(abs(r["s0"] - (r["tn"] - r["fn"])) < 1e-9 for r in rows),
)
check(
    "sep_claim = true - false (claim) for every agent",
    all(abs(r["s1"] - (r["tc"] - r["fc"])) < 1e-9 for r in rows),
)
sharper = [r for r in rows if r["s1"] > r["s0"]]
check("29 agents have sep_claim > sep_noclaim", len(sharper) == 29)
mean = lambda k: float(np.mean([r[k] for r in rows]))
check("mean sep without claim = 15.2", round(mean("s0"), 1) == 15.2)
check("mean sep with claim = 60.2", round(mean("s1"), 1) == 60.2)
check("mean acting, false claim 65.2 -> 29.6",
      round(mean("fn"), 1) == 65.2 and round(mean("fc"), 1) == 29.6)
check("mean acting, true claim 80.4 -> 89.7",
      round(mean("tn"), 1) == 80.4 and round(mean("tc"), 1) == 89.7)
others = [r for r in rows if not r["s1"] > r["s0"]]
names = sorted(r["agent"] for r in others)
check("the two that do not separate more sharply are Llama 4 Scout "
      "and Nemotron Nano 12B", names == ["Llama 4 Scout", "Nemotron Nano 12B"])
nano = next(r for r in rows if r["agent"] == "Nemotron Nano 12B")
scout = next(r for r in rows if r["agent"] == "Llama 4 Scout")
check("Nemotron Nano 12B narrows, Llama 4 Scout is unchanged",
      nano["s1"] < nano["s0"] and scout["s1"] == scout["s0"])
for r in rows:
    r["red"] = r["agent"] in names

# ---------------------------------------------------------------- style
plt.rcParams.update(
    {
        "font.family": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
        "font.size": 7.5,
        "text.color": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
    }
)

# ---------------------------------------------------------------- geometry
# One axes; x = separation in points, y = typographic points (1 unit = 1 pt).
FIG_W = 3.25
LEFT_IN, RIGHT_IN = 0.64, 0.10
BOTTOM_IN, TOP_IN = 0.37, 0.03
X0, X1 = -10.0, 100.0
BIN = 10.0

PITCH = 8.0           # pt between dot centres, both ways
DIA = 7.0             # pt dot diameter
ROW1_H = 66.0         # pt, row without claim
ROW2_H = 54.0         # pt, row with claim
GAP = 10.0            # pt between rows

axes_h_pt = ROW1_H + ROW2_H + GAP
FIG_H = BOTTOM_IN + TOP_IN + axes_h_pt / 72.0
assert FIG_H <= 2.9, FIG_H

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=300)
ax = fig.add_axes(
    [LEFT_IN / FIG_W, BOTTOM_IN / FIG_H,
     (FIG_W - LEFT_IN - RIGHT_IN) / FIG_W, axes_h_pt / 72.0 / FIG_H]
)
ax_w_in = FIG_W - LEFT_IN - RIGHT_IN
pt_per_x = ax_w_in * 72.0 / (X1 - X0)      # points per unit of separation
ax.set_xlim(X0, X1)
ax.set_ylim(0, axes_h_pt)

y2 = 0.0                      # baseline of the claim row (bottom axis)
y1 = ROW2_H + GAP             # baseline of the no-claim row
top1 = y1 + ROW1_H
top2 = y2 + ROW2_H

# recessive grid at bin edges, zero line a little stronger
for xv in np.arange(0, 101, 10):
    ax.plot([xv, xv], [y2, top1], color="#E6E6E6", lw=0.45, zorder=0)
ax.plot([0, 0], [y2, top1], color=MID, lw=0.7, ls=(0, (2.5, 2)), zorder=0.5)
for yb in (y1, y2):
    ax.plot([X0, X1], [yb, yb], color=LIGHT, lw=0.7, zorder=0.6)


drawn = {}


def stack(values, reds, base, color, label):
    """Dot histogram: one dot per agent, 10-point bins, two dots per level."""
    bins = {}
    for v, rd in zip(values, reds):
        b = int(np.floor(v / BIN + 1e-9))
        bins.setdefault(b, []).append((v, rd))
    dx = (PITCH / 2.0) / pt_per_x            # half pitch in data units
    drawn[label] = [0, 0]                    # [dots, red dots]
    for b, items in bins.items():
        items.sort(key=lambda t: (not t[1], t[0]))   # reds sit at the base
        cx = b * BIN + BIN / 2.0
        m = len(items)
        for i, (v, rd) in enumerate(items):
            lvl, col = divmod(i, 2)
            last_alone = (i == m - 1) and (m % 2 == 1)
            x = cx if last_alone else cx + (-dx if col == 0 else dx)
            y = base + 4.5 + PITCH * lvl
            # the two exceptions keep their row color and get a navy ring (red is reserved for harm)
            ax.scatter([x], [y], s=DIA ** 2, color=color, edgecolor=(INK if rd else "white"),
                       linewidth=(1.1 if rd else 0.5), zorder=3, clip_on=False)
            drawn[label][0] += 1
            drawn[label][1] += int(rd)
    return bins


v0 = [r["s0"] for r in rows]
v1 = [r["s1"] for r in rows]
rd = [r["red"] for r in rows]
b0 = stack(v0, rd, y1, "#8C8C8C", "noclaim")
b1 = stack(v1, rd, y2, ORANGE, "claim")

# ---------------------------------------------------------------- means
m0, m1 = mean("s0"), mean("s1")
for mv, base, top in ((m0, y1, top1), (m1, y2, top2)):
    ax.plot([mv, mv], [base, top - 9], color=INK, lw=1.0, zorder=4,
            solid_capstyle="butt")
    ax.scatter([mv], [top - 9], marker="v", s=22, color=INK, zorder=4,
               linewidth=0, clip_on=False)
ax.text(m0 + 2.2, top1 - 9, "mean %d" % round(m0), ha="left", va="center",
        fontsize=7.5, fontweight="bold", color=INK, zorder=5)
ax.text(m1 + 2.2, top2 - 9, "mean %d" % round(m1), ha="left", va="center",
        fontsize=7.5, fontweight="bold", color=INK, zorder=5)

# ---------------------------------------------------------------- row labels
ax.text(X0 - 3.0, y1 + 4.5 + PITCH * 1.5, "No claim", ha="right", va="center",
        fontsize=8, color=INK, clip_on=False)
ax.text(X0 - 3.0, y2 + 4.5 + PITCH * 1.0, "Claim\nstated", ha="right",
        va="center", fontsize=8, color=INK, clip_on=False, linespacing=1.0)


# ---------------------------------------------------------------- the 2 red agents
WHITE = dict(fc='white', ec='none', pad=0.8)
red_dx = (PITCH / 2.0) / pt_per_x
ty = y2 + 27.0
ax.text(14.0, ty, "2 of 31 agents do not\nseparate more sharply", ha="left",
        va="center", fontsize=7.5, color=INK, linespacing=1.05, zorder=5,
        bbox=WHITE)
lx = 11.4
ax.plot([13.4, lx, lx, 5.0 + red_dx + 2.3], [ty + 4.0, ty + 4.0, y2 + 4.5, y2 + 4.5],
        color=INK, lw=0.7, zorder=2.5, solid_joinstyle="miter")

# ---------------------------------------------------------------- how it happens
fa0, fa1 = mean("fn"), mean("fc")
ta0, ta1 = mean("tn"), mean("tc")
bx = 100.0
yy = y1 + 30.0
ax.text(bx, yy + 10.5, "Mean acting (%), no claim \u2192 claim", ha="right",
        va="center", fontsize=7.5, color=INK, style="italic", zorder=5, bbox=WHITE)
ax.text(bx, yy, "claim true  %d \u2192 %d" % (round(ta0), round(ta1)),
        ha="right", va="center", fontsize=7.5, color=INK, zorder=5,
        bbox=WHITE)
ax.text(bx, yy - 10.5, "claim false  %d \u2192 %d" % (round(fa0), round(fa1)),
        ha="right", va="center", fontsize=7.5, color=INK, zorder=5,
        bbox=WHITE)

# ---------------------------------------------------------------- axis
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MID)
ax.spines["bottom"].set_linewidth(0.8)
ax.set_yticks([])
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.tick_params(axis="x", length=2.5, width=0.7, color=MID, labelsize=7.5,
               pad=2)
ax.set_xlabel("Acting if claim true minus if claim false (points)",
              fontsize=7.5, labelpad=2)

# ---------------------------------------------------------------- layout checks
fig.canvas.draw()
from matplotlib import font_manager as fm
from matplotlib.text import Text

check("one dot per agent in each row (31 + 31)",
      drawn["noclaim"][0] == 31 and drawn["claim"][0] == 31)
check("exactly 2 red dots in each row",
      drawn["noclaim"][1] == 2 and drawn["claim"][1] == 2)
sizes = [t.get_fontsize() for t in fig.findobj(Text)
         if t.get_text().strip() and t.get_visible()]
check("every text element >= 7 pt (min %.1f pt)" % min(sizes), min(sizes) >= 7.0)
check("figure width exactly 3.25 in", fig.get_size_inches()[0] == 3.25)
check("figure height <= 2.9 in (%.2f in)" % fig.get_size_inches()[1],
      fig.get_size_inches()[1] <= 2.9)
fpath = fm.findfont("Times New Roman")  # matplotlib falls back to its default font where Times New Roman is missing
if "Times New Roman" not in fpath:
    print("note: Times New Roman not found; the figure uses", os.path.basename(fpath))
# no text box overlaps another text box
bb = [(t.get_text().replace("\n", " ")[:24], t.get_window_extent())
      for t in fig.findobj(Text)
      if t.get_text().strip() and t.get_visible()
      and t.get_window_extent().width > 0]
clash = [(a[0], b[0]) for i, a in enumerate(bb) for b in bb[i + 1:]
         if a[1].overlaps(b[1])]
check("no two text boxes overlap", not clash)

os.makedirs(os.path.dirname(OUT_PDF) or ".", exist_ok=True)
fig.savefig(OUT_PDF)
if OUT_PNG:
    fig.savefig(OUT_PNG, dpi=300)
print("ok", FIG_W, round(FIG_H, 3))
print("\n".join(checks))
