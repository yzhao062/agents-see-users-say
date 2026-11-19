"""Crossover summary as one full-width, two-panel figure (all text >= 7 pt).

(a) dot plot of Clear, Counted, No-fetch and Shown / listed per condition, with every value annotated;
(b) forest plot of the no-fetch gap with its 95% CI, both exactly as the table prints them (no value is computed).
Agents and fetch share are small text columns after the labels. Reads data.csv beside this file.

Value labels in (a) are placed by a global search: the widest label-label gap that any assignment of
above / below / side positions can reach, then the most conventional assignment among those.

    python make_fig.py        # writes fig.pdf and fig.png (300 dpi) beside it
"""
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

HERE = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 7, "font.family": "serif", "font.serif": ["Times New Roman"], "pdf.fonttype": 42})  # explicit prose face

TEAL, ORANGE, GRAY, MINT, CORAL, INK, LGRAY = "#2F7F93", "#D9692F", "#9A9A9A", "#5FAF8F", "#ED8D5A", "#333333", "#C9C9C9"
D_TEAL, D_ORANGE, D_GRAY = "#1F5F6E", "#A84E1F", "#555555"  # dark value-label shades
NAVY = "#1F2A44"  # Counted and No-fetch in navy (shapes tell them apart); teal and orange are reserved
GRID = "#E6E6E6"
TXT = 7.0  # every text in the figure is at least this size (pt)
AXL = 7.5  # axis labels and group headings (pt)

W = 6.75  # figure width (in)
UNIT = 0.30  # one row pitch (in)
YLIM = (5.3, -0.95)  # data y range, drawn top to bottom
Y_BOT = 0.37  # room for tick labels and the axis label (in)
TOP_BAND = 0.34  # header band above the axes (in)
H = TOP_BAND + (YLIM[0] - YLIM[1]) * UNIT + Y_BOT  # figure height (in)
Y_TOP = H - TOP_BAND
# horizontal layout (in)
X_TICK, X_HEAD, X_LABEL = 0.07, 0.07, 0.15
X_AGENTS, X_FETCH = 2.12, 2.55  # right edges of the two small text columns
A_L, A_W = 2.70, 2.05  # panel (a)
B_L, B_W = 5.08, 0.72  # panel (b)
X_GAPTXT = B_L + B_W + 0.07
XA = (-4.0, 104.0)  # panel (a) x range (%)
PT_PER_X = A_W * 72 / (XA[1] - XA[0])  # points per percent unit in panel (a)

# series: column, marker, size, face, edge, edge width, zorder, preferred annotation sides, text colour
SERIES = [
    ("Clear", "o", 3.6, "white", GRAY, 0.9, 3, ["above", "below", "right", "above_l", "below_l"], INK),
    ("Counted", "D", 2.7, "white", NAVY, 0.6, 4, ["below", "above", "left", "below_l", "above_l"], NAVY),
    ("No-fetch", "o", 4.0, NAVY, "white", 0.35, 6, ["above", "below", "above_r", "right", "below_r"], NAVY),
    ("Shown / listed", "s", 3.4, GRAY, "white", 0.35, 5, ["below", "above", "left", "below_l", "above_l"], D_GRAY),
]
PLACE_ORDER = ["No-fetch", "Shown / listed", "Counted", "Clear"]
OFF = 3.4  # label offset from marker centre (pt)
CANDS = {  # name: (dx, dy, ha, va, extra cost) in points
    "above": (0, OFF, "center", "bottom", 0.0), "below": (0, -OFF, "center", "top", 0.0),
    "right": (4.4, 0, "left", "center", 1.0), "left": (-4.4, 0, "right", "center", 1.0),
    "above_r": (0.8, OFF, "left", "bottom", 0.4), "above_l": (-0.8, OFF, "right", "bottom", 0.4),
    "below_r": (0.8, -OFF, "left", "top", 0.4), "below_l": (-0.8, -OFF, "right", "top", 0.4),
}
PAD_MARK = 1.0  # minimum gap between a label and any marker (pt)
PAD_BAR = 1.0  # minimum gap between a label and a gap connector (pt)
GAP_MARKS = 0.6  # minimum gap between two markers (pt)
LABEL_GAP_TRY = [5.0, 4.5, 4.0, 3.5, 3.0, 2.75, 2.5, 2.25, 2.0, 1.75, 1.5, 1.25]  # label-label gap targets (pt)


def fr(x_in, y_in):
    return x_in / W, y_in / H


def minus(s):
    return s.replace("-", "−")


def style(ax):
    ax.grid(color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=TXT, pad=1.5, length=2.5, width=0.5)
    for s in ("left", "bottom"):
        ax.spines[s].set_linewidth(0.5)
        ax.spines[s].set_color("#555555")


def short_label(name):
    return name.split(": ", 1)[1]


def half_extent(mk, ms, mew):
    """Rendered half-width of a marker in pt (diamond: half-diagonal)."""
    return (0.7071 if mk == "D" else 0.5) * ms + mew / 2


class Box:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1


def gap_between(a, b):
    """Gap in px between two boxes (0 if they overlap)."""
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return math.hypot(dx, dy)


def solve_labels(options, tl_px):
    """options: list (one per label) of [(cost, bbox, cname), ...]; pick one per label so that every pair of
    chosen boxes is at least tl_px apart, with minimum total cost (branch and bound). None if infeasible."""
    n = len(options)
    best = {"cost": float("inf"), "pick": None}
    chosen = []

    def rec(k, cost):
        if cost >= best["cost"]:
            return
        if k == n:
            best["cost"], best["pick"] = cost, list(chosen)
            return
        for c, bb, nm in sorted(options[k], key=lambda t: t[0]):
            if all(gap_between(bb, pb) >= tl_px for _, pb, _ in chosen):
                chosen.append((nm, bb, c))
                rec(k + 1, cost + c)
                chosen.pop()

    rec(0, 0.0)
    return best["pick"], best["cost"]


def main():
    df = pd.read_csv(HERE / "data.csv", dtype=str, keep_default_na=False)
    assert len(df) == 5
    # table's own printed strings for annotation, floats for position
    num = {c: df[c].astype(float) for c in ["Clear", "Counted", "Fetch share", "No-fetch", "Shown / listed", "Gap",
                                            "Gap CI low", "Gap CI high"]}
    groups = [g.split(": ", 1)[0] for g in df["Structure and condition"]]
    assert groups == ["Which record"] * 3 + ["Whether to act"] * 2, groups
    ypos = [0.0, 1.0, 2.0, 3.7, 4.7]
    head_y = {"Which record": -0.60, "Whether to act": 2.95}
    sep = 2.62
    pt_per_y = UNIT * 72  # points per y unit

    # --- spread markers that would overlap (vertical beeswarm dodge, Counted up and Shown / listed down) ---
    spec = {s[0]: s for s in SERIES}
    shift = {c: [0.0] * 5 for c in spec}  # pt, positive = up
    for i in range(5):
        r1 = half_extent(*[spec["Counted"][k] for k in (1, 2, 5)])
        r2 = half_extent(*[spec["Shown / listed"][k] for k in (1, 2, 5)])
        need = r1 + r2 + GAP_MARKS
        dx = abs(num["Counted"][i] - num["Shown / listed"][i]) * PT_PER_X
        if dx < need:
            s = math.sqrt(need ** 2 - dx ** 2)
            shift["Counted"][i], shift["Shown / listed"][i] = s / 2, -s / 2
    ymark = {c: [ypos[i] - shift[c][i] / pt_per_y for i in range(5)] for c in spec}  # y is inverted: up = smaller y

    fig = plt.figure(figsize=(W, H), dpi=300)
    axa = fig.add_axes([*fr(A_L, Y_BOT), A_W / W, (Y_TOP - Y_BOT) / H])
    axb = fig.add_axes([*fr(B_L, Y_BOT), B_W / W, (Y_TOP - Y_BOT) / H], sharey=axa)
    for ax in (axa, axb):
        style(ax)
        ax.set_ylim(*YLIM)
        ax.set_yticks(ypos)
        ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
    axa.set_xlim(*XA)
    axa.set_xticks([0, 50, 100])
    axa.set_xticklabels(["0%", "50%", "100%"])
    axa.set_xlabel("% of rollouts", labelpad=1.5, fontsize=AXL)
    axb.set_xlim(-8, 74)
    axb.set_xticks([0, 20, 40, 60])
    axb.set_xlabel("No-fetch gap (points)", labelpad=1.5, fontsize=AXL)

    # ---- (a) dot plot ----
    for i, y in enumerate(ypos):  # gap connector first (behind the markers)
        axa.plot([num["Shown / listed"][i], num["No-fetch"][i]], [y, y], color=NAVY, alpha=0.25, linewidth=1.6,
                 solid_capstyle="butt", zorder=2)
    handles = []
    for col, mk, ms, fc, ec, mew, z, _, _ in SERIES:
        axa.plot(num[col], ymark[col], ls="", marker=mk, ms=ms, mfc=fc, mec=ec, mew=mew, zorder=z, clip_on=False)
        handles.append(Line2D([], [], ls="", marker=mk, ms=ms, mfc=fc, mec=ec, mew=mew, label=col))

    # ---- (b) forest plot ----
    axb.plot([0, 0], [-0.28, YLIM[0]], color=INK, linewidth=0.6, linestyle=(0, (3, 2)), zorder=2)
    for i, y in enumerate(ypos):
        lo, hi = num["Gap CI low"][i], num["Gap CI high"][i]
        axb.plot([lo, hi], [y, y], color=INK, linewidth=0.9, solid_capstyle="butt", zorder=3)
        axb.plot([lo, hi], [y, y], ls="", marker="|", ms=3.6, mec=INK, mew=0.8, zorder=3)
        # family colors and shapes matching the main-text figures (which-record filled teal circle, empty-claim open orange square)
        if groups[i] == "Which record":
            axb.plot([num["Gap"][i]], [y], ls="", marker="o", ms=3.8, mfc=TEAL, mec="white", mew=0.4, zorder=4)
        else:
            axb.plot([num["Gap"][i]], [y], ls="", marker="s", ms=3.6, mfc="white", mec=ORANGE, mew=0.9, zorder=4)

    # ---- labels, ticks, headings, text columns (figure x, data y) ----
    ty = blended_transform_factory(fig.transFigure, axa.transData)
    label_texts = []
    for i, y in enumerate(ypos):
        colr = MINT if df["Row color"][i] == "mint" else CORAL
        # label tick: a short vertical bar in row coordinates
        fig.add_artist(Line2D([X_TICK / W + 0.0045] * 2, [y - 0.36, y + 0.36], transform=ty, color=colr, linewidth=2.4,
                              solid_capstyle="butt"))
        label_texts.append(axa.text(X_LABEL / W, y, short_label(df["Structure and condition"][i]), transform=ty,
                                    ha="left", va="center", fontsize=TXT, color=INK, clip_on=False))
        axa.text(X_AGENTS / W, y, df["Agents"][i], transform=ty, ha="right", va="center", fontsize=TXT, color=INK,
                 clip_on=False)
        axa.text(X_FETCH / W, y, f"{df['Fetch share'][i]}%", transform=ty, ha="right", va="center", fontsize=TXT,
                 color=INK, clip_on=False)
        gtxt = minus(f"{df['Gap'][i]} [{df['Gap CI low'][i]}, {df['Gap CI high'][i]}]")
        axb.text(X_GAPTXT / W, y, gtxt, transform=ty, ha="left", va="center", fontsize=TXT, color=INK, clip_on=False)
    # headings use the paper's task-family names (data.csv keeps its row keys)
    head_text = {"Which record": "Which-record tasks", "Whether to act": "Empty-claim tasks"}
    for g, y in head_y.items():
        axa.text(X_HEAD / W, y, head_text[g], transform=ty, ha="left", va="center", fontsize=AXL, fontweight="bold", color=INK,
                 clip_on=False)
    # group separator across the full width
    fig.add_artist(Line2D([0.05 / W, (W - 0.05) / W], [sep, sep], transform=ty, color=LGRAY, linewidth=0.5))

    # panel letters inside the (empty) heading row
    for ax, tt in ((axa, "(a)"), (axb, "(b)")):
        ax.text(0.0, head_y["Which record"], tt, transform=blended_transform_factory(ax.transAxes, ax.transData),
                ha="left", va="center", fontsize=TXT, clip_on=False, bbox=dict(fc="white", ec="none", pad=0.6),
                zorder=6)

    # ---- header band: tick legend, column headers, marker legend, gap header ----
    top_y = H - 0.165
    for k, (colr, txt) in enumerate(((MINT, "reference"), (CORAL, "variant"))):
        yy = top_y + 0.058 - k * 0.116
        fig.add_artist(Line2D([(X_TICK + 0.02) / W] * 2, [(yy - 0.045) / H, (yy + 0.045) / H], transform=fig.transFigure,
                              color=colr, linewidth=2.4, solid_capstyle="butt"))
        fig.text((X_TICK + 0.075) / W, yy / H, txt, ha="left", va="center", fontsize=TXT, color=INK)
    fig.text(X_AGENTS / W, top_y / H, "agents", ha="right", va="center", fontsize=TXT, color=D_GRAY)
    fig.text(X_FETCH / W, top_y / H, "fetch\nshare", ha="right", va="center", multialignment="right", fontsize=TXT,
             color=D_GRAY, linespacing=1.05)
    fig.legend(handles=handles, loc="center", bbox_to_anchor=((A_L + A_W / 2 + 0.03) / W, top_y / H), ncol=2,
               frameon=False, fontsize=TXT, handletextpad=0.25, columnspacing=0.9, borderaxespad=0, handlelength=1.0,
               borderpad=0, labelspacing=0.2)
    fig.text(X_GAPTXT / W, top_y / H, "gap [95% CI]", ha="left", va="center", fontsize=TXT, color=INK)

    # ---- value annotations: global search for the widest label gaps ----
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    px = fig.dpi / 72.0
    obst = []  # (kind, box, row): every marker (rendered extent) and every gap connector, in px
    for col, mk, ms, fc, ec, mew, *_ in SERIES:
        for i in range(5):
            cx, cy = axa.transData.transform((num[col][i], ymark[col][i]))
            r = half_extent(mk, ms, mew) * px
            obst.append(("marker", Box(cx - r, cy - r, cx + r, cy + r), i))
    for i, y in enumerate(ypos):
        x0, cy = axa.transData.transform((min(num["Shown / listed"][i], num["No-fetch"][i]), y))
        x1, _ = axa.transData.transform((max(num["Shown / listed"][i], num["No-fetch"][i]), y))
        obst.append(("bar", Box(x0, cy - 0.8 * px, x1, cy + 0.8 * px), i))
    axbb = axa.get_window_extent(rend)
    keys = [(col, i) for col in PLACE_ORDER for i in range(5)]
    options = []
    for col, i in keys:
        sp = spec[col]
        opts = []
        for cname, (dx, dy, ha, va, extra) in CANDS.items():
            t = axa.annotate(minus(df[col][i]), (num[col][i], ymark[col][i]), xytext=(dx, dy),
                             textcoords="offset points", ha=ha, va=va, fontsize=TXT, annotation_clip=False)
            bb = t.get_window_extent(rend)
            t.remove()
            if bb.x0 < axbb.x0 + 1 or bb.x1 > axbb.x1 - 1:
                continue
            if any(gap_between(bb, ob) < (PAD_MARK if kind == "marker" else PAD_BAR) * px for kind, ob, _ in obst):
                continue
            rank = sp[7].index(cname) if cname in sp[7] else len(sp[7])
            opts.append((extra + 0.1 * rank, bb, cname))
        assert opts, (col, i)
        options.append(opts)
    pick, tl_used = None, None
    for tl in LABEL_GAP_TRY:
        pick, cost = solve_labels(options, tl * px)
        if pick is not None:
            tl_used = tl
            break
    assert pick is not None, "no feasible label placement"
    report = []
    for (col, i), (cname, bb, _) in zip(keys, pick):
        dx, dy, ha, va, _ = CANDS[cname]
        axa.annotate(minus(df[col][i]), (num[col][i], ymark[col][i]), xytext=(dx, dy), textcoords="offset points",
                     ha=ha, va=va, fontsize=TXT, color=spec[col][8], annotation_clip=False)
        report.append((col, i, cname))
    boxes = [bb for _, bb, _ in pick]
    min_ll = min(gap_between(a, b) for k, a in enumerate(boxes) for b in boxes[k + 1:]) / px
    rows_of = [i for _, i in keys]
    min_lm = min(gap_between(a, ob) for a in boxes for kind, ob, _ in obst if kind == "marker") / px
    min_lm_other = min(gap_between(a, ob) for a, r in zip(boxes, rows_of) for kind, ob, rr in obst
                       if kind == "marker" and rr != r) / px
    min_lb = min(gap_between(a, ob) for a in boxes for kind, ob, _ in obst if kind == "bar") / px
    print(f"label-label gap target reached: {tl_used} pt; measured min label-label {min_ll:.2f} pt, "
          f"label-marker {min_lm:.2f} pt (other rows' markers {min_lm_other:.2f} pt), label-connector {min_lb:.2f} pt "
          f"(label boxes include font descent)")
    print("marker dodge (pt, +up):", {c: [round(v, 2) for v in shift[c]] for c in ("Counted", "Shown / listed")})

    for ext in ("pdf", "png"):
        fig.savefig(HERE / f"fig.{ext}", dpi=300)
    fig.canvas.draw()
    lab_r = max(t.get_window_extent(rend).x1 for t in label_texts) / fig.dpi
    print(f"figure {W} x {H:.3f} in; label column right edge {lab_r:.2f} in (agents column right edge {X_AGENTS:.2f} in)")
    print("placements:", report)


if __name__ == "__main__":
    main()
