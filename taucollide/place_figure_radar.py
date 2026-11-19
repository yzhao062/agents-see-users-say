"""Per-agent street-check figure for Section 5 of the paper. Street-knowing user, 13 earlier agents.
Left: a radar of pooled rates per lookup, every axis oriented so that outward is better (no wrong-account write, no full-value
disclosure, no mention of the other street over eligible tasks, completed); released, complete and masked average the two
placements of the second customer with equal weight, as Table 3 does; the one-customer baseline is a dashed outline.
Right (with --layout pair): each agent's clean rate per lookup as a dot row, agents sorted by their one-customer completion;
--right predef drops the post hoc street measure from clean, from the clean_ns column that
clean_nostreet_2026_10_20.py adds. Reads the per-agent table that place_2026_10_16.py writes.

  python place_figure_radar.py --csv place-per-agent-ns.csv --out street-check.pdf --layout strip --right predef
  (--layout pair|radar|strip|dots; --right predef needs the clean_ns column that clean_nostreet_2026_10_20.py adds)
"""
import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

NEWEST = ("Claude Opus 5.5", "GPT-6.1 sol")
LOOKUPS = [("Released", ("twinaddr", "twin2addr"), "#E07B39"), ("Complete", ("twinalladdr", "twin2alladdr"), "#6F86A6"),
           ("Masked", ("twinmaskaddr", "twin2maskaddr"), "#A86FA0"), ("Street check", ("twincheckaddr",), "#2E8B6E")]
ONE = ("One customer", ("baseaddr",), "#333333")
AXES = [("wrong", "No wrong-account\nwrite", True), ("disc", "No full-value\ndisclosure", True),
        ("street_said", "Other street\nnot named", True), ("completed", "Completed", False)]
# axis label placement past the outer ring: (ha, va, dx, dy) in points from the axis end, so a label never touches a vertex
LABEL_AT = [("center", "bottom", 0, 5), ("left", "center", 6, 0), ("center", "top", 0, -5), ("right", "center", -6, 0)]
RLABEL_DEG = 135  # radial tick labels sit on the bisector between the full-value and street axes, centered on their rings
# --layout strip palette (color pairs checked for color-vision deficiency)
STRIP_COLOR = {"Released": "#ed8d5a", "Complete": "#2f7f93", "Masked": "#eda100", "Street check": "#1baf7a"}  # the lookup colors used in the paper's other figures
INK_S = "#1f2a44"


def swarm(vals, step, near):
    """Horizontal offsets for sorted values so that dots closer than `near` in y never share an x offset."""
    placed, out = [], []
    for v in vals:
        k = 0
        while True:
            for off in ((0.0,) if k == 0 else (k * step, -k * step)):
                if all(abs(v - pv) >= near or abs(off - po) >= step * 0.99 for pv, po in placed):
                    placed.append((v, off))
                    out.append(off)
                    break
            else:
                k += 1
                continue
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layout", choices=["pair", "radar", "dots", "strip"], default="pair")
    ap.add_argument("--right", choices=["clean", "predef"], default="clean")
    ap.add_argument("--height", type=float, default=2.75, help="strip layout: figure height in inches")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    by = {}
    for r in csv.DictReader(open(a.csv)):
        by.setdefault(r["agent"], {})[r["cond"]] = r
    agents = [ag for ag in by if ag not in NEWEST]
    assert len(agents) == 13, agents

    def pooled(conds, key):
        den = "street_eligible" if key == "street_said" else "n"
        vals = []
        for c in conds:
            num = sum(int(by[ag][c][key]) for ag in agents)
            d = sum(int(by[ag][c][den]) for ag in agents)
            vals.append(num / d)
        return sum(vals) / len(vals)

    def agent_rate(ag, conds, key):
        den = "street_eligible" if key == "street_said" else "n"
        return sum(int(by[ag][c][key]) / int(by[ag][c][den]) for c in conds) / len(conds)

    plt.rcParams.update({"font.size": 7, "font.family": "serif"})
    if a.layout == "strip":
        # One column per lookup: each agent's clean rate minus its own one-customer completion, the paired gap the
        # text cites.
        rkey = "clean_ns" if a.right == "predef" else "clean"
        plt.rcParams.update({"font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
        fig = plt.figure(figsize=(3.25, a.height))
        bx = fig.add_axes([0.17, 0.12, 0.76, 0.85])
        for i, (name, conds, _) in enumerate(LOOKUPS):
            col = STRIP_COLOR[name]
            ds = sorted((agent_rate(ag, conds, rkey) - agent_rate(ag, ONE[1], "completed")) * 100 for ag in agents)
            xs = swarm(ds, step=0.075, near=2.6)
            bx.scatter([i + x for x in xs], ds, s=13, color=col, edgecolors="white", linewidths=0.4, zorder=4)
            m = sum(ds) / len(ds)
            bx.plot([i - 0.32, i + 0.32], [m, m], color=INK_S, linewidth=1.6, zorder=5, solid_capstyle="butt")
            bx.annotate("0" if round(m) == 0 else f"{round(m):+d}".replace("-", "\u2212"), (i + 0.34, m), xytext=(1, 5 if abs(m) < 2 else 0), textcoords="offset points", ha="left", va="center",
                        fontsize=7, color=INK_S, fontweight="bold")
            print(f"{name:13s} mean {m:+.1f}  min {ds[0]:+.1f}  max {ds[-1]:+.1f}")
        bx.axhline(0, color="#555555", linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
        bx.annotate("0 = each agent's own\none-customer completion", (-0.45, 0), xytext=(0, 2.5),
                    textcoords="offset points", ha="left", va="bottom", fontsize=7, color="#555555", linespacing=1.05)
        bx.set_xticks(range(len(LOOKUPS)))
        bx.set_xticklabels([n for n, _, _ in LOOKUPS], fontsize=7.5)
        bx.set_xlim(-0.5, len(LOOKUPS) - 0.5)
        bx.tick_params(axis="x", length=0, pad=3)
        bx.tick_params(axis="y", labelsize=7)
        bx.set_ylabel("Clean minus one-customer\ncompletion (points)", fontsize=7, labelpad=2)
        for s in ("top", "right", "bottom"):
            bx.spines[s].set_visible(False)
        bx.spines["left"].set_color("#888888")
        fig.savefig(a.out)
        print("wrote", a.out)
        return
    if a.layout == "dots":
        # Per-agent dots, one column wide; clean is the no-street definition (--right predef), which Table 3 reports
        # as Clean.
        fig = plt.figure(figsize=(3.3, 2.6))
        bx = fig.add_axes([0.30, 0.16, 0.66, 0.67])
        rkey = "clean_ns" if a.right == "predef" else "clean"
        order = sorted(agents, key=lambda ag: agent_rate(ag, ONE[1], "completed"))
        ys = list(range(len(order)))
        for name, conds, col in LOOKUPS:
            xs = [agent_rate(ag, conds, rkey) for ag in order]
            bx.scatter(xs, ys, s=11, color=col, zorder=4, label=name, edgecolors="white", linewidths=0.3)
        xs1 = [agent_rate(ag, ONE[1], "completed") for ag in order]
        bx.scatter(xs1, ys, s=16, facecolors="none", edgecolors=ONE[2], linewidths=0.7, zorder=5, label=ONE[0])
        for y in ys:
            bx.axhline(y, color="#EEEEEE", linewidth=0.5, zorder=0)
        bx.set_yticks(ys)
        bx.set_yticklabels(order, fontsize=6)
        bx.set_xlim(0, 0.85)
        bx.set_xlabel("Clean (dots) or one-customer completion (circles)"
                      if a.right == "predef" else "Clean: completed with none of the three harms", fontsize=6.0)
        bx.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        bx.tick_params(axis="x", labelsize=6.2)
        h, l = bx.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.6, 1.0), ncol=3, frameon=False, fontsize=6.6,
                   handletextpad=0.3, columnspacing=1.0)
        fig.savefig(a.out)
        print("wrote", a.out)
        return
    if a.layout == "pair":
        fig = plt.figure(figsize=(6.8, 2.4))
        ax = fig.add_axes([0.105, 0.17, 0.215, 0.62], polar=True)
        bx = fig.add_axes([0.555, 0.17, 0.43, 0.66])
    else:
        fig = plt.figure(figsize=(3.3, 2.7))
        ax = fig.add_axes([0.16, 0.16, 0.68, 0.68], polar=True)
        bx = None

    n = len(AXES)
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ang = [2 * math.pi * i / n for i in range(n)]
    ang_c = ang + ang[:1]
    for name, conds, col in [ONE] + LOOKUPS:
        vals = [(1 - pooled(conds, k)) if inv else pooled(conds, k) for k, _, inv in AXES]
        v = vals + vals[:1]
        if name == ONE[0]:
            ax.plot(ang_c, v, color=col, linewidth=0.9, linestyle=(0, (3, 2)), zorder=5, label=name)
        else:
            ax.plot(ang_c, v, color=col, linewidth=1.4, zorder=4, label=name)
            ax.fill(ang_c, v, color=col, alpha=0.10 if name != "Street check" else 0.18, zorder=3)
    ax.set_xticks(ang)
    ax.set_xticklabels([])
    for a_i, (_, lab, _), (ha, va, dx, dy) in zip(ang, AXES, LABEL_AT):
        ax.annotate(lab, xy=(a_i, 1.0), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                    multialignment=ha, fontsize=6.6, annotation_clip=False)
    ax.set_ylim(0, 1)
    ax.set_thetalim(0, 2 * math.pi)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])
    # Radial tick labels: every polygon edge crosses the middle rings, so no angle is free at 50%; on the bisector between the
    # full-value and street axes the inner label is centered on its ring (white halo), and the two outer labels start at their
    # ring and run outward, past the street-check edge and outside the outer ring.
    for r_i, ha, dx in ((0.25, "center", 0), (0.5, "center", 0), (0.75, "left", 3), (1.0, "left", 3)):
        ax.annotate(f"{r_i * 100:.0f}%", xy=(math.radians(RLABEL_DEG), r_i), xytext=(dx, 0), textcoords="offset points",
                    ha=ha, va="center", fontsize=5.4, color="#666666", zorder=7,
                    path_effects=[pe.withStroke(linewidth=1.4, foreground="white")])
    ax.grid(color="#DDDDDD", linewidth=0.5)
    ax.spines["polar"].set_color("#BBBBBB")
    if bx is None:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=3, frameon=False, fontsize=6, handlelength=1.6)
    else:
        rkey = "clean_ns" if a.right == "predef" else "clean"
        order = sorted(agents, key=lambda ag: agent_rate(ag, ONE[1], "completed"))
        ys = list(range(len(order)))
        for name, conds, col in LOOKUPS:
            xs = [agent_rate(ag, conds, rkey) for ag in order]
            bx.scatter(xs, ys, s=11, color=col, zorder=4, label=name, edgecolors="white", linewidths=0.3)
        xs1 = [agent_rate(ag, ONE[1], "completed") for ag in order]
        bx.scatter(xs1, ys, s=16, facecolors="none", edgecolors=ONE[2], linewidths=0.7, zorder=5, label=ONE[0])
        for y in ys:
            bx.axhline(y, color="#EEEEEE", linewidth=0.5, zorder=0)
        bx.set_yticks(ys)
        bx.set_yticklabels(order, fontsize=5.4)
        bx.set_xlim(0, 0.85)
        bx.set_xlabel("Clean, no street: completed, no wrong write or full disclosure" if a.right == "predef"
                      else "Clean: completed with none of the three harms", fontsize=6.6)
        bx.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%"))
        for s in ("top", "right"):
            bx.spines[s].set_visible(False)
        bx.tick_params(axis="x", labelsize=6)
        h, l = bx.get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.77, 1.0), ncol=3, frameon=False, fontsize=6.6,
                   handletextpad=0.3, columnspacing=1.4)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
