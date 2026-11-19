"""The per-agent figure of the paper (Figure 3 in arXiv v1): one single-column figure with two panels showing the per-agent points (while the
contract table and the crossover summary table appear in compact form in the main text).
  (a) fetched the second record before acting against acting, page at K = 2.
  (b) acting when the competing evidence is counted against listed or shown, per clear-acting agent.

  python grid_figure.py --art path/to/artifacts \
      --out figures/grid-panels.pdf
"""
import os
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from all_agents import frontier_manifest, load, manifest  # noqa: E402
from seeing import cell, corr  # noqa: E402

GRAY, MINT, CORAL, INK = "#9A9A9A", "#5FAF8F", "#ED8D5A", "#333333"
NAVY = "#1F2A44"  # newest models in navy
TEAL, ORANGE = "#2F7F93", "#D9692F"  # which-record family teal (tool shows), empty-claim family orange (user says)
NEWEST = ("GPT-6 astra", "Claude Opus 5.5", "GPT-6.1 sol")
SQ, DI = 40, 12  # marker areas (pt^2) of the open Gemini squares and the solid newest-model diamonds


def seeing_point(rows):
    p2 = cell(rows["none"], "page", 2)
    if not p2:
        return None
    return sum(r["s"]["seen"] >= 2 for r in p2) / len(p2), sum(r["s"]["acted"] for r in p2) / len(p2)


def style(ax):
    ax.grid(color="#E6E6E6", linewidth=0.5, zorder=0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(labelsize=7, pad=1.5)
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(-0.04, 1.04)
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    pct = plt.FuncFormatter(lambda v, _: f"{v * 100:.0f}%")  # percent of rollouts, as in the text and tables
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(Path(os.environ.get("WTA_WORK", "data/work"))))
    ap.add_argument("--art", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--height", type=float, default=1.95, help="figure height in inches")
    a = ap.parse_args()
    loaded = {lab: load(fs) for lab, fs in manifest(a.work).items()}
    assert len(loaded) == 36, len(loaded)
    front = {lab: seeing_point(load(fs)) for lab, fs in frontier_manifest().items()}

    # Every text element is at least 7 pt at column width, in Times to match Figures 5
    # and 7; the legends sit above the panels so the larger text does not cover points.
    plt.rcParams.update({"font.size": 7.5, "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"]})
    fig = plt.figure(figsize=(3.25, a.height))
    axa = fig.add_axes([0.15, 0.19, 0.32, 0.58])  # left margin wide enough for the percent tick labels and the y label
    axb = fig.add_axes([0.65, 0.19, 0.32, 0.58])

    # (a) fetched against acted under page. The three newest models share one point (100%, 0%) and the two Gemini 3.1 Pro
    # settings sit within 4 points of it, so all of them overlap. No position moves: the Gemini settings are open squares
    # underneath (zorder 3) and the newest models are one solid diamond on top (zorder 5), inside the squares. The legend
    # swatches are smaller than the plotted markers and carry the counts, so the stack at (100%, 0%) reads as 3 and 2 points.
    pts = [p for p in (seeing_point(r) for r in loaded.values()) if p]
    axa.scatter(*zip(*pts), s=8, color=TEAL, alpha=0.85, edgecolor="white", linewidth=0.3, zorder=2)  # which-record data
    new = [p for lab, p in front.items() if p and lab in NEWEST]
    gem = [p for lab, p in front.items() if p and lab.startswith("Gemini")]
    # Gemini as open navy triangles, the newest Claude/GPT as solid navy diamonds; positions unchanged.
    # A callout with a leader names the stack instead of jittering it.
    axa.scatter(*zip(*gem), s=SQ, marker="^", facecolor="none", edgecolor=NAVY, linewidth=0.7, zorder=3, clip_on=False)
    axa.scatter(*zip(*new), s=DI, marker="D", color=NAVY, edgecolor="white", linewidth=0.3, zorder=5, clip_on=False)
    axa.annotate("3 newest (0%)\nGemini (1%, 4%)", xy=(1.0, 0.03), xytext=(0.99, 0.80), ha="right", va="center",
                 fontsize=7, color=NAVY, linespacing=1.0,
                 arrowprops=dict(arrowstyle="-", color=NAVY, linewidth=0.5, shrinkA=1, shrinkB=4), zorder=6)
    axa.set_xlabel("Opened the second record", labelpad=1)
    axa.set_ylabel("Acted (count)", labelpad=1)
    swatch = [Line2D([], [], ls="", marker="o", ms=2.8, mfc=TEAL, mec="white", mew=0.3),
              Line2D([], [], ls="", marker="D", ms=3.2, mfc=NAVY, mec="white", mew=0.3),
              Line2D([], [], ls="", marker="^", ms=4.4, mfc="none", mec=NAVY, mew=0.7)]
    axa.legend(swatch, [f"Earlier ({len(pts)})", f"Newest models ({len(new)})", f"Gemini 3.1 Pro ({len(gem)})"],
               loc="lower left", bbox_to_anchor=(-0.32, 1.02), frameon=False, fontsize=7, handlelength=1.0,
               handletextpad=0.3, borderaxespad=0.0, labelspacing=0.15)
    style(axa)
    print(f"(a) earlier agents={len(pts)} r={corr(pts):.2f} newest={new} gemini={gem}")

    # (b) counted against listed or shown, per clear-acting agent
    art = Path(a.art)
    ref = [r for r in csv.DictReader(open(art / "panel-r2-statedrec-2026-10-15.csv")) if r["clear_referent"] == "True"]
    absn = list(csv.DictReader(open(art / "panel-r2-absence-agents-2026-10-15.csv")))
    rx, ry = [float(r["k2_page"]) for r in ref], [float(r["k2_listed"]) for r in ref]
    ax_, ay = [float(r["W1_counted_stated"]) for r in absn], [float(r["W1_shown_stated"]) for r in absn]
    assert len(rx) == 18 and len(ax_) == 27, (len(rx), len(ax_))
    axb.plot([0, 1], [0, 1], color="#BBBBBB", linewidth=0.6, linestyle="--", zorder=1)
    # Task families by color and shape, the same throughout the paper.
    axb.scatter(ax_, ay, s=10, marker="s", facecolor="white", edgecolor=ORANGE, linewidth=0.8, zorder=2, label="Empty claim (27)")
    axb.scatter(rx, ry, s=12, marker="o", color=TEAL, edgecolor="white", linewidth=0.3, zorder=3, label="Which record (18)")
    axb.set_xlabel("Acted, count", labelpad=1)
    axb.set_ylabel("Acted, listing", labelpad=1)
    axb.legend(loc="lower left", bbox_to_anchor=(-0.32, 1.02), frameon=False, fontsize=7, handletextpad=0.2,
               borderaxespad=0.0, labelspacing=0.15)
    style(axb)
    print(f"(b) referent below diagonal {sum(y < x for x, y in zip(rx, ry))}/18")

    for ax, t in ((axa, "(a)"), (axb, "(b)")):
        ax.text(-0.40, 1.02, t, transform=ax.transAxes, fontsize=7.5, fontweight="bold", va="bottom", ha="left")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
