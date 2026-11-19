"""Stated absence over all 31 Bedrock agents, no clear-acting selection: per agent,
the separation of the two worlds under the count (W0 counted acting minus W1 counted acting), with the claim stated (the base
absence run) and without it (the unstated crossover run), and its change. Agent-cluster bootstrap as in crossover.py.

  python two_world_all_2026_10_15.py

Expected summary: 31 agents, mean change +0.450 (agent-cluster 95% CI 0.367 to 0.536); 29 of 31 separate more sharply
when the user states the claim.
"""
import os
import sys
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent))
from all_agents import LABEL  # noqa: E402
from score_absence import boot, cell, load as load_absence, read_raw  # noqa: E402

WORK = Path(os.environ.get("WTA_WORK", "data/work"))


def separation(rows):
    acted = lambda w: mean(r["s"]["acted"] for r in cell(rows, "counted", w))
    return acted(0) - acted(1)


def main():
    change = {}
    for s in LABEL:
        fn = s.replace(":", "_") + ".jsonl"
        sep = {}
        for sub, claim in (("absence", True), ("unstated", False)):
            rows = [r for r in load_absence(read_raw([WORK / sub / fn])) if r["kind"] == "act" and r.get("stated", True) == claim]
            sep[claim] = separation(rows)
        change[s] = sep[True] - sep[False]
    lo, hi = boot(change, mean)
    print(f"agents {len(change)}; mean change in separation (stated minus unstated) {mean(change.values()):+.3f} "
          f"[{lo:.3f}, {hi:.3f}]; larger when stated: {sum(v > 0 for v in change.values())} of {len(change)}")


if __name__ == "__main__":
    main()
