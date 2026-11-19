"""Per-agent compound (K=2, irreversible) acting and the share of acted rollouts that acted on both matches,
and the send (share verb, listed) acting, for the 36 earlier agents. Reuses licgrid/all_agents.py loaders.

  python compound_both.py   (reads $WTA_WORK; no network)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import all_agents as A  # noqa: E402

work = str(Path(os.environ.get("WTA_WORK", "data/work")))
man = A.manifest(work)
out = []
for lab, files in man.items():
    rows = A.load(files)["none"]
    s = [r["s"] for r in rows if r["contract"] == "compound" and r["k"] == 2 and r["consequence"] == "irreversible"]
    acted = [x for x in s if x["acted"]]
    both = [x for x in acted if x["acted_all"]]
    snd = [r["s"] for r in rows if r["contract"] == "all" and r["k"] == 2 and r["consequence"] == "share"]
    sa = [x for x in snd if x["acted"]]
    out.append((lab, len(s), len(acted), len(both), len(snd), len(sa)))
tot_a = tot_b = 0
n_third = 0
n_both_major = 0
print("agent | compound n | acted | both | send n | send acted")
for lab, n, a, b, m, sa in out:
    print(f"{lab} | {n} | {a} | {b} | {m} | {sa} | acted share {a/n:.3f}" + (f" | both share {b/a:.3f}" if a else ""))
    tot_a += a
    tot_b += b
    if n and a / n >= 1 / 3:
        n_third += 1
        if a and b / a > 0.5:
            n_both_major += 1
print("agents with compound acting >= 1/3:", n_third, "of", len(out))
print("of those, agents with more than half of acted rollouts on both:", n_both_major)
print("pooled acted:", tot_a, "pooled both:", tot_b, f"{tot_b / tot_a:.3f}")
print("send acting >= 0.90:", [(lab, round(sa / m, 3)) for lab, n, a, b, m, sa in out if m and sa / m >= 0.895])
