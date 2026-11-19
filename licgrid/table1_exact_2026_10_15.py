"""Exact contract means for Table 1 (36 earlier agents), from the rollouts rather than the rounded appendix cells."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import all_agents as aa  # noqa: E402

work = str(Path(os.environ.get("WTA_WORK", "data/work")))
loaded = {lab: aa.load(fs) for lab, fs in aa.manifest(work).items()}
print("agents:", len(loaded))
for key, c, k in [("clear", "all", 1), ("bare", "first", 2), ("noted", "firstnote", 2), ("doc", "firstdoc", 2),
                  ("page", "page", 2), ("listed", "all", 2)]:
    acted = n = most = agents = 0
    for lab, rows in loaded.items():
        xs = [r["s"]["acted"] for r in rows["none"] if r["contract"] == c and r["k"] == k and r["consequence"] == "irreversible"]
        if len(xs) < 20:
            continue
        agents += 1
        acted += sum(xs)
        n += len(xs)
        most += sum(xs) / len(xs) > 0.5
    print(f"{key:7s} agents={agents} most={most} acted={acted}/{n} = {acted / n:.4f}")
