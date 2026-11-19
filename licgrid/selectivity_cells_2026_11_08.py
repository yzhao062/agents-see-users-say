"""Export the per-agent cells behind Figure 5 (selectivity) to CSV."""
import os
import csv
import sys
from pathlib import Path
from statistics import mean

LIC = Path(__file__).resolve().parent
sys.path.insert(0, str(LIC))
from all_agents import LABEL  # noqa: E402
from score_absence import cell, load as load_absence, read_raw  # noqa: E402

WORK = Path(os.environ.get("WTA_WORK", "data/work"))
out = Path(sys.argv[1])
rows_out = []
for s in LABEL:
    fn = s.replace(":", "_") + ".jsonl"
    rec = {"agent": LABEL[s]}
    for sub, claim in (("absence", True), ("unstated", False)):
        rows = [r for r in load_absence(read_raw([WORK / sub / fn])) if r["kind"] == "act" and r.get("stated", True) == claim]
        tag = "claim" if claim else "noclaim"
        for w, wn in ((0, "true"), (1, "false")):
            for tier, col in (("counted", "counted"), ("shown", "listed")):
                c = cell(rows, tier, w)
                rec[f"{col}_{wn}_{tag}"] = round(100 * mean(r["s"]["acted"] for r in c), 2) if c else ""
    rec["sep_noclaim"] = round(rec["counted_true_noclaim"] - rec["counted_false_noclaim"], 2)
    rec["sep_claim"] = round(rec["counted_true_claim"] - rec["counted_false_claim"], 2)
    rows_out.append(rec)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0]))
    w.writeheader()
    w.writerows(rows_out)
n = len(rows_out)
print("agents", n, "wider", sum(r["sep_claim"] > r["sep_noclaim"] for r in rows_out))
for k in ("counted_true_noclaim", "counted_false_noclaim", "counted_true_claim", "counted_false_claim", "sep_noclaim", "sep_claim"):
    print(k, round(mean(r[k] for r in rows_out), 1))
