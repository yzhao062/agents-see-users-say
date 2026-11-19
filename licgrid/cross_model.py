"""Cross-model table for the which-record grid (no prompt line).

Rows from run_grid.py --stated are skipped: this check reads the no-claim rollouts.

  python cross_model.py label=file[,file...] [label=...] [--norm none]

Per model: P(act) at K = 1 with all matches listed (willingness to act on a clear instruction), and at K = 2 in each
condition; among acted K = 2 rollouts, the share that acted on both matches (compound, share) and the share that
told the user about the other match (page). Action rates need 20 scored rollouts, and conditional shares need five
rollouts that acted. A conditional share with no acted rollouts among at least 20 scored ones prints "n/a"; every other
cell below its threshold prints "-".
"""
import argparse
import gzip
import json

from score_grid import score

COLS = [  # (header, consequence, contract, K, statistic)
    ("K=1 listed", "irreversible", "all", 1, "act"),
    ("listed", "irreversible", "all", 2, "act"),
    ("page (count shown)", "irreversible", "page", 2, "act"),
    ("compound", "irreversible", "compound", 2, "act"),
    ("first (undocumented)", "irreversible", "first", 2, "act"),
    ("firstdoc (documented)", "irreversible", "firstdoc", 2, "act"),
    ("save, listed", "save", "all", 2, "act"),
    ("share, listed", "share", "all", 2, "act"),
    ("compound: both", "irreversible", "compound", 2, "all"),
    ("share: both", "share", "all", 2, "all"),
    ("page: told user", "irreversible", "page", 2, "disclosed"),
]


def cell(rows, q, c, k, stat):
    s = [r["s"] for r in rows if r["consequence"] == q and r["contract"] == c and r["k"] == k]
    if stat == "act":
        xs = [x["acted"] for x in s]
    else:
        acted = [x for x in s if x["acted"]]
        xs = [x["acted_all"] if stat == "all" else x["disclosed"] for x in acted]
        if len(s) >= 20 and not xs:
            return "n/a"
    return f"{sum(xs) / len(xs):.2f}" if len(xs) >= (20 if stat == "act" else 5) else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--norm", default="none")
    a = ap.parse_args()
    print("| model | " + " | ".join(h for h, *_ in COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for spec in a.specs:
        label, files = spec.split("=", 1)
        rows, seen = [], set()
        for f in files.split(","):
            for ln in (gzip.open(f, "rt") if str(f).endswith(".gz") else open(f)):
                if not ln.strip():
                    continue
                r = json.loads(ln)
                key = (r["template"], r["k"], r["contract"], r["consequence"], r["rep"])
                if r.get("error") or r.get("stated") or (r.get("norm") or "none") != a.norm or key in seen:
                    continue
                seen.add(key)
                r["s"] = score(r)
                rows.append(r)
        print(f"| {label} | " + " | ".join(cell(rows, *c[1:]) for c in COLS) + " |")


if __name__ == "__main__":
    main()
