"""Reversible actions: does a reversible verb change acting at K = 2, and do agents that act report the other match?

Rows from run_grid.py --stated are skipped: this check reads the no-claim rollouts.

  python reversible.py label=file[,file...] [label=...]

Per model and contract (page, compound, firstdoc): P(act | irreversible), P(act | reversible), their difference, and
among acted reversible rollouts the share whose final message told the user about the other match. Cells with fewer
than 20 scored rollouts print as "-".
"""
import argparse
import gzip
import json

from score_grid import score

CONTRACTS = ["page", "compound", "firstdoc"]


def rate(xs, n=20):
    return sum(xs) / len(xs) if len(xs) >= n else None


def fmt(x, signed=False):
    return "-" if x is None else (f"{x:+.2f}" if signed else f"{x:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    head = []
    for c in CONTRACTS:
        head += [f"{c} irrev", f"{c} rev", f"{c} diff", f"{c} rev: told"]
    print("| model | " + " | ".join(head) + " |")
    print("|---|" + "---|" * len(head))
    for spec in a.specs:
        label, files = spec.split("=", 1)
        rows, seen = [], set()
        for f in files.split(","):
            for ln in (gzip.open(f, "rt") if str(f).endswith(".gz") else open(f)):
                if not ln.strip():
                    continue
                r = json.loads(ln)
                key = (r["template"], r["k"], r["contract"], r["consequence"], r["rep"])
                if r.get("error") or r.get("stated") or (r.get("norm") or "none") != "none" or key in seen or r["k"] != 2:
                    continue
                seen.add(key)
                r["s"] = score(r)
                rows.append(r)
        cells = []
        for c in CONTRACTS:
            irr = rate([r["s"]["acted"] for r in rows if r["contract"] == c and r["consequence"] == "irreversible"])
            rev_rows = [r["s"] for r in rows if r["contract"] == c and r["consequence"] == "reversible"]
            rev = rate([s["acted"] for s in rev_rows])
            told = rate([s["disclosed"] for s in rev_rows if s["acted"]], n=5)
            diff = rev - irr if irr is not None and rev is not None else None
            cells += [fmt(irr), fmt(rev), fmt(diff, signed=True), fmt(told)]
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
