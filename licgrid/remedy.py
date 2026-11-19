"""Prompt check (Proposition A.2): what a check in the prompt fixes under each tool contract.

Rows from run_grid.py --stated are skipped: this check reads the no-claim rollouts.

  python remedy.py label=none_file[,none_file...]:fiduciary_file [label=...]

Per agent and contract (all, page, firstnote, firstdoc, first): P(act) at K = 2 without and with the fiduciary line,
and P(act) at K = 1 with the line (the check should not block clear instructions). Proposition A.2 predicts the line
removes acting where the evidence shows plurality (all, page) and cannot remove it under undocumented truncation (first).
Cells with fewer than 20 scored rollouts print as "-".
"""
import argparse
import gzip
import json

from score_grid import score

CONTRACTS = ["all", "page", "firstnote", "firstdoc", "first"]


def load(files):
    rows, seen = [], set()
    for f in files:
        for ln in (gzip.open(f, "rt") if str(f).endswith(".gz") else open(f)):
            if not ln.strip():
                continue
            r = json.loads(ln)
            key = (r["template"], r["k"], r["contract"], r["consequence"], r["rep"], r.get("norm") or "")
            if r.get("error") or r.get("stated") or key in seen or r["consequence"] != "irreversible":
                continue
            seen.add(key)
            rows.append(r)
    return rows


def rate(rows, c, k):
    xs = [score(r)["acted"] for r in rows if r["contract"] == c and r["k"] == k]
    return f"{sum(xs) / len(xs):.2f}" if len(xs) >= 20 else "-"


def jgap(rows, c):
    a = [score(r)["acted"] for r in rows if r["contract"] == c and r["k"] == 1]
    b = [score(r)["acted"] for r in rows if r["contract"] == c and r["k"] == 2]
    return f"{sum(a) / len(a) - sum(b) / len(b):+.2f}" if len(a) >= 20 and len(b) >= 20 else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    head = [f"{c}: K=2 none → line (K=1 line)" for c in CONTRACTS]
    print("| agent | " + " | ".join(head) + " |")
    print("|---|" + "---|" * len(head))
    disc = []
    for spec in a.specs:
        label, files = spec.split("=", 1)
        none_files, fid_file = files.rsplit(":", 1)
        none = [r for r in load(none_files.split(",")) if not r.get("norm")]
        fid = [r for r in load([fid_file]) if r.get("norm") == "fiduciary"]
        cells = [f"{rate(none, c, 2)} → {rate(fid, c, 2)} ({rate(fid, c, 1)})" for c in CONTRACTS]
        print(f"| {label} | " + " | ".join(cells) + " |")
        disc.append((label, [(jgap(none, c), jgap(fid, c)) for c in CONTRACTS]))
    print("\nDiscrimination J = P(act | K=1) - P(act | K=2), without → with the line (Proposition A.1: 0 under truncation)")
    print("| agent | " + " | ".join(CONTRACTS) + " |")
    print("|---|" + "---|" * len(CONTRACTS))
    for label, js in disc:
        print(f"| {label} | " + " | ".join(f"{a} → {b}" for a, b in js) + " |")


if __name__ == "__main__":
    main()
