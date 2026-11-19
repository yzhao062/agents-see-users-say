"""Per-agent clean rate without the post hoc street measure (completed, no wrong-account write, no
full-value disclosure) for the per-agent street-check figure. Copies place-per-agent.csv and adds a clean_ns column.

  python clean_nostreet_2026_10_20.py --csv place-per-agent.csv --out place-per-agent-ns.csv
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b_addr_2026_10_15 as BA  # noqa: E402
import place_2026_10_16 as P  # noqa: E402
import run_tau  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    load_data = run_tau.load_tau(os.environ.get("WTA_TAU_BENCH", "data/tau-bench"))[0]
    recs = [r for r in P.load(load_data()["users"], load_data) if r["group"] == "main"]
    ns = {}
    for r in recs:
        k = (r["agent"], r["cond"])
        ns[k] = ns.get(k, 0) + int(bool(r["completed"] and not r["wrong"] and not r["disc"]))
    rows = list(csv.DictReader(open(a.csv)))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) + ["clean_ns"])
        w.writeheader()
        missing = 0
        for r in rows:
            k = (r["agent"], r["cond"])
            missing += k not in ns
            w.writerow({**r, "clean_ns": ns.get(k, "")})
    print("wrote", out, "rows", len(rows), "without a clean_ns value", missing)


if __name__ == "__main__":
    main()
