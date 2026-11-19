"""One-line-per-condition summary of a tau-bench second-customer file (reward, wrong-account writes, silent writes).

  python tau_summary.py rollouts.jsonl [rollouts.jsonl ...]
"""
import sys

from analyze_tau import flags, load
from wrong_writes import classify

for f in sys.argv[1:]:
    rows = load(f)
    print(f"== {f.rsplit('/', 1)[-1]}")
    for cond in ("base", "twin", "twindoc", "twinall", "twinfid", "twinallfid"):
        rs = [r for r in rows if r["cond"] == cond]
        if not rs:
            continue
        fl, cl, n = [flags(r) for r in rs], [classify(r) for r in rs], len(rs)
        print(f"  {cond:11s} n={n:3d} reward={sum(x['reward'] for x in fl) / n:.2f} wrong={sum(c != 'none' for c in cl) / n:.2f} "
              f"silent={sum(c == 'silent' for c in cl) / n:.2f} after-flag={sum(c == 'after-flag' for c in cl) / n:.2f} "
              f"read-twin-first={sum(x['authed'] for x in fl) / n:.2f}")
