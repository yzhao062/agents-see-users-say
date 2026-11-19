"""Split wrong-customer writes by what the agent had said before the first one, and bootstrap twin minus twinall.

  python wrong_writes.py label=rollouts.jsonl [label=...]

silent      the first wrong write came before any agent message that questions the account (see analyze_tau.NOTICE)
after-flag  the agent questioned the account first, and the write followed the simulated user's reply
Bootstrap: 2000 resamples of tasks, paired across conditions.
"""
import argparse
import json
import random

from analyze_tau import NOTICE, load
from run_tau import WRITE_TOOLS


def first_wrong_index(r):
    """Index in the log of the first successful write that touched the twin, or None."""
    if not r["twin_id"]:
        return None
    twin_orders = {o for o, owner in r["changed_orders_by_owner"].items() if owner == r["twin_id"]}
    for i, m in enumerate(r["log"]):
        if m.get("tool") in WRITE_TOOLS and m["role"] == "assistant":
            kw = m.get("kwargs") or {}
            res = r["log"][i + 1]["content"] if i + 1 < len(r["log"]) else ""
            if str(res).startswith("Error"):
                continue
            if kw.get("order_id") in twin_orders or kw.get("user_id") == r["twin_id"]:
                return i
    return None


def classify(r):
    if not (r["twin_records_changed"] or r["twin_payment_used"]):
        return "none"
    i = first_wrong_index(r)
    if i is None:
        return "silent"  # payment-method-only cases: no record of the twin was the target
    said = " ".join(m.get("content", "") for m in r["log"][:i] if m["role"] == "assistant")
    return "after-flag" if NOTICE.search(said) else "silent"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    a = ap.parse_args()
    rng = random.Random(0)
    print("| agent | condition | wrong | silent | after-flag | twin - twinall (wrong) [95% CI] |")
    print("|---|---|---|---|---|---|")
    for spec in a.specs:
        label, f = spec.split("=", 1)
        rows = load(f)
        by = {(r["task"], r["cond"]): classify(r) for r in rows}
        tasks = sorted({t for t, c in by if c == "twin"} & {t for t, c in by if c == "twinall"})
        for cond in ("twin", "twindoc", "twinall"):
            cl = [by[(t, cond)] for t in tasks if (t, cond) in by]
            n = len(cl)
            w = sum(c != "none" for c in cl) / n
            s = sum(c == "silent" for c in cl) / n
            af = sum(c == "after-flag" for c in cl) / n
            diff = ""
            if cond == "twin":
                d = [(by[(t, "twin")] != "none") - (by[(t, "twinall")] != "none") for t in tasks]
                boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
                diff = f"{sum(d) / len(d):+.2f} [{boots[50]:+.2f}, {boots[1949]:+.2f}]"
            print(f"| {label} | {cond} | {w:.2f} | {s:.2f} | {af:.2f} | {diff} |")


if __name__ == "__main__":
    main()
