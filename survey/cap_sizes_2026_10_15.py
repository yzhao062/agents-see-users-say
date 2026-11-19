#!/usr/bin/env python
"""Cap sizes of the capped MCP and toolkit tools of the API truncation survey (breaking down the caps of the
85 no-signal tools and how many could hide a two-record collision). Reads the per-tool labels of
<artifacts>/data/U*.jsonl with the nine unique-key corrections of
reaggregate_2026_10_13.py applied, and tabulates result_cap over the 211 capped MCP and toolkit tools, split by whether the
response the agent sees gives a sign that more exist. Standard library only, no network.

  python -I cap_sizes_2026_10_15.py [--artifacts DIR]

result_cap values are "first_match", "top_k:<n>", "page:<n>", or other labels; a numeric cap n hides a second match only when
more than n records match, so a two-record collision is hidden by first_match or a cap of 1.
"""
import argparse
import collections
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import reaggregate_2026_10_13 as rg  # noqa: E402


def size(cap):
    m = re.search(r"(\d+)", cap)
    return 1 if cap == "first_match" else int(m.group(1)) if m else None


def bucket(n):
    if n is None:
        return "no number"
    for hi, name in ((1, "1"), (10, "2 to 10"), (25, "11 to 25"), (50, "26 to 50"), (100, "51 to 100")):
        if n <= hi:
            return name
    return "over 100"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(rg.DEFAULT_ART))
    a = ap.parse_args()
    rows = rg.load(a.artifacts)
    nine = set(rg.NINE_IDS)
    corr = [dict(r, result_cap="none") if r["_id"] in nine else r for r in rows]
    capped = [r for r in corr if r["unit"] in rg.MCP_UNITS and r["result_cap"] != "none"]
    silent = [r for r in capped if not rg.prop_a(r)]
    assert (len(capped), len(silent)) == (211, 85), (len(capped), len(silent))
    order = ["1", "2 to 10", "11 to 25", "26 to 50", "51 to 100", "over 100", "no number"]
    print(f"# Cap Sizes of the 211 Capped MCP and Toolkit Tools ({len(silent)} without a sign)\n")
    print("| cap | all capped | no sign | sign |")
    print("|---|---|---|---|")
    for b in order:
        n_all = sum(bucket(size(r["result_cap"])) == b for r in capped)
        n_sil = sum(bucket(size(r["result_cap"])) == b for r in silent)
        print(f"| {b} | {n_all} | {n_sil} | {n_all - n_sil} |")
    print("\nCap kinds among the no-sign tools:", dict(collections.Counter(r["result_cap"].split(":")[0] for r in silent)))
    print("Routes among the no-sign tools (B1 page/cursor/offset, B2 limit only, B0 neither):",
          dict(collections.Counter(rg.prop_b(r) for r in silent)))
    small = [r for r in silent if (size(r["result_cap"]) or 999) <= 10]
    print(f"\nNo-sign tools with a cap of at most 10: {len(small)}")
    for r in small:
        print(f"- {r['_id']} {r['_repo']} {r['tool']} ({r['kind']}): {r['result_cap']}, route {rg.prop_b(r)}")
    print("\nNo-sign tools whose label has no number:", [(r["_id"], r["tool"], r["result_cap"]) for r in silent
                                                          if size(r["result_cap"]) is None])
    print("Kinds of the no-sign tools:", dict(collections.Counter(r["kind"] for r in silent)))


if __name__ == "__main__":
    main()
