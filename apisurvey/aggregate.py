"""Aggregate the API truncation survey (units U1-U8) with unique-key spot-check corrections."""
import collections
import glob
import json
import sys

A = sys.argv[1]
rows = []
for f in sorted(glob.glob(f"{A}/data/U*.jsonl")):
    rows += [json.loads(line) for line in open(f) if line.strip()]

# Spot-check corrections: these "first_match" tools look up a key that is unique by construction
# (dict key, email, Confluence title within a space, meeting join URL), so a cap of 1 hides nothing.
UNIQUE_KEY = {"confluence_get_page", "list-online-meetings", "find_user_id_by_email", "get_customer_by_phone",
              "get_user_id", "get_symbol_by_name", "get_nearest_airport_by_city", "get_zipcode_based_on_city"}


def group(r):
    if r["unit"] == "U8":
        return "upstream API"
    if r["unit"] == "U7":
        return "benchmark"
    return "MCP / toolkit"


def stats(rs):
    capped = [r for r in rs if r["result_cap"] != "none"]
    silent = [r for r in capped if r["agent_visible_signal"] == ["none"]]
    dropped = [r for r in rs if r["upstream_signal_dropped"] == "yes"]
    stated = [r for r in capped if r["cap_stated_in_description"]]
    first = [r for r in rs if r["result_cap"] == "first_match" and r["tool"] not in UNIQUE_KEY]
    unreach = [r for r in silent if r["more_results_reachable"] in ("no", "limit_param_only")]
    return dict(n=len(rs), capped=len(capped), silent=len(silent), silent_unreach=len(unreach),
                stated=len(stated), dropped=len(dropped), first=len(first))


by = collections.defaultdict(list)
for r in rows:
    by[group(r)].append(r)
    by["source:" + r["source"].split("@")[0]].append(r)
print("| group | tools | capped | capped, no signal | of those, no way to reach more | cap in description "
      "| wrapper dropped upstream signal | first match on a non-unique key |")
print("|---|---|---|---|---|---|---|---|")
for g in ["MCP / toolkit", "benchmark", "upstream API"] + sorted(k for k in by if k.startswith("source:")):
    s = stats(by[g])
    print(f"| {g} | {s['n']} | {s['capped']} | {s['silent']} ({s['silent'] / max(s['capped'], 1):.0%}) "
          f"| {s['silent_unreach']} | {s['stated']} | {s['dropped']} | {s['first']} |")
agent = [r for r in rows if r["unit"] != "U8"]
s = stats(agent)
print(f"\nAgent-facing total (U1-U7): {s}")
print("first match on a non-unique key:",
      [(r["source"].split("@")[0], r["tool"]) for r in agent
       if r["result_cap"] == "first_match" and r["tool"] not in UNIQUE_KEY])
print("reachability values:", collections.Counter(r["more_results_reachable"] for r in agent))
