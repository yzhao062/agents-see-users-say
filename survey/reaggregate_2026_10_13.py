#!/usr/bin/env python
"""Re-aggregate the API truncation survey (units U1-U8).

Reads the per-tool labels in <artifacts>/data/U*.jsonl and prints markdown summary tables for the survey.

What it does
  1. Reproduces the initial coding and asserts the baseline counts.
  2. Applies the nine unique-key corrections (a lookup on an identifier cannot hide a second match, so the tool
     leaves the capped set) and recomputes every column of the original table.
  3. Splits the single "no completeness signal" judgment into three properties, counted over capped tools:
       (a) signal availability: the response says that more results exist or gives a total;
       (b) access to further results: a page, cursor, offset or limit parameter reaches the rest;
       (c) ability to certify completeness: the agent can establish that it has seen every match.
  4. Prints the second-reading tables (seven first-match tools, nine corrections, disputed categories) with the
     decisive text pulled from the artifacts, and a sensitivity table over the disputed categories.

Run (no network, standard library only):
  python -I reaggregate_2026_10_13.py [--artifacts DIR] [--tau-root DIR]
      [--json OUT.json]

--tau-root points at a local tau-bench checkout. It is a supplementary check outside the artifact folder and is
skipped when the folder is absent. The script never fetches from the web.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_ART = Path(os.environ.get("WTA_ARTIFACTS", "data/artifacts")) / "api-survey-2026-10-12"
DEFAULT_TAU = Path(os.environ.get("WTA_TAU_BENCH", "data/tau-bench"))

INITIAL_DATE = "2026-10-12"
MCP_UNITS = ("U1", "U2", "U3", "U4", "U5", "U6")
UNITS = MCP_UNITS + ("U7", "U8")

# Tool names excluded by the initial aggregate.py from the first-match count (eight names, nine tools:
# find_user_id_by_email exists in tau-bench and in tau2-bench).
OLD_UNIQUE_NAMES = {"confluence_get_page", "list-online-meetings", "find_user_id_by_email", "get_customer_by_phone",
                    "get_user_id", "get_symbol_by_name", "get_nearest_airport_by_city", "get_zipcode_based_on_city"}

# The nine unique-key corrections. Tier T1: the key is a dictionary key or a platform-enforced identifier.
# Tier T2: the key is an identifier attribute whose uniqueness is a property of the data, not of the code.
NINE = [
    dict(id="U3:28", repo="sooperset/mcp-atlassian", tool="confluence_get_page", tier="T1",
         why="A page title is unique within a Confluence space, so exact title plus space key names one page.",
         spot="spot-checked"),
    dict(id="U4:72", repo="softeria/ms-365-mcp-server", tool="list-online-meetings", tier="T1",
         why="The required JoinWebUrl filter identifies one meeting; the description says the collection cannot be listed or paged.",
         spot="not spot-checked"),
    dict(id="U7:5", repo="sierra-research/tau-bench", tool="find_user_id_by_email", tier="T2",
         why="Email is an identifier attribute; the loop returns the first hit, and a second hit would be a data error.",
         spot="not spot-checked; re-read here (supplementary)"),
    dict(id="U7:11", repo="sierra-research/tau2-bench", tool="find_user_id_by_email", tier="T2",
         why="Same as tau-bench; the survey notes state that the data holds no duplicate email.",
         spot="spot-checked"),
    dict(id="U7:13", repo="sierra-research/tau2-bench", tool="get_customer_by_phone", tier="T2",
         why="A phone number identifies one customer; the tool matches contact numbers and line IDs, so uniqueness across both fields rests on the data.",
         spot="spot-checked"),
    dict(id="U7:27", repo="ShishirPatil/gorilla", tool="get_user_id", tier="T1",
         why="Dictionary lookup by user name; one value by construction.", spot="spot-checked"),
    dict(id="U7:35", repo="ShishirPatil/gorilla", tool="get_symbol_by_name", tier="T1",
         why="Dictionary lookup by company name; one value by construction.", spot="spot-checked"),
    dict(id="U7:43", repo="ShishirPatil/gorilla", tool="get_nearest_airport_by_city", tier="T1",
         why="Function from a city name to its one nearest airport; single-valued by definition.",
         spot="spot-checked"),
    dict(id="U7:45", repo="ShishirPatil/gorilla", tool="get_zipcode_based_on_city", tier="T1",
         why="Dictionary lookup by city name; one value by construction.", spot="not spot-checked"),
]
NINE_IDS = [d["id"] for d in NINE]
T2_IDS = [d["id"] for d in NINE if d["tier"] == "T2"]

# The seven first-match tools on a key that several records can share.
SEVEN = [
    dict(id="U7:6", repo="sierra-research/tau-bench", tool="find_user_id_by_name_zip", status="confirmed",
         key="Name and zip are not an identifier; two users can share both.",
         spot="spot-checked; code and data re-read here at the same commit (supplementary)"),
    dict(id="U7:10", repo="sierra-research/tau2-bench", tool="find_user_id_by_name_zip", status="confirmed",
         key="Same key as tau-bench. The tau2 data are not in the artifacts, so a collision-free table is not assumed.",
         spot="spot-checked"),
    dict(id="U1:30", repo="modelcontextprotocol/servers-archived", tool="maps_geocode", status="confirmed",
         key="One address string can resolve to several places.", spot="spot-checked (results[0])"),
    dict(id="U1:31", repo="modelcontextprotocol/servers-archived", tool="maps_reverse_geocode",
         status="labels confirmed; key status flagged",
         key="The discarded results may be coarser enclosing places and not rival candidates for the same target; the API schema is not in the artifacts.",
         spot="not spot-checked"),
    dict(id="U7:18", repo="apple/ToolSandbox", tool="search_stock", status="confirmed",
         key="A stock search string can match several tickers.", spot="spot-checked ([\"stock\"][0])"),
    dict(id="U7:16", repo="apple/ToolSandbox", tool="search_lat_lon", status="labels confirmed; key status flagged",
         key="Reverse geocoding again: whether the dropped matches are rivals or enclosures is not in the artifacts.",
         spot="not spot-checked"),
    dict(id="U7:19", repo="apple/ToolSandbox", tool="search_holiday", status="labels confirmed; key status flagged",
         key="Top match of a fuzzy ranking, so the cap is top-1 of a ranking (top_k:1) and not first in iteration order; ties are not visible in the artifacts.",
         spot="not spot-checked"),
]
SEVEN_IDS = [d["id"] for d in SEVEN]
REVERSE_GEOCODE_IDS = ["U1:31", "U7:16"]

# Capped tools whose output carries a count that equals the page just returned (a count that reads like a total).
PAGELEN = {
    "U1:17": "notes", "U3:35": "notes", "U3:36": "notes", "U3:39": "notes", "U5:22": "notes", "U5:23": "notes",
    "U5:24": "notes", "U5:26": "notes", "U6:3": "record", "U6:16": "notes",
}
PAGELEN_RECORD_QUOTE = {
    "U6:3": "Salesforce query prints `Query returned ${result.records.length} records` (drops `totalSize` and `done`)",
}

# Ranked or similarity retrieval: the match set is not defined by a filter, so "every match" has no fixed meaning.
RANKED = {
    "U1:14": "knowledge-base chunks", "U1:15": "web search results", "U1:16": "local POI search",
    "U1:32": "place search", "U2:10": "semantic similarity", "U3:8": "fuzzy search", "U5:9": "documentation search",
    "U6:15": "vector search", "U7:17": "places around a point", "U7:19": "fuzzy name match",
    "U7:46": "search engine query", "U7:48": "BM25+ similarity", "U7:51": "BM25+ similarity",
    "U7:52": "vector similarity", "U7:54": "vector similarity",
}
FIXTURE_IDS = ["U1:5"]  # the mock resource links of the "everything" test server

# What the 18 stated-in-description flags of the units actually say.
STATED_KIND = {
    "U1:5": "cap sentence", "U1:15": "cap sentence", "U2:11": "cap sentence", "U3:38": "cap sentence",
    "U4:4": "cap sentence", "U4:24": "cap sentence",
    "U4:11": "signal sentence", "U4:15": "signal sentence", "U4:19": "signal sentence", "U4:21": "signal sentence",
    "U4:22": "signal sentence", "U4:23": "signal sentence",
    "U4:20": "parameter-level default or maximum", "U4:58": "parameter-level default or maximum",
    "U4:60": "parameter-level default or maximum", "U6:8": "parameter-level default or maximum",
    "U7:46": "parameter-level default or maximum",
    "U4:72": "not a cap statement",
}

PUBLISHED_INITIAL = {  # n, capped, silent, silent_unreach, stated, dropped, first (non-unique)
    "mcp": (272, 213, 87, 54, 17, 84, 2),
    "benchmark": (146, 89, 89, 7, 1, 0, 5),
    "upstream": (18, 18, 0, 0, 1, 0, 0),
}
PUBLISHED_AGENT_INITIAL = dict(n=418, capped=302, silent=176, silent_unreach=61, stated=18, dropped=84, first=7)


ELIGIBILITY = [
    "Population: a lookup tool is any tool that searches, lists or looks records up by a name or filter. Pure get-by-id tools are skipped unless they resolve names. One Gemini Flash reader coded each unit; a consolidated tool with several methods is one row.",
    "Units: U1-U6 are MCP servers and toolkits, U7 is five agent benchmarks, U8 is 18 upstream endpoints read from documentation. The agent-facing total is U1-U7 (418 tools); U8 is a reference and never enters it.",
    "Capped: result_cap is not none. This includes first_match, top_k:k, page:n and unknown, because the reader prompt counts a default page size as a cap even when a parameter can raise it. Two agent-facing tools carry unknown (U5:9, U5:16) and count as capped.",
    "Correction: the nine unique-key tools are re-coded result_cap none. They stay in the lookup-tool counts (272, 146) and leave the capped counts and every property count.",
    "No signal (the initial silent set): capped and agent_visible_signal equal to [none]. A signal is any of total_count, has_more, next_cursor_or_token, truncation_notice. A count equal to the returned page is coded as no signal.",
    "Old column no way to get more: silent and more_results_reachable in {no, limit_param_only}. It is kept for continuity and replaced by property (b).",
    "Cap stated: capped and cap_stated_in_description. The flag was set by each unit: U4 counted parameter-level defaults, U7 did not count AppWorld's page_limit parameter.",
    "Wrapper dropped: upstream_signal_dropped is yes, counted over all rows of the group and not only capped rows. Benchmarks are coded na.",
    "First match on a non-unique key: result_cap is first_match after the correction.",
    "Properties (a), (b) and (c) are defined for capped tools only. An uncapped tool returns every match by construction and has nothing to certify.",
    "Sample: popular open repositories chosen by hand, not a random sample; one reader labeled each unit, and a second reader spot-checked about 140 of the 418 agent-facing rows.",
]


def load(art):
    rows = []
    for f in sorted(glob.glob(os.path.join(art, "data", "U*.jsonl"))):
        with open(f, encoding="utf-8") as fh:
            for n, line in enumerate(fh, 1):
                if line.strip():
                    r = json.loads(line)
                    r["_id"] = f"{r['unit']}:{n}"
                    r["_repo"] = r["source"].split("@")[0]
                    rows.append(r)
    return rows


def group(r):
    return "upstream" if r["unit"] == "U8" else "benchmark" if r["unit"] == "U7" else "mcp"


def pct(n, d, digits=0):
    """Percentage rounded half up (31.25 prints as 31.3, as a reader would round by hand)."""
    if not d:
        return "n/a"
    q = Decimal(10) ** -digits
    return f"{(Decimal(100 * n) / Decimal(d)).quantize(q, rounding=ROUND_HALF_UP)}%"


def cell(n, d, digits=0):
    return f"{n} ({pct(n, d, digits)})"


def esc(s):
    return str(s).replace("|", "\\|").replace("\n", " ")


def md(headers, body):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(esc(c) for c in row) + " |" for row in body]
    return "\n".join(out)


# ---------------------------------------------------------------- initial coding (as in aggregate.py)

def stats_initial(rs):
    capped = [r for r in rs if r["result_cap"] != "none"]
    silent = [r for r in capped if r["agent_visible_signal"] == ["none"]]
    return dict(
        n=len(rs), capped=len(capped), silent=len(silent),
        silent_unreach=len([r for r in silent if r["more_results_reachable"] in ("no", "limit_param_only")]),
        stated=len([r for r in capped if r["cap_stated_in_description"]]),
        dropped=len([r for r in rs if r["upstream_signal_dropped"] == "yes"]),
        first=len([r for r in rs if r["result_cap"] == "first_match" and r["tool"] not in OLD_UNIQUE_NAMES]))


def stats_corrected(rs):
    """Same columns on the corrected labels; the first-match column then needs no name exclusion."""
    s = stats_initial(rs)
    s["first"] = len([r for r in rs if r["result_cap"] == "first_match"])
    return s


# ---------------------------------------------------------------- the three properties

def prop_a(r):
    """(a) the response says that more results exist or gives a total."""
    return r["agent_visible_signal"] != ["none"]


def prop_b(r):
    """(b) B1: page, cursor or offset parameter; B2: limit parameter only (bounded reach); B0: neither."""
    reach = r["more_results_reachable"]
    return "B1" if reach == "pagination_param" else "B2" if reach == "limit_param_only" else "B0"


def prop_c(r):
    """(c) can the agent decide whether it has seen every match?
    yes_signal: the signal decides; yes_paging: no signal, but paging to an empty page decides;
    conditional: no signal, a limit can be raised, so the agent decides only when the matches fit under the
    tool's maximum limit; no: neither route exists."""
    if prop_a(r):
        return "yes_signal"
    b = prop_b(r)
    return "yes_paging" if b == "B1" else "conditional" if b == "B2" else "no"


def cellkey(r):
    return ("A1" if prop_a(r) else "A0") + prop_b(r)


def scope_stats(rs):
    """rs: rows of one scope on the corrected labels. Properties are counted over capped tools only."""
    capped = [r for r in rs if r["result_cap"] != "none"]
    cells = collections.Counter(cellkey(r) for r in capped)
    c = collections.Counter(prop_c(r) for r in capped)
    a1 = sum(v for k, v in cells.items() if k.startswith("A1"))
    a0 = len(capped) - a1
    b = collections.Counter(prop_b(r) for r in capped)
    return dict(lookup=len(rs), capped=len(capped), cells=dict(cells), a_signal=a1, a_none=a0,
                b_paging=b["B1"], b_limit=b["B2"], b_neither=b["B0"],
                b_any=b["B1"] + b["B2"], b_neither_no=sum(1 for r in capped if prop_b(r) == "B0" and r["more_results_reachable"] == "no"),
                b_neither_refine=sum(1 for r in capped if prop_b(r) == "B0" and r["more_results_reachable"] == "query_refinement_only"),
                c_signal=c["yes_signal"], c_paging=c["yes_paging"], c_cond=c["conditional"], c_no=c["no"],
                silent_paging=cells.get("A0B1", 0), silent_limit=cells.get("A0B2", 0), silent_neither=cells.get("A0B0", 0),
                stated=sum(1 for r in capped if r["cap_stated_in_description"]),
                dropped_all=sum(1 for r in rs if r["upstream_signal_dropped"] == "yes"),
                first=sum(1 for r in capped if r["result_cap"] == "first_match"))


def scopes(rows):
    s = [("MCP servers and toolkits (U1-U6)", lambda r: r["unit"] in MCP_UNITS),
         ("Agent benchmarks (U7)", lambda r: r["unit"] == "U7"),
         ("Agent-facing total (U1-U7)", lambda r: r["unit"] != "U8"),
         ("Upstream APIs (U8, reference)", lambda r: r["unit"] == "U8")]
    s += [(u, (lambda u: lambda r: r["unit"] == u)(u)) for u in UNITS]
    return [(name, [r for r in rows if f(r)]) for name, f in s]


def variant(orig_by_id, corr_by_id, order, exclude=(), revert=(), silence=()):
    """Rows of one sensitivity variant: exclude some ids, revert some corrections, or treat a signal as absent."""
    out = []
    for i in order:
        if i in exclude:
            continue
        r = orig_by_id[i] if i in revert else corr_by_id[i]
        if i in silence:
            r = dict(r)
            r["agent_visible_signal"] = ["none"]
        out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default=str(DEFAULT_ART))
    ap.add_argument("--tau-root", default=str(DEFAULT_TAU))
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = load(args.artifacts)
    assert len(rows) == 436, len(rows)
    for r in rows:
        assert ("none" not in r["agent_visible_signal"]) or r["agent_visible_signal"] == ["none"], r["_id"]
    by_id = {r["_id"]: r for r in rows}
    order = [r["_id"] for r in rows]
    result = {}

    print(f"Re-aggregation of 2026-10-13 from {len(rows)} per-tool labels in {args.artifacts}\n")
    print("## Eligibility rules\n")
    for i, rule in enumerate(ELIGIBILITY, 1):
        print(f"{i}. {rule}")
    print()

    # ------------------------------------------------------------ 1. initial coding reproduced
    print(f"## Initial coding ({INITIAL_DATE}), reproduced\n")
    hdr = ["Group", "Lookup tools", "Capped", "Capped, no signal", "Of those, no way to get more",
           "Cap stated in description", "Wrapper dropped an upstream signal", "First match on a non-unique key"]
    body = []
    init = {}
    for g, label in (("mcp", "MCP servers and toolkits (U1-U6)"), ("benchmark", "Agent benchmarks (U7)"),
                     ("upstream", "Upstream APIs (U8)")):
        s = stats_initial([r for r in rows if group(r) == g])
        init[g] = s
        assert (s["n"], s["capped"], s["silent"], s["silent_unreach"], s["stated"], s["dropped"], s["first"]) == PUBLISHED_INITIAL[g], (g, s)
        body.append([label, s["n"], s["capped"], cell(s["silent"], s["capped"]), s["silent_unreach"], s["stated"],
                     s["dropped"], s["first"]])
    agent = [r for r in rows if r["unit"] != "U8"]
    s = stats_initial(agent)
    assert s == PUBLISHED_AGENT_INITIAL, s
    init["agent"] = s
    body.append(["Agent-facing total (U1-U7)", s["n"], s["capped"], cell(s["silent"], s["capped"]), s["silent_unreach"],
                 s["stated"], s["dropped"], s["first"]])
    print(md(hdr, body))
    print("\nAll published numbers of the 2026-10-12 table reproduce from the per-tool labels.\n")
    result["initial"] = init

    # ------------------------------------------------------------ 2. the nine corrections
    first_all = [r for r in rows if r["result_cap"] == "first_match"]
    assert len(first_all) == 16
    old_nine = sorted(r["_id"] for r in first_all if r["tool"] in OLD_UNIQUE_NAMES)
    assert old_nine == sorted(NINE_IDS), (old_nine, NINE_IDS)
    for d in NINE:
        r = by_id[d["id"]]
        assert r["tool"] == d["tool"] and r["_repo"] == d["repo"], d
        assert r["result_cap"] == "first_match" and r["agent_visible_signal"] == ["none"], d
    seven_found = sorted(r["_id"] for r in first_all if r["_id"] not in NINE_IDS)
    assert seven_found == sorted(SEVEN_IDS), (seven_found, SEVEN_IDS)
    for d in SEVEN:
        assert by_id[d["id"]]["tool"] == d["tool"] and by_id[d["id"]]["_repo"] == d["repo"], d

    corr = []
    for r in rows:
        r2 = dict(r)
        if r["_id"] in NINE_IDS:
            r2["result_cap"] = "none"
            r2["_corrected"] = True
        corr.append(r2)
    corr_by_id = {r["_id"]: r for r in corr}

    print("## The nine unique-key corrections\n")
    body = []
    for i, d in enumerate(NINE, 1):
        r = by_id[d["id"]]
        quote = r["description_quote"] or r["notes"]
        body.append([i, f"`{d['tool']}` ({d['repo']}, {d['id']})",
                     f"{r['result_cap']}; signal {','.join(r['agent_visible_signal'])}; reach {r['more_results_reachable']}",
                     "result_cap none (leaves the capped set)", d["tier"], d["why"], f"\"{quote}\"", d["spot"]])
    print(md(["#", "Tool (source, row)", "Original label", "Corrected label", "Tier", "Reason", "Decisive text in the artifacts",
              "Independent check"], body))
    print()

    # ------------------------------------------------------------ 3. corrected totals
    print("## Corrected totals, same columns as the table of 2026-10-12\n")
    cor = {}
    body = []
    for g, label in (("mcp", "MCP servers and toolkits (U1-U6)"), ("benchmark", "Agent benchmarks (U7)"),
                     ("upstream", "Upstream APIs (U8)")):
        s = stats_corrected([r for r in corr if group(r) == g])
        cor[g] = s
        body.append([label, s["n"], s["capped"], cell(s["silent"], s["capped"]), s["silent_unreach"], s["stated"],
                     s["dropped"], s["first"]])
    s = stats_corrected([r for r in corr if r["unit"] != "U8"])
    cor["agent"] = s
    body.append(["Agent-facing total (U1-U7)", s["n"], s["capped"], cell(s["silent"], s["capped"]), s["silent_unreach"],
                 s["stated"], s["dropped"], s["first"]])
    print(md(hdr, body))
    assert (cor["mcp"]["capped"], cor["mcp"]["silent"]) == (211, 85)
    assert (cor["benchmark"]["capped"], cor["benchmark"]["silent"]) == (82, 82)
    assert cor["agent"]["first"] == 7
    print()
    result["corrected"] = cor

    # ------------------------------------------------------------ 4. three properties
    print("## Three properties on the corrected labels (capped tools only)\n")
    sc = scopes(corr)
    sstats = {name: scope_stats(rs) for name, rs in sc}
    result["scopes"] = sstats
    print("### (a) signal and (c) certification\n")
    body = []
    for name, rs in sc:
        s = sstats[name]
        d = s["capped"]
        body.append([name, s["lookup"], d, cell(s["a_signal"], d), cell(s["a_none"], d),
                     cell(s["c_signal"], d), cell(s["c_paging"], d), cell(s["c_cond"], d), cell(s["c_no"], d)])
    print(md(["Scope", "Lookup tools", "Capped (denominator)", "(a) signal available", "(a) no signal",
              "(c) decided by the signal", "(c) decided by paging to the end", "(c) only if matches fit under the limit",
              "(c) cannot be decided"], body))
    print("\n### (b) access to further results\n")
    body = []
    for name, rs in sc:
        s = sstats[name]
        d = s["capped"]
        body.append([name, d, cell(s["b_paging"], d), cell(s["b_limit"], d), cell(s["b_neither"], d),
                     s["a_none"], cell(s["silent_paging"], s["a_none"]), cell(s["silent_limit"], s["a_none"]),
                     cell(s["silent_neither"], s["a_none"])])
    print(md(["Scope", "Capped", "(b) page, cursor or offset", "(b) limit parameter only", "(b) neither",
              "No-signal tools", "of which page, cursor or offset", "of which limit only", "of which neither"], body))
    print("\n### Signal by access (corrected, capped tools)\n")
    body = []
    for name in ("MCP servers and toolkits (U1-U6)", "Agent benchmarks (U7)", "Agent-facing total (U1-U7)",
                 "Upstream APIs (U8, reference)"):
        s = sstats[name]
        c = s["cells"]
        body.append([name, s["capped"], c.get("A1B1", 0), c.get("A1B2", 0), c.get("A1B0", 0), c.get("A0B1", 0),
                     c.get("A0B2", 0), c.get("A0B0", 0)])
    print(md(["Scope", "Capped", "signal + paging", "signal + limit only", "signal, no access", "no signal + paging",
              "no signal + limit only", "no signal, no access"], body))

    print("\n### Silent tools that wrap an upstream signal (MCP, corrected)\n")
    a0m = [r for r in corr if r["unit"] in MCP_UNITS and r["result_cap"] != "none" and not prop_a(r)]
    dr = collections.Counter(r["upstream_signal_dropped"] for r in a0m)
    print(f"{len(a0m)} silent MCP tools: upstream_signal_dropped = {dict(dr)}\n")
    result["silent_mcp_dropped"] = dict(dr)

    print("### Per source repository (sources with at least one no-signal capped tool)\n")
    bysrc = collections.defaultdict(list)
    for r in corr:
        if r["unit"] != "U8":
            bysrc[(group(r), r["_repo"])].append(r)
    body = []
    persource = {}
    for (g, repo), rs in sorted(bysrc.items(), key=lambda kv: (kv[0][0] != "mcp", -scope_stats(kv[1])["a_none"], kv[0][1])):
        s = scope_stats(rs)
        persource[repo] = s
        if s["a_none"]:
            body.append([repo, "MCP" if g == "mcp" else "benchmark", s["lookup"], s["capped"], s["a_none"],
                         s["silent_paging"], s["silent_limit"], s["silent_neither"]])
    print(md(["Source", "Group", "Lookup tools", "Capped", "No signal", "no signal + paging", "no signal + limit only",
              "no signal, no access"], body))
    result["per_source"] = persource
    n_src = collections.Counter(g for (g, _repo) in bysrc)
    src_a0 = collections.Counter(g for (g, repo), rs in bysrc.items() if scope_stats(rs)["a_none"])
    src_c = collections.Counter(g for (g, repo), rs in bysrc.items() if scope_stats(rs)["c_no"])
    print(f"\nSources: MCP {n_src['mcp']}, benchmark {n_src['benchmark']}. Sources with at least one no-signal capped tool: "
          f"MCP {src_a0['mcp']}, benchmark {src_a0['benchmark']}. Sources with at least one tool whose completeness "
          f"cannot be decided: MCP {src_c['mcp']}, benchmark {src_c['benchmark']}.\n")
    result["sources"] = dict(total=dict(n_src), a0=dict(src_a0), c_no=dict(src_c))

    ev = collections.Counter(r["evidence"] for r in corr if r["unit"] != "U8" and r["result_cap"] != "none")
    ms365 = sum(c for e, c in ev.items() if e.startswith("src/endpoints.json, src/graph-tools.ts:2678-2908"))
    appworld = sum(1 for r in corr if r["unit"] != "U8" and r["result_cap"] != "none" and "pagination.py:11-19" in r["evidence"])
    print(f"Clusters: {ms365} capped Microsoft 365 tools share one evidence span (graph-tools.ts:2678-2908) and "
          f"{appworld} capped AppWorld tools share one helper (pagination.py:11-19); together "
          f"{ms365 + appworld} of {sstats['Agent-facing total (U1-U7)']['capped']} capped agent-facing tools.\n")
    result["clusters"] = dict(ms365=ms365, appworld=appworld)

    # ------------------------------------------------------------ 5. second reading
    print("## Second reading\n")
    nq = sum(1 for r in rows if r["description_quote"])
    print(f"Rows with a description quote: {nq} of {len(rows)}. No row stores a tool input schema; the evidence cell holds "
          f"file and line numbers only. The second reading therefore tests each label against the unit's own note, the "
          f"quoted description and the spot checks.\n")

    print("### Seven first-match tools on a non-unique key\n")
    body = []
    for d in SEVEN:
        r = by_id[d["id"]]
        quote = r["description_quote"] or r["notes"]
        body.append([f"`{d['tool']}` ({d['repo']}, {d['id']})",
                     f"cap {r['result_cap']}; signal {','.join(r['agent_visible_signal'])}; reach {r['more_results_reachable']}; "
                     f"can_return_multiple {str(r['can_return_multiple']).lower()}; "
                     f"(a) {'signal' if prop_a(r) else 'none'}, (b) {prop_b(r)}, (c) {prop_c(r)}",
                     d["status"], d["key"], f"\"{quote}\" ({r['evidence']})", d["spot"]])
    print(md(["Tool", "Labels as coded", "Second-reading verdict", "Why the key is shared", "Decisive text", "Independent check"], body))
    print()

    print("### Disputed categories\n")
    body = []

    # signal without access
    sa = [r for r in corr if r["unit"] != "U8" and r["result_cap"] != "none" and prop_a(r) and prop_b(r) != "B1"]
    for r in sa:
        text = f"\"{r['notes']}\""
        if r["description_quote"]:
            text = f"description: \"{r['description_quote']}\"; unit note: {text}"
        body.append(["signal without paging", f"`{r['tool']}` ({r['_id']})",
                     f"signal {','.join(r['agent_visible_signal'])}; reach {r['more_results_reachable']}",
                     "consistent with its note" if r["_id"] != "U4:4" else "consistent with its note; the notice appears only with OAuth user tokens (xoxp)",
                     text])
    # page-length counts
    for i, src in PAGELEN.items():
        r = by_id[i]
        if src == "notes":
            assert re.search(r"page (length|size|array)|returned (count|items)|array count|current page|returned count", r["notes"], re.I), i
            txt = r["notes"]
        else:
            txt = PAGELEN_RECORD_QUOTE[i]
        body.append(["count equals page length", f"`{r['tool']}` ({i})",
                     f"signal {','.join(r['agent_visible_signal'])}; reach {r['more_results_reachable']}",
                     "confirmed; counts as no signal" if prop_a(r) is False else "confirmed; the has_more flag is real, the count is not",
                     f"\"{txt}\"" + ("" if src == "notes" else " (spot check)")])
    # cap unknown
    for r in corr:
        if r["unit"] != "U8" and r["result_cap"] == "unknown":
            body.append(["cap coded unknown", f"`{r['tool']}` ({r['_id']})",
                         f"signal {','.join(r['agent_visible_signal'])}; reach {r['more_results_reachable']}",
                         "kept as capped under the rule result_cap != none; see sensitivity", f"\"{r['notes']}\""])
    # misleading description or schema
    for i, verdict in (("U1:15", "reach coded from code; a schema-only reading would say paging"),
                       ("U1:37", "the unit's note says the tool claims all replies; cap coded from code")):
        r = by_id[i]
        text = f"\"{r['notes']}\""
        if r["description_quote"]:
            text = f"description: \"{r['description_quote']}\"; unit note: {text}"
        body.append(["description or schema disagrees with the code", f"`{r['tool']}` ({i})",
                     f"cap {r['result_cap']}; signal {','.join(r['agent_visible_signal'])}; reach {r['more_results_reachable']}",
                     verdict, text])
    # shared mechanisms
    aw = [r for r in corr if r["unit"] == "U7" and "pagination.py:11-19" in r["evidence"]]
    assert len(aw) == 70 and all(r["result_cap"] == "page:5" and r["more_results_reachable"] == "pagination_param"
                                 and r["agent_visible_signal"] == ["none"] for r in aw)
    body.append(["shared mechanism", f"{len(aw)} AppWorld tools (U7:56-U7:125)", "page:5; signal none; reach pagination_param",
                 "confirmed; silent but pageable. The page_limit and page_index parameter text is not in the artifacts",
                 f"\"{aw[0]['notes']}\" (all {len(aw)} rows cite pagination.py:11-19); survey note: \"AppWorld's 70 silent tools accept `page_index`, so an agent that pages until an empty page sees everything.\""])
    m365 = [r for r in corr if r["_repo"] == "softeria/ms-365-mcp-server" and r["result_cap"] != "none"]
    m365_same = [r for r in m365 if r["evidence"].startswith("src/endpoints.json, src/graph-tools.ts:2678-2908")]
    m365_a1 = [r for r in m365 if prop_a(r)]
    body.append(["shared mechanism", f"{len(m365)} capped Microsoft 365 tools (U4:27-U4:80)",
                 f"{len(m365_a1)} with next_cursor_or_token, {len(m365) - len(m365_a1)} without",
                 f"confirmed against the notes; all {len(m365_same)} rows cite one span, so per-tool schemas are not in the artifacts",
                 f"\"{m365[0]['notes']}\" ({m365[0]['evidence']})"])
    # mock fixture
    r = by_id["U1:5"]
    body.append(["test fixture", f"`{r['tool']}` ({r['_id']})", f"cap {r['result_cap']}; reach {r['more_results_reachable']}",
                 "kept; see sensitivity", f"\"{r['notes']}\""])
    print(md(["Category", "Tool (row)", "Labels as coded", "Second-reading verdict", "Decisive text"], body))
    print()

    print("### Ranked or similarity retrieval tools (the match set is not defined by a filter)\n")
    body = []
    for i, what in RANKED.items():
        r = by_id[i]
        body.append([f"`{r['tool']}` ({i})", what, f"cap {r['result_cap']}; (a) {'signal' if prop_a(r) else 'none'}, (b) {prop_b(r)}",
                     f"\"{r['notes']}\""])
    print(md(["Tool (row)", "Kind", "Labels as coded", "Unit note"], body))
    ra = collections.Counter(cellkey(corr_by_id[i]) for i in RANKED if corr_by_id[i]["result_cap"] != "none")
    print(f"\n{len(RANKED)} tools; cells {dict(ra)}.\n")

    print("### What the 18 stated-in-description flags say (capped agent-facing tools, initial coding)\n")
    st_rows = [r for r in rows if r["unit"] != "U8" and r["result_cap"] != "none" and r["cap_stated_in_description"]]
    assert sorted(r["_id"] for r in st_rows) == sorted(STATED_KIND), sorted(set(STATED_KIND) ^ {r["_id"] for r in st_rows})
    kinds = collections.Counter(STATED_KIND[r["_id"]] for r in st_rows)
    body = []
    for k in ("cap sentence", "signal sentence", "parameter-level default or maximum", "not a cap statement"):
        ids = [r for r in st_rows if STATED_KIND[r["_id"]] == k]
        body.append([k, len(ids), ", ".join(f"`{r['tool']}`" for r in ids)])
    print(md(["What the sentence says", "Tools", "Which"], body))
    den = cor["agent"]["capped"]
    n_cap = kinds["cap sentence"]
    n_sig = kinds["signal sentence"]
    n_par = kinds["parameter-level default or maximum"]
    stated_report = dict(denominator=den, cap_sentence=n_cap, plus_signal_sentence=n_cap + n_sig,
                         plus_parameter_level=n_cap + n_sig + n_par, appworld_parameter_level=appworld,
                         with_appworld=n_cap + n_sig + n_par + appworld)
    result["stated"] = stated_report
    print(f"\nAfter the correction the denominator is {den} capped agent-facing tools. Cap sentence only: {n_cap} "
          f"({pct(n_cap, den, 1)}). Cap or signal sentence: {n_cap + n_sig} ({pct(n_cap + n_sig, den, 1)}). "
          f"All unit flags that survive the correction: {n_cap + n_sig + n_par} ({pct(n_cap + n_sig + n_par, den, 1)}). "
          f"If AppWorld's page_limit parameter counts as documentation (the U4 reading), add {appworld}: "
          f"{n_cap + n_sig + n_par + appworld} ({pct(n_cap + n_sig + n_par + appworld, den, 1)}).\n")

    # ------------------------------------------------------------ 6. sensitivity
    print("### Sensitivity of the headline counts to the disputed categories\n")
    capunk = [r["_id"] for r in rows if r["unit"] != "U8" and r["result_cap"] == "unknown"]
    variants = [
        ("Corrected base", dict()),
        ("Revert the three data-unique identifier lookups (T2) to capped", dict(revert=T2_IDS)),
        ("Drop the two tools with cap unknown", dict(exclude=capunk)),
        ("Drop the 15 ranked or similarity retrieval tools", dict(exclude=list(RANKED))),
        ("Drop the mock tool of the everything test server", dict(exclude=FIXTURE_IDS)),
        ("Treat the xoxp-only notice of conversations_unreads as no signal", dict(silence=["U4:4"])),
        ("Drop the two reverse-geocoding tools", dict(exclude=REVERSE_GEOCODE_IDS)),
        ("Drop cap unknown, ranked, mock and reverse-geocoding tools", dict(
            exclude=capunk + list(RANKED) + FIXTURE_IDS + REVERSE_GEOCODE_IDS)),
    ]
    body = []
    sens = {}
    for label, kw in variants:
        rs = variant(by_id, corr_by_id, order, **kw)
        rs = [r for r in rs if r["unit"] != "U8"]
        m = scope_stats([r for r in rs if r["unit"] in MCP_UNITS])
        b = scope_stats([r for r in rs if r["unit"] == "U7"])
        t = scope_stats(rs)
        sens[label] = dict(mcp=m, benchmark=b, total=t)
        body.append([label, f"{m['a_none']} of {m['capped']}", f"{m['c_no']}", f"{b['a_none']} of {b['capped']}",
                     f"{b['c_no']}", f"{t['c_no']} of {t['capped']}", t["first"]])
    print(md(["Variant", "MCP: no signal", "MCP: cannot decide", "Benchmark: no signal", "Benchmark: cannot decide",
              "Total: cannot decide", "First-match tools left"], body))
    result["sensitivity"] = {k: {g: {kk: vv for kk, vv in v.items() if kk in ("capped", "a_none", "c_no", "c_cond", "first")}
                                 for g, v in d.items()} for k, d in sens.items()}
    print()

    # ------------------------------------------------------------ 7. supplementary tau-bench check
    tau = Path(args.tau_root)
    print("### Supplementary check on the local tau-bench checkout (outside the artifact folder)\n")
    users_json = tau / "tau_bench" / "envs" / "retail" / "data" / "users.json"
    tools_dir = tau / "tau_bench" / "envs" / "retail" / "tools"
    if users_json.exists() and tools_dir.exists():
        users = json.load(open(users_json, encoding="utf-8"))
        emails = collections.Counter(v["email"].lower() for v in users.values())
        names = collections.Counter((v["name"]["first_name"].lower(), v["name"]["last_name"].lower()) for v in users.values())
        trip = collections.Counter((v["name"]["first_name"].lower(), v["name"]["last_name"].lower(), v["address"]["zip"])
                                   for v in users.values())
        shared = {k: c for k, c in names.items() if c > 1}
        nz = (tools_dir / "find_user_id_by_name_zip.py").read_text(encoding="utf-8")
        em = (tools_dir / "find_user_id_by_email.py").read_text(encoding="utf-8")
        loop = "for user_id, profile in users.items():" in nz and "return user_id" in nz
        loop_e = "for user_id, profile in users.items():" in em and "return user_id" in em
        desc_silent = not re.search(r"multiple|several|first match|all match|more than one", nz, re.I)
        info = dict(users=len(users), distinct_emails=len(emails), duplicate_emails=sum(1 for c in emails.values() if c > 1),
                    distinct_full_names=len(names), shared_full_names=len(shared), users_in_shared_names=sum(shared.values()),
                    repeated_name_zip_triples=sum(1 for c in trip.values() if c > 1),
                    name_zip_returns_first_in_loop=loop, email_returns_first_in_loop=loop_e,
                    name_zip_description_silent_on_multiple_matches=desc_silent)
        mdesc = re.search(r'"description":\s*\((.*?)\),\s*"parameters"', nz, re.S)
        desc_text = "".join(re.findall(r'"([^"]*)"', mdesc.group(1))) if mdesc else ""
        info["name_zip_description"] = desc_text
        info["name_zip_loop_lines"] = "tools/find_user_id_by_name_zip.py:11-18 (for user_id, profile in users.items(): if ...: return user_id)"
        print("```\n" + json.dumps(info, indent=1) + "\n```\n")
        result["tau_supplementary"] = info
    else:
        print(f"Skipped: {tau} not found.\n")
        result["tau_supplementary"] = None

    # ------------------------------------------------------------ 8. figures for the paper
    print("## Figures for the paper\n")
    M = sstats["MCP servers and toolkits (U1-U6)"]
    B = sstats["Agent benchmarks (U7)"]
    T = sstats["Agent-facing total (U1-U7)"]
    fig = []
    fig.append(f"(a) No signal: MCP {M['a_none']} of {M['capped']} capped tools ({pct(M['a_none'], M['capped'], 1)}); "
               f"benchmarks {B['a_none']} of {B['capped']} ({pct(B['a_none'], B['capped'], 1)}). "
               f"Initial coding: 87 of 213 ({pct(87, 213, 1)}) and 89 of 89.")
    fig.append(f"(a) Share of all lookup tools: MCP {M['a_none']} of {M['lookup']} ({pct(M['a_none'], M['lookup'], 1)}); "
               f"benchmarks {B['a_none']} of {B['lookup']} ({pct(B['a_none'], B['lookup'], 1)}); "
               f"total {T['a_none']} of {T['lookup']} ({pct(T['a_none'], T['lookup'], 1)}).")
    fig.append(f"(b) Among the {M['a_none']} silent MCP tools: {M['silent_paging']} take a page, cursor or offset, "
               f"{M['silent_limit']} a limit only, {M['silent_neither']} neither. Among the {B['a_none']} silent benchmark tools: "
               f"{B['silent_paging']}, {B['silent_limit']}, {B['silent_neither']}.")
    fig.append(f"(b) Over all capped tools, page, cursor or offset: MCP {M['b_paging']} of {M['capped']} "
               f"({pct(M['b_paging'], M['capped'], 1)}), benchmarks {B['b_paging']} of {B['capped']} "
               f"({pct(B['b_paging'], B['capped'], 1)}); with limit-only access added: MCP {M['b_any']}, benchmarks {B['b_any']}.")
    fig.append(f"(c) Completeness cannot be established: MCP {M['c_no']} of {M['capped']} ({pct(M['c_no'], M['capped'], 1)}), "
               f"benchmarks {B['c_no']} of {B['capped']} ({pct(B['c_no'], B['capped'], 1)}), total {T['c_no']} of {T['capped']} "
               f"({pct(T['c_no'], T['capped'], 1)}). Established only if the matches fit under the limit: MCP {M['c_cond']}, "
               f"benchmarks {B['c_cond']}, total {T['c_cond']}.")
    assert all(cellkey(corr_by_id[i]) == "A0B0" for i in SEVEN_IDS)
    strict = [i for i in SEVEN_IDS if i not in REVERSE_GEOCODE_IDS and i != "U7:19"]
    fig.append(f"First-match tools on a non-unique key: {T['first']} of {T['lookup']} lookup tools ({pct(T['first'], T['lookup'], 1)}) "
               f"and {T['first']} of {T['capped']} capped tools ({pct(T['first'], T['capped'], 1)}); all {T['first']} sit in the "
               f"no-signal, no-access cell. Strict reading (name or address string, first in order; reverse geocoding and "
               f"the fuzzy top-1 left out): {len(strict)}.")
    fig.append(f"Tool-level counts are clustered: {ms365 + appworld} of {T['capped']} capped tools rest on two shared mechanisms; "
               f"{src_a0['mcp']} of {n_src['mcp']} MCP sources and {src_a0['benchmark']} of {n_src['benchmark']} benchmark sources "
               f"have a no-signal capped tool; {src_c['mcp']} and {src_c['benchmark']} have a tool whose completeness cannot be established.")
    for line in fig:
        print(f"- {line}")
    print()
    result["figures"] = fig

    print("### Citeable figures (corrected labels, denominator = capped tools)\n")
    rowsdef = [("Capped tools (denominator)", "capped"), ("(a) signal available", "a_signal"), ("(a) no signal", "a_none"),
               ("(b) page, cursor or offset", "b_paging"), ("(b) limit parameter only", "b_limit"), ("(b) neither", "b_neither"),
               ("(c) decided by the signal", "c_signal"), ("(c) decided by paging to the end", "c_paging"),
               ("(c) only if the matches fit under the limit", "c_cond"), ("(c) cannot be established", "c_no")]
    body = []
    for label, key in rowsdef:
        row = [label]
        for S in (M, B, T):
            row.append(S[key] if key == "capped" else cell(S[key], S["capped"], 1))
        body.append(row)
    print(md(["Figure", "MCP servers and toolkits", "Agent benchmarks", "Agent-facing total"], body))
    print()

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1, sort_keys=True, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
