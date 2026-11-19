"""Route crossing, descriptive: contract "reclist" (the count on the record in hand, the other matches behind a dedicated list
tool keyed by the record's id, as in the stated-absence counted tier) against "pagerec" (the same count, the other match on the
next page of the same search) and the archived "page".
The contrast with pagerec is an interface contrast: besides the route, reclist drops has_more,
rewords the search description, adds a tool before the act tool, and lists uncounted records that repeat the first match.

  python reclist.py [--work $WTA_WORK] [--agents STEM,STEM] [--per-agent] [--out PATH]

Sources (an agent stem is the all_agents.LABEL key, ":" written "_"):
  page      archived grid rollouts (all_agents.manifest and load), rows without a "stated" field
  pagerec   <work>/onrecord/<stem>.jsonl (run_grid.py --contracts pagerec)
  reclist   <work>/reclist/<stem>.jsonl (run_grid.py --contracts reclist)
Counted = K = 2 irreversible under the contract; listed = the archived listed K = 2 cell, the baseline of all three (as in
onrecord.py and itt.py). The gap is itt.py's gap over all rollouts: counted minus listed acting, agent means, with the
agent-cluster and the crossed (agents x 16 templates) bootstrap intervals of itt.analyse (2,000 resamples, seed 0).
Levels per variant: K = 1 acting, K = 2 acting, fetch share (second record seen before acting), and two-share (K = 2 count
answers that state two, mechanism.classify). Paired rows: reclist minus pagerec and reclist minus page.
Agent sets: the referent clear-acting agents (crossover.clear_sets on the archive, fixed before this run) and all agents with
data in the three variants. Self-check: the page and pagerec gaps of the clear-acting set reproduce itt.py (0.415, 0.385).
Exit status 1 when a self-check fails.
"""
import argparse
from pathlib import Path

import numpy as np

import itt
from all_agents import LABEL, load, manifest
from crossover import amean, ci3, ci_of, clear_sets, f3, note, ref_data, table
from mechanism import classify
from score_grid import rate  # (rate, n), or (nan, 0) for an empty list

VARS = [("referent", "page"), ("referent", "pagerec"), ("referent", "reclist")]
CONTRACT = {v: v[1] for v in VARS}
NAME = {("referent", "page"): "page (archived)", ("referent", "pagerec"): "pagerec (count on the record, next page)",
        ("referent", "reclist"): "reclist (count on the record, list tool)"}
PAIRS = [(("referent", "reclist"), ("referent", "pagerec"), "reclist minus pagerec"),
         (("referent", "reclist"), ("referent", "page"), "reclist minus page")]
TIDS = itt.TIDS["referent"]


def f2(p):
    """A (rate, n) pair as two decimals; '-' when n is 0."""
    return f"{p[0]:.2f}" if p[1] else "-"


def levels(rows, c):
    cs = lambda k, q: [r for r in rows if r["contract"] == c and r["k"] == k and r["consequence"] == q]
    return {"k1": rate([r["s"]["acted"] for r in cs(1, "irreversible")]),
            "k2": rate([r["s"]["acted"] for r in cs(2, "irreversible")]),
            "fetch": rate([r["s"]["seen"] >= 2 for r in cs(2, "irreversible")]),
            "two": rate([classify(r["final"]) == "two" for r in cs(2, "count")]),
            "n": {(k, q): len(cs(k, q)) for k in (1, 2) for q in ("irreversible", "count")}}


def load_all(work, stems):
    """recs[variant][stem] = {"counted": [(template, acted, fetched)], "shown": [(template, acted)]}; lev[variant][stem];
    base: the archived referent data that fixes the clear-acting set."""
    work, arch = Path(work), manifest(work)
    recs, lev, base = {v: {} for v in VARS}, {v: {} for v in VARS}, {("referent", "unstated"): {}, ("absence", "stated"): {}}
    for s in stems:
        fn = s.replace(":", "_") + ".jsonl"
        rows = [r for r in load(arch[LABEL[s]])["none"] if not r.get("stated")]
        if not rows:
            note(f"no archived rows for {s}")
            continue
        base[("referent", "unstated")][s] = ref_data(rows)
        listed = itt.listed_ref(rows)
        recs[VARS[0]][s] = {"counted": itt.counted_ref(rows, "page"), "shown": listed}
        lev[VARS[0]][s] = levels(rows, "page")
        for v, sub in ((VARS[1], "onrecord"), (VARS[2], "reclist")):
            p = work / sub / fn
            if not p.exists():
                note(f"missing {p}")
                continue
            got = load([p])["none"]
            kept = [r for r in got if r["contract"] == v[1] and not r.get("stated")]
            if len(kept) < len(got):
                note(f"{p}: {len(got) - len(kept)} rows of another contract or stated left out")
            recs[v][s] = {"counted": itt.counted_ref(kept, v[1]), "shown": listed}
            lev[v][s] = levels(kept, v[1])
            for (k, q), n in lev[v][s]["n"].items():
                if n < itt.CELL:
                    note(f"{NAME[v]} {s}: K = {k} {q} cell has {n} of {itt.CELL} rollouts")
    return recs, lev, base


def level_table(lev, agents):
    rows = []
    for v in VARS:
        by = {k: {a: lev[v][a][k][0] for a in agents if lev[v][a][k][1]} for k in ("k1", "k2", "fetch", "two")}
        rows.append([NAME[v], str(len(agents))] + [f"{f3(amean(by[k].values()))} {ci3(ci_of(by[k]))}" if by[k] else "-"
                                                    for k in ("k1", "k2", "fetch", "two")])
    return table(["variant", "agents", "K=1 acting", "K=2 acting", "fetch share", "two-share (count K=2)"], rows)


def per_agent(lev, recs, clear, agents):
    rows = []
    for s in sorted(agents, key=lambda x: LABEL[x]):
        listed = rate([a for _, a in recs[VARS[0]][s]["shown"]])
        for v in VARS:
            m = lev[v][s]
            rows.append([LABEL[s], v[1], "yes" if s in clear else "no", f2(m["k1"]), f2(m["k2"]), f2(listed),
                         f2((m["k2"][0] - listed[0], m["k2"][1])).replace("-0.00", "0.00"), f2(m["fetch"]), f2(m["two"])])
    return table(["agent", "contract", "clear set", "K=1 acting", "K=2 acting", "listed K=2", "gap", "fetch share", "two-share"], rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=itt.WORK)
    ap.add_argument("--agents", default="", help="comma-separated agent stems (all_agents.LABEL keys); default: all 31")
    ap.add_argument("--per-agent", action="store_true")
    ap.add_argument("--out", default="", help="also write the markdown to this path")
    a = ap.parse_args()
    stems = [s for s in a.agents.split(",") if s] or list(LABEL)
    bad = [s for s in stems if s not in LABEL]
    if bad:
        raise SystemExit(f"unknown agent stems: {bad}")
    recs, lev, base = load_all(a.work, stems)
    clear = clear_sets(base)["referent"]
    itt.PAIRS[:] = PAIRS  # itt.analyse reads the module's PAIRS and set_tables its NAME
    itt.NAME.update(NAME)
    have = sorted(s for s in stems if all(s in recs[v] for v in VARS))
    out = ["# Route Crossing: reclist vs pagerec vs page (descriptive)", "",
           f"Agents with all three variants: {len(have)} of {len(stems)} requested. Clear-acting (archive, K = 1 listed and page "
           f"pooled >= 0.80): {len(clear & set(have))} of them.", ""]
    ok = True
    for title, sel in (("Clear-Acting Agents", sorted(set(have) & clear)), ("All Agents with Data", have)):
        if not sel:
            continue
        Xs = {v: np.stack([itt.counts(recs[v][s], TIDS) for s in sel]) for v in VARS}
        res, pairs = itt.analyse(sel, Xs)
        out += [f"## {title} ({len(sel)})", "", "Levels (agent mean [agent-cluster bootstrap 95% CI])", ""]
        out += level_table(lev, sel)
        out += itt.set_tables(res, pairs)
        if title.startswith("Clear") and len(sel) == 18:
            for v, want in ((VARS[0], 0.415), (VARS[1], 0.385)):
                got = res[v]["itt"]
                good = abs(got - want) <= 0.0006
                ok &= good
                out += [f"Self-check, {NAME[v]} gap over all rollouts: expected {want:.3f}, computed {got:.4f}: "
                        f"{'PASS' if good else 'FAIL'}", ""]
    if a.per_agent:
        out += ["## Per Agent", "", "K = 1 and K = 2 acting are irreversible; listed = archived K = 2 listed acting; gap = K = 2 "
                "acting minus listed; fetch share = second record seen before acting; two-share = K = 2 count answers that "
                "state two.", ""]
        out += per_agent(lev, recs, clear, have)
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
