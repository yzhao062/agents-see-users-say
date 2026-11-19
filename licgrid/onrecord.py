"""The count on the record, descriptive only: contract "pagerec" (page with the count moved from the result onto each record,
as <entity>s_with_this_<key_field>, e.g. reservations_with_this_passenger_name) against the archived contract "page". No threshold, no pass/fail.

  python onrecord.py [--work $WTA_WORK] [--agents STEM,STEM] [--per-agent]

Sources (an agent stem is the all_agents.LABEL key, with ":" written "_"):
  page      archived grid rollouts (all_agents.manifest and load), rows without a "stated" field: contract page, irreversible
            K = 1 and 2, the count probe (consequence count, K = 2), and contract all (listed) K = 2 irreversible
  pagerec   <work>/onrecord/<stem>.jsonl from run_grid.py --contracts pagerec (irreversible and count, K = 1 and 2)
A missing file is skipped, and a short cell (under 80 rollouts) is kept, each with a note on stderr. Statistics weight agents equally.

Measures, for each of page and pagerec (K = 2 irreversible unless marked; the fetch split is crossover.py's referent one):
  K = 1 / K = 2 acting   P(action call)
  fetch share            s["seen"] >= 2 on K = 2 (seeing.py): the agent saw the second record before acting
  no-fetch / fetched     K = 2 acting split by the fetch score above (pooled acted/n, and the agent mean over the agents with
                         at least 5 rollouts in the group)
  gap                    no-fetch K = 2 acting minus archived listed K = 2 acting (agents with at least 5 no-fetch rollouts)
  drop                   no-fetch minus fetched acting (at least 5 in each group)
  two-share              share of K = 2 count-probe answers that state two (mechanism.classify, all_agents "two|page")
Intervals are agent-cluster bootstrap 95% intervals (score_absence.boot); paired rows bootstrap the per-agent difference
(pagerec minus page) over agents that have the measure in both. The listed baseline of the gap is the archived one for both.

Clear-acting agents are the referent set of crossover.py, fixed from the archive (K = 1 listed and page pooled, at least 0.80),
so the new rows cannot pick them. Rows are reported over all agents with data and over the clear-acting set.
"""
import os
import argparse
from pathlib import Path

from all_agents import LABEL, load, manifest
from all_agents import fmt as f2  # two decimals, half up, as in the paper's tables
from crossover import CELL, MIN, amean, ci3, ci_of, clear_sets, drop_of, f3, gap_of, note, pooled, ref_data, table
from mechanism import classify
from score_absence import rate

CONTRACTS = ("page", "pagerec")
NAME = {"page": "page (archived)", "pagerec": "pagerec (new)"}


def contract_data(rows, c):
    """Per-rollout lists of one agent's rows under contract c: acting on K = 1, (acted, fetched) on K = 2, K = 2 count answers."""
    cs = lambda k, q: [r for r in rows if r["contract"] == c and r["k"] == k and r["consequence"] == q]
    return {"k1": [r["s"]["acted"] for r in cs(1, "irreversible")],
            "k2": [(r["s"]["acted"], r["s"]["seen"] >= 2) for r in cs(2, "irreversible")],
            "two": [classify(r["final"]) == "two" for r in cs(2, "count")]}


def short(tag, stem, d):
    for k, n in (("k1", "K = 1"), ("k2", "K = 2"), ("two", "count K = 2"), ("listed", "listed K = 2")):
        if len(d[k]) < CELL:
            note(f"{tag} {stem}: {n} cell has {len(d[k])} of {CELL} rollouts")


def load_all(work, stems):
    """data[contract][stem]; listed[stem] (archived listed K = 2 acting, per rollout); the referent ref_data of the archive."""
    arch, ref, listed, data = manifest(work), {}, {}, {c: {} for c in CONTRACTS}
    for s in stems:
        rows = [r for r in load(arch[LABEL[s]])["none"] if not r.get("stated")]
        if not rows:
            note(f"no archived rows for {s}")
        else:
            ref[s] = ref_data(rows)
            listed[s] = [r["s"]["acted"] for r in rows if r["contract"] == "all" and r["k"] == 2 and r["consequence"] == "irreversible"]
            data["page"][s] = contract_data(rows, "page")
            short(NAME["page"], s, {**data["page"][s], "listed": listed[s]})
        p = Path(work) / "onrecord" / f"{s.replace(':', '_')}.jsonl"
        if not p.exists():
            note(f"missing {p}")
            continue
        rows = load([p])["none"]  # all_agents.load: no errors, no duplicates, scored by score_grid.score
        kept = [r for r in rows if r["contract"] == "pagerec" and not r.get("stated")]
        if len(kept) < len(rows):
            note(f"{p}: {len(rows) - len(kept)} rows other than contract pagerec (or with a \"stated\" field) left out")
        if kept:
            data["pagerec"][s] = contract_data(kept, "pagerec")
            short(NAME["pagerec"], s, {**data["pagerec"][s], "listed": listed.get(s, [])})
    return data, listed, ref


def metrics(d, listed):
    """gap_of and drop_of of crossover.py read these keys: nf, fe, shown (here the archived listed K = 2 acting)."""
    return {"k1": rate(d["k1"]), "k2": rate([a for a, _ in d["k2"]]), "fetch": rate([f for _, f in d["k2"]]),
            "nf": [a for a, f in d["k2"] if not f], "fe": [a for a, f in d["k2"] if f], "shown": rate(listed),
            "two": rate(d["two"])}


def mci(by):
    """Agent mean and its agent-cluster bootstrap interval over a dict agent -> value (None values left out)."""
    by = {a: x for a, x in by.items() if x is not None}
    return "-" if not by else f"{f3(amean(by.values()))} {ci3(ci_of(by))}"


# paired measures, each a per-agent value or None
PAIRED = [("K = 2 acting", lambda m: m["k2"]),
          ("no-fetch K = 2 acting (at least 5 in both)", lambda m: rate(m["nf"]) if len(m["nf"]) >= MIN else None),
          ("fetch share", lambda m: m["fetch"]), ("gap: no-fetch K = 2 minus listed K = 2", gap_of),
          ("two-share (K = 2 count answers)", lambda m: m["two"])]


def block(title, ms, keep):
    """ms: contract -> agent -> metrics; keep: predicate on agent. Levels, fetch split, contrasts, paired rows."""
    sel = {c: {a: m for a, m in ms[c].items() if keep(a)} for c in CONTRACTS}
    out = [f"## {title}", ""]
    rows = [[NAME[c], str(len(s)), *(mci({a: m[k] for a, m in s.items()}) for k in ("k1", "k2", "fetch", "two"))]
            for c, s in sel.items()]
    out += ["Levels (agent mean [agent-cluster bootstrap 95% CI])", ""]
    out += table(["contract", "agents", "K=1 acting", "K=2 acting", "fetch share", "two-share"], rows)
    rows = []
    for c, s in sel.items():
        nf = {a: rate(m["nf"]) for a, m in s.items() if len(m["nf"]) >= MIN}
        fe = {a: rate(m["fe"]) for a, m in s.items() if len(m["fe"]) >= MIN}
        rows.append([NAME[c], str(len(nf)), mci(nf), pooled([x for m in s.values() for x in m["nf"]]), str(len(fe)), mci(fe),
                     pooled([x for m in s.values() for x in m["fe"]])])
    out += ["K = 2 acting by the fetch split (agent mean over the agents with at least 5 rollouts in the group, with its CI; "
            "pooled columns are acted/n over every rollout of the agents)", ""]
    out += table(["contract", "no-fetch agents", "no-fetch acting", "no-fetch, pooled", "fetched agents", "fetched acting",
                  "fetched, pooled"], rows)
    rows = []
    for c, s in sel.items():
        gaps = {a: gap_of(m) for a, m in s.items() if gap_of(m) is not None}
        drops = {a: drop_of(m) for a, m in s.items() if drop_of(m) is not None}
        rows.append([NAME[c], str(len(gaps)), f3(amean(rate(s[a]["nf"]) for a in gaps)), f3(amean(s[a]["shown"] for a in gaps)),
                     f3(amean(gaps.values())), ci3(ci_of(gaps)), str(len(drops)), f3(amean(rate(s[a]["nf"]) for a in drops)),
                     f3(amean(rate(s[a]["fe"]) for a in drops)), f3(amean(drops.values())), ci3(ci_of(drops))])
    out += ["Contrasts (agent means over the agents that meet the floor of 5 rollouts per group; listed = archived K = 2)", ""]
    out += table(["contract", "gap agents", "no-fetch K=2 acting", "listed K=2 acting", "gap", "gap 95% CI", "drop agents",
                  "no-fetch acting", "fetched acting", "drop", "drop 95% CI"], rows)
    a, b = sel["pagerec"], sel["page"]
    rows = []
    for name, fn in PAIRED:
        diff, pv, bv = {}, [], []
        for g in sorted(set(a) & set(b)):
            x, y = fn(a[g]), fn(b[g])
            if x is not None and y is not None:
                diff[g] = x - y
                pv.append(x)
                bv.append(y)
        rows.append([name, str(len(diff)), f3(amean(pv)), f3(amean(bv)), f3(amean(diff.values())), ci3(ci_of(diff))])
    out += ["Paired, pagerec minus page (agents with the measure under both contracts)", ""]
    out += table(["measure", "agents", "pagerec", "page", "difference", "95% CI"], rows)
    return out


def diff2(x):  # a difference of rates, two decimals, no "-0.00"
    return f2(x).replace("-0.00", "0.00")


def per_agent(ms, clear):
    out = ["## Per Agent", "", "K=1 and K=2 acting are irreversible; fetched = second record seen before acting (s[\"seen\"] >= 2); "
           "no-fetch and fetched cells are acting rate (acted/n); listed = archived K = 2 listed acting; gap and drop need at "
           "least 5 rollouts per group; two = share of K = 2 count answers that state two. Clear set = referent set of "
           "crossover.py (archive).", ""]
    rows = []
    for stem in LABEL:
        for c in CONTRACTS:
            if stem not in ms[c]:
                continue
            m = ms[c][stem]
            rows.append([LABEL[stem], NAME[c], "yes" if stem in clear else "no", f2(m["k1"]), f2(m["k2"]), f2(m["fetch"]),
                         pooled(m["nf"], f2), pooled(m["fe"], f2), f2(m["shown"]), diff2(gap_of(m)), diff2(drop_of(m)), f2(m["two"])])
    out += table(["agent", "contract", "clear set", "K=1 acting", "K=2 acting", "fetch share", "no-fetch acting",
                  "fetched acting", "listed K=2", "gap", "drop", "two-share"], rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(Path(os.environ.get("WTA_WORK", "data/work"))))
    ap.add_argument("--agents", default="", help="comma-separated agent stems (all_agents.LABEL keys); default: all 31")
    ap.add_argument("--per-agent", action="store_true", help="also print the per-agent table")
    a = ap.parse_args()
    stems = [s for s in a.agents.split(",") if s] or list(LABEL)
    bad = [s for s in stems if s not in LABEL]
    if bad:
        raise SystemExit(f"unknown agent stems: {bad}")
    data, listed, ref = load_all(a.work, stems)
    # the crossover.py rule (K = 1 listed and page pooled >= 0.80), applied to the archive alone; the absence side is not used
    clear = clear_sets({("referent", "unstated"): ref, ("absence", "stated"): {}})["referent"]
    ms = {c: {g: metrics(d, listed.get(g, [])) for g, d in data[c].items()} for c in CONTRACTS}
    print("# On-record count: pagerec vs page (descriptive)\n")
    print("Agents with data: " + "; ".join(f"{NAME[c]} {len(data[c])}" for c in CONTRACTS) + f" (of {len(stems)} requested).")
    print(f"Clear-acting agents (archive, K = 1 listed and page pooled acting >= 0.80): {len(clear & set(stems))} of {len(ref)}.\n")
    out = block("All Agents with Data", ms, lambda g: True)
    out += block("Clear-Acting Agents", ms, lambda g: g in clear)
    if a.per_agent:
        out += per_agent(ms, clear)
    print("\n".join(out))


if __name__ == "__main__":
    main()
