"""The 2x2 crossover, descriptive only: structure (referent grid, stated absence) x claim (stated, unstated). Inside each
structure one factor flips: whether the user states the claim the structure turns on. No threshold, no pass/fail.

  python crossover.py [--work $WTA_WORK] [--agents STEM,STEM] [--per-agent]

Conditions and sources (an agent stem is the all_agents.LABEL key, with ":" written "_"):
  referent, unstated  archived grid rollouts (all_agents.manifest and load), rows without a "stated" field
  referent, stated    <work>/stated/<stem>.jsonl from run_grid.py --stated ("stated": true), scored by all_agents.load
  absence, stated     <work>/absence/<stem>.jsonl from run_absence.py, act rows (score_absence loaders; no field = stated)
  absence, unstated   <work>/unstated/<stem>.jsonl from run_absence.py --unstated ("stated": false)
A missing file is skipped with a note on stderr (so are short cells: 80 rollouts each). Statistics weight agents equally.

Measures (referent | absence); referent cells are irreversible, contracts all (listed) and page:
  clear acting     K = 1 listed                          | W0 counted
  counted acting   K = 2 page                            | W1 counted
  fetch share      s["seen"] >= 2 on K = 2 page (seeing.py) | the fetched score of score_absence on W1 counted
  no-fetch / fetched acting   counted acting split by the fetch score above
  shown acting     K = 2 listed                          | W1 shown
Contrasts as in score_absence: gap = no-fetch counted acting minus shown acting (S2: agents with at least 5 no-fetch
rollouts); drop = no-fetch minus fetched acting (S3: at least 5 in each group). Intervals are agent-cluster bootstrap 95%
intervals (score_absence.boot); paired rows bootstrap the per-agent difference (stated minus unstated) over agents that
have the measure in both conditions.

Clear-acting agents, fixed from the archived condition of each structure so the new data cannot pick them:
  absence   the V rule of score_absence (W0 acting, counted and shown pooled, at least 0.80) on the stated data
  referent  the same rule with W0 mapped to K = 1 (listed and page pooled, 160 rollouts) on the archive (18 of 31 agents)
"""
import os
import argparse
import sys
from pathlib import Path
from statistics import mean

from all_agents import LABEL, load, manifest
from all_agents import fmt as f2  # two decimals, half up, as in the paper's tables
from score_absence import boot, cell, load as load_absence, rate, read_raw, w0_acting

MIN = 5  # no-fetch / fetched rollouts an agent needs for a gap or a drop (score_absence S2, S3)
CLEAR = 0.80
CELL = 80  # rollouts in a full cell: 16 templates x 5 reps
STRUCT = ("referent", "absence")
CONDS = [("referent", "unstated"), ("referent", "stated"), ("absence", "stated"), ("absence", "unstated")]  # archived first
BASE = {"referent": "unstated", "absence": "stated"}  # the archived claim condition of each structure
NAME = {c: f"{c[0]}, {c[1]}" + (" (archived)" if BASE[c[0]] == c[1] else " (new)") for c in CONDS}


def note(msg):
    print("note: " + msg, file=sys.stderr)


def ref_data(rows):
    """Per-rollout lists of one agent's referent rows: acting on K = 1 listed, (acted, fetched) on K = 2 page, K = 2 listed."""
    cs = lambda ct, k: [r["s"] for r in rows if r["contract"] == ct and r["k"] == k and r["consequence"] == "irreversible"]
    return {"clear": [s["acted"] for s in cs("all", 1)], "counted": [(s["acted"], s["seen"] >= 2) for s in cs("page", 2)],
            "shown": [s["acted"] for s in cs("all", 2)], "v": rate([s["acted"] for s in cs("all", 1) + cs("page", 1)])}


def abs_data(rows):
    cs = lambda tier, w: [r["s"] for r in cell(rows, tier, w)]
    return {"clear": [s["acted"] for s in cs("counted", 0)], "counted": [(s["acted"], s["fetched"]) for s in cs("counted", 1)],
            "shown": [s["acted"] for s in cs("shown", 1)], "v": w0_acting(rows)}


def short(tag, stem, d):
    for k in ("clear", "counted", "shown"):
        if len(d[k]) < CELL:
            note(f"{tag} {stem}: {k} cell has {len(d[k])} of {CELL} rollouts")


def load_all(work, stems):
    work, arch = Path(work), manifest(work)
    data = {c: {} for c in CONDS}

    def put(cond, stem, rows, builder):
        if rows:
            data[cond][stem] = builder(rows)
            short(NAME[cond], stem, data[cond][stem])

    for s in stems:
        put(("referent", "unstated"), s, [r for r in load(arch[LABEL[s]])["none"] if not r.get("stated")], ref_data)
        for cond, sub in ((("referent", "stated"), "stated"), (("absence", "stated"), "absence"),
                          (("absence", "unstated"), "unstated")):
            p = work / sub / f"{s.replace(':', '_')}.jsonl"
            if not p.exists():
                note(f"missing {p}")
                continue
            if cond[0] == "referent":
                rows = load([p])["none"]  # all_agents.load: no errors, no duplicates, scored by score_grid.score
                kept = [r for r in rows if r.get("stated") is True]
                if len(kept) < len(rows):
                    note(f"{p}: {len(rows) - len(kept)} rows without \"stated\": true left out")
                put(cond, s, kept, ref_data)
            else:
                rows = [r for r in load_absence(read_raw([p])) if r["kind"] == "act"]
                kept = [r for r in rows if r.get("stated", True) == (cond[1] == "stated")]
                if len(kept) < len(rows):
                    note(f"{p}: {len(rows) - len(kept)} rows with the other \"stated\" value left out")
                put(cond, s, kept, abs_data)
    return data


def clear_sets(data):
    ref = {a for a, d in data[("referent", "unstated")].items() if (d["v"] or 0) >= CLEAR}
    ab = {a for a, d in data[("absence", "stated")].items() if (d["v"] or 0) >= CLEAR}
    return {"referent": ref, "absence": ab}


def metrics(d):
    nf = [a for a, f in d["counted"] if not f]
    fe = [a for a, f in d["counted"] if f]
    return {"clear": rate(d["clear"]), "counted": rate([a for a, _ in d["counted"]]), "fetch": rate([f for _, f in d["counted"]]),
            "nf": nf, "fe": fe, "shown": rate(d["shown"])}


def amean(xs):
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def ci_of(by):
    return boot(by, amean) if by else None


def f3(x):
    return "-" if x is None else f"{round(x, 3) + 0.0:.3f}"  # + 0.0: no "-0.000"


def ci3(c):
    return "-" if c is None else f"[{f3(c[0])}, {f3(c[1])}]"


def pooled(xs, f=f3):
    return "-" if not xs else f"{f(sum(xs) / len(xs))} ({sum(xs)}/{len(xs)})"


def table(head, rows):
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows] + [""]


def gap_of(m):
    return rate(m["nf"]) - m["shown"] if len(m["nf"]) >= MIN and m["shown"] is not None else None


def drop_of(m):
    return rate(m["nf"]) - rate(m["fe"]) if len(m["nf"]) >= MIN and len(m["fe"]) >= MIN else None


# paired measures, each a per-agent value or None (the two named in the task first)
PAIRED = [("counted acting", lambda m: m["counted"]),
          ("no-fetch counted acting (at least 5 in both)", lambda m: rate(m["nf"]) if len(m["nf"]) >= MIN else None),
          ("fetch share", lambda m: m["fetch"]), ("shown acting", lambda m: m["shown"]),
          ("gap: no-fetch counted minus shown", gap_of)]


def block(title, ms, keep):
    """ms: condition -> agent -> metrics; keep: structure -> predicate on agent. Levels, contrasts, paired rows."""
    sel = {c: {a: m for a, m in ms[c].items() if keep[c[0]](a)} for c in CONDS}
    out = [f"## {title}", ""]
    rows = []
    for c in CONDS:
        s = sel[c]
        rows.append([NAME[c], str(len(s)), f3(amean(m["clear"] for m in s.values())), f3(amean(m["counted"] for m in s.values())),
                     f3(amean(m["fetch"] for m in s.values())), pooled([a for m in s.values() for a in m["nf"]]),
                     pooled([a for m in s.values() for a in m["fe"]])])
    out += ["Levels (agent means; pooled columns are acted/n over every rollout of the agents)", ""]
    out += table(["condition", "agents", "clear acting", "counted acting", "fetch share", "no-fetch acting, pooled",
                  "fetched acting, pooled"], rows)
    rows = []
    for c in CONDS:
        s = sel[c]
        gaps = {a: gap_of(m) for a, m in s.items() if gap_of(m) is not None}
        drops = {a: drop_of(m) for a, m in s.items() if drop_of(m) is not None}
        rows.append([NAME[c], str(len(gaps)), f3(amean(rate(s[a]["nf"]) for a in gaps)), f3(amean(s[a]["shown"] for a in gaps)),
                     f3(amean(gaps.values())), ci3(ci_of(gaps)), str(len(drops)), f3(amean(rate(s[a]["nf"]) for a in drops)),
                     f3(amean(rate(s[a]["fe"]) for a in drops)), f3(amean(drops.values())), ci3(ci_of(drops))])
    out += ["Contrasts (agent means over the agents that meet the floor of 5 rollouts per group)", ""]
    out += table(["condition", "gap agents", "no-fetch counted acting", "shown acting", "gap", "gap 95% CI", "drop agents",
                  "no-fetch acting", "fetched acting", "drop", "drop 95% CI"], rows)
    rows = []
    for st in STRUCT:
        a, b = sel[(st, "stated")], sel[(st, "unstated")]
        for name, fn in PAIRED:
            diff, sv, uv = {}, [], []
            for g in sorted(set(a) & set(b)):
                x, y = fn(a[g]), fn(b[g])
                if x is not None and y is not None:
                    diff[g] = x - y
                    sv.append(x)
                    uv.append(y)
            rows.append([st, name, str(len(diff)), f3(amean(sv)), f3(amean(uv)), f3(amean(diff.values())), ci3(ci_of(diff))])
    out += ["Paired, stated minus unstated (agents with the measure in both conditions)", ""]
    out += table(["structure", "measure", "agents", "stated", "unstated", "difference", "95% CI"], rows)
    return out


def per_agent(ms, clear):
    out = ["## Per agent", "", "Referent: clear = K = 1 listed, counted = K = 2 page, shown = K = 2 listed, fetched = second "
           "record seen before acting. Absence: clear = W0 counted, counted = W1 counted, shown = W1 shown, fetched = score_absence. "
           "No-fetch and fetched cells are acting rate (acted/n). Clear set: absence = V rule (W0 counted and shown pooled), "
           "referent = the same rule at K = 1 (listed and page pooled), both on the archived condition.", ""]
    for st in STRUCT:
        rows = []
        for stem in LABEL:
            for claim in ("stated", "unstated"):
                if stem not in ms[(st, claim)]:
                    continue
                m = ms[(st, claim)][stem]
                rows.append([LABEL[stem], claim, "yes" if stem in clear[st] else "no", f2(m["clear"]), f2(m["counted"]),
                             f2(m["fetch"]), pooled(m["nf"], f2), pooled(m["fe"], f2), f2(m["shown"])])
        out += [f"### {st.capitalize()}", ""]
        out += table(["agent", "claim", "clear set", "clear acting", "counted acting", "fetch share", "no-fetch acting",
                      "fetched acting", "shown acting"], rows)
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
    data = load_all(a.work, stems)
    clear = clear_sets(data)
    ms = {c: {g: metrics(d) for g, d in data[c].items()} for c in CONDS}
    print("# Crossover: structure x claim (descriptive)\n")
    print("Agents with data: " + "; ".join(f"{NAME[c]} {len(data[c])}" for c in CONDS) + f" (of {len(stems)} requested).")
    print("Clear-acting agents (archived conditions, acting >= 0.80): " + "; ".join(
        f"{st} {len(clear[st] & set(stems))} of {len(data[(st, BASE[st])])}" for st in STRUCT) + ".\n")
    out = block("All agents with data", ms, {st: (lambda g: True) for st in STRUCT})
    out += block("Clear-acting agents", ms, {st: (lambda g, st=st: g in clear[st]) for st in STRUCT})
    if a.per_agent:
        out += per_agent(ms, clear)
    print("\n".join(out))


if __name__ == "__main__":
    main()
