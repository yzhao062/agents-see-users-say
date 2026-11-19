"""The intention-to-treat (ITT) contrast for the referent and stated-absence structures, descriptive only.

The contract is randomized over rollouts, but the headline gap of crossover.py and onrecord.py (no-fetch counted acting minus
listed or shown acting) conditions on an agent-chosen behaviour, fetching before acting. The ITT gap does not: counted acting
over ALL rollouts minus listed (referent) or shown (absence) acting.

  python itt.py [--work $WTA_WORK] [--out [PATH]] [--frontier]

Variants (a rollout is scored by the loaders of crossover.py and onrecord.py; counted = K = 2 irreversible under the named
contract for the referent, W1 counted for absence; listed = K = 2 contract all, shown = W1 shown):
  referent page      archived rollouts, counted = page, listed = archived listed
  referent stated    <work>/stated/<stem>.jsonl ("There is only one ..."), counted = page, listed = stated listed
  referent pagerec   <work>/onrecord/<stem>.jsonl, counted = pagerec, listed = archived listed (as onrecord.py)
  absence stated     <work>/absence/<stem>.jsonl (archived), counted and shown from W1
  absence unstated   <work>/unstated/<stem>.jsonl
Measures: counted and listed or shown acting, the ITT gap (agent means; also pooled over rollouts), the no-fetch counted acting
and the no-fetch gap exactly as crossover.py defines them (agents with at least 5 no-fetch rollouts). Paired differences:
stated minus unstated (both structures) and pagerec minus page.
Agent sets: (a) the clear-acting set of each structure (crossover.clear_sets), (b) the agents clear-acting in both structures,
(c) all 31 Bedrock agents. Two bootstrap intervals for every gap, 2,000 resamples each:
  agent-cluster  score_absence.boot as in crossover.py (same resamples and seed 0)
  crossed        agents and the 16 templates resampled independently with replacement (numpy, seed 0); every agent and variant
                 of a resample shares one template draw, and the agent-mean statistic is recomputed on the resampled agents
                 x templates (a template drawn twice counts its rollouts twice, including for the floor of 5 no-fetch rollouts)
Sets of clear-acting agents are fixed from the archived conditions and are not re-drawn inside a bootstrap.
--frontier adds the frontier models (work/stated-nairr and work/onrecord-nairr). --out writes the markdown to PATH
(defaults to OUT). Exit status 1 when a self-check fails.
"""
import os
import argparse
from pathlib import Path

import numpy as np

from absence_templates import ABSENCE
from all_agents import LABEL, load, manifest
from crossover import MIN, abs_data, amean, ci3, ci_of, clear_sets, drop_of, f3, gap_of, metrics, note, pooled, ref_data, table
from mechanism import classify
from score_absence import boot, cell, load as load_absence, rate, read_raw
from templates import TEMPLATES

WORK = str(Path(os.environ.get("WTA_WORK", "data/work")))
OUT = WORK + "/logs/itt-2026-10-15.md"
CELL = 80  # rollouts in a full cell: 16 templates x 5 reps
BOOT_N, SEED = 2000, 0
STRUCT = ("referent", "absence")
TIDS = {"referent": [t.tid for t in TEMPLATES], "absence": [t.tid for t in ABSENCE]}
VARIANTS = {"referent": [("referent", "page"), ("referent", "stated"), ("referent", "pagerec")],
            "absence": [("absence", "stated"), ("absence", "unstated")]}
NAME = {("referent", "page"): "referent, page (archived)", ("referent", "stated"): "referent, stated (new)",
        ("referent", "pagerec"): "referent, pagerec (new)", ("absence", "stated"): "absence, stated (archived)",
        ("absence", "unstated"): "absence, unstated (new)"}
PAIRS = [(("referent", "stated"), ("referent", "page"), "referent: stated minus unstated (page)"),
         (("referent", "pagerec"), ("referent", "page"), "referent: pagerec minus page"),
         (("absence", "stated"), ("absence", "unstated"), "absence: stated minus unstated")]
FRONT = ["claude-opus-5.5", "gpt-6.1-sol", "gpt-6-astra"]


# ---------------------------------------------------------------- loading (scored rollouts as in crossover.py, onrecord.py)

def counted_ref(rows, contract):
    """(template, acted, fetched) on K = 2 irreversible under a contract; fetched = second record seen before acting."""
    return [(r["template"], r["s"]["acted"], r["s"]["seen"] >= 2) for r in rows
            if r["contract"] == contract and r["k"] == 2 and r["consequence"] == "irreversible"]


def listed_ref(rows):
    return [(r["template"], r["s"]["acted"]) for r in rows
            if r["contract"] == "all" and r["k"] == 2 and r["consequence"] == "irreversible"]


def absence_part(rows):
    return {"counted": [(r["template"], r["s"]["acted"], r["s"]["fetched"]) for r in cell(rows, "counted", 1)],
            "shown": [(r["template"], r["s"]["acted"]) for r in cell(rows, "shown", 1)]}


def load_all(work, stems):
    """recs[variant][stem] = {"counted": [(template, acted, fetched)], "shown": [(template, acted)]}, and base: the
    crossover.py data of the two archived conditions (their "v" values select the clear-acting agents)."""
    work, arch = Path(work), manifest(work)
    recs = {v: {} for vs in VARIANTS.values() for v in vs}
    base = {("referent", "unstated"): {}, ("absence", "stated"): {}}
    for s in stems:
        fn = s.replace(":", "_") + ".jsonl"
        rows = [r for r in load(arch[LABEL[s]])["none"] if not r.get("stated")]
        if rows:
            base[("referent", "unstated")][s] = ref_data(rows)
            recs[("referent", "page")][s] = {"counted": counted_ref(rows, "page"), "shown": listed_ref(rows)}
        else:
            note(f"no archived rows for {s}")
        p = work / "stated" / fn
        if p.exists():
            kept = [r for r in load([p])["none"] if r.get("stated") is True]
            recs[("referent", "stated")][s] = {"counted": counted_ref(kept, "page"), "shown": listed_ref(kept)}
        else:
            note(f"missing {p}")
        p = work / "onrecord" / fn
        if p.exists() and rows:
            kept = [r for r in load([p])["none"] if r["contract"] == "pagerec" and not r.get("stated")]
            recs[("referent", "pagerec")][s] = {"counted": counted_ref(kept, "pagerec"), "shown": listed_ref(rows)}
        else:
            note(f"missing {p}")
        for var, sub, claim in ((("absence", "stated"), "absence", True), (("absence", "unstated"), "unstated", False)):
            p = work / sub / fn
            if not p.exists():
                note(f"missing {p}")
                continue
            kept = [r for r in load_absence(read_raw([p])) if r["kind"] == "act" and r.get("stated", True) == claim]
            if kept:
                recs[var][s] = absence_part(kept)
                if var == ("absence", "stated"):
                    base[var][s] = abs_data(kept)
    for var, by in recs.items():
        for s, d in by.items():
            for k in ("counted", "shown"):
                if len(d[k]) < CELL:
                    note(f"{NAME[var]} {s}: {k} cell has {len(d[k])} of {CELL} rollouts")
    return recs, base


def counts(d, tids):
    """(6, T) array of one agent's weighted-count inputs per template: counted n, acted; no-fetch n, acted; listed or shown
    n, acted."""
    ix = {t: i for i, t in enumerate(tids)}
    X = np.zeros((6, len(tids)))
    for t, a, f in d["counted"]:
        X[0, ix[t]] += 1
        X[1, ix[t]] += a
        if not f:
            X[2, ix[t]] += 1
            X[3, ix[t]] += a
    for t, a in d["shown"]:
        X[4, ix[t]] += 1
        X[5, ix[t]] += a
    return X


def crossover_metrics(d):
    """The crossover.py metrics of one agent (its gap_of and drop_of read nf, fe, shown)."""
    return metrics({"clear": [], "counted": [(a, f) for _, a, f in d["counted"]], "shown": [a for _, a in d["shown"]]})


# ---------------------------------------------------------------- statistics on weighted counts S[..., agents, 6]

def nmean(x):
    """Mean over the last axis without NaN; NaN when nothing is left."""
    ok = ~np.isnan(x)
    n = ok.sum(-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(ok, x, 0.0).sum(-1) / np.maximum(n, 1), np.nan)


def agent_values(S):
    """Per-agent measures; the no-fetch ones need at least MIN no-fetch rollouts (crossover.gap_of)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        c, l = S[..., 1] / S[..., 0], S[..., 5] / S[..., 4]
        nf = np.where(S[..., 2] >= MIN, S[..., 3] / S[..., 2], np.nan)
    return {"counted": c, "listed": l, "itt": c - l, "nf": nf, "nfgap": nf - l}


def summarize(S):
    """Agent means of every measure, and the ITT gap pooled over the rollouts of the agents."""
    out = {k: nmean(v) for k, v in agent_values(S).items()}
    with np.errstate(invalid="ignore", divide="ignore"):
        out["pooled"] = S[..., 1].sum(-1) / S[..., 0].sum(-1) - S[..., 5].sum(-1) / S[..., 4].sum(-1)
    return out


def crossed(Xs, n=BOOT_N, seed=SEED):
    """Xs: variant -> (agents, 6, templates). Crossed bootstrap: agents and templates drawn independently with replacement,
    one draw shared by all variants; returns variant -> summarize() of the n resamples."""
    A, _, T = next(iter(Xs.values())).shape
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, A, size=(n, A))
    W = np.stack([np.bincount(rng.integers(0, T, size=T), minlength=T) for _ in range(n)])
    rows = np.arange(n)[:, None]
    return {v: summarize(np.einsum("amt,rt->ram", X, W)[rows, pick]) for v, X in Xs.items()}


def pct(v):
    """95% interval with the indexing of score_absence.boot."""
    v = np.sort(v[~np.isnan(v)])
    return (float(v[int(0.025 * len(v))]), float(v[int(0.975 * len(v)) - 1])) if len(v) else None


def pooled_stat(ts):
    ca, cn, la, ln = (sum(t[i] for t in ts) for i in range(4))
    return ca / cn - la / ln if cn and ln else None


def by_agent(agents, arr):
    return {a: float(x) for a, x in zip(agents, arr) if not np.isnan(x)}


def analyse(agents, Xs):
    """Both intervals for every variant and pair. agents: stems in the row order of the arrays in Xs."""
    cr, res, avs = crossed(Xs), {}, {}
    for v, X in Xs.items():
        S0 = X.sum(-1)
        av = avs[v] = agent_values(S0)
        sm = summarize(S0)
        gap_by = by_agent(agents, av["nfgap"])
        keep = ~np.isnan(av["nfgap"])
        res[v] = {"n": len(agents), "counted": float(sm["counted"]), "listed": float(sm["listed"]), "itt": float(sm["itt"]),
                  "itt_ci": ci_of(by_agent(agents, av["itt"])), "itt_x": pct(cr[v]["itt"]), "pooled": float(sm["pooled"]),
                  "pooled_ci": boot({a: tuple(S0[i, [1, 0, 5, 4]]) for i, a in enumerate(agents)}, pooled_stat),
                  "pooled_x": pct(cr[v]["pooled"]), "gap_n": len(gap_by), "nf": float(nmean(av["nf"][keep])),
                  "gap_listed": float(nmean(av["listed"][keep])), "gap": amean(gap_by.values()), "gap_ci": ci_of(gap_by),
                  "gap_x": pct(cr[v]["nfgap"])}
    pairs = []
    for a, b, name in PAIRS:
        if a not in Xs or b not in Xs:
            continue
        for key, label in (("itt", "ITT gap"), ("counted", "counted acting")):
            diff = {g: x - y for g, x, y in zip(agents, avs[a][key], avs[b][key]) if not (np.isnan(x) or np.isnan(y))}
            pairs.append({"pair": name, "measure": label, "n": len(diff), "first": res[a][key], "second": res[b][key],
                          "diff": amean(diff.values()), "ci": ci_of(diff), "x": pct(cr[a][key] - cr[b][key])})
    return res, pairs


# ---------------------------------------------------------------- formatting

def g3(x):
    return "-" if x is None or np.isnan(x) else f3(float(x))


def cx(c):  # an interval or None
    return "-" if c is None else ci3(c)


def set_tables(res, pairs):
    rows = [[NAME[v], str(r["n"]), g3(r["counted"]), g3(r["listed"]), g3(r["itt"]), cx(r["itt_ci"]), cx(r["itt_x"]),
             g3(r["pooled"]), cx(r["pooled_ci"]), cx(r["pooled_x"])] for v, r in res.items()]
    out = ["ITT gap: counted acting minus listed (referent) or shown (absence) acting, over ALL rollouts (agent means; the "
           "pooled gap weights every rollout equally)", ""]
    out += table(["variant", "agents", "counted acting", "listed or shown acting", "ITT gap", "agent-cluster 95% CI",
                  "crossed 95% CI", "pooled ITT gap", "pooled, agent-cluster CI", "pooled, crossed CI"], rows)
    rows = [[NAME[v], str(r["gap_n"]), g3(r["nf"]), g3(r["gap_listed"]), g3(r["gap"]), cx(r["gap_ci"]), cx(r["gap_x"])]
            for v, r in res.items()]
    out += ["No-fetch gap as crossover.py and onrecord.py define it (agents with at least 5 no-fetch rollouts; the listed or "
            "shown column is the mean over those agents)", ""]
    out += table(["variant", "gap agents", "no-fetch counted acting", "listed or shown acting", "no-fetch gap",
                  "agent-cluster 95% CI", "crossed 95% CI"], rows)
    rows = [[p["pair"], p["measure"], str(p["n"]), g3(p["first"]), g3(p["second"]), g3(p["diff"]), cx(p["ci"]), cx(p["x"])]
            for p in pairs]
    out += ["Paired differences (first minus second, per agent over all rollouts)", ""]
    out += table(["pair", "measure", "agents", "first", "second", "difference", "agent-cluster 95% CI", "crossed 95% CI"], rows)
    return out


# ---------------------------------------------------------------- per-template and S3 floor (absence)

def per_template(Xs, agents):
    rows, gaps = [], {}
    role = {t.tid: t.role for t in ABSENCE}
    tot = {v: X.sum(0) for v, X in Xs.items()}  # (6, templates), pooled over the agents
    for i, tid in enumerate(TIDS["absence"]):
        row = [tid, role[tid]]
        for v in Xs:
            c, l = tot[v][1, i] / tot[v][0, i], tot[v][5, i] / tot[v][4, i]
            gaps.setdefault(v, []).append((float(c - l), tid))
            row += [g3(c), g3(l), g3(c - l)]
        rows.append(row)
    head = ["template", "role"] + [f"{lab} {m}" for lab in ("stated", "unstated") for m in ("counted", "shown", "ITT gap")]
    out = [f"Absence, W1 acting per template, pooled over the {len(agents)} clear-acting agents (counted and shown over all "
           "rollouts; ITT gap = counted minus shown)", ""] + table(head, rows)
    for v, g in gaps.items():
        lo, hi = min(g), max(g)
        out.append(f"- {NAME[v]}: per-template ITT gap ranges from {g3(lo[0])} ({lo[1]}) to {g3(hi[0])} ({hi[1]}); "
                   f"{sum(x > 0 for x, _ in g)} of {len(g)} templates above 0.")
    return out + [""]


def s3_floor(recs, agents):
    """Absence stated, clear-acting agents with at least 5 rollouts in each of fetched and no-fetch counted W1: how many do
    not act less after fetching, and how many of those have no-fetch acting at or below 0.05."""
    have = {}
    for s in agents:
        m = crossover_metrics(recs[("absence", "stated")][s])
        if drop_of(m) is not None:
            have[s] = (rate(m["nf"]), rate(m["fe"]))
    flat = [s for s, (nf, fe) in have.items() if nf <= fe]
    floor = [s for s in flat if have[s][0] <= 0.05]
    return have, flat, floor


# ---------------------------------------------------------------- frontier

def frontier_section(work):
    work = Path(work)
    cs = lambda rows, ct, k, q="irreversible": [r for r in rows if r["contract"] == ct and r["k"] == k and r["consequence"] == q]
    acts = lambda rs: [r["s"]["acted"] for r in rs]
    cnt, st_rows, pr_rows = [], [], []
    for m in FRONT:
        ps, po = work / "stated-nairr" / f"{m}.jsonl", work / "onrecord-nairr" / f"{m}.jsonl"
        raw = lambda p: sum(1 for ln in open(p) if ln.strip()) if p.exists() else 0
        st = [r for r in load([ps])["none"] if r.get("stated") is True]
        pr = [r for r in load([po])["none"] if r["contract"] == "pagerec" and not r.get("stated")]
        cnt.append([m, f"{raw(ps)} ({len(st)})", f"{raw(po)} ({len(pr)})"])
        l1, l2, p1, p2 = cs(st, "all", 1), cs(st, "all", 2), cs(st, "page", 1), cs(st, "page", 2)
        itt = rate(acts(p2)) - rate(acts(l2)) if p2 and l2 else None
        st_rows.append([m, pooled(acts(l1)), pooled(acts(l2)), pooled(acts(p1)), pooled(acts(p2)),
                        pooled([r["s"]["seen"] >= 2 for r in p2]), pooled([r["s"]["acted"] for r in p2 if r["s"]["seen"] < 2]),
                        g3(itt)])
        q1, q2 = cs(pr, "pagerec", 1), cs(pr, "pagerec", 2)
        pr_rows.append([m, pooled(acts(q1)), pooled(acts(q2)), pooled([r["s"]["seen"] >= 2 for r in q2]),
                        pooled([r["s"]["acted"] for r in q2 if r["s"]["seen"] < 2]),
                        pooled([classify(r["final"]) == "two" for r in cs(pr, "pagerec", 2, "count")])])
    out = ["## Frontier Models (NAIRR)", "",
           "Rollouts of the three frontier models, reported apart from the 31 Bedrock agents; files may be partial, so every "
           "rate carries acted/n. Fetch share = K = 2 rollouts that saw the second record before acting (s[\"seen\"] >= 2); "
           "no-fetch acting = acted/n among those that did not. The ITT gap here is page K = 2 minus listed K = 2 acting, "
           "both from the stated file.", "", "Rows in the file (rows scored: no error, no duplicate, stated or pagerec only)", ""]
    out += table(["model", "stated-nairr", "onrecord-nairr"], cnt)
    out += ["Stated (\"There is only one ...\"), irreversible", ""]
    out += table(["model", "listed K=1", "listed K=2", "page K=1", "page K=2", "page K=2 fetch share",
                  "page K=2 no-fetch acting", "ITT gap (page K=2 minus listed K=2)"], st_rows)
    out += ["Count on the record (pagerec)", ""]
    out += table(["model", "pagerec K=1", "pagerec K=2", "pagerec K=2 fetch share", "pagerec K=2 no-fetch acting",
                  "count probe states two (K=2, mechanism.classify)"], pr_rows)
    return out


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--out", nargs="?", const=OUT, default=None, help=f"write the markdown to PATH (default {OUT})")
    ap.add_argument("--frontier", action="store_true", help="add the NAIRR frontier models")
    a = ap.parse_args()
    stems = list(LABEL)
    for st, tids in TIDS.items():
        if len(tids) != 16:
            note(f"{st}: {len(tids)} templates, expected 16")
    recs, base = load_all(a.work, stems)
    clear = clear_sets(base)
    both = clear["referent"] & clear["absence"]
    sets = [("A) Clear-Acting Agents of Each Structure", {"referent": clear["referent"], "absence": clear["absence"]}),
            ("B) Common Cohort: Clear-Acting in Both Structures", {"referent": both, "absence": both}),
            ("C) All 31 Bedrock Agents", {st: set(stems) for st in STRUCT})]
    # the array statistics must equal the crossover.py list statistics, agent by agent
    for st in STRUCT:
        for v in VARIANTS[st]:
            for s, d in recs[v].items():
                av = agent_values(counts(d, TIDS[st]).sum(-1)[None, :])
                m = crossover_metrics(d)
                want, got = gap_of(m), av["nfgap"][0]
                assert (want is None and np.isnan(got)) or abs(want - got) < 1e-12, (v, s, want, got)
                assert abs(m["counted"] - av["counted"][0]) < 1e-12 and abs(m["shown"] - av["listed"][0]) < 1e-12, (v, s)
    results = {}
    for title, sel in sets:
        for st in STRUCT:
            agents = sorted(s for s in sel[st] if all(s in recs[v] for v in VARIANTS[st]))
            if len(agents) < len(sel[st]):
                note(f"{title} {st}: {len(sel[st]) - len(agents)} agents lack a variant")
            Xs = {v: np.stack([counts(recs[v][s], TIDS[st]) for s in agents]) for v in VARIANTS[st]}
            results[(title, st)] = (agents, Xs) + analyse(agents, Xs)
    ab_agents, ab_X = results[(sets[0][0], "absence")][:2]
    have, flat, floor = s3_floor(recs, ab_agents)
    # self-check against the published crossover values and agent-cluster intervals
    ra, rr = results[(sets[0][0], "absence")][2], results[(sets[0][0], "referent")][2]
    ref_v, abs_v = VARIANTS["referent"], VARIANTS["absence"]
    checks = [("clear-acting agents, referent", 18, len(clear["referent"])), ("clear-acting agents, absence", 27, len(clear["absence"])),
              ("common cohort", 17, len(both)),
              ("no-fetch gap, referent page", 0.547, rr[ref_v[0]]["gap"]), ("no-fetch gap, referent stated", 0.564, rr[ref_v[1]]["gap"]),
              ("no-fetch gap, referent pagerec", 0.524, rr[ref_v[2]]["gap"]),
              ("no-fetch gap, absence stated", 0.058, ra[abs_v[0]]["gap"]), ("no-fetch gap, absence unstated", 0.040, ra[abs_v[1]]["gap"]),
              ("counted acting, referent page (18)", 0.654, rr[ref_v[0]]["counted"]),
              ("counted acting, absence stated (27)", 0.278, ra[abs_v[0]]["counted"]),
              ("listed acting, referent page (18)", 0.240, rr[ref_v[0]]["listed"]),
              ("shown acting, absence stated (27)", 0.295, ra[abs_v[0]]["listed"]),
              ("S3 floor: agents not acting less after fetching", 10, len(flat)),
              ("S3 floor: of those, no-fetch acting at most 0.05", 6, len(floor))]
    for v, lo, hi in ((ref_v[0], 0.436, 0.653), (ref_v[1], 0.438, 0.692), (ref_v[2], 0.408, 0.640)):
        checks += [(f"no-fetch gap CI low, {NAME[v]}", lo, rr[v]["gap_ci"][0]), (f"no-fetch gap CI high, {NAME[v]}", hi, rr[v]["gap_ci"][1])]
    for v, lo, hi in ((abs_v[0], -0.012, 0.129), (abs_v[1], 0.004, 0.074)):
        checks += [(f"no-fetch gap CI low, {NAME[v]}", lo, ra[v]["gap_ci"][0]), (f"no-fetch gap CI high, {NAME[v]}", hi, ra[v]["gap_ci"][1])]
    ok = [abs(want - got) <= 0.0005 + 1e-12 for _, want, got in checks]
    out = ["# ITT Contrast: Counted Minus Listed or Shown Acting (descriptive)", "",
           "The contract is randomized; the no-fetch gap conditions on fetching, which the agent chooses. The ITT gap is counted "
           "acting over all rollouts minus listed (referent) or shown (absence) acting, so it needs no conditioning. Counted = "
           "K = 2 under page, pagerec (referent) or W1 counted (absence). Agents weigh equally in agent means; intervals are "
           f"95% percentile intervals of {BOOT_N} resamples (agent-cluster: score_absence.boot as in crossover.py; crossed: "
           "agents and the 16 templates resampled independently). Clear-acting sets are fixed from the archived conditions.", "",
           "## Self-Check", "", "Reproduced values, tolerance 0.0005 (the agent-cluster intervals are those of "
           "crossover.py and onrecord.py)", ""]
    out += table(["check", "expected", "computed", "pass"],
                 [[n, f"{w:g}" if isinstance(w, int) else f"{w:.3f}", f"{g:g}" if isinstance(g, int) else f"{g:.4f}",
                   "PASS" if o else "FAIL"] for (n, w, g), o in zip(checks, ok)])
    out += [f"Self-check: {'all ' + str(len(ok)) + ' checks pass' if all(ok) else 'FAILED, ' + str(ok.count(False)) + ' of ' + str(len(ok))}.", ""]
    out += [f"Common cohort ({len(both)}): " + ", ".join(LABEL[s] for s in sorted(both, key=lambda s: LABEL[s])) + ".", ""]
    allres = [r for key in results for r in results[key][2].values()]
    same = lambda x, y: np.allclose(x, y, rtol=0, atol=1e-9)  # float noise between a pooled ratio and an agent mean
    if all(same(r["pooled"], r["itt"]) and same(r["pooled_ci"], r["itt_ci"]) and same(r["pooled_x"], r["itt_x"]) for r in allres):
        out += ["Every agent has the same number of counted and of listed or shown rollouts in every set, so the pooled ITT "
                "gap equals the agent-mean ITT gap and the two share their intervals.", ""]
    out += ["Pagerec and page share the archived listed baseline, so their ITT difference equals their difference in counted "
            "acting.", ""]
    for title, sel in sets:
        out += [f"## {title}", ""]
        res, pairs = {}, []
        for st in STRUCT:
            agents, Xs, r, p = results[(title, st)]
            res.update(r)
            pairs += p
        n_ref, n_abs = len(results[(title, "referent")][0]), len(results[(title, "absence")][0])
        out += [f"Agents: referent {n_ref}, absence {n_abs}.", ""]
        out += set_tables(res, pairs)
    out += ["## Absence: Per-Template ITT Gap and the S3 Floor", ""]
    out += per_template(ab_X, ab_agents)
    out += [f"S3 floor (absence stated, {len(ab_agents)} clear-acting agents, at least 5 fetched and 5 no-fetch counted W1 "
            f"rollouts): {len(have)} agents qualify; {len(flat)} do not act less after fetching (no-fetch acting at or below "
            f"fetched acting); of those, {len(floor)} have no-fetch acting at or below 0.05.", ""]
    out += table(["agent", "no-fetch acting", "fetched acting", "acts less after fetching", "no-fetch at most 0.05"],
                 [[LABEL[s], g3(nf), g3(fe), "no" if s in flat else "yes", "yes" if s in floor else "no"]
                  for s, (nf, fe) in sorted(have.items(), key=lambda kv: (kv[1][0] > kv[1][1], LABEL[kv[0]]))])
    if a.frontier:
        out += frontier_section(a.work)
    text = "\n".join(out)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n")
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
