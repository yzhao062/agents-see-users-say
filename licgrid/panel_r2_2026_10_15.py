"""Existing-log robustness and follow-up analyses, descriptive only.

  python panel_r2_2026_10_15.py [--work $WTA_WORK] [--out DIR] [--no-write]

Inputs (read only; no model or network call). Rows come from the loaders the neighbouring scripts use: all_agents.manifest and
all_agents.load (the 36 archived agents; the 31 Bedrock agents' <work>/stated, onrecord and statedrec files), itt.load_all (the
referent and absence rollouts of itt.py), score_absence.read_raw and load (<work>/absence = claim stated, <work>/unstated).
Clear-acting sets are crossover.clear_sets on the archived conditions (referent 18, absence 27 at 0.80). "Fetched" is
seeing.split (s["seen"] >= 2 on K = 2 page, the response-boundary score of score_grid.step_turns). Agent-cluster intervals are
score_absence.boot (2,000 resamples, seed 0). Crossed intervals are those of itt.py: agents and the 16 templates drawn
independently with replacement (numpy, seed 0, 2,000 resamples), one template draw per resample shared by every measure, the
agent-mean statistic recomputed on the resampled agents x templates. Intervals are 95% percentile intervals.

Analyses
  (a) Fetch effect within (agent, template) cells. 36 archived agents, K = 2 page irreversible rollouts (5 per cell). For each
      cell with both fetched and not-fetched rollouts: acting(fetched) minus acting(not fetched). Mean over cells weighted by
      cell size (rollouts in the cell), unweighted, and weighted by n_f n_nf / (n_f + n_nf) (precision, a third view because
      every cell has 5 rollouts, which makes the first two equal). Agent-cluster bootstrap. Also restricted to agents with at
      least 5 rollouts in each group (all their cells with both groups).
  (b) The user's claim in both worlds of stated absence. 27 clear-acting agents, claim stated (archived) and unstated. W0 and W1
      acting by tier (counted, shown); discrimination = W0 acting minus W1 acting under the same contract; paired change =
      stated minus unstated. "W0 counted" is the "clear acting" column of crossover.py (0.946 and 0.835).
  (c) Stated absence per agent. 27 clear-acting agents, W1 counted vs W1 shown acting over all rollouts (claim stated). Agents
      with shown minus counted >= 0.10 and counted minus shown >= 0.10. Reference: how many agents would pass each cut if the
      contract label carried no information (permutation of the label within template, 2,000 shuffles per agent).
  (d) For the record: family-clustered bootstrap (families resampled with replacement) and leave-one-family-out ranges for the
      referent page ITT gap (18 agents), the stated-absence ITT gap (27), and the referent "only one" counted change (18; 31),
      and the same at clear-acting thresholds 0.70 and 0.90. ITT gap = counted minus listed (referent) or shown (absence) acting
      over all rollouts, agent mean, as itt.py.
  (e) The 2 x 2 of experiment A. 18 clear-acting referent agents, K = 2 counted acting over all rollouts in page, stated page,
      pagerec and stated pagerec (<work>/statedrec, contract pagerec, stated true), listed baselines (archived; stated),
      gaps, paired differences with crossed intervals, and K = 1 acting in the new cell. The primary set is the agents whose
      new K = 2 cell is complete (80 rollouts, 5 per template); a provisional line covers every agent with any rows.

Outputs, in --out (default <work>/logs): the markdown report (also printed), a JSON file with every number, and
per-cell / per-agent CSV files (fetch-cells, absence-agents, family, statedrec). Self-check: reference values from
itt.py, crossover.py, and seeing.py are recomputed (tolerance 0.0005); the exit status is 1 when one fails (the
outputs are still written, with FAIL marked).
"""
import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import crossover
import seeing
from all_agents import LABEL, load, manifest
from crossover import CELL, amean, ci_of, clear_sets, note, table
from itt import BOOT_N, SEED, TIDS, WORK, cx, counted_ref, g3, load_all, nmean, pct
from score_absence import boot, cell, load as load_absence, rate, read_raw

DATE = "2026-10-15"
THRESHOLDS = (0.70, 0.80, 0.90)
CUT = 0.10  # per-agent difference of (c)
REPS = CELL // 16  # rollouts per template in a full cell
# (b): tag, subdirectory of <work>, value of the row field "stated"
CLAIMS = (("s", "absence", True), ("u", "unstated", False))
ABS_CELLS = {"w0c": ("counted", 0), "w0s": ("shown", 0), "w1c": ("counted", 1), "w1s": ("shown", 1)}
# Model families of the 31 Bedrock agents, by substring of the all_agents.LABEL key (first match wins). Mistral, Ministral and
# Devstral are one family; gpt-oss and Gemma are in the rules but have no Bedrock stem among the 31. An agent that matches no
# rule is its own family (singleton).
FAMILY = [("nova", "Nova"), ("mistral", "Mistral"), ("minimax", "MiniMax"), ("qwen", "Qwen"), ("llama", "Llama"),
          ("glm", "GLM"), ("kimi", "Kimi"), ("nemotron", "Nemotron"), ("deepseek", "DeepSeek"), ("gpt-oss", "gpt-oss"),
          ("gemma", "Gemma")]


def family(stem):
    return next((name for key, name in FAMILY if key in stem), LABEL[stem])


# ---------------------------------------------------------------- counts of acted/n per agent, cell and template

class Counts:
    """X[agent, 2 i (n) or 2 i + 1 (acted), template] for the named cells i; the crossed bootstrap resamples the last axis."""

    def __init__(self, agents, tids, names):
        self.agents, self.tids, self.names = list(agents), list(tids), list(names)
        self.row = {a: i for i, a in enumerate(self.agents)}
        self.col = {t: i for i, t in enumerate(self.tids)}
        self.X = np.zeros((len(self.agents), 2 * len(self.names), len(self.tids)))

    def add(self, agent, name, template, acted):
        j = 2 * self.names.index(name)
        self.X[self.row[agent], j, self.col[template]] += 1
        self.X[self.row[agent], j + 1, self.col[template]] += bool(acted)

    def n(self, agent, name):
        return int(self.X[self.row[agent], 2 * self.names.index(name)].sum())

    def subset(self, agents):
        new = Counts(agents, self.tids, self.names)
        for a in new.agents:
            new.X[new.row[a]] = self.X[self.row[a]]
        return new

    def rate(self, S, *names):
        """Acting rate pooled over the named cells, from sums S[..., agents, 2 cells]; NaN where a cell is empty."""
        js = [2 * self.names.index(n) for n in names]
        with np.errstate(invalid="ignore", divide="ignore"):
            return sum(S[..., j + 1] for j in js) / sum(S[..., j] for j in js)


def draws(A, T, n=BOOT_N, seed=SEED):
    """The resampling of itt.crossed: an agent draw, then one template draw per resample (the same order of numpy calls)."""
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, A, size=(n, A))
    W = np.stack([np.bincount(rng.integers(0, T, size=T), minlength=T) for _ in range(n)])
    return pick, W


def resample(X, dr):
    pick, W = dr
    return np.einsum("amt,rt->ram", X, W)[np.arange(len(pick))[:, None], pick]


def evaluate(cnt, measures):
    """measures: key -> fn(S) giving per-agent values from sums S[..., agents, cells]. Returns key -> n agents, agent mean,
    agent-cluster CI, crossed CI, by-agent values. One template draw is shared by all measures."""
    S0, Sr = cnt.X.sum(-1), resample(cnt.X, draws(len(cnt.agents), len(cnt.tids)))
    out = {}
    for k, fn in measures.items():
        v = fn(S0)
        by = {a: float(x) for a, x in zip(cnt.agents, v) if not np.isnan(x)}
        out[k] = {"n": len(by), "mean": amean(by.values()), "ci": ci_of(by), "x": pct(nmean(fn(Sr))), "by": by}
    return out


def point(cnt, fn, agents):
    v = fn(cnt.X.sum(-1))
    return {a: float(v[cnt.row[a]]) for a in agents if not np.isnan(v[cnt.row[a]])}


# ---------------------------------------------------------------- self-check

CHECKS = []


def check(name, want, got, tol=0.0005):
    CHECKS.append((name, want, got, tol))


def check_ci(name, want, got):
    check(name + " low", want[0], got[0] if got else None)
    check(name + " high", want[1], got[1] if got else None)


def checks_ok():
    return [isinstance(w, int) and w == g or (not isinstance(w, int) and g is not None and abs(w - g) <= t + 1e-12)
            for _, w, g, t in CHECKS]


# ---------------------------------------------------------------- (a) fetch effect within cells

def fetch_cells(work):
    """cells[agent] = [(template, n_fetched, acted_fetched, n_not, acted_not)] and tot[agent] = the sums, from the K = 2 page
    irreversible rollouts of the 36 archived agents, split by seeing.split (fetched = the second record seen before acting)."""
    cells, tot = {}, {}
    for lab, fs in manifest(work).items():
        rows = [r for r in load(fs)["none"] if not r.get("stated")]
        saw, one = seeing.split(rows)
        tot[lab] = (len(saw), sum(saw), len(one), sum(one))
        cells[lab] = []
        for tid in TIDS["referent"]:
            saw, one = seeing.split([r for r in rows if r["template"] == tid])
            cells[lab].append((tid, len(saw), sum(saw), len(one), sum(one)))
        assert tuple(sum(c[i] for c in cells[lab]) for i in (1, 2, 3, 4)) == tot[lab], lab
    return cells, tot


WEIGHT = {"size": lambda nf, nn: nf + nn, "unweighted": lambda nf, nn: 1.0, "precision": lambda nf, nn: nf * nn / (nf + nn)}


def within(weight):
    """Statistic over a list of per-agent lists of cells: the weighted mean of acting(fetched) - acting(not fetched)."""
    def stat(lists):
        cs = [c for l in lists for c in l]
        if not cs:
            return None
        ws = [WEIGHT[weight](c[1], c[3]) for c in cs]
        return sum(w * (c[2] / c[1] - c[4] / c[3]) for w, c in zip(ws, cs)) / sum(ws)
    return stat


def fetch_effect(cells, tot, agents):
    both = {a: [c for c in cells[a] if c[1] and c[3]] for a in agents}
    flat = [c for l in both.values() for c in l]
    res = {"agents": len(agents), "agents_with_cells": sum(bool(l) for l in both.values()), "cells_all": 16 * len(agents),
           "cells": len(flat),
           "pooled_fetched": (sum(tot[a][1] for a in agents), sum(tot[a][0] for a in agents)),
           "pooled_not": (sum(tot[a][3] for a in agents), sum(tot[a][2] for a in agents))}
    for w in WEIGHT:
        res[w] = (within(w)(list(both.values())), boot(both, within(w)))
    diffs = [c[2] / c[1] - c[4] / c[3] for c in flat]
    res["lower"], res["equal"], res["higher"] = (sum(d < 0 for d in diffs), sum(d == 0 for d in diffs), sum(d > 0 for d in diffs))
    per = {a: within("size")([l]) for a, l in both.items() if l}
    res["agents_negative"] = sum(v < 0 for v in per.values())
    res["per_agent"] = per
    return res


def fetch_section(cells, tot):
    sets = [("all 36 archived agents", sorted(cells)),
            ("agents with at least 5 rollouts in each group", sorted(a for a in cells if tot[a][0] >= 5 and tot[a][2] >= 5))]
    res = {title: fetch_effect(cells, tot, ag) for title, ag in sets}
    f = lambda p: f"{p[0]}/{p[1]} = {p[0] / p[1]:.3f}"
    out = ["## (a) Fetch Effect Within (Agent, Template) Cells", "",
           "K = 2 page irreversible rollouts, 5 per (agent, template) cell. Fetched = the second record seen before acting "
           "(seeing.split). Difference = acting(fetched) minus acting(not fetched) in a cell that has both. Agent-cluster 95% CI "
           "(2,000 resamples of agents; an agent without such a cell contributes no cell).", ""]
    rows = []
    for title, r in res.items():
        rows.append([title, str(r["agents"]), str(r["agents_with_cells"]), f"{r['cells']} of {r['cells_all']}",
                     f(r["pooled_fetched"]), f(r["pooled_not"])])
    out += table(["agent set", "agents", "agents with a both-group cell", "cells with both groups",
                  "pooled acting, fetched (acted/n)", "pooled acting, not fetched"], rows)
    rows = []
    for title, r in res.items():
        rows.append([title] + [f"{g3(r[w][0])} {cx(r[w][1])}" for w in WEIGHT]
                    + [f"{r['lower']} / {r['equal']} / {r['higher']}", f"{r['agents_negative']} of {len(r['per_agent'])}"])
    out += table(["agent set", "mean, weighted by cell size", "mean, unweighted", "mean, precision-weighted",
                  "cells fetched lower / equal / higher", "agents with a negative mean"], rows)
    out.append("Every cell has 5 rollouts, so the size-weighted and unweighted means are equal; the precision weight gives "
               "cells with a 2-3 split more weight than cells with a 1-4 split.")
    out.append("")
    return out, res


# ---------------------------------------------------------------- (b) the claim in both worlds of stated absence

def absence_counts(work, stems):
    """Counts of W0 and W1 acting per template, claim stated and unstated, for each stem that has both files."""
    keep = [s for s in stems if all((Path(work) / sub / (s.replace(":", "_") + ".jsonl")).exists() for _, sub, _ in CLAIMS)]
    if len(keep) < len(stems):
        note(f"{len(stems) - len(keep)} agents lack an absence or unstated file")
    cnt = Counts(keep, TIDS["absence"], [f"{c}_{k}" for c, _, _ in CLAIMS for k in ABS_CELLS])
    for s in keep:
        for tag, sub, claim in CLAIMS:
            p = Path(work) / sub / (s.replace(":", "_") + ".jsonl")
            kept = [r for r in load_absence(read_raw([p])) if r["kind"] == "act" and r.get("stated", True) == claim]
            for k, (tier, w) in ABS_CELLS.items():
                rs = cell(kept, tier, w)
                if len(rs) < CELL:
                    note(f"absence {tag} {s}: {tier} W{w} cell has {len(rs)} of {CELL} rollouts")
                for r in rs:
                    cnt.add(s, f"{tag}_{k}", r["template"], r["s"]["acted"])
    return cnt


def absence_measures(cnt):
    """(name, claim) -> fn(S); claim in s (stated), u (unstated), d (stated minus unstated)."""
    R = cnt.rate
    spec = {"W0 counted acting": lambda S, c: R(S, f"{c}_w0c"),
            "W0 shown acting": lambda S, c: R(S, f"{c}_w0s"),
            "W0 acting, counted and shown pooled": lambda S, c: R(S, f"{c}_w0c", f"{c}_w0s"),
            "W1 counted acting": lambda S, c: R(S, f"{c}_w1c"),
            "W1 shown acting": lambda S, c: R(S, f"{c}_w1s"),
            "discrimination, counted contract (W0 counted minus W1 counted)": lambda S, c: R(S, f"{c}_w0c") - R(S, f"{c}_w1c"),
            "discrimination, shown contract (W0 shown minus W1 shown)": lambda S, c: R(S, f"{c}_w0s") - R(S, f"{c}_w1s"),
            "W1 counted minus W1 shown (ITT gap)": lambda S, c: R(S, f"{c}_w1c") - R(S, f"{c}_w1s")}
    ms = {}
    for name, f in spec.items():
        ms[(name, "s")] = lambda S, f=f: f(S, "s")
        ms[(name, "u")] = lambda S, f=f: f(S, "u")
        ms[(name, "d")] = lambda S, f=f: f(S, "s") - f(S, "u")
    return list(spec), ms


def claim_section(abs_cnt, agents):
    cnt = abs_cnt.subset(agents)
    names, ms = absence_measures(cnt)
    res = evaluate(cnt, ms)
    out = ["## (b) The User's Claim in Both Worlds of Stated Absence", "",
           f"{len(agents)} clear-acting agents (W0 acting >= 0.80 on the archived stated data). Stated = \"It is not attached ...\" "
           "(archived); unstated = the same without the claim. W0 = no dependent exists, W1 = one exists. Agent means over "
           "all rollouts; crossed 95% CI (agents x 16 templates); the change is stated minus unstated per agent.", ""]
    rows = []
    for nm in names:
        s, u, d = res[(nm, "s")], res[(nm, "u")], res[(nm, "d")]
        rows.append([nm, f"{g3(s['mean'])} {cx(s['x'])}", f"{g3(u['mean'])} {cx(u['x'])}", g3(d["mean"]), cx(d["ci"]), cx(d["x"])])
    out += table(["measure", "claim stated: mean [crossed CI]", "claim unstated: mean [crossed CI]", "change (stated minus unstated)",
                  "agent-cluster CI of the change", "crossed CI of the change"], rows)
    out += ["Agent-cluster 95% CIs of the levels", ""]
    out += table(["measure", "claim stated", "claim unstated"], [[nm, cx(res[(nm, "s")]["ci"]), cx(res[(nm, "u")]["ci"])] for nm in names])
    return out, res, names


# ---------------------------------------------------------------- (c) stated absence per agent

def absence_agent_rows(recs, agents):
    """One row per agent that has W1 rollouts under both contracts; an agent with an empty W1 cell is left out with a note."""
    rows = []
    for s in agents:
        d = recs[("absence", "stated")][s]
        ca, cn, sa, sn = sum(a for _, a, _ in d["counted"]), len(d["counted"]), sum(a for _, a in d["shown"]), len(d["shown"])
        if not (cn and sn):
            note(f"absence stated {s}: W1 counted {cn}, shown {sn} rollouts; left out of (c)")
            continue
        rows.append({"stem": s, "agent": LABEL[s], "counted_acted": ca, "counted_n": cn, "shown_acted": sa, "shown_n": sn,
                     "counted": ca / cn, "shown": sa / sn, "shown_minus_counted": sa / sn - ca / cn})
    return rows


def permutation_reference(recs, agents, n=BOOT_N, seed=SEED):
    """If the contract label (counted or shown) carried no information, how many agents would show shown minus counted at
    least +0.10 and at most -0.10? Labels are permuted within template, separately for each agent (n shuffles per agent)."""
    rng = np.random.default_rng(seed)
    hi, lo = np.zeros(n), np.zeros(n)
    for s in agents:
        d = recs[("absence", "stated")][s]
        cn, sn = len(d["counted"]), len(d["shown"])
        c_acts, total = np.zeros(n), 0.0
        for tid in TIDS["absence"]:
            c = [a for t, a, _ in d["counted"] if t == tid]
            pool = np.array(c + [a for t, a in d["shown"] if t == tid], float)
            if len(pool):
                c_acts += rng.permuted(np.tile(pool, (n, 1)), axis=1)[:, :len(c)].sum(1)
                total += pool.sum()
        diff = (total - c_acts) / sn - c_acts / cn
        hi += diff >= CUT - 1e-9
        lo += diff <= -CUT + 1e-9
    return hi, lo


def agents_section(recs, agents):
    rows = sorted(absence_agent_rows(recs, agents), key=lambda r: -r["shown_minus_counted"])
    below = [r for r in rows if r["shown_minus_counted"] >= CUT - 1e-9]
    above = [r for r in rows if -r["shown_minus_counted"] >= CUT - 1e-9]
    used = {r["stem"] for r in rows}
    hi, lo = permutation_reference(recs, [s for s in agents if s in used])  # in the order of agents: the shuffles draw from one stream
    q = lambda v: f"{v.mean():.1f} [{np.percentile(v, 2.5):.0f}, {np.percentile(v, 97.5):.0f}]"
    res = {"agents": len(rows), "clear_agents": len(agents), "counted_below_shown": [r["agent"] for r in below], "counted_above_shown": [r["agent"] for r in above],
           "n_below": len(below), "n_above": len(above), "null_below": float(hi.mean()), "null_above": float(lo.mean()),
           "shuffles_below": int((hi >= len(below)).sum()), "shuffles_above": int((lo >= len(above)).sum())}
    out = ["## (c) Stated Absence per Agent: Counted vs Shown Acting", "",
           f"{len(rows)} clear-acting agents"
           + ("" if len(rows) == len(agents) else f" (of {len(agents)}; the rest have an empty W1 cell)")
           + ", W1 with the claim stated, acting over all rollouts ("
           + (f"{CELL} per contract" if all(r["counted_n"] == r["shown_n"] == CELL for r in rows) else "acted/n in the table")
           + f"). Difference = shown acting minus counted acting; cut {CUT:.2f}.", ""]
    out.append(f"- Shown minus counted >= {CUT:.2f} (the count stops more than the listing): {len(below)} agents"
               + (": " + ", ".join(f"{r['agent']} ({r['shown_minus_counted']:+.3f})" for r in below) if below else "") + ".")
    out.append(f"- Counted minus shown >= {CUT:.2f} (the listing stops more than the count): {len(above)} agents"
               + (": " + ", ".join(f"{r['agent']} ({-r['shown_minus_counted']:+.3f})" for r in above) if above else "") + ".")
    out.append(f"- Within {CUT:.2f} either way: {len(rows) - len(below) - len(above)} agents.")
    out.append(f"- Reference, contract label carrying no information (label permuted within template, {BOOT_N:,} shuffles per agent): "
               f"{q(hi)} agents at shown minus counted >= {CUT:.2f} ({res['shuffles_below']} of {BOOT_N:,} shuffles reach the observed {len(below)}) and "
               f"{q(lo)} at counted minus shown >= {CUT:.2f} ({res['shuffles_above']} of {BOOT_N:,} reach the observed {len(above)}); mean [2.5th, 97.5th percentile].")
    out.append("")
    out += table(["agent", "counted acting", "shown acting", "shown minus counted"],
                 [[r["agent"], f"{r['counted_acted']}/{r['counted_n']} = {r['counted']:.3f}",
                   f"{r['shown_acted']}/{r['shown_n']} = {r['shown']:.3f}", f"{r['shown_minus_counted']:+.3f}"] for r in rows])
    return out, res, rows


# ---------------------------------------------------------------- (d) family-clustered intervals, leave-one-family-out

def fam_boot(vals, fam):
    members = defaultdict(list)
    for a, v in vals.items():
        members[fam[a]].append(v)
    return boot(dict(members), lambda lists: amean([x for l in lists for x in l]))


def lofo(vals, fam):
    return {f: amean([v for a, v in vals.items() if fam[a] != f]) for f in sorted({fam[a] for a in vals})
            if any(fam[a] != f for a in vals)}


def clear_at(base, thr):
    """crossover.clear_sets with the clear-acting threshold set to thr (CLEAR is read when the function is called)."""
    old = crossover.CLEAR
    crossover.CLEAR = thr
    try:
        return clear_sets(base)
    finally:
        crossover.CLEAR = old


def family_rows(ref_cnt, abs_cnt, clear, everyone):
    R, A = ref_cnt.rate, abs_cnt.rate
    stats = [("referent page ITT gap (counted page minus listed)", ref_cnt, lambda S: R(S, "page") - R(S, "listed"), "referent"),
             ("absence stated ITT gap (W1 counted minus W1 shown)", abs_cnt, lambda S: A(S, "s_w1c") - A(S, "s_w1s"), "absence"),
             ("referent 'only one' counted change (stated page minus page)", ref_cnt, lambda S: R(S, "spage") - R(S, "page"), "referent"),
             ("referent 'only one' counted change, all agents", ref_cnt, lambda S: R(S, "spage") - R(S, "page"), "all")]
    rows = []
    for name, cnt, fn, which in stats:
        agents = sorted(everyone if which == "all" else clear[which])
        vals = point(cnt, fn, [a for a in agents if a in cnt.row])
        fam = {a: family(a) for a in vals}
        lo = lofo(vals, fam)
        lo_min, lo_max = min(lo, key=lo.get), max(lo, key=lo.get)
        rows.append({"statistic": name, "agents": len(vals), "families": len(set(fam.values())), "estimate": amean(vals.values()),
                     "agent_ci": ci_of(vals), "family_ci": fam_boot(vals, fam), "lofo_min": lo[lo_min], "lofo_min_family": lo_min,
                     "lofo_max": lo[lo_max], "lofo_max_family": lo_max})
    return rows


def family_section(ref_cnt, abs_cnt, base, everyone):
    out = ["## (d) Family-Clustered Intervals and Leave-One-Family-Out (for the record)", "",
           "Families (from all_agents.LABEL): " + "; ".join(
               f"{f} ({', '.join(LABEL[s] for s in everyone if family(s) == f)})" for f in sorted({family(s) for s in everyone})) + ".", "",
           "Estimate = agent mean. Agent-cluster CI = score_absence.boot over agents; family CI = the same bootstrap over families "
           "(a drawn family brings all its agents in the set); leave-one-family-out = the estimate with one family removed, "
           "min and max with the family removed.", ""]
    res = {}
    for thr in THRESHOLDS:
        clear = clear_at(base, thr)
        rows = family_rows(ref_cnt, abs_cnt, clear, everyone)
        out +=[f"### Clear-Acting Threshold {thr:.2f}" + (" (the paper's)" if thr == 0.80 else ""), "",
                f"Clear-acting agents: referent {len(clear['referent'])}, absence {len(clear['absence'])}.", ""]
        out += table(["statistic", "agents", "families", "estimate", "agent-cluster CI", "family-cluster CI",
                      "leave-one-family-out min (family removed)", "max (family removed)"],
                     [[r["statistic"], str(r["agents"]), str(r["families"]), g3(r["estimate"]), cx(r["agent_ci"]), cx(r["family_ci"]),
                       f"{g3(r['lofo_min'])} ({r['lofo_min_family']})", f"{g3(r['lofo_max'])} ({r['lofo_max_family']})"] for r in rows])
        res[thr] = {"clear_referent": len(clear["referent"]), "clear_absence": len(clear["absence"]), "rows": rows}
    return out, res


# ---------------------------------------------------------------- (e) the 2 x 2 of experiment A

def load_statedrec(work, stems, ref_cnt, k1_cnt):
    """Fill the sprec cell of ref_cnt and the K = 1 cell of k1_cnt from <work>/statedrec; return rows seen per agent."""
    seen = {}
    for s in stems:
        seen[s] = {"k2": 0, "k1": 0}
        p = Path(work) / "statedrec" / (s.replace(":", "_") + ".jsonl")
        if s not in ref_cnt.row:
            continue
        if not p.exists():
            note(f"missing {p}")
            continue
        rows = load([p])["none"]  # all_agents.load: no errors, no duplicates, scored by score_grid.score
        kept = [r for r in rows if r["contract"] == "pagerec" and r.get("stated") is True]
        if len(kept) < len(rows):
            note(f"{p}: {len(rows) - len(kept)} rows without contract pagerec and \"stated\": true left out")
        for t, a, _ in counted_ref(kept, "pagerec"):
            ref_cnt.add(s, "sprec", t, a)
            seen[s]["k2"] += 1
        for r in kept:
            if r["k"] == 1 and r["consequence"] == "irreversible":
                k1_cnt.add(s, "k1", r["template"], r["s"]["acted"])
                seen[s]["k1"] += 1
        for k, lab in (("k2", "K = 2"), ("k1", "K = 1")):
            if 0 < seen[s][k] < CELL:
                note(f"statedrec {s}: {lab} cell has {seen[s][k]} of {CELL} rollouts")
    return seen


def complete_cell(cnt, agent, name):
    j = 2 * cnt.names.index(name)
    return bool((cnt.X[cnt.row[agent], j] == REPS).all())


def cell_rate(cnt, agent, name):
    """One agent's acting rate in a cell over all its rollouts, rounded for a csv file; empty when the cell has none."""
    if agent not in cnt.row:
        return ""
    v = float(cnt.rate(cnt.X[cnt.row[agent]].sum(-1), name))
    return "" if np.isnan(v) else round(v, 6)


def tbl_rows(res, keys):
    return [[lab, str(res[k]["n"]), g3(res[k]["mean"]), cx(res[k]["ci"]), cx(res[k]["x"])] for lab, k in keys]


def two_by_two(ref_cnt, agents, tag):
    cnt = ref_cnt.subset(agents)
    R = cnt.rate
    cellrate = {"page": ("page",), "stated page": ("spage",), "pagerec": ("prec",), "stated pagerec": ("sprec",),
                "listed (archived)": ("listed",), "listed (stated)": ("slisted",)}
    ms = {lab: (lambda S, nm=nm: R(S, *nm)) for lab, nm in cellrate.items()}
    ms.update({
        "gap, page (counted minus archived listed)": lambda S: R(S, "page") - R(S, "listed"),
        "gap, stated page (counted minus stated listed)": lambda S: R(S, "spage") - R(S, "slisted"),
        "gap, pagerec (counted minus archived listed)": lambda S: R(S, "prec") - R(S, "listed"),
        "gap, stated pagerec (counted minus archived listed)": lambda S: R(S, "sprec") - R(S, "listed"),
        "gap, stated pagerec (counted minus stated listed)": lambda S: R(S, "sprec") - R(S, "slisted"),
        "stated pagerec minus pagerec": lambda S: R(S, "sprec") - R(S, "prec"),
        "stated pagerec minus stated page": lambda S: R(S, "sprec") - R(S, "spage"),
        "stated page minus page": lambda S: R(S, "spage") - R(S, "page"),
        "pagerec minus page": lambda S: R(S, "prec") - R(S, "page"),
        "interaction: (stated pagerec minus pagerec) minus (stated page minus page)":
            lambda S: (R(S, "sprec") - R(S, "prec")) - (R(S, "spage") - R(S, "page")),
        "gap, stated pagerec (stated listed) minus gap, pagerec (archived listed)":
            lambda S: (R(S, "sprec") - R(S, "slisted")) - (R(S, "prec") - R(S, "listed")),
        "gap, stated page minus gap, page": lambda S: (R(S, "spage") - R(S, "slisted")) - (R(S, "page") - R(S, "listed"))})
    res = evaluate(cnt, ms)
    out = [f"### {tag}: {len(agents)} agents", "",
           "K = 2 counted acting over all rollouts (agent means), crossed 95% CI (agents x 16 templates) and agent-cluster CI.", ""]
    out += table(["cell", "agents", "mean", "agent-cluster CI", "crossed CI"], tbl_rows(res, [(k, k) for k in cellrate]))
    out += ["Gaps: counted acting minus a listed baseline (the new cell against both)", ""]
    out += table(["gap", "agents", "mean", "agent-cluster CI", "crossed CI"], tbl_rows(res, [(k, k) for k in ms if k.startswith("gap,") and "minus gap" not in k]))
    out += ["Paired differences (per agent, over all rollouts)", ""]
    out += table(["difference", "agents", "mean", "agent-cluster CI", "crossed CI"],
                 tbl_rows(res, [(k, k) for k in list(ms)[len(cellrate) + 5:]]))
    return out, res


def statedrec_section(work, ref_cnt, clear18, stems, base):
    k1_cnt = Counts(ref_cnt.agents, ref_cnt.tids, ["k1"])
    seen = load_statedrec(work, stems, ref_cnt, k1_cnt)
    out = ["## (e) The 2 x 2 of Experiment A: Claim x Count Location", "",
           "Cells (K = 2 irreversible, counted acting over all rollouts): page (archived, count in the result), stated page "
           "(<work>/stated), pagerec (<work>/onrecord, count on the record), stated pagerec (<work>/statedrec, new). Listed "
           "baselines: archived listed (K = 2, contract all) and stated listed (<work>/stated). The clear-acting referent "
           "agents are fixed from the archive.", ""]
    ref = sorted(a for a in clear18 if a in ref_cnt.row)  # an agent without every archived variant is not in ref_cnt
    if len(ref) < len(clear18):
        note(f"{len(clear18) - len(ref)} clear-acting referent agents lack a variant of the referent data; left out of (e)")
    complete = [a for a in ref if complete_cell(ref_cnt, a, "sprec")]
    some = [a for a in ref if seen[a]["k2"] > 0]
    k1_done = [a for a in ref if complete_cell(k1_cnt, a, "k1")]
    out.append(f"Data in <work>/statedrec: {sum(seen[a]['k2'] > 0 for a in stems)} of {len(stems)} agents have K = 2 rows; "
               f"among the {len(ref)} clear-acting referent agents {len(some)} have K = 2 rows, {len(complete)} have a complete "
               f"K = 2 cell ({CELL} rollouts, {REPS} per template) and {len(k1_done)} a complete K = 1 cell.")
    out.append("")
    out += table(["agent", "K=2 rows", "K=1 rows", "K=2 complete", "K=1 complete"],
                 [[LABEL[a], f"{seen[a]['k2']}/{CELL}", f"{seen[a]['k1']}/{CELL}", "yes" if a in complete else "no",
                   "yes" if a in k1_done else "no"] for a in ref])
    res = {"clear_agents": len(ref), "with_rows": len(some), "complete_k2": len(complete), "complete_k1": len(k1_done), "seen": seen}
    if complete:
        o, r = two_by_two(ref_cnt, complete, "Primary (complete K = 2 cells)")
        out += o
        res["primary"] = r
    else:
        out += ["No agent has a complete K = 2 cell yet; the 2 x 2 is not computed.", ""]
    if some and len(some) > len(complete):
        o, r = two_by_two(ref_cnt, some, "Provisional (every agent with any K = 2 rows; partial cells, uneven template coverage)")
        out += o
        res["provisional"] = r
    if k1_done:
        cnt = k1_cnt.subset(k1_done)
        r = evaluate(cnt, {"K = 1 acting, stated pagerec": lambda S: cnt.rate(S, "k1")})
        r = r["K = 1 acting, stated pagerec"]
        # reference: the archived K = 1 listed acting (crossover.ref_data "clear") of the same agents
        arch1 = {a: rate(base[("referent", "unstated")][a]["clear"]) for a in k1_done}
        diff1 = {a: r["by"][a] - arch1[a] for a in k1_done}
        out += [f"K = 1 acting in the new cell (irreversible, {len(k1_done)} agents with a complete K = 1 cell): {g3(r['mean'])}, "
                f"agent-cluster CI {cx(r['ci'])}, crossed CI {cx(r['x'])}. Archived K = 1 listed acting of the same agents: "
                f"{g3(amean(arch1.values()))} {cx(ci_of(arch1))}; new cell minus archived: {g3(amean(diff1.values()))} {cx(ci_of(diff1))} "
                "(agent-cluster).", ""]
        res["k1"] = r
        res["k1_archived_listed"] = {"mean": amean(arch1.values()), "ci": ci_of(arch1), "diff": amean(diff1.values()), "diff_ci": ci_of(diff1)}
    elif any(seen[a]["k1"] for a in ref):
        out += ["No agent has a complete K = 1 cell yet.", ""]
    return out, res, seen, k1_cnt


# ---------------------------------------------------------------- csv and json

def write_csv(path, head, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(head)
        w.writerows(rows)


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items() if k != "by"}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


# ---------------------------------------------------------------- main

def main():
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=WORK)
    ap.add_argument("--out", default=None, help="output directory (default <work>/logs)")
    ap.add_argument("--no-write", action="store_true", help="print only; write no file")
    a = ap.parse_args()
    stems = sorted(LABEL)
    outdir = Path(a.out) if a.out else Path(a.work) / "logs"

    recs, base = load_all(a.work, stems)
    clear = clear_sets(base)
    everyone = sorted(s for s in stems if all(s in recs[v] for v in recs))
    if len(everyone) < len(stems):
        note(f"{len(stems) - len(everyone)} agents lack a variant of the referent or absence data")
    # referent counts: page, stated page, pagerec (counted K = 2 irreversible), the archived and the stated listed baselines
    ref_cnt = Counts(everyone, TIDS["referent"], ["page", "spage", "prec", "sprec", "listed", "slisted"])
    for s in everyone:
        for var, name, shown in ((("referent", "page"), "page", "listed"), (("referent", "stated"), "spage", "slisted"),
                                 (("referent", "pagerec"), "prec", None)):
            d = recs[var][s]
            for t, acted, _ in d["counted"]:
                ref_cnt.add(s, name, t, acted)
            for t, acted in (d["shown"] if shown else []):
                ref_cnt.add(s, shown, t, acted)
    abs_cnt = absence_counts(a.work, stems)
    clear_ref, clear_abs = sorted(clear["referent"] & set(everyone)), sorted(clear["absence"] & set(abs_cnt.agents))

    cells, tot = fetch_cells(a.work)
    fetch_md, fetch_res = fetch_section(cells, tot)
    claim_md, claim_res, claim_names = claim_section(abs_cnt, clear_abs)
    agents_md, agents_res, agent_rows = agents_section(recs, clear_abs)
    family_md, family_res = family_section(ref_cnt, abs_cnt, base, everyone)
    rec_md, rec_res, seen, k1_cnt = statedrec_section(a.work, ref_cnt, clear["referent"], stems, base)

    # ---- self-check against reference values (itt.py, crossover.py, seeing.py)
    all36 = fetch_res["all 36 archived agents"]
    check("archived agents", 36, all36["agents"])
    check("fetched-and-acted, 36 agents", 72, all36["pooled_fetched"][0])
    check("fetched, 36 agents", 855, all36["pooled_fetched"][1])
    check("not-fetched-and-acted, 36 agents", 1496, all36["pooled_not"][0])
    check("not fetched, 36 agents", 2025, all36["pooled_not"][1])
    sizes = [c[1] + c[3] for cs in cells.values() for c in cs]  # the text of (a) says every cell has 5 rollouts
    check("(a): smallest (agent, template) cell, rollouts", 5, min(sizes))
    check("(a): largest (agent, template) cell, rollouts", 5, max(sizes))
    check("clear-acting agents, referent", 18, len(clear_ref))
    check("clear-acting agents, absence", 27, len(clear_abs))
    cr = claim_res
    for nm, want_s, want_u in (("W0 counted acting", 0.946, 0.835), ("W1 counted acting", 0.278, 0.666)):
        check(nm + ", stated", want_s, cr[(nm, "s")]["mean"])
        check(nm + ", unstated", want_u, cr[(nm, "u")]["mean"])
    check("W1 shown acting, stated", 0.295, cr[("W1 shown acting", "s")]["mean"])
    gap = "W1 counted minus W1 shown (ITT gap)"
    check("absence ITT gap, stated", -0.017, cr[(gap, "s")]["mean"])
    check_ci("absence ITT gap, stated, crossed CI", (-0.080, 0.047), cr[(gap, "s")]["x"])
    check("absence ITT gap, unstated", -0.021, cr[(gap, "u")]["mean"])
    check_ci("absence ITT gap, unstated, crossed CI", (-0.077, 0.030), cr[(gap, "u")]["x"])
    check("W1 counted change, stated minus unstated", -0.388, cr[("W1 counted acting", "d")]["mean"])
    check_ci("W1 counted change, agent-cluster CI", (-0.485, -0.292), cr[("W1 counted acting", "d")]["ci"])
    check_ci("W1 counted change, crossed CI", (-0.493, -0.291), cr[("W1 counted acting", "d")]["x"])
    check("agents in (c)", 27, agents_res["agents"])
    # the referent 2 x 2 machinery on the archived cells of all 18 clear-acting agents, independent of <work>/statedrec
    sub = ref_cnt.subset(clear_ref)
    R = sub.rate
    arch = evaluate(sub, {"page": lambda S: R(S, "page"), "spage": lambda S: R(S, "spage"), "prec": lambda S: R(S, "prec"),
                          "listed": lambda S: R(S, "listed"), "slisted": lambda S: R(S, "slisted"),
                          "gap": lambda S: R(S, "page") - R(S, "listed"), "sgap": lambda S: R(S, "spage") - R(S, "slisted"),
                          "pgap": lambda S: R(S, "prec") - R(S, "listed"),
                          "only one": lambda S: R(S, "spage") - R(S, "page"), "pagerec": lambda S: R(S, "prec") - R(S, "page")})
    for k, want in (("page", 0.654), ("spage", 0.776), ("prec", 0.625), ("listed", 0.240), ("slisted", 0.333), ("gap", 0.415),
                    ("sgap", 0.444), ("pgap", 0.385), ("only one", 0.122), ("pagerec", -0.029)):
        check(f"referent 18 agents: {k}", want, arch[k]["mean"])
    check_ci("referent page ITT gap, crossed CI", (0.289, 0.543), arch["gap"]["x"])
    check_ci("referent 'only one' counted change, crossed CI", (0.016, 0.247), arch["only one"]["x"])
    check_ci("referent pagerec minus page, crossed CI", (-0.103, 0.046), arch["pagerec"]["x"])
    all31 = evaluate(ref_cnt, {"only one": lambda S: ref_cnt.rate(S, "spage") - ref_cnt.rate(S, "page")})["only one"]
    check("referent 'only one' counted change, all 31", 0.071, all31["mean"])
    check_ci("referent 'only one' counted change, all 31, crossed CI", (-0.031, 0.173), all31["x"])
    check("family table, 0.80: referent page ITT gap", 0.415, family_res[0.80]["rows"][0]["estimate"])
    check("family table, 0.80: absence stated ITT gap", -0.017, family_res[0.80]["rows"][1]["estimate"])
    check("family table, 0.80: 'only one' change, 18", 0.122, family_res[0.80]["rows"][2]["estimate"])
    check("family table, 0.80: 'only one' change, all", 0.071, family_res[0.80]["rows"][3]["estimate"])
    ok = checks_ok()

    out = [f"# Robustness Analyses of the Existing Rollouts ({DATE}, descriptive)", "",
           "Agents weigh equally in agent means. Intervals are 95% percentile intervals of 2,000 resamples (agent-cluster: "
           "score_absence.boot; crossed: agents and the 16 templates resampled independently, as itt.py). Clear-acting sets are "
           "fixed from the archived conditions at 0.80 unless a threshold is named.", "",
           "## Self-Check", "", "Archived values recomputed (tolerance 0.0005; intervals as in itt.py and "
           "crossover.py)", ""]
    out += table(["check", "expected", "computed", "pass"],
                 [[n, f"{w:g}" if isinstance(w, int) else f"{w:.3f}", "-" if g is None else (f"{g:g}" if isinstance(g, int) else f"{g:.4f}"),
                   "PASS" if o else "FAIL"] for (n, w, g, _), o in zip(CHECKS, ok)])
    out += [f"Self-check: {'all ' + str(len(ok)) + ' checks pass' if all(ok) else 'FAILED, ' + str(ok.count(False)) + ' of ' + str(len(ok))}.", ""]
    out += fetch_md + claim_md + agents_md + family_md + rec_md
    text = "\n".join(out)
    print(text)

    if not a.no_write:
        outdir.mkdir(parents=True, exist_ok=True)
        stem = lambda n, e: outdir / f"panel-r2-{n}{DATE}.{e}"
        stem("", "md").write_text(text + "\n")
        write_csv(stem("fetch-cells-", "csv"), ["agent", "template", "n_fetched", "acted_fetched", "n_not_fetched", "acted_not_fetched",
                                                "difference"],
                  [[ag, t, nf, af, nn, an, "" if not (nf and nn) else round(af / nf - an / nn, 6)]
                   for ag, cs in cells.items() for t, nf, af, nn, an in cs])
        cols = ("s_w0c", "u_w0c", "s_w1c", "u_w1c", "s_w1s", "u_w1s")
        write_csv(stem("absence-agents-", "csv"),
                  ["agent", "stem", "counted_acted", "counted_n", "shown_acted", "shown_n", "shown_minus_counted", "W0_counted_stated",
                   "W0_counted_unstated", "W1_counted_stated", "W1_counted_unstated", "W1_shown_stated", "W1_shown_unstated"],
                  [[r["agent"], r["stem"], r["counted_acted"], r["counted_n"], r["shown_acted"], r["shown_n"],
                    round(r["shown_minus_counted"], 6)] + [cell_rate(abs_cnt, r["stem"], k) for k in cols]
                   for r in sorted(agent_rows, key=lambda r: r["stem"])])
        write_csv(stem("family-", "csv"),
                  ["threshold", "statistic", "agents", "families", "estimate", "agent_ci_low", "agent_ci_high", "family_ci_low",
                   "family_ci_high", "lofo_min", "lofo_min_family", "lofo_max", "lofo_max_family"],
                  [[thr, r["statistic"], r["agents"], r["families"], round(r["estimate"], 6), *map(lambda x: round(x, 6), r["agent_ci"]),
                    *map(lambda x: round(x, 6), r["family_ci"]), round(r["lofo_min"], 6), r["lofo_min_family"], round(r["lofo_max"], 6),
                    r["lofo_max_family"]] for thr, v in family_res.items() for r in v["rows"]])
        write_csv(stem("statedrec-", "csv"), ["agent", "stem", "clear_referent", "k2_rows", "k1_rows", "k2_complete", "k2_stated_pagerec",
                                              "k2_page", "k2_stated_page", "k2_pagerec", "k2_listed", "k2_listed_stated", "k1_stated_pagerec"],
                  [[LABEL[s], s, s in clear["referent"], seen[s]["k2"], seen[s]["k1"],
                    s in ref_cnt.row and complete_cell(ref_cnt, s, "sprec"),
                    *(cell_rate(ref_cnt, s, c) for c in ("sprec", "page", "spage", "prec", "listed", "slisted")),
                    cell_rate(k1_cnt, s, "k1")] for s in stems])
        json_res = {"self_check": [{"check": n, "expected": w, "computed": g, "pass": o} for (n, w, g, _), o in zip(CHECKS, ok)],
                    "a_fetch_within_cells": fetch_res, "b_claim_both_worlds": {f"{k[0]} | {k[1]}": v for k, v in claim_res.items()},
                    "c_stated_absence_agents": agents_res, "d_family": {str(k): v for k, v in family_res.items()},
                    "e_statedrec": rec_res}
        stem("", "json").write_text(json.dumps(clean(json_res), indent=1) + "\n")
        print(f"\nwrote {stem('', 'md')} and companion .json and .csv files", file=sys.stderr)
    print(f"runtime {time.perf_counter() - t0:.1f} s", file=sys.stderr)
    return 0 if all(ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
