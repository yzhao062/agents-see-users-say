"""Draw the hand-read audit samples from the gate rollouts. Writes two files per sample
kind: the transcripts to read (no scorer output) and the scorer's labels keyed by sample number, so the reader labels
first and compares after.

  python sample_absence.py FILE_OR_DIR ... --out DIR [--seed 0]

Action sample: 40 rollouts, 10 per act cell (counted W0, counted W1, shown W0, shown W1), spread over agents and
templates; within counted W1, five fetched and five not fetched (by the scorer) when both groups allow it, the other group
filling any shortfall; shuffled before numbering. Probe sample: 50 answers spread over agents. Refuses undersized input.
"""
import argparse
import json
import random
from pathlib import Path

from score_absence import cell, load, probe_count, read_raw


def spread(rs, n, rng):
    """n rows, round-robin over agents, a random template order within each agent."""
    by = {}
    for r in rs:
        by.setdefault(r["model"], []).append(r)
    for v in by.values():
        rng.shuffle(v)
    out, keys = [], sorted(by)
    while len(out) < n and any(by.values()):
        for k in keys:
            if by[k] and len(out) < n:
                out.append(by[k].pop())
    return out


def show(r):
    lines = [f"model {r['model']} | template {r['template']} | tier {r['tier']} | world {r['world']} | rep {r['rep']}",
             f"falsifier token T = {r['token']}; act tool {r['act_tool']}; list tool {r['list_tool']}",
             f"instruction: {r['instruction']}"]
    for s in r["steps"]:
        res = json.dumps(s["result"])
        lines.append(f"  [response {s.get('turn', '?')}] {s['tool']}({json.dumps(s['args'])}) -> {res[:700]}")
    lines.append(f"  final: {(r['final'] or '<none>')[:1200]}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    rows = load(read_raw(a.paths))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    act = []
    for tier, world in (("counted", 0), ("shown", 0), ("shown", 1)):
        act += spread(cell(rows, tier, world), 10, rng)
    c1 = cell(rows, "counted", 1)
    fp = [r for r in c1 if r["s"]["fetched"]]
    nf = [r for r in c1 if not r["s"]["fetched"]]
    n_f = min(len(fp), max(5, 10 - len(nf)))  # 5 and 5 when both groups allow it; the other group fills a shortfall
    act += spread(fp, n_f, rng) + spread(nf, 10 - n_f, rng)
    probe = spread(cell(rows, "counted", 1, "probe"), 50, rng)
    if len(act) != 40 or len(probe) != 50:
        raise SystemExit("G5 requires 10 action transcripts per cell and 50 probe answers")
    rng.shuffle(act)  # sample numbers carry no scorer label
    with open(out / "action-transcripts.txt", "w") as fh:
        for i, r in enumerate(act, 1):
            fh.write(f"=== A{i:02d}\n{show(r)}\n\n")
    with open(out / "action-scorer.json", "w") as fh:
        json.dump({f"A{i:02d}": {"acted": r["s"]["acted"], "fetched": r["s"]["fetched"]} for i, r in enumerate(act, 1)}, fh, indent=1)
    with open(out / "probe-answers.txt", "w") as fh:
        for i, r in enumerate(probe, 1):
            fh.write(f"=== P{i:02d} | {r['model']} | {r['template']}\nquestion: {r['instruction']}\nanswer: {r['final']}\n\n")
    with open(out / "probe-scorer.json", "w") as fh:
        json.dump({f"P{i:02d}": probe_count(r["final"], r["template"]) for i, r in enumerate(probe, 1)}, fh, indent=1)
    print(f"wrote {len(act)} action transcripts and {len(probe)} probe answers to {out}")


if __name__ == "__main__":
    main()
