"""tau-bench second-customer table over every agent configuration: silent wrong-account writes (before any agent message that
questions the account) per condition and reward, from the archived rollouts and local results (work/results/).
Prints markdown, and LaTeX rows with --latex.

  python tau_table.py [--work $WTA_WORK] [--latex out.tex]
  python tau_table.py --paper out.tex     # count rows of the paper's tau-bench tables (see paper_rows)
"""
import os
import argparse
import gzip
import json
from pathlib import Path

from analyze_tau import flags
from wrong_writes import classify

ART = Path(os.environ.get("WTA_ARTIFACTS", "data/artifacts")) / "taucollide-rollouts-2026-10-12"
# label, file (archive name ending .gz, or a work/results name), group
AGENTS = [
    ("GLM-5", "zai.glm-5.jsonl.gz", "main"),
    ("DeepSeek V3.2", "deepseek.v3.2.jsonl.gz", "main"),
    ("Kimi K2.5", "moonshotai.kimi-k2.5.jsonl.gz", "main"),
    ("Qwen3-Next", "qwen.qwen3-next-80b-a3b.jsonl.gz", "main"),
    ("Nova 2 Lite", "us.amazon.nova-2-lite-v1_0.jsonl.gz", "main"),
    ("MiniMax M2.5", "minimax.minimax-m2.5.jsonl.gz", "main"),
    ("Nemotron Super 3", "nvidia.nemotron-super-3-120b.jsonl.gz", "main"),
    ("Mistral Large 3", "mistral.mistral-large-3-675b-instruct.jsonl.gz", "main"),
    ("Qwen3.5-9B", "qwen35-9b-b2.jsonl.gz", "main"),
    ("Qwen3.5-27B", "qwen35-27b-vm6.jsonl.gz", "main"),
    ("Qwen3.5-122B", "qwen35-122b-b2.jsonl.gz", "main"),
    ("gpt-oss-20b", "gptoss20b-vm6.jsonl.gz", "main"),
    ("gpt-oss-120b", "gptoss120b-medium-b2.jsonl.gz", "main"),
    ("Qwen3.5-27B, no thinking", "qwen35-27b-nothink-b2.jsonl.gz", "reasoning"),
    ("gpt-oss-20b, low", "gptoss20b-low-b2.jsonl.gz", "reasoning"),
    ("gpt-oss-20b, high", "gptoss20b-high-vm6.jsonl.gz", "reasoning"),
    ("gpt-oss-120b, low", "gptoss120b-low-b2.jsonl.gz", "reasoning"),
    ("gpt-oss-120b, high", "gptoss120b-high-b2.jsonl.gz", "reasoning"),
    # frontier configurations
    ("Gemini 3.1 Pro, high", "nairr-gemini-3.1-pro-high.jsonl.gz", "frontier"),
    ("Gemini 3.1 Pro, low", "nairr-gemini-3.1-pro-low.jsonl.gz", "frontier"),
    ("Gemini 3.8 Flash", "nairr-gemini-3.8-flash-default.jsonl.gz", "frontier"),
]
WRONG = ["twin", "twindoc", "twinfid", "twinall", "twinallfid"]
REWARD = ["base", "twin", "twinall", "twinallfid"]


def rows_of(name, work):
    """analyze_tau.load semantics (skip errored and repeated (task, cond) rows), reading the archive in place."""
    p = ART / name if name.endswith(".gz") else Path(work) / "results" / name
    if not p.exists():
        return None
    out, seen = [], set()
    with (gzip.open(p, "rt") if p.suffix == ".gz" else open(p)) as fh:
        for ln in fh:
            if not ln.strip():
                continue
            r = json.loads(ln)
            if r.get("error") or (r["task"], r["cond"]) in seen:
                continue
            seen.add((r["task"], r["cond"]))
            out.append(r)
    return out


def row(rows):
    out = {}
    for c in set(WRONG + REWARD):
        rs = [r for r in rows if r["cond"] == c]
        if len(rs) < 80:  # a condition counts once at least 80 of the 87 tasks are in
            out[c] = (None, None, len(rs))
            continue
        out[c] = (sum(classify(r) == "silent" for r in rs) / len(rs), sum(flags(r)["reward"] for r in rs) / len(rs), len(rs))
    return out


def paper_rows(work, out):
    """LaTeX rows of the main-text and appendix tau-bench tables, as counts (cells read
    silent/total wrong-account writes; reward is the number of tasks completed). Definitions come from audit_2026_10_13 (which
    imports analyze_tau, wrong_writes and run_tau), so the paper, the audit script and this script agree.
    Writes six blocks to `out`:
    COHORT (ranges across agents: minimum to maximum over the 13 historical agents, over the 12 with a twinnote run for that condition;
    silent writes, total wrong-account writes, reward), FRONTIER (appendix newest-model table: Claude Opus 5.5 and GPT-6.1 sol; columns
    twin, twinnote, twinall, then reward base, twin, twinnote, twinall; the n of each row is in its label), PANEL (policy comparison on matched tasks:
    twinnote against twinnotepol paired by task: silent, total, tasks that lost / gained a wrong-account write,
    conversations with any successful write, transfers to a human agent, reward), AGENTS (appendix all-agents table: every historical
    agent under every condition, plus a sum row over the 12 agents with a twinnote run and one over all 13), GEMINI (write and reward columns of the appendix Gemini table)
    and REASON (appendix reasoning and scale table: twin and twinall silent/total, then reward base, twin, twinall for the
    reasoning and scale settings). Cells that no run covers read --; Qwen3.5-122B has no twinnote run yet."""
    import audit_2026_10_13 as au  # imported here: the audit imports AGENTS and ART from this module

    au.RES = Path(work) / "results"
    agents = au.load_all()
    cnt = {}
    for label, a in agents.items():
        for c, rs in a["rows"].items():
            if rs:
                cnt[(label, c)] = au.counts(rs)

    def ok(label, cond, min_n=80):
        k = cnt.get((label, cond))
        return k is not None and k["n"] >= min_n

    def writes(label, cond, color="", min_n=80):
        if not ok(label, cond, min_n):
            return f"{color}--"
        k = cnt[(label, cond)]
        return f"{color}{k['silent']}/{k['wrong']}"

    def reward(label, cond, min_n=80):
        return str(cnt[(label, cond)]["reward"]) if ok(label, cond, min_n) else "--"

    mint, coral = r"\cellcolor{tabmint}", r"\cellcolor{tabcoral}"
    main = [l for l, a in agents.items() if a["group"] == "main"]
    lines = ["% COHORT"]
    span = lambda c, key: (lambda v: f"{min(v)} to {max(v)}")([cnt[(l, c)][key] for l in main if ok(l, c)])
    n_of = lambda c: sum(ok(l, c) for l in main)
    lines.append(rf"base (released data) & -- & -- & {span('base', 'reward')} \\")
    lines.append(rf"\cellcolor{{tabmint}}twin & \cellcolor{{tabmint}}{span('twin', 'silent')} & \cellcolor{{tabmint}}{span('twin', 'wrong')} & \cellcolor{{tabmint}}{span('twin', 'reward')} \\")
    lines.append(rf"twinfid (prompt line) & {span('twinfid', 'silent')} & {span('twinfid', 'wrong')} & {span('twinfid', 'reward')} \\")
    lines.append(rf"\cellcolor{{tabcoral}}twinnote (note, {n_of('twinnote')} agents) & \cellcolor{{tabcoral}}{span('twinnote', 'silent')} & \cellcolor{{tabcoral}}{span('twinnote', 'wrong')} & \cellcolor{{tabcoral}}{span('twinnote', 'reward')} \\")
    lines.append(rf"\cellcolor{{tabcoral}}twinall (complete lookup) & \cellcolor{{tabcoral}}{span('twinall', 'silent')} & \cellcolor{{tabcoral}}{span('twinall', 'wrong')} & \cellcolor{{tabcoral}}{span('twinall', 'reward')} \\")
    lines.append(f"% agents per row: base {n_of('base')}, twin {n_of('twin')}, twinfid {n_of('twinfid')}, twinnote {n_of('twinnote')}, twinall {n_of('twinall')}; "
                 f"sums over 13 agents (silent/total of {sum(cnt[(l, 'twin')]['n'] for l in main):,}): twin {sum(cnt[(l, 'twin')]['silent'] for l in main)}/{sum(cnt[(l, 'twin')]['wrong'] for l in main)}, "
                 f"twinall {sum(cnt[(l, 'twinall')]['silent'] for l in main)}/{sum(cnt[(l, 'twinall')]['wrong'] for l in main)} (after-flag {sum(cnt[(l, 'twinall')]['after'] for l in main)})")

    lines.append("% FRONTIER")
    for label, tag in (("Claude Opus 5.5", "Opus 5.5"), ("GPT-6.1 sol", "GPT-6.1 sol")):
        n = cnt[(label, "twin")]["n"]
        rw = lambda c: f"{reward(label, c, 1)}/{cnt[(label, c)]['n']}" if (label, c) in cnt else "--"
        cells = [writes(label, "twin", mint, 1), writes(label, "twinnote", coral, 1), writes(label, "twinall", coral, 1),
                 rw("base"), rw("twin"), rw("twinnote"), rw("twinall")]
        lines.append(f"{tag} ({n}) & " + " & ".join(cells) + r" \\")

    lines.append("% PANEL")
    notes = []
    for label, tag in (("Claude Opus 5.5", "Opus 5.5"), ("GPT-6.1 sol", "GPT-6.1 sol")):
        by = {c: {r["task"]: r for r in agents[label]["rows"][c]} for c in ("twinnote", "twinnotepol")}
        tasks = sorted(set(by["twinnote"]) & set(by["twinnotepol"]))
        cl = {c: {t: au.classify(by[c][t]) for t in tasks} for c in by}
        silent = {c: sum(v == "silent" for v in cl[c].values()) for c in by}
        total = {c: sum(v != "none" for v in cl[c].values()) for c in by}
        lost = [t for t in tasks if cl["twinnote"][t] != "none" and cl["twinnotepol"][t] == "none"]
        gained = [t for t in tasks if cl["twinnote"][t] == "none" and cl["twinnotepol"][t] != "none"]
        wrote = {c: sum(any(w["ok"] for w in by[c][t]["writes"]) for t in tasks) for c in by}
        trans = {c: sum(any(m.get("tool") == "transfer_to_human_agents" for m in by[c][t]["log"]) for t in tasks) for c in by}
        rew = {c: sum(by[c][t]["reward"] == 1 for t in tasks) for c in by}
        arrow = lambda d: f"{d['twinnote']} $\\to$ {d['twinnotepol']}"
        lines.append(f"{tag} & {arrow(silent)} & {arrow(total)} & {len(lost)} / {len(gained)} & {arrow(wrote)} & {arrow(trans)} & {arrow(rew)} \\\\")
        notes.append(f"% {tag}: n = {len(tasks)}; tasks that lost a wrong-account write {lost}, gained {gained}; exact sign test p = {au.sign_p(len(lost), len(gained)):.4f}")
    lines += notes

    lines.append("% AGENTS")  # appendix all-agents table
    wconds = ("twin", "twindoc", "twinfid", "twinnote", "twinall", "twinallfid")
    rconds = ("base", "twin", "twindoc", "twinfid", "twinnote", "twinall", "twinallfid")
    for i, label in enumerate(main):
        if i == 8:  # the first eight agents ran on Amazon Bedrock, the next five on self-hosted GPUs
            lines.append(r"\midrule")
        cells = [writes(label, c, mint if c == "twin" else (coral if c in ("twinnote", "twinall") else "")) for c in wconds] + [reward(label, c) for c in rconds]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    def sum_row(ag, name):
        t = lambda c, key: sum(cnt[(l, c)][key] for l in ag if ok(l, c))
        n_conv = sum(cnt[(l, "twin")]["n"] for l in ag)
        cells = [f"{t(c, 'silent')}/{t(c, 'wrong')}" if all(ok(l, c) for l in ag) else "--" for c in wconds]
        cells += [str(t(c, "reward")) if all(ok(l, c) for l in ag) else "--" for c in rconds]
        return rf"\rowcolor{{tabgray}} {name} ({n_conv:,}) & " + " & ".join(cells) + r" \\"

    full = [l for l in main if ok(l, "twinnote")]  # agents with every condition (twinnote complete)
    lines += [r"\midrule", sum_row(full, f"Sum, {len(full)} agents")] + ([sum_row(main, f"Sum, {len(main)} agents")] if len(full) != len(main) else [])

    lines.append("% GEMINI")  # appendix Gemini table
    for label in ("Gemini 3.1 Pro, high", "Gemini 3.1 Pro, low", "Gemini 3.8 Flash"):
        cells = [writes(label, "twin", mint), writes(label, "twinall", coral), reward(label, "base"), reward(label, "twin"), reward(label, "twinall")]
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")

    lines.append("% REASON")  # appendix reasoning and scale table
    prev = None
    for label, name, setting in au.REASONING_ROWS:
        if prev is not None and name != prev:
            lines.append(r"\midrule")
        prev = name
        cells = [writes(label, "twin"), writes(label, "twinall"), reward(label, "base"), reward(label, "twin"), reward(label, "twinall")]
        lines.append(f"{name} & {setting} & " + " & ".join(cells) + r" \\")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(Path(os.environ.get("WTA_WORK", "data/work"))))
    ap.add_argument("--latex", default="")
    ap.add_argument("--paper", default="", help="write the LaTeX rows of the paper's tau-bench tables to this file")
    a = ap.parse_args()
    if a.paper:
        paper_rows(a.work, a.paper)
        return
    f = lambda x: "-" if x is None else f"{x:.2f}"
    lines = []
    print("| agent | group | " + " | ".join(f"wrong {c}" for c in WRONG) + " | " + " | ".join(f"reward {c}" for c in REWARD) + " | n |")
    print("|---|---|" + "---|" * (len(WRONG) + len(REWARD) + 1))
    for label, name, group in AGENTS:
        rs = rows_of(name, a.work)
        if rs is None:
            print(f"| {label} | {group} | missing |")
            continue
        s = row(rs)
        print(f"| {label} | {group} | " + " | ".join(f(s[c][0]) for c in WRONG) + " | " + " | ".join(f(s[c][1]) for c in REWARD)
              + f" | {min(s[c][2] for c in s)} |")
        lines.append(f"{label} & " + " & ".join(f(s[c][0]) for c in WRONG) + " & " + " & ".join(f(s[c][1]) for c in REWARD) + " \\\\")
    if a.latex:
        Path(a.latex).parent.mkdir(parents=True, exist_ok=True)
        Path(a.latex).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
