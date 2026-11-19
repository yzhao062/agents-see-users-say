"""Every grid agent in one table: the paper's 14 agents (archived rollouts) and the further Bedrock agents run with the
full protocol. Prints a markdown table, the claim checks the paper states over all agents,
and with --latex the appendix table rows.

  python all_agents.py [--work $WTA_WORK] [--latex out.tex]

Columns (K = 2 unless marked; acting is P(action call), 80 rollouts per cell):
  K1      acting on one listed match (a clear instruction)
  listed, page, compound, first, firstdoc, firstnote, send   acting under each contract (send: the share verb, listed)
  one|doc   share of firstdoc count answers that state a single match (the tool description did not reach belief)
  hedge|note  share of firstnote count answers that say more may exist (the note in the result did)
  two|page  share of page count answers that state two
  |J| trunc   max over first, firstdoc, firstnote of |P(act | K=1) - P(act | K=2)|, without / with the fiduciary line
  K1 line   acting on one listed match with the line; J page line: the K = 1 minus K = 2 gap under page with the line
"""
import os
import argparse
import gzip
import json
from decimal import ROUND_HALF_UP, Decimal
from fractions import Fraction
from pathlib import Path

from mechanism import classify
from score_grid import score

ART = Path(os.environ.get("WTA_ARTIFACTS", "data/artifacts")) / "licgrid-rollouts-2026-10-12"
BEDROCK9 = ["deepseek.v3.2", "minimax.minimax-m2.5", "nvidia.nemotron-super-3-120b", "us.amazon.nova-2-lite-v1_0",
            "us.meta.llama4-maverick-17b-instruct-v1_0", "zai.glm-5", "qwen.qwen3-next-80b-a3b", "moonshotai.kimi-k2.5",
            "mistral.mistral-large-3-675b-instruct"]
LABEL = {"deepseek.v3.2": "DeepSeek V3.2", "minimax.minimax-m2.5": "MiniMax M2.5",
         "nvidia.nemotron-super-3-120b": "Nemotron Super 3", "us.amazon.nova-2-lite-v1_0": "Nova 2 Lite",
         "us.meta.llama4-maverick-17b-instruct-v1_0": "Llama 4 Maverick", "zai.glm-5": "GLM-5",
         "qwen.qwen3-next-80b-a3b": "Qwen3-Next 80B", "moonshotai.kimi-k2.5": "Kimi K2.5",
         "mistral.mistral-large-3-675b-instruct": "Mistral Large 3",
         "amazon.nova-micro-v1_0": "Nova Micro", "amazon.nova-lite-v1_0": "Nova Lite", "amazon.nova-pro-v1_0": "Nova Pro",
         "minimax.minimax-m2": "MiniMax M2", "minimax.minimax-m2.1": "MiniMax M2.1",
         "mistral.ministral-3-3b-instruct": "Ministral 3 3B", "mistral.ministral-3-8b-instruct": "Ministral 3 8B",
         "mistral.ministral-3-14b-instruct": "Ministral 3 14B", "mistral.devstral-2-123b": "Devstral 2 123B",
         "moonshot.kimi-k2-thinking": "Kimi K2 Thinking", "us.moonshotai.kimi-k3": "Kimi K3",
         "nvidia.nemotron-nano-9b-v2": "Nemotron Nano 9B", "nvidia.nemotron-nano-12b-v2": "Nemotron Nano 12B",
         "nvidia.nemotron-nano-3-30b": "Nemotron Nano 3 30B", "qwen.qwen3-32b-v1_0": "Qwen3 32B",
         "qwen.qwen3-coder-30b-a3b-v1_0": "Qwen3 Coder 30B", "qwen.qwen3-coder-next": "Qwen3 Coder Next",
         "qwen.qwen3-vl-235b-a22b": "Qwen3 VL 235B", "us.meta.llama3-3-70b-instruct-v1_0": "Llama 3.3 70B",
         "us.meta.llama4-scout-17b-instruct-v1_0": "Llama 4 Scout", "zai.glm-4.7-flash": "GLM-4.7 Flash",
         "zai.glm-4.7": "GLM-4.7"}
NEW = [k for k in LABEL if k not in BEDROCK9]


def manifest(work):
    a = lambda n: ART / f"{n}.jsonl.gz"
    g = Path(work) / "grid"
    m = {
        # Qwen3.5 in the grid ran with thinking off (enable_thinking false), so the mechanism and prompt-check cells
        # come from the no-thinking rollouts; the thinking-on runs are the scale sweep, not mixed in here.
        "Qwen3.5-27B": [a("qwen27b__none"), a("qwen27b_r2__none"), a("qwen27b_r3__none"), a("deltaai-qwen27b_r4__none"),
                        a("b2-qwen35_27b_nothink__none"), a("b2-qwen35_27b_nothink__none__scope"), a("qwen27b__fiduciary"),
                        a("qwen27b_r2__fiduciary"), a("qwen27b_r3__fiduciary"), a("b2-qwen35_27b_nothink__fiduciary"),
                        a("b2-qwen35_27b_nothink__fiduciary__scope")],
        "Qwen3.5-122B": [a("deltaai-qwen122b__none"), a("deltaai-qwen122b_r2__none"), a("deltaai-qwen122b_r3__none"),
                         a("deltaai-qwen122b_r4__none"), a("deltaai-qwen122b__fiduciary"), a("deltaai-qwen122b_r2__fiduciary"),
                         a("deltaai-qwen122b_r3__fiduciary")],
    }
    for s in ("120b", "20b"):
        m[f"gpt-oss-{s}"] = [a(f"gptoss{s}__none"), a(f"gptoss{s}_r2__none"), a(f"gptoss{s}_r3__none"),
                             a(f"deltaai-gptoss{s}_r4__none"), a(f"b2-gptoss{s}_medium__none"),
                             a(f"b2-gptoss{s}_medium__none__scope"), a(f"gptoss{s}__fiduciary"), a(f"gptoss{s}_r2__fiduciary"),
                             a(f"gptoss{s}_r3__fiduciary"), a(f"b2-gptoss{s}_medium__fiduciary"),
                             a(f"b2-gptoss{s}_medium__fiduciary__scope")]
    for b in BEDROCK9:
        m[LABEL[b]] = [a(f"bedrock-{b}__none"), a(f"r5-{b}"), a(f"r6-{b}__fiduciary"), a(f"bedrock-full-{b}__fiduciary")]
    # Gemini 3.8 Flash: no-line cells from the Gemini API (AI Studio); line cells from Vertex AI
    m["Gemini 3.8 Flash"] = [a("gemini-3.8-flash__none"), a("r5-gemini-3.8-flash"), a("nairr-gemini-3.8-flash-default__fiduciary")]
    for b in NEW:
        m[LABEL[b]] = [a(f"bedrock-full-{b}__none"), a(f"bedrock-full-{b}__fiduciary")]
    return m


# frontier configurations, reported apart from the 36 agents
# the Claude and GPT models ran the no-line cells only
FRONTIER = {"Gemini 3.1 Pro, high": "nairr-gemini-3.1-pro-high", "Gemini 3.1 Pro, low": "nairr-gemini-3.1-pro-low",
            "GPT-6 astra": "nairr-precheck-gpt-6-astra", "Claude Opus 5.5": "nairr-precheck-claude-opus-5.5",
            "GPT-6.1 sol": "nairr-precheck-gpt-6.1-sol"}


def frontier_manifest():  # missing files are skipped by load()
    return {lab: [ART / f"{t}__none.jsonl.gz", ART / f"{t}__listpage.jsonl.gz", ART / f"{t}__fiduciary.jsonl.gz"]
            for lab, t in FRONTIER.items()}


def load(files):
    rows, seen = {"none": [], "fiduciary": []}, set()
    for f in files:
        f = Path(f)
        if not f.exists():
            continue
        op = gzip.open if f.suffix == ".gz" else open
        with op(f, "rt") as fh:
            for ln in fh:
                if not ln.strip():
                    continue
                r = json.loads(ln)
                norm = r.get("norm") or "none"
                key = (r["template"], r["k"], r["contract"], r["consequence"], r["rep"], norm)
                if r.get("error") or key in seen or norm not in rows:
                    continue
                seen.add(key)
                r["s"] = score(r)
                rows[norm].append(r)
    return rows


def rate(rows, c, k, q="irreversible"):
    xs = [r["s"]["acted"] for r in rows if r["contract"] == c and r["k"] == k and r["consequence"] == q]
    return sum(xs) / len(xs) if len(xs) >= 20 else None


def answers(rows, c, kind):
    xs = [classify(r["final"]) for r in rows if r["contract"] == c and r["k"] == 2 and r["consequence"] == "count"]
    return xs.count(kind) / len(xs) if len(xs) >= 20 else None


def jgap(rows, c):
    a, b = rate(rows, c, 1), rate(rows, c, 2)
    return None if a is None or b is None else a - b


def stats(rows):
    n, f = rows["none"], rows["fiduciary"]
    jt = lambda rs: [abs(j) for j in (jgap(rs, c) for c in ("first", "firstdoc", "firstnote")) if j is not None]
    return {
        "K1": rate(n, "all", 1), "listed": rate(n, "all", 2), "page": rate(n, "page", 2),
        "compound": rate(n, "compound", 2), "first": rate(n, "first", 2), "firstdoc": rate(n, "firstdoc", 2),
        "firstnote": rate(n, "firstnote", 2), "send": rate(n, "all", 2, "share"),
        "one|doc": answers(n, "firstdoc", "one"), "hedge|note": answers(n, "firstnote", "hedge"),
        "two|page": answers(n, "page", "two"),
        "|J| trunc": max(jt(n)) if jt(n) else None, "|J| trunc line": max(jt(f)) if jt(f) else None,
        "K1 line": rate(f, "all", 1), "J page line": jgap(f, "page"),
    }


COLS = ["K1", "listed", "page", "compound", "first", "firstdoc", "firstnote", "send", "one|doc", "hedge|note", "two|page",
        "|J| trunc", "|J| trunc line", "K1 line", "J page line"]


def fmt(x):
    """Two decimals, rounded half up from the exact fraction (rates are k/n with n <= 80 per cell)."""
    if x is None:
        return "-"
    f = Fraction(x).limit_denominator(10000)
    return str((Decimal(f.numerator) / Decimal(f.denominator)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=str(Path(os.environ.get("WTA_WORK", "data/work"))))
    ap.add_argument("--latex", default="")
    ap.add_argument("--frontier-latex", default="")
    a = ap.parse_args()
    table = {lab: stats(load(fs)) for lab, fs in manifest(a.work).items()}
    print("| agent | " + " | ".join(COLS) + " |")
    print("|---|" + "---|" * len(COLS))
    for lab, s in table.items():
        print(f"| {lab} | " + " | ".join(fmt(s[c]) for c in COLS) + " |")
    v = lambda c: [(lab, s[c]) for lab, s in table.items() if s[c] is not None]
    print(f"\nagents: {len(table)}")
    print("listed <= 0.24:", sum(x <= 0.24 for _, x in v("listed")), "of", len(v("listed")),
          "| listed >= 0.5:", [lab for lab, x in v("listed") if x >= 0.5])
    print("|first - K1| max:", max(abs(s["first"] - s["K1"]) for s in table.values() if s["first"] is not None and s["K1"] is not None))
    print("|J| trunc (none) max:", max(x for _, x in v("|J| trunc")), "| with line max:", max(x for _, x in v("|J| trunc line")))
    print("page >= 0.24:", sum(x >= 0.24 for _, x in v("page")), "of", len(v("page")))
    print("firstdoc - first > 0.10:", [(lab, round(s["firstdoc"] - s["first"], 2)) for lab, s in table.items()
                                       if s["firstdoc"] is not None and s["first"] is not None and s["firstdoc"] - s["first"] > 0.10])
    know = [(lab, s["hedge|note"], s["firstnote"]) for lab, s in table.items() if s["hedge|note"] is not None]
    print("hedge|note >= 0.5:", len([k for k in know if k[1] >= 0.5]), "of", len(know),
          "; their firstnote acting:", sorted(round(k[2], 2) for k in know if k[1] >= 0.5 and k[2] is not None))
    print("|firstnote - first| max:", max(abs(s["firstnote"] - s["first"]) for s in table.values()
                                         if s["firstnote"] is not None and s["first"] is not None))
    print("two|page >= 0.8 and page >= 0.24:", [lab for lab, s in table.items() if (s["two|page"] or 0) >= 0.8 and (s["page"] or 0) >= 0.24])
    print("line: check (K1 line >= 0.8, J page line >= 0.5):", [lab for lab, s in table.items() if (s["K1 line"] or 0) >= 0.8 and (s["J page line"] or 0) >= 0.5])
    print("line: brake (K1 line <= 0.25):", [lab for lab, s in table.items() if s["K1 line"] is not None and s["K1 line"] <= 0.25])
    if a.latex:
        Path(a.latex).parent.mkdir(parents=True, exist_ok=True)
        with open(a.latex, "w") as fh:
            for lab, s in table.items():
                fh.write(lab + " & " + " & ".join(fmt(s[c]).replace("0.", ".") if s[c] is not None else "--" for c in COLS) + " \\\\\n")
    front = {lab: stats(load(fs)) for lab, fs in frontier_manifest().items()}
    print("\nfrontier (NAIRR):")
    for lab, s in front.items():
        print(f"| {lab} | " + " | ".join(fmt(s[c]) for c in COLS) + " |")
    if a.frontier_latex:
        Path(a.frontier_latex).parent.mkdir(parents=True, exist_ok=True)
        with open(a.frontier_latex, "w") as fh:
            for lab, s in front.items():
                fh.write(lab + " & " + " & ".join(fmt(s[c]).replace("0.", ".") if s[c] is not None else "--" for c in COLS) + " \\\\\n")


if __name__ == "__main__":
    main()
