"""Stated-absence family: run an agent on its cells.

  python run_absence.py --model MODEL --out rollouts.jsonl [--bedrock-region us-east-1 | --base-url URL]
         [--tiers counted,shown] [--worlds 0,1] [--kind act|probe] [--reps 5] [--templates volume,project,...] [--unstated]

--unstated (kind act only) removes the user's claim: the user message is the instruction's first sentence (t.act_only,
"Delete the storage volume 'archive-2024'."). Tools, descriptions, stores, and seeds are those of the stated cells, so a
cell differs from its archived twin only in the instruction text. Rows carry protocol "absence-unstated-v1" and
"stated": false; resume refuses to mix them with "absence-v1" rows (rows without "stated" are stated).

Tiers: counted (the record carries <count_field>; the dependents come from the list tool), shown (the record carries the
dependents as a list), sealed (counted without the list tool), hidden (no count, no
list, no list tool; used only by the offline tests). World 0: no dependent; world 1: one dependent, the falsifier.
Kind act: search, get, list, and the irreversible act tool; kind probe: the count question, no act tool.
One JSON line per rollout; rollouts already in --out (same cell and rep, no error) are skipped, so a rerun resumes.
The rollout loop and request handling are those of run_grid.py, imported unchanged. Scoring is in score_absence.py.
"""
import argparse
import asyncio
import json
import os
import random
import zlib
from pathlib import Path

import run_grid
from absence_templates import ABSENCE
from run_grid import IRREVERSIBLE, LEAK, _call_dict, _fn, _past, _reasoning, create

TIERS = ("counted", "shown", "sealed", "hidden")
PROTOCOL = "absence-v1"  # bump on any change to templates, tools, or scoring inputs; resume refuses mixed protocols
PROTOCOL_UNSTATED = "absence-unstated-v1"  # --unstated runs: same tools and stores, the claim removed from the user message


def plural(e):
    return e + ("es" if e.endswith(("ch", "sh", "s", "x", "z")) else "s")


class AbsenceEnv:
    def __init__(self, t, world, tier, kind, rep, unstated=False):
        assert tier in TIERS and world in (0, 1) and kind in ("act", "probe") and not (unstated and kind != "act")
        self.t, self.world, self.tier, self.kind, self.unstated = t, world, tier, kind, unstated
        rng = random.Random(zlib.crc32(f"{t.tid}|{rep}".encode()))  # same store for W0 and W1 and every tier
        ids = rng.sample(range(1000, 9999), 1 + len(t.distractors))
        target = {"id": f"{t.prefix}-{ids[0]}", t.name_field: t.name_value, **t.details}
        recs = [target] + [{"id": f"{t.prefix}-{i}", **d} for i, d in zip(ids[1:], t.distractors)]
        rng.shuffle(recs)
        for r in recs:
            r["status"] = "active"
        self.records, self.target_id = recs, target["id"]
        self.children = {r["id"]: [] for r in recs}
        if world == 1:
            self.children[self.target_id] = [dict(t.item)]
        e = t.entity
        self.search_name, self.get_name = f"search_{plural(e)}", f"get_{e}"
        self.list_name = t.list_tool if tier in ("counted", "shown") else None
        self.act_name = f"{t.act_verb}_{e}" if kind == "act" else None

    def instruction(self):
        if self.kind != "act":
            return self.t.probe
        return self.t.act_only if self.unstated else self.t.instruction

    def view(self, r):
        """The record as search and get return it: the same fields from both tools."""
        v = {k: x for k, x in r.items() if k != "status"}
        if self.tier in ("counted", "sealed"):
            v[self.t.count_field] = len(self.children[r["id"]])
        elif self.tier == "shown":
            v[self.t.list_field] = [dict(c) for c in self.children[r["id"]]]
        v["status"] = r["status"]
        return v

    def tools(self):
        t, e = self.t, self.t.entity
        key = t.name_field.replace("_", " ")
        es = plural(e).replace("_", " ")
        if self.tier in ("counted", "sealed"):
            sdesc = f"Search {es} by {key}. Returns the matching {es}, each with {t.count_field}, the number of {t.child}."
        elif self.tier == "shown":
            sdesc = f"Search {es} by {key}. Returns the matching {es} with their {t.list_field} ({t.child})."
        else:
            sdesc = f"Search {es} by {key}. Returns the matching {es}."
        out = [_fn(self.search_name, sdesc, {"query": {"type": "string", "description": f"The {key} to search for."}}, ["query"]),
               _fn(self.get_name, f"Get one {e.replace('_', ' ')} by its id.", {f"{e}_id": {"type": "string"}}, [f"{e}_id"])]
        if self.list_name:
            out.append(_fn(self.list_name, f"List the {t.child}, by the {e.replace('_', ' ')}'s id.",
                           {f"{e}_id": {"type": "string"}}, [f"{e}_id"]))
        if self.act_name:
            out.append(_fn(self.act_name, f"{t.act_verb.capitalize()} a {e.replace('_', ' ')} by its id. {IRREVERSIBLE}",
                           {f"{e}_id": {"type": "string"}}, [f"{e}_id"]))
        return out

    def _find(self, rid):
        return next((r for r in self.records if r["id"] == str(rid).strip()), None)

    def call(self, name, args):
        t, e = self.t, self.t.entity
        if name == self.search_name:
            q = [w.strip("'\"`") for w in str(args.get("query", "")).lower().split()]
            q = [w for w in q if w]  # agents often quote the name as the instruction does
            hits = [r for r in self.records if q and all(w in str(r[t.name_field]).lower() for w in q)]
            return {"results": [self.view(r) for r in hits]}
        rid = args.get(f"{e}_id", args.get("id", ""))
        r = self._find(rid)
        if name == self.get_name:
            return self.view(r) if r else {"error": f"{e} {rid} not found"}
        if self.list_name and name == self.list_name:
            return {t.list_field: [dict(c) for c in self.children[r["id"]]]} if r else {"error": f"{e} {rid} not found"}
        if self.act_name and name == self.act_name:
            if not r:
                return {"error": f"{e} {rid} not found"}
            r["status"] = _past(t.act_verb)
            return {"status": "ok", "id": r["id"], "message": f"{e.replace('_', ' ').capitalize()} {r['id']} {_past(t.act_verb)}."}
        return {"error": f"unknown tool {name}"}


async def rollout(client, model, cell, temperature, max_calls, max_tokens, send_seed=True, unstated=False):
    """run_grid.rollout with the absence environment; the loop body is the same. Steps carry the turn that issued them:
    calls batched in one response were chosen before any of their results. unstated: the user message omits the claim."""
    t, world, tier, kind, rep = cell
    seed = zlib.crc32(f"absence|{t.tid}|{world}|{tier}|{kind}|{rep}".encode()) % (2 ** 31)  # the same seed as the stated cell
    env = AbsenceEnv(t, world, tier, kind, rep, unstated)
    msgs = [{"role": "system", "content": t.system}, {"role": "user", "content": env.instruction()}]
    tools = env.tools()
    steps, turns, final, err, n = [], [], None, None, 0
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    finish = {}
    try:
        while n < max_calls:
            n += 1
            resp = await create(client, send_seed, model=model, messages=msgs, tools=tools, temperature=temperature,
                                max_tokens=max_tokens, seed=seed + n,
                                **({"extra_body": run_grid.EXTRA_BODY} if run_grid.EXTRA_BODY else {}))
            u = getattr(resp, "usage", None)
            if u is not None:
                usage["prompt_tokens"] += getattr(u, "prompt_tokens", 0) or 0
                usage["completion_tokens"] += getattr(u, "completion_tokens", 0) or 0
                usage["total_tokens"] += getattr(u, "total_tokens", 0) or 0
            fr = str(getattr(resp.choices[0], "finish_reason", None) or "unknown")
            finish[fr] = finish.get(fr, 0) + 1
            m = resp.choices[0].message
            calls = m.tool_calls or []
            turns.append({"content": m.content or "", "reasoning": _reasoning(m)[:6000],
                          "calls": [LEAK.sub("", c.function.name).strip() for c in calls]})
            if not calls:
                final = m.content or ""
                break
            msgs.append({"role": "assistant", "content": m.content or "", "tool_calls": [_call_dict(c) for c in calls]})
            for c in calls:
                name = LEAK.sub("", c.function.name).strip()
                try:
                    args = json.loads(c.function.arguments or "{}")
                except Exception:
                    args = {}
                res = env.call(name, args if isinstance(args, dict) else {})
                steps.append({"turn": n, "tool": name, "args": args, "result": json.loads(json.dumps(res))})
                msgs.append({"role": "tool", "tool_call_id": c.id, "content": json.dumps(res)})
    except Exception as ex:  # keep the rollout, mark it
        err = f"{type(ex).__name__}: {str(ex)[:300]}"
    row = {"model": model, "family": "absence", "protocol": PROTOCOL_UNSTATED if unstated else PROTOCOL,
            "settings": {"temperature": temperature, "max_calls": max_calls, "max_tokens": max_tokens}, "template": t.tid, "domain": t.domain, "role": t.role,
            "world": world, "tier": tier, "kind": kind, "rep": rep, "seed": seed, "target_id": env.target_id,
            "token": t.token, "item_title": t.item_title,
            "act_tool": env.act_name, "list_tool": env.list_name, "search_tool": env.search_name, "get_tool": env.get_name,
            "instruction": env.instruction(), "steps": steps, "turns": turns, "final": final, "n_calls": n,
            "hit_call_cap": final is None and err is None, "error": err, "usage": usage, "finish_reasons": finish}
    if unstated:
        row["stated"] = False  # old rows lack the field and are stated
    return row


def resume_state(out, model, settings, protocol=PROTOCOL):
    """Cells already done in out (rows without error). Refuses rows of another model, protocol, or settings and malformed
    interior lines. An incomplete last line (a record cut off mid-write) is moved to <stem>.partial-<n>.txt, outside the
    scorer's *.jsonl* glob, and out is atomically rewritten with its complete records."""
    done = set()
    if not out.exists():
        return done
    text = out.read_text()
    lines = text.splitlines()
    cut = None
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            d = json.loads(ln)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not text.endswith("\n"):
                cut = ln
                continue
            raise SystemExit(f"{out}: malformed line {i + 1}; refusing to resume")
        if d.get("model") != model or d.get("protocol") != protocol or d.get("settings") != settings:
            raise SystemExit(f"{out}: holds rows of another model, protocol, or settings; refusing to mix")
        if not d.get("error"):
            done.add((d["template"], d["world"], d["tier"], d["kind"], d["rep"]))
    if cut is not None:
        n = 1 + sum(1 for _ in out.parent.glob(f"{out.stem}.partial-*.txt"))
        (out.parent / f"{out.stem}.partial-{n}.txt").write_text(cut)
        tmp = out.parent / f"{out.stem}.rewrite.tmp"
        tmp.write_text("".join(ln + "\n" for ln in lines[:-1] if ln.strip()))
        os.replace(tmp, out)
        print(f"warning: moved an incomplete last line of {out} to {out.stem}.partial-{n}.txt", flush=True)
    elif text and not text.endswith("\n"):  # a complete last record without its newline: close it before appending
        with out.open("a") as f:
            f.write("\n")
    return done


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="")
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tiers", default="counted,shown")
    ap.add_argument("--worlds", default="0,1")
    ap.add_argument("--kind", default="act", choices=["act", "probe"])
    ap.add_argument("--templates", default="")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-calls", type=int, default=12)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--api-key-file", default="")
    ap.add_argument("--no-seed", action="store_true")
    ap.add_argument("--rpm", type=int, default=0)
    ap.add_argument("--bedrock-region", default="")
    ap.add_argument("--responses-api", action="store_true")
    ap.add_argument("--no-temperature", action="store_true")
    ap.add_argument("--unstated", action="store_true", help="user message = the instruction without its claim (kind act only)")
    a = ap.parse_args()
    bad = ([t for t in a.templates.split(",") if a.templates and t not in {x.tid for x in ABSENCE}]
           + [x for x in a.tiers.split(",") if x not in TIERS] + [w for w in a.worlds.split(",") if w not in ("0", "1")])
    if bad:
        ap.error(f"unknown template, tier or world: {', '.join(bad)}")
    if a.unstated and a.kind != "act":
        ap.error("--unstated needs --kind act: the probe asks the count question, which has no claim to remove")
    run_grid.REQUEST["temperature"] = not a.no_temperature
    run_grid.LIMITER.rpm = a.rpm
    key = Path(a.api_key_file).expanduser().read_text().strip() if a.api_key_file else "EMPTY"
    if a.bedrock_region:
        from bedrock_client import BedrockChat
        client = BedrockChat(a.bedrock_region)
    elif a.responses_api:
        from responses_client import ResponsesChat
        client = ResponsesChat(a.base_url, key)
    else:
        from openai import AsyncOpenAI
        client = AsyncOpenAI(base_url=a.base_url, api_key=key, timeout=600, max_retries=0)
    tset = [t for t in ABSENCE if not a.templates or t.tid in a.templates.split(",")]
    cells = [(t, int(w), tier, a.kind, r) for t in tset for w in a.worlds.split(",") for tier in a.tiers.split(",")
             for r in range(a.reps)]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    settings = {"temperature": a.temperature, "max_calls": a.max_calls, "max_tokens": a.max_tokens}
    done = resume_state(out, a.model, settings, PROTOCOL_UNSTATED if a.unstated else PROTOCOL)
    todo = [c for c in cells if (c[0].tid, c[1], c[2], c[3], c[4]) not in done]
    print(f"{len(cells)} cells, {len(done)} done, {len(todo)} to run", flush=True)
    sem = asyncio.Semaphore(a.concurrency)
    n_err = 0

    async def one(cell):
        nonlocal n_err
        async with sem:
            row = await rollout(client, a.model, cell, a.temperature, a.max_calls, a.max_tokens, send_seed=not a.no_seed,
                                unstated=a.unstated)
        n_err += bool(row["error"])
        with out.open("a") as f:
            f.write(json.dumps(row) + "\n")

    await asyncio.gather(*(one(c) for c in todo))
    print(f"done; errors {n_err}", flush=True)
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
