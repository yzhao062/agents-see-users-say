"""The OpenAI Responses API behind the slice of the chat-completions interface that run_grid.py uses.

Azure's GPT reasoning models (gpt-6-astra) refuse function tools with reasoning on /v1/chat/completions, so they run
through /v1/responses. Requests are stateless (store=False); each response's output items, including the encrypted
reasoning, ride back into the history on the first tool call (key "responses_items") and are sent again verbatim on
the next turn, so the model keeps its reasoning across tool calls. Reasoning summaries ("summary": "auto") give the
in-episode text. Temperature is not sent (these models accept only the default); seed is not sent.

  client = ResponsesChat(base_url, api_key)
  resp = await client.chat.completions.create(model=..., messages=[...], tools=[...], max_tokens=...,
                                              extra_body={"reasoning_effort": "high"})  # effort optional
"""
import json
from types import SimpleNamespace as NS

from openai import AsyncOpenAI, OpenAI


class _Call:
    """A tool call shaped like openai's ChatCompletionMessageToolCall, carrying the response's output items."""

    def __init__(self, cid, name, arguments, items=None):
        self.id, self.type = cid, "function"
        self.function = NS(name=name, arguments=arguments)
        self.items = items

    def model_dump(self, exclude_none=True):
        d = {"id": self.id, "type": "function", "function": {"name": self.function.name, "arguments": self.function.arguments}}
        if self.items is not None:
            d["responses_items"] = self.items
        return d


def _tools(tools):
    return [{"type": "function", "name": t["function"]["name"], "description": t["function"].get("description", ""),
             "parameters": t["function"]["parameters"]} for t in tools or []]


def _to_input(messages):
    # a turn's function calls go back only if a tool result for them follows (tau-bench executes the first call only)
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    out = []
    for m in messages:
        role = m["role"]
        if role in ("system", "user"):
            out.append({"role": role, "content": m["content"]})
        elif role == "assistant":
            calls = m.get("tool_calls") or []
            items = calls[0].get("responses_items") if calls else None
            if items is not None:  # reasoning (encrypted), any message, and the answered function calls of that turn
                out.extend(it for it in items if it.get("type") != "function_call" or it.get("call_id") in answered)
            else:
                if m.get("content"):
                    out.append({"role": "assistant", "content": m["content"]})
                for c in calls:
                    out.append({"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                                "arguments": c["function"]["arguments"] or "{}"})
        elif role == "tool":
            out.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
    return out


def _request(model, messages, tools, max_tokens, extra_body):
    eff = (extra_body or {}).get("reasoning_effort")
    return dict(model=model, input=_to_input(messages), tools=_tools(tools), max_output_tokens=max_tokens, store=False,
                reasoning={"summary": "auto", **({"effort": eff} if eff else {})}, include=["reasoning.encrypted_content"])


class _Completions:
    def __init__(self, base_url, api_key):
        self.c = AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=600, max_retries=0)
        self.s = OpenAI(base_url=base_url, api_key=api_key, timeout=600, max_retries=0)  # for threaded callers (run_tau)

    async def create(self, model, messages, tools=None, max_tokens=4096, extra_body=None, **_):
        return _parse(await self.c.responses.create(**_request(model, messages, tools, max_tokens, extra_body)))

    def create_sync(self, model, messages, tools=None, max_tokens=4096, extra_body=None, **_):
        return _parse(self.s.responses.create(**_request(model, messages, tools, max_tokens, extra_body)))


def _parse(r):
    """A Responses result as a chat-completions response; output items ride on the first tool call."""
    items = [it.model_dump(exclude_none=True) for it in r.output]
    texts, thoughts, calls = [], [], []
    for it in r.output:
        if it.type == "message":
            texts += [p.text for p in it.content if getattr(p, "type", "") == "output_text"]
        elif it.type == "reasoning":
            thoughts += [s.text for s in (it.summary or []) if getattr(s, "text", None)]
        elif it.type == "function_call":
            calls.append(_Call(it.call_id, it.name, it.arguments, items if not calls else None))
    u = r.usage
    usage = NS(prompt_tokens=getattr(u, "input_tokens", 0) or 0, completion_tokens=getattr(u, "output_tokens", 0) or 0,
               total_tokens=getattr(u, "total_tokens", 0) or 0)
    fin = "length" if r.status == "incomplete" else ("tool_calls" if calls else "stop")
    msg = NS(content="\n".join(texts) if texts else None, tool_calls=calls or None, reasoning="\n".join(thoughts))
    return NS(choices=[NS(message=msg, finish_reason=fin)], usage=usage)


class ResponsesChat:
    def __init__(self, base_url, api_key):
        self.chat = NS(completions=_Completions(base_url, api_key))
