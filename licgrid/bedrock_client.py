"""Amazon Bedrock (Converse API) behind the small slice of the OpenAI chat-completions interface that run_grid.py uses.

  client = BedrockChat(region="us-east-1")
  resp = await client.chat.completions.create(model=..., messages=[...], tools=[...], temperature=..., max_tokens=...)

Credentials come from the default AWS chain (~/.aws) or AWS_BEARER_TOKEN_BEDROCK. Throttling and server errors are raised with
an HTTP-like status_code so run_grid.create() backs off and retries them.
"""
import asyncio
import json
from types import SimpleNamespace as NS

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

NO_TEMPERATURE = set()  # models that reject inferenceConfig.temperature (filled on the first refusal)
RETRYABLE = {"ThrottlingException": 429, "ServiceUnavailableException": 503, "ModelNotReadyException": 503,
             "InternalServerException": 500, "ModelTimeoutException": 408}


class BedrockError(Exception):
    def __init__(self, msg, status_code=None):
        super().__init__(msg)
        self.status_code = status_code


class _Call:
    """A tool call shaped like openai's ChatCompletionMessageToolCall (id, function.name/arguments, model_dump)."""

    def __init__(self, cid, name, arguments):
        self.id, self.type = cid, "function"
        self.function = NS(name=name, arguments=arguments)

    def model_dump(self, exclude_none=True):
        return {"id": self.id, "type": "function", "function": {"name": self.function.name,
                                                                "arguments": self.function.arguments}}


def _to_converse(messages):
    system, out = [], []
    for m in messages:
        role = m["role"]
        if role == "system":
            system.append({"text": m["content"]})
        elif role == "user":
            out.append({"role": "user", "content": [{"text": m["content"]}]})
        elif role == "assistant":
            blocks = [{"text": m["content"]}] if m.get("content") else []
            for c in m.get("tool_calls") or []:
                try:
                    args = json.loads(c["function"]["arguments"] or "{}")
                except Exception:
                    args = {}
                blocks.append({"toolUse": {"toolUseId": c["id"], "name": c["function"]["name"],
                                           "input": args if isinstance(args, dict) else {}}})
            out.append({"role": "assistant", "content": blocks or [{"text": " "}]})
        elif role == "tool":
            try:
                body = json.loads(m["content"])
                content = [{"json": body}] if isinstance(body, dict) else [{"text": m["content"]}]
            except Exception:
                content = [{"text": m["content"]}]
            block = {"toolResult": {"toolUseId": m["tool_call_id"], "content": content}}
            if out and out[-1]["role"] == "user" and any("toolResult" in b for b in out[-1]["content"]):
                out[-1]["content"].append(block)  # consecutive tool results go in one user turn
            else:
                out.append({"role": "user", "content": [block]})
    return system, out


class _Completions:
    def __init__(self, region):
        self.rt = boto3.client("bedrock-runtime", region_name=region,
                               config=Config(read_timeout=600, retries={"max_attempts": 1}))

    def _call(self, model, messages, tools, temperature, max_tokens):
        system, msgs = _to_converse(messages)
        cfg = {"maxTokens": max_tokens}
        if model not in NO_TEMPERATURE:
            cfg["temperature"] = temperature
        kw = {"modelId": model, "messages": msgs, "inferenceConfig": cfg}
        if system:
            kw["system"] = system
        if tools:
            kw["toolConfig"] = {"tools": [{"toolSpec": {"name": t["function"]["name"],
                                                        "description": t["function"]["description"],
                                                        "inputSchema": {"json": t["function"]["parameters"]}}}
                                          for t in tools]}
        try:
            return self.rt.converse(**kw)
        except ClientError as ex:
            code = ex.response.get("Error", {}).get("Code", "")
            if "support the temperature field" in str(ex) and "temperature" in cfg:
                NO_TEMPERATURE.add(model)  # e.g. Kimi K3: the model runs at its own default temperature
                return self._call(model, messages, tools, temperature, max_tokens)
            raise BedrockError(f"{code}: {ex}", RETRYABLE.get(code)) from None

    async def create(self, model, messages, tools=None, temperature=0.7, max_tokens=4096, seed=None, **_):
        return await asyncio.to_thread(self.create_sync, model, messages, tools, temperature, max_tokens)

    def create_sync(self, model, messages, tools=None, temperature=0.7, max_tokens=4096, **_):
        r = self._call(model, messages, tools, temperature, max_tokens)
        texts, calls, thoughts = [], [], []
        for b in r["output"]["message"]["content"]:
            if "text" in b:
                texts.append(b["text"])
            elif "reasoningContent" in b:
                thoughts.append((b["reasoningContent"].get("reasoningText") or {}).get("text", ""))
            elif "toolUse" in b:
                tu = b["toolUse"]
                calls.append(_Call(tu["toolUseId"], tu["name"], json.dumps(tu.get("input") or {})))
        u = r.get("usage", {})
        usage = NS(prompt_tokens=u.get("inputTokens", 0), completion_tokens=u.get("outputTokens", 0),
                   total_tokens=u.get("totalTokens", 0))
        msg = NS(content="\n".join(texts) if texts else None, tool_calls=calls or None, reasoning="\n".join(thoughts))
        return NS(choices=[NS(message=msg, finish_reason=r.get("stopReason"))], usage=usage)


class BedrockChat:
    def __init__(self, region="us-east-1"):
        self.chat = NS(completions=_Completions(region))
