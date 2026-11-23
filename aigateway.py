"""AI Gateway Lite (free core): call any OpenAI-compatible LLM API with retries and provider fallback.

Standard library only. Python 3.9+. MIT licensed.
Works with Google Gemini (free tier), Groq, OpenRouter, local Ollama and anything else that speaks the OpenAI chat format.
"""
from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

__all__ = ["Provider", "Gateway", "Result", "GatewayError", "from_env"]


class GatewayError(RuntimeError):
    """All providers failed, or the request was invalid. Messages never contain API keys."""


@dataclass
class Provider:
    name: str
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 60.0
    extra_headers: dict = field(default_factory=dict)

    def redact(self, text: str) -> str:
        return text.replace(self.api_key, "***") if self.api_key else text


@dataclass
class Result:
    text: str
    provider: str
    model: str
    latency_s: float = 0.0
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


class _Retryable(Exception):
    def __init__(self, msg: str, retry_after: Optional[float] = None):
        super().__init__(msg)
        self.retry_after = retry_after


class Gateway:
    """Try providers in order. Retry temporary errors with backoff; skip a failing provider for `cooldown` seconds."""

    def __init__(self, providers: list[Provider], retries: int = 2, backoff: float = 1.0, cooldown: float = 30.0,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic):
        if not providers:
            raise ValueError("Give at least one provider")
        self.providers, self.retries, self.backoff, self.cooldown = providers, retries, backoff, cooldown
        self.sleep, self.clock, self.down_until = sleep, clock, {}

    def chat(self, prompt, system: Optional[str] = None, temperature: float = 0.0, max_tokens: int = 1024) -> Result:
        """prompt: a string, or a list of {"role": ..., "content": ...} messages."""
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else list(prompt)
        now = self.clock()
        live = [p for p in self.providers if self.down_until.get(p.name, 0) <= now]
        errors = []
        for p in (live or self.providers):  # if all are cooling down, try them all anyway
            try:
                r = self._try(p, messages, system, temperature, max_tokens)
                self.down_until.pop(p.name, None)
                return r
            except GatewayError as e:
                errors.append(f"{p.name}: {e}")
                self.down_until[p.name] = self.clock() + self.cooldown
        raise GatewayError("All providers failed. " + " | ".join(errors))

    def _try(self, p: Provider, messages, system, temperature, max_tokens) -> Result:
        last = ""
        for attempt in range(self.retries + 1):
            t0 = time.monotonic()
            try:
                text, pt, ct = self._call(p, messages, system, temperature, max_tokens)
                return Result(text, p.name, p.model, round(time.monotonic() - t0, 3), pt, ct)
            except _Retryable as e:
                last = str(e)
                if attempt < self.retries:
                    delay = e.retry_after if e.retry_after is not None else self.backoff * (2 ** attempt) * (0.5 + random.random() / 2)
                    self.sleep(min(delay, 60))
        raise GatewayError(f"gave up after {self.retries + 1} tries ({last})")

    def _call(self, p: Provider, messages, system, temperature, max_tokens):
        body = {"model": p.model, "max_tokens": max_tokens, "temperature": temperature,
                "messages": ([{"role": "system", "content": system}] if system else []) + messages}
        headers = {"Authorization": "Bearer " + p.api_key} if p.api_key else {}
        req = urllib.request.Request(p.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers, **p.extra_headers})
        try:
            with urllib.request.urlopen(req, timeout=p.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = p.redact(e.read().decode("utf-8", "replace")[:200])
            if e.code in (408, 409, 425, 429) or e.code >= 500:
                ra = e.headers.get("Retry-After")
                raise _Retryable(f"HTTP {e.code} {detail}", float(ra) if ra and ra.replace(".", "", 1).isdigit() else None) from None
            raise GatewayError(f"HTTP {e.code} (not retried): {detail}") from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise _Retryable(p.redact(f"network error: {e}")) from None
        except json.JSONDecodeError:
            raise _Retryable("provider returned invalid JSON") from None
        try:
            text = payload["choices"][0]["message"]["content"] or ""
            u = payload.get("usage", {})
            return text, u.get("prompt_tokens"), u.get("completion_tokens")
        except (KeyError, IndexError, TypeError):
            raise GatewayError("unexpected response shape from provider") from None


def from_env(prefix: str = "AI_") -> Provider:
    """Provider from AI_BASE_URL, AI_MODEL and AI_API_KEY (the key is optional for local servers)."""
    return Provider("env", os.environ[prefix + "BASE_URL"], os.environ[prefix + "MODEL"], os.environ.get(prefix + "API_KEY", ""))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print('usage: AI_BASE_URL=... AI_MODEL=... AI_API_KEY=... python aigateway.py "your prompt"')
        raise SystemExit(2)
    try:
        print(Gateway([from_env()]).chat(" ".join(sys.argv[1:])).text)
    except KeyError as e:
        raise SystemExit(f"Set the environment variable {e}")
    except GatewayError as e:
        raise SystemExit(f"Failed: {e}")
