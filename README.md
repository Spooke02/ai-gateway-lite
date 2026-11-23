# AI Gateway Lite

A tiny, dependency-free Python library for calling LLM APIs **reliably**: provider fallback, retries with backoff, and clear errors. One file, standard library only, Python 3.9+, MIT licensed.

It works with anything that speaks the OpenAI chat format: **Google Gemini (free tier), Groq, OpenRouter, local Ollama**, and more.

```python
from aigateway import Gateway, Provider

gw = Gateway([
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai", "<model>", "<KEY>"),
    Provider("local",  "http://127.0.0.1:11434/v1", "<ollama-model>"),     # fallback, no key needed
])
r = gw.chat("Explain caching in one sentence.", system="Be brief.")
print(r.text, r.provider, r.latency_s, r.prompt_tokens, r.completion_tokens)
```

## What you get

- **Fallback chain:** if provider A is down or rate-limited, it tries provider B. A failing provider is skipped for 30 seconds (`cooldown`) so later calls don't wait on it again.
- **Retries done properly:** exponential backoff with jitter, honours the `Retry-After` header, and never retries mistakes (400 bad request, 401 wrong key).
- **Safe errors:** API keys are removed from error messages.
- **Zero dependencies**, about 120 lines you can read in a few minutes.
- **10 tests** that run offline against a local fake server (no key needed).

## Quick start

1. Copy `aigateway.py` into your project.
2. Set environment variables (use any OpenAI-compatible service):

```
AI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
AI_MODEL=<a model name from your provider's dashboard>
AI_API_KEY=<your key>
```

3. Run it:

```
python aigateway.py "Say hello in five words"
```

or in code: `Gateway([from_env()]).chat("...")`.

## API

```python
Provider(name, base_url, model, api_key="", timeout=60.0, extra_headers={})
Gateway(providers, retries=2, backoff=1.0, cooldown=30.0)
Gateway.chat(prompt, system=None, temperature=0.0, max_tokens=1024) -> Result
Result(text, provider, model, latency_s, prompt_tokens, completion_tokens)
```

`prompt` can be a string or a list of `{"role": ..., "content": ...}` messages. If every provider fails, `GatewayError` lists each provider's error.

## Tests

```
pip install pytest
pytest
```

## Free core vs. the full version

This repo is the **free core**. The paid package adds the things you usually end up writing next:

| | Free core | **AI Gateway Lite (full)** |
|---|---|---|
| OpenAI-compatible providers, retries, fallback, cooldown | yes | yes |
| Anthropic (Claude) provider | | yes |
| Disk cache (SQLite), so repeat requests cost nothing | | yes |
| Per-provider rate limit (stay inside free-tier quotas) | | yes |
| Validated JSON output with schema check and one repair retry | | yes |
| Usage log (calls, cache hits, tokens, latency) | | yes |
| 5 runnable examples and 26 tests | | yes |
| 2-page AI integration cheat sheet (PDF) | | yes |

Get the full version: https://spooke0.gumroad.com/l/ai-gateway-lite

## Notes and limits

- No streaming, no tool/function calling, no async. For those, use a larger library.
- Model names, URLs and limits change. Confirm them in your provider's documentation.
- Keep keys in environment variables. Never commit them.

## License

MIT. See `LICENSE`.
