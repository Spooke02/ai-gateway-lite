import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

import aigateway as ag


class Fake:
    """Scriptable local API server: plan[prefix] is a list of (status, headers, body) consumed in order; the last repeats."""
    def __init__(self):
        self.plan, self.requests = {}, []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])).decode())
                outer.requests.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "body": body})
                steps = outer.plan[self.path.split("/v1")[0]]
                n = sum(1 for r in outer.requests if r["path"] == self.path)
                status, headers, payload = steps[min(n - 1, len(steps) - 1)]
                raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
                self.send_response(status)
                for k, v in headers.items(): self.send_header(k, v)
                self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.srv.server_address[1]}"
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def stop(self): self.srv.shutdown()


def ok(text, pt=5, ct=7):
    return (200, {}, {"choices": [{"message": {"content": text}}], "usage": {"prompt_tokens": pt, "completion_tokens": ct}})


@pytest.fixture
def fake():
    f = Fake(); yield f; f.stop()


def prov(fake, prefix, name="a", key="SECRET-KEY-123"):
    return ag.Provider(name, fake.url + prefix, "m1", key)


def calls(fake, prefix):
    return sum(1 for q in fake.requests if q["path"].startswith(prefix))


def test_success_sends_auth_system_and_parses_usage(fake):
    fake.plan["/a"] = [ok("hello")]
    r = ag.Gateway([prov(fake, "/a/v1")]).chat("hi", system="be brief")
    assert r.text == "hello" and r.prompt_tokens == 5 and r.completion_tokens == 7 and r.provider == "a"
    req = fake.requests[0]
    assert req["headers"]["authorization"] == "Bearer SECRET-KEY-123"
    assert req["body"]["messages"][0] == {"role": "system", "content": "be brief"} and req["body"]["model"] == "m1"


def test_429_honours_retry_after_then_succeeds(fake):
    fake.plan["/a"] = [(429, {"Retry-After": "3"}, {"error": "slow"}), ok("done")]
    slept = []
    assert ag.Gateway([prov(fake, "/a/v1")], sleep=slept.append).chat("hi").text == "done" and slept == [3.0]


def test_5xx_falls_back_to_next_provider(fake):
    fake.plan["/a"] = [(503, {}, {"e": 1})]; fake.plan["/b"] = [ok("from b")]
    g = ag.Gateway([prov(fake, "/a/v1", "a"), prov(fake, "/b/v1", "b")], retries=1, sleep=lambda s: None)
    r = g.chat("hi")
    assert r.provider == "b" and calls(fake, "/a") == 2


def test_400_not_retried_and_key_redacted(fake):
    fake.plan["/a"] = [(400, {}, {"error": "bad key SECRET-KEY-123 rejected"})]
    with pytest.raises(ag.GatewayError) as e:
        ag.Gateway([prov(fake, "/a/v1")], sleep=lambda s: None).chat("hi")
    assert len(fake.requests) == 1 and "SECRET-KEY-123" not in str(e.value) and "***" in str(e.value)


def test_all_fail_aggregates_errors(fake):
    fake.plan["/a"] = [(500, {}, {})]; fake.plan["/b"] = [(500, {}, {})]
    with pytest.raises(ag.GatewayError) as e:
        ag.Gateway([prov(fake, "/a/v1", "a"), prov(fake, "/b/v1", "b")], retries=0, sleep=lambda s: None).chat("x")
    assert "a:" in str(e.value) and "b:" in str(e.value)


def test_cooldown_skips_a_failed_provider(fake):
    fake.plan["/a"] = [(500, {}, {})]; fake.plan["/b"] = [ok("b")]
    now = [0.0]
    g = ag.Gateway([prov(fake, "/a/v1", "a"), prov(fake, "/b/v1", "b")], retries=0, sleep=lambda s: None, cooldown=30, clock=lambda: now[0])
    g.chat("q1"); a1 = calls(fake, "/a")
    g.chat("q2"); assert calls(fake, "/a") == a1          # skipped during cooldown
    now[0] = 31; g.chat("q3"); assert calls(fake, "/a") == a1 + 1


def test_invalid_json_from_server_is_retried(fake):
    fake.plan["/a"] = [(200, {}, b"<html>oops</html>"), ok("fine")]
    assert ag.Gateway([prov(fake, "/a/v1")], sleep=lambda s: None).chat("x").text == "fine"


def test_unexpected_shape_is_an_error(fake):
    fake.plan["/a"] = [(200, {}, {"weird": True})]
    with pytest.raises(ag.GatewayError):
        ag.Gateway([prov(fake, "/a/v1")]).chat("x")


def test_connection_refused_is_retried_then_fails():
    with pytest.raises(ag.GatewayError):
        ag.Gateway([ag.Provider("dead", "http://127.0.0.1:9/v1", "m", "k")], retries=1, sleep=lambda s: None).chat("x")


def test_requires_a_provider():
    with pytest.raises(ValueError):
        ag.Gateway([])
