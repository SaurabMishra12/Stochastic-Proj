"""Azure Responses HTTP transport with explicit per-attempt budget accounting."""

import json
import os
import random
import socket
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .ledger import DuplicateLogicalCall


class TransportError(RuntimeError):
    def __init__(self, message, *, outcome, attempts=1, result=None):
        super().__init__(message)
        self.outcome = outcome
        self.attempts = attempts
        self.result = result


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


class RateLimiter:
    """Thread-safe request spacing and a rolling 60-second token limit."""

    def __init__(self, rpm=10, tpm=None, *, clock=time.monotonic, sleep=time.sleep):
        for value, name in ((rpm, "rpm"), (tpm, "tpm")):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
                raise ValueError(f"{name} must be a positive integer or None")
        self.rpm = rpm
        self.tpm = tpm
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()
        self.next_request = 0.
        self.events = []

    def acquire(self, token_bound):
        if isinstance(token_bound, bool) or not isinstance(token_bound, int) or token_bound < 0:
            raise ValueError("token bound must be a nonnegative integer")
        if self.tpm is not None and token_bound > self.tpm:
            raise ValueError("request token bound exceeds the per-minute token quota")
        with self.lock:
            while True:
                now = self.clock()
                self.events = [(stamp, tokens) for stamp, tokens in self.events if stamp > now - 60]
                wait = max(0, self.next_request - now)
                if self.tpm is not None and sum(tokens for _, tokens in self.events) + token_bound > self.tpm:
                    wait = max(wait, self.events[0][0] + 60 - now)
                if wait <= 0:
                    break
                self.sleep(wait)
            self.next_request = now + (60 / self.rpm if self.rpm is not None else 0)
            if self.tpm is not None:
                self.events.append((now, token_bound))


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _endpoint(base_url):
    parsed = urlsplit(base_url)
    host = (parsed.hostname or "").lower()
    suffixes = (".openai.azure.com", ".cognitiveservices.azure.com", ".services.ai.azure.com",
                ".openai.azure.us", ".openai.azure.cn")
    if (parsed.scheme != "https" or not any(host.endswith(s) for s in suffixes)
            or parsed.username is not None or parsed.password is not None
            or parsed.port not in (None, 443) or parsed.query or parsed.fragment):
        raise ValueError("base_url must be a credential-free HTTPS Azure endpoint")
    path = parsed.path.rstrip("/")
    if not path:
        path = "/openai/v1"
    if path != "/openai/v1":
        raise ValueError("base_url must use the Azure /openai/v1/ endpoint")
    return urlunsplit(("https", parsed.netloc, path + "/responses", "", ""))


def _redact(value, secret):
    if isinstance(value, str):
        return value.replace(secret, "[REDACTED]") if secret else value
    if isinstance(value, dict):
        return {_redact(k, secret): _redact(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def _atomic_json(path, value):
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".attempt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _valid_usage(usage):
    if not isinstance(usage, dict):
        return None
    if any(isinstance(usage.get(key), bool) or not isinstance(usage.get(key), int)
           or usage[key] < 0 for key in ("input_tokens", "output_tokens")):
        return None
    return usage


def _extract_text(raw):
    parts = []
    refused = False
    output = raw.get("output", [])
    if not isinstance(output, list):
        return "", False
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content", [])
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "refusal":
                refused = True
            if part.get("type") == "output_text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
    return "\n".join(parts), refused


def _content_filter(value):
    if isinstance(value, dict):
        code = value.get("code")
        if isinstance(code, str) and code.lower() in {"content_filter", "responsibleaipolicyviolation"}:
            return True
        return any(_content_filter(item) for item in value.values())
    if isinstance(value, list):
        return any(_content_filter(item) for item in value)
    return False


class AzureResponsesTransport:
    """Send no automatic retries after an ambiguous timeout or connection error.

    Only 429/500/502/503/504 may be retried, with a fresh durable reservation
    for every attempt. urllib has no implicit retry mechanism, and redirects
    are disabled so an API credential cannot follow a redirect to another host.
    """

    def __init__(self, base_url, deployment, key_loader, ledger, output_dir, *,
                 max_output_tokens=4096, max_input_token_bound=12288, opener=None,
                 sleep=time.sleep, max_attempts=3, timeout=120, max_retry_delay=30,
                 rpm=10, tpm=None, rate_limiter=None,
                 expected_models=("gpt-5-mini", "gpt-5-mini-2025-08-07")):
        self.endpoint = _endpoint(base_url)
        if not isinstance(deployment, str) or not deployment.strip():
            raise ValueError("deployment must be a nonempty string")
        for value, name in ((max_output_tokens, "max_output_tokens"),
                            (max_input_token_bound, "max_input_token_bound"),
                            (max_attempts, "max_attempts")):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_attempts > 3:
            raise ValueError("at most three HTTP attempts are permitted per logical call")
        if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("timeout must be finite and positive")
        if not isinstance(max_retry_delay, (int, float)) or not 0 <= max_retry_delay <= 60:
            raise ValueError("max_retry_delay must be between zero and 60 seconds")
        self.deployment = deployment
        self.key_loader = key_loader
        self.ledger = ledger
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.output_dir, 0o700)
        self.max_output_tokens = max_output_tokens
        self.max_input_token_bound = max_input_token_bound
        self.opener = opener if opener is not None else build_opener(_NoRedirect())
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.timeout = timeout
        self.max_retry_delay = max_retry_delay
        self.rate_limiter = rate_limiter or RateLimiter(rpm=rpm, tpm=tpm)
        if not expected_models or isinstance(expected_models, str) or any(not isinstance(model, str) or not model for model in expected_models):
            raise ValueError("expected_models must contain verified model identities")
        self.expected_models = frozenset(expected_models)

    def _delay(self, retry_after, attempt):
        delay = None
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                try:
                    date = parsedate_to_datetime(retry_after)
                    if date.tzinfo is None:
                        date = date.replace(tzinfo=timezone.utc)
                    delay = (date - datetime.now(timezone.utc)).total_seconds()
                except (TypeError, ValueError, OverflowError):
                    pass
        if delay is None or not 0 <= delay < float("inf"):
            delay = 2 ** (attempt - 1)
        return min(self.max_retry_delay, max(0, delay) + random.uniform(0, .25))

    def request(self, input, *, instructions=None, logical_id=None, reasoning_effort="low",
                input_token_bound=None):
        if reasoning_effort not in {"minimal", "low", "medium", "high"}:
            raise ValueError("unsupported reasoning effort")
        bound = self.max_input_token_bound if input_token_bound is None else input_token_bound
        if isinstance(bound, bool) or not isinstance(bound, int) or not 0 < bound <= self.max_input_token_bound:
            raise ValueError("input_token_bound exceeds the configured bound or is invalid")
        if not isinstance(input, (str, list)) or not input:
            raise ValueError("input must be a nonempty string or message list")
        if instructions is not None and not isinstance(instructions, str):
            raise ValueError("instructions must be a string")
        payload = {"model": self.deployment, "input": input,
            "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": self.max_output_tokens, "store": False}
        if instructions is not None:
            payload["instructions"] = instructions
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")
        input_bytes = len(json.dumps(input, ensure_ascii=False).encode("utf-8")) + len((instructions or "").encode("utf-8")) + 512
        if input_bytes > bound:
            raise ValueError("prompt UTF-8 byte upper bound exceeds input_token_bound")
        logical_id = logical_id or uuid.uuid4().hex
        if not isinstance(logical_id, str) or not logical_id:
            raise ValueError("logical_id must be a nonempty string")
        key = self.key_loader()
        if not isinstance(key, str) or not key.strip() or "\n" in key or "\r" in key:
            raise ValueError("key_loader must return a nonempty single-line credential")
        if key in body.decode("utf-8"):
            raise ValueError("request payload unexpectedly contains the API credential")
        for attempt_number in range(1, self.max_attempts + 1):
            self.rate_limiter.acquire(bound + self.max_output_tokens)
            attempt_id = uuid.uuid4().hex
            try:
                reservation = self.ledger.reserve(attempt_id, logical_id, bound, self.max_output_tokens,
                    allow_existing_logical=attempt_number > 1)
            except DuplicateLogicalCall as error:
                raise TransportError(str(error), outcome="existing_logical_call", attempts=0,
                    result={"logical_id": logical_id, "previous_attempts": self.ledger.logical_attempts(logical_id)}) from None
            prefix = self.output_dir / attempt_id
            _atomic_json(prefix.with_suffix(".request.json"), payload)
            started_at = _utc_now()
            started = time.monotonic()
            record = {"logical_id": logical_id, "attempt_id": attempt_id,
                "attempt_count": attempt_number, "started_at": started_at,
                "url": self.endpoint, "request_file": prefix.with_suffix(".request.json").name,
                "reservation_usd": reservation["reserved_nusd"] / 1_000_000_000,
                "input_token_bound": bound, "output_token_bound": self.max_output_tokens,
                "deployment": self.deployment}
            _atomic_json(prefix.with_suffix(".attempt.json"), record)
            request = Request(self.endpoint, data=body, method="POST",
                headers={"api-key": key, "Content-Type": "application/json"})
            raw = None
            error_message = None
            retry_after = None
            http_status = None
            try:
                open_call = self.opener.open if hasattr(self.opener, "open") else self.opener
                with open_call(request, timeout=self.timeout) as response:
                    http_status = response.status
                    raw_bytes = response.read()
                try:
                    raw = json.loads(raw_bytes)
                except (ValueError, UnicodeError):
                    raw = {"unparsed_body": raw_bytes.decode("utf-8", errors="replace")}
                    outcome = "invalid_json"
                else:
                    outcome = "response"
            except HTTPError as error:
                http_status = error.code
                retry_after = error.headers.get("Retry-After") if error.headers else None
                try:
                    error_body = error.read().decode("utf-8", errors="replace")
                except Exception:
                    error_body = "[unreadable error body]"
                try:
                    raw = json.loads(error_body)
                except ValueError:
                    raw = {"unparsed_body": error_body}
                error_message = f"HTTP {error.code}: {str(error.reason)}"
                outcome = "http_error"
            except (TimeoutError, socket.timeout) as error:
                error_message = f"Timeout: {error}"
                outcome = "timeout_unknown"
            except (URLError, OSError) as error:
                error_message = f"Transport failure: {error}"
                outcome = "transport_unknown"
            except Exception as error:
                error_message = f"Transport failure ({type(error).__name__}): {error}"
                outcome = "transport_unknown"
            finished_at = _utc_now()
            latency = max(0, time.monotonic() - started)
            raw = _redact(raw, key)
            if outcome == "http_error" and _content_filter(raw):
                outcome = "content_filter"
            usage = _valid_usage(raw.get("usage")) if isinstance(raw, dict) else None
            text = ""
            status = raw.get("status") if isinstance(raw, dict) else None
            model = raw.get("model") if isinstance(raw, dict) else None
            if outcome == "response":
                if not isinstance(raw, dict):
                    outcome = "invalid_response"
                else:
                    text, refusal = _extract_text(raw)
                    if refusal:
                        outcome = "content_filter"
                    elif status == "incomplete":
                        reason = raw.get("incomplete_details", {})
                        outcome = "content_filter" if isinstance(reason, dict) and reason.get("reason") == "content_filter" else "incomplete"
                    elif status != "completed":
                        outcome = "failed_response"
                    elif not text.strip():
                        outcome = "missing_output_text"
                    elif model not in self.expected_models:
                        outcome = "model_mismatch"
                    else:
                        outcome = "completed"
            settled = self.ledger.settle(attempt_id, usage, outcome=outcome)
            result = {"raw": raw, "text": text, "model": model, "status": status,
                "usage": usage, "usage_known": usage is not None, "outcome": outcome,
                "logical_id": logical_id, "attempt_id": attempt_id,
                "attempt_count": attempt_number, "http_status": http_status,
                "cost_usd": (settled["actual_nusd"] if settled["actual_nusd"] is not None else settled["reserved_nusd"]) / 1_000_000_000,
                "latency_seconds": latency, "started_at": started_at, "finished_at": finished_at}
            if raw is not None:
                _atomic_json(prefix.with_suffix(".response.json"), raw)
            record.update({k: v for k, v in result.items() if k not in {"raw", "text"}})
            if error_message:
                record["error"] = _redact(error_message, key)
            _atomic_json(prefix.with_suffix(".attempt.json"), record)
            if outcome == "completed":
                return result
            if outcome == "http_error" and http_status in {429, 500, 502, 503, 504} and attempt_number < self.max_attempts:
                self.sleep(self._delay(retry_after, attempt_number))
                continue
            message = _redact(error_message or f"Responses outcome: {outcome}", key)
            raise TransportError(message, outcome=outcome, attempts=attempt_number, result=result)
        raise AssertionError("unreachable")
