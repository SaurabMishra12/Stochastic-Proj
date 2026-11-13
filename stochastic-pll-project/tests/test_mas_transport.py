import io
import json
import socket
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError

import pytest

from src.llm_mas.ledger import BudgetExceeded, Ledger
from src.llm_mas.transport import AzureResponsesTransport, RateLimiter, TransportError


def make_ledger(tmp_path, **kwargs):
    return Ledger(tmp_path / "budget.sqlite", "approved-local-test", kwargs.pop("max_usd", 1),
                  kwargs.pop("max_attempts", 10), **kwargs)


def test_unsettled_reservation_survives_reopen_and_blocks_spend(tmp_path):
    ledger = make_ledger(tmp_path, max_usd=.012)
    ledger.reserve("a", "logical", 12288, 4096)
    reopened = make_ledger(tmp_path, max_usd=.012)
    assert reopened.summary()["charged_usd"] == pytest.approx(.011264)


def test_budget_is_checked_before_reservation_and_unknown_usage_stays_charged(tmp_path):
    ledger = make_ledger(tmp_path, max_usd=.02)
    ledger.reserve("a", "logical", 12288, 4096)
    ledger.settle("a", None, outcome="timeout")
    assert ledger.summary()["uncertain_attempts"] == 1
    with pytest.raises(BudgetExceeded):
        ledger.reserve("b", "logical", 12288, 4096)
    assert ledger.summary()["attempts"] == 1


def test_actual_usage_releases_unused_reservation_and_policy_cannot_change(tmp_path):
    ledger = make_ledger(tmp_path, max_usd=.02)
    ledger.reserve("a", "logical", 12288, 4096)
    ledger.settle("a", {"input_tokens": 1000, "output_tokens": 500}, outcome="completed")
    assert ledger.summary()["charged_usd"] == pytest.approx(.00125)
    with pytest.raises(ValueError, match="policy"):
        Ledger(tmp_path / "budget.sqlite", "different-approval", .02, 10)


def test_concurrent_reservations_cannot_exceed_attempt_limit(tmp_path):
    ledger = make_ledger(tmp_path, max_attempts=1)
    def reserve(i):
        try:
            ledger.reserve(str(i), str(i), 10, 10)
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=4) as executor:
        assert sum(executor.map(reserve, range(4))) == 1


def test_logical_call_limit_counts_retries_as_one_call(tmp_path):
    ledger = make_ledger(tmp_path, max_logical_calls=1)
    ledger.reserve("a", "logical", 10, 10)
    ledger.reserve("b", "logical", 10, 10)
    with pytest.raises(BudgetExceeded):
        ledger.reserve("c", "another", 10, 10)


class Response:
    def __init__(self, value):
        self.value = value
        self.headers = {}
        self.status = 200
    def read(self):
        return json.dumps(self.value).encode()
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass


def response(status="completed", usage=True):
    value = {"id": "resp_local", "model": "gpt-5-mini-2025-08-07", "status": status,
             "output": [{"type": "message", "content": [{"type": "output_text", "text": "42"}]}]}
    if usage:
        value["usage"] = {"input_tokens": 100, "output_tokens": 30,
                          "output_tokens_details": {"reasoning_tokens": 20}}
    return value


def transport(tmp_path, opener, **kwargs):
    return AzureResponsesTransport("https://unit-test.openai.azure.com/openai/v1/", "gpt-5-mini",
        lambda: "secret-local-key", make_ledger(tmp_path), tmp_path / "outputs", opener=opener,
        sleep=lambda _: None, rpm=None, **kwargs)


def test_transport_reserves_before_dispatch_and_preserves_request_usage_and_raw_response(tmp_path):
    def opener(request, timeout):
        assert make_ledger(tmp_path).summary()["attempts"] == 1
        payload = json.loads(request.data)
        assert payload["store"] is False
        assert payload["max_output_tokens"] == 4096
        assert payload["reasoning"] == {"effort": "low"}
        assert not ({"temperature", "top_p", "seed"} & payload.keys())
        return Response(response())
    result = transport(tmp_path, opener).request("What is 6 times 7?", logical_id="item-1")
    assert result["text"] == "42"
    assert result["outcome"] == "completed"
    assert make_ledger(tmp_path).summary()["charged_usd"] == pytest.approx(.000085)
    saved = list((tmp_path / "outputs").rglob("*.json"))
    assert len(saved) >= 3
    assert all("secret-local-key" not in p.read_text() for p in saved)
    assert any(json.loads(p.read_text()).get("id") == "resp_local" for p in saved)


def test_retryable_http_errors_consume_independent_attempts_and_redact_errors(tmp_path):
    attempts = []
    def opener(request, timeout):
        attempts.append(1)
        if len(attempts) < 3:
            raise HTTPError(request.full_url, 429, "secret-local-key echoed", {"Retry-After": "0"},
                            io.BytesIO(b'{"error":"secret-local-key echoed"}'))
        return Response(response())
    result = transport(tmp_path, opener).request("question", logical_id="item")
    assert result["text"] == "42"
    summary = make_ledger(tmp_path).summary()
    assert summary["attempts"] == 3
    assert summary["logical_calls"] == 1
    assert summary["uncertain_attempts"] == 2
    assert all("secret-local-key" not in p.read_text() for p in (tmp_path / "outputs").rglob("*.json"))


def test_content_filter_provider_error_never_retries_even_with_retryable_http_status(tmp_path):
    calls = []
    def opener(request, timeout):
        calls.append(1)
        raise HTTPError(request.full_url, 500, "filtered", {},
            io.BytesIO(b'{"error":{"code":"content_filter","message":"filtered"}}'))
    with pytest.raises(TransportError) as error:
        transport(tmp_path, opener).request("question")
    assert error.value.outcome == "content_filter"
    assert len(calls) == 1


def test_all_retryable_failures_stop_after_three_attempts(tmp_path):
    def opener(request, timeout):
        raise HTTPError(request.full_url, 503, "unavailable", {}, io.BytesIO(b'{}'))
    with pytest.raises(TransportError) as error:
        transport(tmp_path, opener).request("question")
    assert error.value.attempts == 3
    assert make_ledger(tmp_path).summary()["attempts"] == 3


def test_success_without_usage_keeps_reservation_and_missing_text_is_not_accepted(tmp_path):
    result = transport(tmp_path, lambda request, timeout: Response(response(usage=False))).request("question")
    assert result["usage_known"] is False
    assert make_ledger(tmp_path).summary()["reserved_usd"] == pytest.approx(.011264)
    missing = response()
    missing["output"] = []
    with pytest.raises(TransportError) as error:
        transport(tmp_path, lambda request, timeout: Response(missing)).request("question")
    assert error.value.outcome == "missing_output_text"


def test_reopened_transport_refuses_existing_logical_call_and_exposes_resume_evidence(tmp_path):
    calls = []
    def opener(request, timeout):
        calls.append(1)
        return Response(response())
    transport(tmp_path, opener).request("question", logical_id="fixed-step")
    with pytest.raises(TransportError) as error:
        transport(tmp_path, opener).request("question", logical_id="fixed-step")
    assert error.value.outcome == "existing_logical_call"
    assert len(calls) == 1
    prior = make_ledger(tmp_path).logical_attempts("fixed-step")
    assert len(prior) == 1
    assert prior[0]["outcome"] == "completed"


def test_unexpected_response_model_fails_but_is_still_billed(tmp_path):
    changed = response()
    changed["model"] = "different-model"
    with pytest.raises(TransportError) as error:
        transport(tmp_path, lambda request, timeout: Response(changed)).request("question")
    assert error.value.outcome == "model_mismatch"
    assert make_ledger(tmp_path).summary()["charged_usd"] == pytest.approx(.000085)


@pytest.mark.parametrize("failure", [socket.timeout("secret-local-key"),
    HTTPError("https://unit-test.openai.azure.com", 400, "bad request", {}, io.BytesIO(b'{}')),
    HTTPError("https://unit-test.openai.azure.com", 408, "timeout", {}, io.BytesIO(b'{}'))])
def test_ambiguous_timeouts_and_terminal_http_errors_are_never_retried(tmp_path, failure):
    calls = []
    def opener(request, timeout):
        calls.append(1)
        raise failure
    with pytest.raises(TransportError) as error:
        transport(tmp_path, opener).request("question")
    assert len(calls) == 1
    assert "secret-local-key" not in str(error.value)
    assert make_ledger(tmp_path).summary()["uncertain_attempts"] == 1


def test_incomplete_response_fails_but_still_settles_recorded_usage(tmp_path):
    with pytest.raises(TransportError) as error:
        transport(tmp_path, lambda request, timeout: Response(response("incomplete"))).request("question")
    assert error.value.outcome == "incomplete"
    assert make_ledger(tmp_path).summary()["charged_usd"] == pytest.approx(.000085)


@pytest.mark.parametrize("url", ["http://unit.openai.azure.com/", "https://evil.example/", "https://key@unit.openai.azure.com/"])
def test_non_azure_or_insecure_endpoints_are_rejected_before_secret_loading(tmp_path, url):
    def key_loader():
        pytest.fail("invalid endpoint loaded key")
    with pytest.raises(ValueError):
        AzureResponsesTransport(url, "deployment", key_loader, make_ledger(tmp_path), tmp_path / "outputs")


def test_rate_limiter_spaces_attempts_and_enforces_rolling_token_quota():
    now = [0.]
    def sleep(seconds):
        now[0] += seconds
    limiter = RateLimiter(rpm=10, tpm=100, clock=lambda: now[0], sleep=sleep)
    limiter.acquire(60)
    limiter.acquire(40)
    assert now[0] == 6
    limiter.acquire(40)
    assert now[0] == 60
    with pytest.raises(ValueError, match="token"):
        limiter.acquire(101)


def test_input_byte_bound_includes_prompt_framing_and_rejects_before_dispatch(tmp_path):
    def opener(request, timeout):
        pytest.fail("oversized prompt was dispatched")
    with pytest.raises(ValueError, match="upper bound"):
        transport(tmp_path, opener).request("A" * 100, input_token_bound=100)
    assert make_ledger(tmp_path).summary()["attempts"] == 0
