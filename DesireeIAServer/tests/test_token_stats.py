"""Token accounting: timings and usage shapes."""

from __future__ import annotations

import pytest

from desireeiaserver.token_stats import TokenTracker


def test_usage_counts():
    tracker = TokenTracker()
    tracker.prompt_n = 4
    tracker.predicted_n = 10
    usage = tracker.usage("model-id")
    assert usage["prompt_tokens"] == 4
    assert usage["completion_tokens"] == 10
    assert usage["total_tokens"] == 14
    assert usage["prompt_tokens_details"]["cached_tokens"] == 0


def test_timings_rates():
    tracker = TokenTracker()
    tracker.prompt_n = 4
    tracker.prompt_ms = 100.0
    tracker.predicted_n = 10
    tracker.predicted_ms = 100.0
    timings = tracker.timings()
    assert timings["prompt_per_second"] == pytest.approx(40.0)
    assert timings["predicted_per_second"] == pytest.approx(100.0)
    assert timings["prompt_per_token_ms"] == pytest.approx(25.0)
    assert timings["predicted_per_token_ms"] == pytest.approx(10.0)