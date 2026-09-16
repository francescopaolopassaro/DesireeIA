# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""In-process metrics, exposed at GET /metrics in Prometheus text format.

Hand-rolled rather than depending on the `prometheus_client` package - the
exposition format for the handful of counters/gauges this server has is a
few lines of text, not worth a new dependency for a project this size.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, Tuple


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._request_count: Dict[Tuple[str, str, int], int] = defaultdict(int)
        self._request_seconds_sum: Dict[Tuple[str, str], float] = defaultdict(float)
        self._request_seconds_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self.chat_requests_total = 0
        self.prompt_tokens_total = 0
        self.completion_tokens_total = 0
        self.tool_calls_total = 0

    def record_request(self, method: str, route: str, status: int, duration_s: float) -> None:
        with self._lock:
            self._request_count[(method, route, status)] += 1
            self._request_seconds_sum[(method, route)] += duration_s
            self._request_seconds_count[(method, route)] += 1

    def record_generation(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.chat_requests_total += 1
            self.prompt_tokens_total += prompt_tokens
            self.completion_tokens_total += completion_tokens

    def record_tool_call(self) -> None:
        with self._lock:
            self.tool_calls_total += 1

    def render(self, manager) -> str:
        lines = []
        with self._lock:
            lines += [
                "# HELP desireeia_http_requests_total Total HTTP requests handled.",
                "# TYPE desireeia_http_requests_total counter",
            ]
            for (method, route, status), count in sorted(self._request_count.items()):
                lines.append(
                    f'desireeia_http_requests_total{{method="{method}",route="{route}",status="{status}"}} {count}'
                )

            lines += [
                "# HELP desireeia_http_request_duration_seconds_sum Cumulative request handling time.",
                "# TYPE desireeia_http_request_duration_seconds_sum counter",
            ]
            for (method, route), total in sorted(self._request_seconds_sum.items()):
                lines.append(
                    f'desireeia_http_request_duration_seconds_sum{{method="{method}",route="{route}"}} {total:.6f}'
                )

            lines += [
                "# HELP desireeia_http_request_duration_seconds_count Request count backing the sum above.",
                "# TYPE desireeia_http_request_duration_seconds_count counter",
            ]
            for (method, route), count in sorted(self._request_seconds_count.items()):
                lines.append(
                    f'desireeia_http_request_duration_seconds_count{{method="{method}",route="{route}"}} {count}'
                )

            lines += [
                "# HELP desireeia_chat_requests_total Completed chat/completions generations.",
                "# TYPE desireeia_chat_requests_total counter",
                f"desireeia_chat_requests_total {self.chat_requests_total}",
                "# HELP desireeia_prompt_tokens_total Cumulative prompt tokens processed.",
                "# TYPE desireeia_prompt_tokens_total counter",
                f"desireeia_prompt_tokens_total {self.prompt_tokens_total}",
                "# HELP desireeia_completion_tokens_total Cumulative completion tokens generated.",
                "# TYPE desireeia_completion_tokens_total counter",
                f"desireeia_completion_tokens_total {self.completion_tokens_total}",
                "# HELP desireeia_tool_calls_total Tool calls detected in model output.",
                "# TYPE desireeia_tool_calls_total counter",
                f"desireeia_tool_calls_total {self.tool_calls_total}",
            ]

        lines += [
            "# HELP desireeia_model_loaded Whether a model group is currently loaded (1) or not (0).",
            "# TYPE desireeia_model_loaded gauge",
        ]
        for model_id, group in manager.groups.items():
            loaded = 1 if group.aggregate_state() == "loaded" else 0
            lines.append(f'desireeia_model_loaded{{model="{model_id}"}} {loaded}')

        lines += [
            "# HELP desireeia_model_replicas_busy Busy replica count per model.",
            "# TYPE desireeia_model_replicas_busy gauge",
        ]
        for model_id, group in manager.groups.items():
            busy = sum(1 for replica in group.replicas if replica.is_busy)
            lines.append(f'desireeia_model_replicas_busy{{model="{model_id}"}} {busy}')

        return "\n".join(lines) + "\n"


# One instance for the whole process - metrics are process-global state, not
# per-request or per-app-instance (mirrors how every real Prometheus client
# library works: a single default registry).
metrics = Metrics()
