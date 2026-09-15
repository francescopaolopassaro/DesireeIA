"""Token and timing accounting for a single generation request."""

from __future__ import annotations


class TokenTracker:
    def __init__(self) -> None:
        self.prompt_n = 0
        self.cached_tokens = 0
        self.predicted_n = 0
        self.prompt_ms = 0.0
        self.predicted_ms = 0.0
        self.total_ms = 0.0

    def timings(self) -> dict:
        prompt_per_token = (self.prompt_ms / self.prompt_n) if self.prompt_n else 0.0
        predicted_per_token = (self.predicted_ms / self.predicted_n) if self.predicted_n else 0.0
        return {
            "prompt_n": self.prompt_n,
            "predicted_n": self.predicted_n,
            "prompt_ms": round(self.prompt_ms, 3),
            "predicted_ms": round(self.predicted_ms, 3),
            "total_ms": round(self.total_ms, 3),
            "prompt_per_token_ms": round(prompt_per_token, 3),
            "predicted_per_token_ms": round(predicted_per_token, 3),
            "prompt_per_second": round((self.prompt_n / self.prompt_ms) * 1000.0, 2)
            if self.prompt_ms else 0.0,
            "predicted_per_second": round((self.predicted_n / self.predicted_ms) * 1000.0, 2)
            if self.predicted_ms else 0.0,
        }

    def usage(self, model_id: str) -> dict:
        prompt = self.prompt_n + self.cached_tokens
        return {
            "prompt_tokens": prompt,
            "completion_tokens": self.predicted_n,
            "total_tokens": prompt + self.predicted_n,
            "prompt_tokens_details": {"cached_tokens": self.cached_tokens},
            "model": model_id,
        }