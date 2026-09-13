"""Minimal Ollama client over the standard library.

No SDK dependency on purpose: the whole point of this project is that it runs
on a laptop with nothing but Python and a local model server. One fewer
dependency is one fewer thing to break six months from now.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "gemma3:4b"


class OllamaError(RuntimeError):
    """Raised when the local model server is unreachable or misbehaving."""


@dataclass
class Ollama:
    """A thin, retrying wrapper around the Ollama chat endpoint."""

    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    temperature: float = 0.2  # summarisation wants determinism, not flair
    num_ctx: int = 8192
    timeout: float = 300.0
    retries: int = 2

    def _post(self, endpoint: str, payload: dict, timeout: float | None = None) -> dict:
        request = urllib.request.Request(
            f"{self.host}{endpoint}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))

    def available(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.host}/api/tags", timeout=5).read()
            return True
        except (urllib.error.URLError, OSError, TimeoutError):
            return False

    def models(self) -> list[str]:
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
            return [m["name"] for m in data.get("models", [])]
        except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
            raise OllamaError(f"cannot list models at {self.host}: {exc}") from exc

    def preflight(self) -> None:
        """Fail loudly and usefully before we burn time on a long transcript."""
        if not self.available():
            raise OllamaError(
                f"Ollama is not reachable at {self.host}.\n"
                "  Start it with:  ollama serve"
            )
        installed = self.models()
        # Ollama reports "gemma3:4b"; accept a bare "gemma3" request too.
        if not any(m == self.model or m.startswith(f"{self.model}:") for m in installed):
            raise OllamaError(
                f"Model '{self.model}' is not installed.\n"
                f"  Install it with:  ollama pull {self.model}\n"
                f"  Installed: {', '.join(installed) or '(none)'}"
            )

    def chat(
        self,
        prompt: str,
        system: str | None = None,
        schema: dict | None = None,
    ) -> str:
        """Send one prompt and return the raw text response.

        When `schema` is supplied, Ollama constrains decoding to that JSON
        schema. That is the difference between a 4B model that reliably
        returns parseable output and one that wanders off into prose.
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
        }
        if schema:
            payload["format"] = schema

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                data = self._post("/api/chat", payload)
                return data.get("message", {}).get("content", "")
            except (urllib.error.URLError, OSError, TimeoutError, ValueError) as exc:
                last_error = exc
                if attempt < self.retries:
                    time.sleep(2 * (attempt + 1))

        raise OllamaError(f"Ollama request failed after {self.retries + 1} attempts: {last_error}")

    def chat_json(
        self,
        prompt: str,
        schema: dict,
        system: str | None = None,
        default: dict | None = None,
    ) -> dict:
        """Schema-constrained call that never raises on malformed output.

        A single bad chunk must not kill a 45-minute meeting's worth of work,
        so we degrade to `default` and let the caller carry on.
        """
        raw = self.chat(prompt, system=system, schema=schema)
        try:
            parsed = json.loads(raw)
        except ValueError:
            parsed = _salvage_json(raw)

        if not isinstance(parsed, dict):
            return dict(default or {})
        return parsed


def _salvage_json(raw: str) -> dict | None:
    """Last-ditch extraction of a JSON object from a chatty response."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(raw[start : end + 1])
    except ValueError:
        return None
