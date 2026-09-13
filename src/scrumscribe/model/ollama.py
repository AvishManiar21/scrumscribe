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
    # Ollama's default prediction budget cuts long answers off mid-sentence,
    # which surfaces as truncated JSON rather than as an error. Extraction
    # over a dense section legitimately needs more room than the default.
    num_predict: int = 2048
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
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        if schema:
            payload["format"] = schema

        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                data = self._post("/api/chat", payload)
                content = data.get("message", {}).get("content", "")
                if not content.strip():
                    # Ollama occasionally returns a successful response with no
                    # content -- most visibly when it is swapping a model in or
                    # out. Treated as success this silently degrades the caller
                    # (an empty string parses to nothing, no exception is
                    # raised, and the result is a mechanical fallback nobody
                    # notices), so it is treated as a retryable failure.
                    raise OllamaError("empty response from model")
                return content
            except (
                urllib.error.URLError,
                OSError,
                TimeoutError,
                ValueError,
                OllamaError,
            ) as exc:
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
        raw = ""
        for attempt in range(2):
            raw = self.chat(prompt, system=system, schema=schema)
            if raw.strip():
                break

        try:
            # strict=False permits literal newlines and tabs inside strings.
            # Constrained decoding guarantees the JSON shape but not that the
            # model escapes control characters, and a summary written across
            # two lines is otherwise a hard parse failure -- which silently
            # degraded roughly two runs in three before this was found.
            parsed = json.loads(raw, strict=False)
        except ValueError:
            parsed = _salvage_json(raw)

        if not isinstance(parsed, dict):
            return dict(default or {})
        return parsed


def _salvage_json(raw: str) -> dict | None:
    """Recover a usable object from malformed or truncated model output.

    Two distinct failures land here. A chatty model wraps its JSON in prose,
    which the brace scan handles. More importantly, a model that runs out of
    output budget stops mid-string, leaving JSON that is valid right up to the
    point it was cut off. Closing the open string and brackets recovers the
    content that was produced -- a summary missing its last half-sentence is
    far more useful than no summary at all.
    """
    start = raw.find("{")
    if start == -1:
        return None

    end = raw.rfind("}")
    if end > start:
        try:
            return json.loads(raw[start : end + 1], strict=False)
        except ValueError:
            pass

    return _close_truncated(raw[start:])


def _close_truncated(fragment: str) -> dict | None:
    """Close an unterminated JSON fragment and parse what survived."""
    in_string = False
    escaped = False
    stack: list[str] = []

    for char in fragment:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if char in "{[":
            stack.append(char)
        elif char in "}]":
            if stack:
                stack.pop()

    repaired = fragment
    if in_string:
        repaired += '"'
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"

    try:
        return json.loads(repaired, strict=False)
    except ValueError:
        return None
