from __future__ import annotations

import json
import os
import shlex
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LLMClient:
    last_call: dict[str, Any] | None = None

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        raise NotImplementedError


def load_env_file(path: Path | str = ".env") -> None:
    env_path = Path(path)
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum() or key[0].isdigit():
            continue
        try:
            parsed = shlex.split(value, comments=False, posix=True)
            value = parsed[0] if parsed else ""
        except ValueError:
            value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


class OpenAICompatibleLLMClient(LLMClient):
    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
        timeout: int = 120,
        max_retries: int = 0,
        retry_backoff: float = 2.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key_env = api_key_env
        self.timeout = int(timeout)
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff = max(0.0, float(retry_backoff))

    def complete_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {self.api_key_env}")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "Return valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        request_summary = {
            "url": f"{self.base_url}/chat/completions",
            "model": self.model,
            "messages": payload["messages"],
            "response_format": payload["response_format"],
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        self.last_call = {
            "schema_name": schema_name,
            "request": request_summary,
            "attempts": [],
        }

        raw = ""
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw = response.read().decode("utf-8")
                self.last_call["attempts"].append({"attempt": attempt + 1, "status": "success"})
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                self.last_call["attempts"].append(
                    {"attempt": attempt + 1, "status": "http_error", "code": exc.code, "body": body}
                )
                raise RuntimeError(f"LLM API HTTP {exc.code}: {body}") from exc
            except (socket.timeout, TimeoutError, urllib.error.URLError) as exc:
                self.last_call["attempts"].append(
                    {
                        "attempt": attempt + 1,
                        "status": "retryable_error",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if attempt >= self.max_retries:
                    raise TimeoutError(
                        f"LLM API request timed out or failed after {attempt + 1} attempt(s): {exc}"
                    ) from exc
                time.sleep(self.retry_backoff * float(attempt + 1))

        data = json.loads(raw)
        content = data["choices"][0]["message"]["content"]
        if self.last_call is not None:
            self.last_call["raw_response_text"] = raw
            self.last_call["raw_response_json"] = data
            self.last_call["message_content"] = content
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM returned non-JSON content for {schema_name}: {content[:500]}") from exc
        if self.last_call is not None:
            self.last_call["parsed_json"] = parsed
        return parsed
