"""This file contains a small chat client for endpoints such as Gemini, on the standard library."""

from __future__ import annotations
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any
from .logging_utils import LOG


@dataclass
class ChatResult:
    text: str
    ok: bool
    error: str = ""
    latency_ms: float = 0.0


class ChatClient:
    """A single `POST /chat/completions` call, with timeout and one retry."""

    def __init__(
        self,
        base_url: str,
        api_key: str = "",
        model: str = "",
        timeout: float = 30.0,
        user_agent: str = "tngd-faq-rag",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.user_agent = user_agent

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ChatClient(base_url={self.base_url!r}, model={self.model!r})"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 256,
        temperature: float = 0.0,
        retries: int = 1,
        response_format: dict[str, Any] | None = None,
    ) -> ChatResult:
        """One chat completion."""
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }
        if response_format:
            body["response_format"] = response_format
        payload = json.dumps(body).encode("utf-8")

        headers = {"Content-Type": "application/json", "User-Agent": self.user_agent}
        if self.api_key:
            # Omit the header entirely when there is no key, as Ollama needs none
            headers["Authorization"] = f"Bearer {self.api_key}"

        last_error = ""
        for attempt in range(retries + 1):
            started = time.perf_counter()
            try:
                request = urllib.request.Request(
                    f"{self.base_url}/chat/completions", data=payload, headers=headers
                )
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                text = body["choices"][0]["message"]["content"] or ""
                return ChatResult(
                    text=text.strip(),
                    ok=True,
                    latency_ms=round((time.perf_counter() - started) * 1000, 2),
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", "ignore")[:200]
                last_error = f"HTTP {exc.code}: {detail}"
                if (
                    exc.code == 400
                    and response_format is not None
                    and "response_format" in detail.lower()
                ):
                    LOG.debug("endpoint rejected response_format; retrying without it")
                    return self.complete(
                        messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        retries=retries,
                        response_format=None,
                    )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

        LOG.warning("LLM call failed (%s)", last_error)
        return ChatResult(text="", ok=False, error=last_error)

    def health(self) -> ChatResult:
        """One tiny call to report whether the endpoint is actually reachable."""
        return self.complete(
            [{"role": "user", "content": "Reply with the single word OK."}],
            max_tokens=5,
            retries=0,
        )


def llm_is_configured() -> bool:
    """Has the user pointed us at an LLM?"""
    from .config import _resolve_api_key

    return bool(_resolve_api_key()[0]) or bool(os.environ.get("LLM_API_BASE"))


def build_chat_client(cfg: Any, *, model: str = "") -> ChatClient:
    """A client for whichever chat endpoint is configured."""
    return ChatClient(
        base_url=cfg.api_base,
        api_key=cfg.api_key,
        model=model or cfg.api_model,
        timeout=cfg.llm_timeout,
        user_agent=cfg.user_agent,
    )
