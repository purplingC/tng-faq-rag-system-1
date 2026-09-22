"""This file writes answers through any OpenAI-compatible endpoint, using only urllib."""

from __future__ import annotations
import json
from collections.abc import Sequence
from ..config import Config
from ..logging_utils import LOG
from ..models import Candidate, Generation
from ..store import MetadataStore
from ..text import clean_text
from .base import Generator, PromptBuilder


class OpenAICompatibleGenerator(Generator):
    """Any OpenAI-compatible chat completions endpoint, via stdlib urllib."""

    name = "api"

    def __init__(self, cfg: Config, prompt_builder: PromptBuilder) -> None:
        self.cfg = cfg
        self.pb = prompt_builder
        self.pb.max_input_tokens = 3000

    def generate(
        self, question: str, candidates: Sequence[Candidate], store: MetadataStore
    ) -> Generation:
        import urllib.error
        import urllib.request

        prompt, used = self.pb.build(question, candidates)
        payload = {
            "model": self.cfg.api_model,
            "temperature": 0,
            "max_tokens": 400,
            "messages": [
                {"role": "system", "content": self.pb.instructions()},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            self.cfg.api_base.rstrip("/") + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
                "User-Agent": self.cfg.user_agent,
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = clean_text(data["choices"][0]["message"]["content"])
        except Exception as exc:
            LOG.warning("API generation failed (%s); abstaining to the safe path", exc)
            return Generation(text="", backend=self.name, abstained=True)
        if not text or "NOT_IN_KB" in text.upper():
            return Generation(text="", backend=self.name, raw=text, abstained=True)
        return Generation(text=text, backend=self.name, used_source_indices=used, raw=text)
