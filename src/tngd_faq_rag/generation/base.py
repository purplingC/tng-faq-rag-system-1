"""This file defines the generator interface and builds prompts that fit a token budget."""

from __future__ import annotations
from collections.abc import Sequence
from ..constants import SYSTEM_RULES
from ..models import Candidate, Generation
from ..store import MetadataStore
from ..text import approx_token_count


class PromptBuilder:
    """Token-budget-aware prompt assembly."""

    def __init__(self, canary: str, max_input_tokens: int = 480) -> None:
        self.canary = canary
        self.max_input_tokens = max_input_tokens

    def instructions(self) -> str:
        return SYSTEM_RULES.format(canary=self.canary)

    def build(self, question: str, candidates: Sequence[Candidate]) -> tuple[str, list[int]]:
        head = f"{self.instructions()}\n\nQUESTION: {question}\n\nSOURCES:\n"
        budget = self.max_input_tokens - approx_token_count(head) - 24
        parts: list[str] = []
        used: list[int] = []
        for i, c in enumerate(candidates):
            block = f"[{i + 1}] {c.answer_slice}"
            cost = approx_token_count(block)
            if cost > budget and parts:
                break
            if cost > budget:  # first source alone too big
                words = block.split()
                keep = max(16, int(budget / 1.3))
                block = " ".join(words[:keep])
                cost = approx_token_count(block)
            parts.append(block)
            used.append(i)
            budget -= cost
            if budget <= 0:
                break
        return head + "\n\n".join(parts) + "\n\nANSWER:", used


class Generator:
    name = "base"

    def generate(
        self, question: str, candidates: Sequence[Candidate], store: MetadataStore
    ) -> Generation:
        raise NotImplementedError
