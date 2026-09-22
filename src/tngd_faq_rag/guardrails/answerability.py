"""This file asks an LLM whether the retrieved passages actually answer the question."""

from __future__ import annotations
import json
import re
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from ..logging_utils import LOG
from ..text import normalize_question, sha1

if TYPE_CHECKING:  # pragma: no cover
    from ..llm import ChatClient

_SYSTEM_PROMPT = (
    "You grade whether retrieved FAQ passages can answer a user's question. "
    "You never answer the question yourself. You reply with passage numbers only."
)

_USER_TEMPLATE = """QUESTION: {question}

PASSAGES:
{passages}

Which passages contain the information needed to answer the QUESTION?

A passage that is merely about the same TOPIC does not count. It must actually
answer what was asked. For example, an article defining a feature does not
answer a question about how to spell its name, or a request to translate it.

Reply with the passage numbers separated by commas, or the single word NONE.
If your output is constrained to JSON, return {{"supporting": [numbers]}} and an
empty list when no passage answers the question."""

_NONE_RE = re.compile(r"\bnone\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")

# Constrained output, so a provider that honours it cannot return unparsable text
# Providers without support fall back to free text, which the parser still reads
_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "answerability",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "supporting": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": (
                        "1-based numbers of the passages that contain the information "
                        "needed to answer the question. Empty if none of them do."
                    ),
                }
            },
            "required": ["supporting"],
            "additionalProperties": False,
        },
    },
}


@dataclass
class AnswerabilityResult:
    """Outcome of a grading call."""

    answerable: bool
    supporting: list[int] = field(default_factory=list)
    reason: str = ""
    graded: bool = False  # False when no real judgement was made
    latency_ms: float = 0.0

    def as_dict(self) -> dict:
        return {
            "answerable": self.answerable,
            "supporting": self.supporting,
            "graded": self.graded,
            "reason": self.reason,
            "latency_ms": self.latency_ms,
        }


class AnswerabilityGrader:
    """No-op grader: everything is answerable. Used when grading is disabled."""

    name = "disabled"

    def grade(self, question: str, passages: Sequence[str]) -> AnswerabilityResult:
        return AnswerabilityResult(
            answerable=True,
            supporting=list(range(len(passages))),
            reason="answerability grading is disabled",
            graded=False,
        )


class LLMAnswerabilityGrader(AnswerabilityGrader):
    """Grades passages with a chat model over whichever endpoint is configured."""

    name = "llm"

    def __init__(
        self,
        client: ChatClient,
        *,
        max_passages: int = 3,
        max_passage_chars: int = 700,
        cache_size: int = 256,
    ) -> None:
        self.client = client
        self.max_passages = max_passages
        self.max_passage_chars = max_passage_chars
        self._cache: OrderedDict[str, AnswerabilityResult] = OrderedDict()
        self._cache_size = cache_size

    def _cache_key(self, question: str, passages: Sequence[str]) -> str:
        return sha1(normalize_question(question), *passages)

    def _remember(self, key: str, result: AnswerabilityResult) -> AnswerabilityResult:
        self._cache[key] = result
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return result

    @staticmethod
    def _parse(reply: str, n: int) -> list[int] | None:
        """Map the model's reply to passage indices."""
        text = (reply or "").strip()
        if not text:
            return None

        # Structured output first, where an empty list is a real none
        try:
            payload = json.loads(text)
        except (ValueError, TypeError):
            payload = None
        if isinstance(payload, dict) and isinstance(payload.get("supporting"), list):
            raw = [int(x) for x in payload["supporting"] if isinstance(x, (int, float))]
            valid = sorted({i - 1 for i in raw if 1 <= i <= n})
            if valid or not raw:
                # An empty list is a real none, but out of range indices are botched
                # Refusing on the strength of nonsense would be worse than failing open
                return valid
            return None

        numbers = [int(m) for m in _NUMBER_RE.findall(text)]
        valid = sorted({i - 1 for i in numbers if 1 <= i <= n})
        if valid:
            return valid
        if _NONE_RE.search(text):
            return []
        return None

    def grade(self, question: str, passages: Sequence[str]) -> AnswerabilityResult:
        if not passages:
            return AnswerabilityResult(False, [], "no passages to grade", graded=False)

        considered = list(passages[: self.max_passages])
        key = self._cache_key(question, considered)
        cached = self._cache.get(key)
        if cached is not None:
            self._cache.move_to_end(key)
            return cached

        numbered = "\n\n".join(
            f"[{i + 1}] {text[: self.max_passage_chars]}" for i, text in enumerate(considered)
        )
        result = self.client.complete(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _USER_TEMPLATE.format(question=question, passages=numbered),
                },
            ],
            max_tokens=64,
            temperature=0.0,
            response_format=_RESPONSE_SCHEMA,
        )

        if not result.ok:
            # Fail open, keeping prior behaviour when the grader is unreachable
            return self._remember(
                key,
                AnswerabilityResult(
                    answerable=True,
                    supporting=list(range(len(considered))),
                    reason=f"grader unavailable ({result.error})",
                    graded=False,
                    latency_ms=result.latency_ms,
                ),
            )

        parsed = self._parse(result.text, len(considered))
        if parsed is None:
            LOG.debug("answerability grader returned an unparsable reply: %r", result.text)
            return self._remember(
                key,
                AnswerabilityResult(
                    answerable=True,
                    supporting=list(range(len(considered))),
                    reason=f"unparsable grader reply: {result.text[:60]!r}",
                    graded=False,
                    latency_ms=result.latency_ms,
                ),
            )

        answerable = bool(parsed)
        return self._remember(
            key,
            AnswerabilityResult(
                answerable=answerable,
                supporting=parsed,
                reason=(
                    f"graded: passages {[i + 1 for i in parsed]} can answer"
                    if answerable
                    else "graded: no retrieved passage answers this question"
                ),
                graded=True,
                latency_ms=result.latency_ms,
            ),
        )


def build_answerability_grader(cfg: Any) -> AnswerabilityGrader:
    """Pick a grader from configuration and from what is actually reachable."""
    from ..llm import build_chat_client, llm_is_configured

    mode = (cfg.answerability or "auto").lower()
    if mode == "off":
        LOG.debug("Answerability grading: disabled by configuration")
        return AnswerabilityGrader()
    if mode == "auto" and not llm_is_configured():
        LOG.debug(
            "Answerability grading: off (no LLM configured). "
            "Set LLM_API_BASE - e.g. a local Ollama server - to enable it."
        )
        return AnswerabilityGrader()

    client = build_chat_client(cfg, model=cfg.answerability_model)
    grader = LLMAnswerabilityGrader(client, max_passages=cfg.answerability_max_passages)
    health = client.health()
    if health.ok:
        LOG.info(
            "Answerability grading: on (%s via %s, health check %.0f ms)",
            client.model,
            client.base_url,
            health.latency_ms,
        )
    else:
        # Returned anyway, since a slow starting endpoint must not disable the feature
        LOG.warning(
            "Answerability grading: configured but the endpoint is unreachable (%s). "
            "Requests will fall back to retrieval scoring alone.",
            health.error,
        )
    return grader
