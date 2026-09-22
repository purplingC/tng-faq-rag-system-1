"""This file checks questions before answering and answers before sending them."""

from __future__ import annotations
import re
from collections.abc import Sequence
from ..logging_utils import LOG
from ..models import Action, Rule, Verdict
from ..text import clean_text, luhn_valid
from .normalization import _rx, deobfuscate
from .rules import BENIGN_CONTEXT, INPUT_RULES


class InputPolicy:
    """Screens untrusted user input before any retrieval or generation happens."""

    def __init__(self, rules: Sequence[Rule] = tuple(INPUT_RULES)) -> None:
        self.rules = list(rules)

    def evaluate(self, question: str) -> Verdict:
        raw = clean_text(question)
        if not raw:
            return Verdict(reason="empty")
        norm = deobfuscate(raw)
        benign = bool(BENIGN_CONTEXT.search(raw))

        # Check cards arithmetically so a reference number is not mistaken for a PAN
        for m in re.finditer(r"(?:\d[ \-]?){13,19}", raw):
            digits = re.sub(r"\D", "", m.group(0))
            if luhn_valid(digits):
                return Verdict(
                    action=Action.BLOCK,
                    category="payment_data",
                    severity=1.0,
                    reason="message contains a valid payment card number",
                    matched=["luhn:" + digits[:4] + "..." + digits[-2:]],
                )

        best: Verdict | None = None
        for rule in self.rules:
            hit = rule.pattern.search(raw)
            if hit is None and rule.on_normalized:
                hit = rule.pattern.search(norm)
            if hit is None:
                continue
            if rule.exempt is not None and rule.exempt.search(raw):
                LOG.debug("rule %s matched but exempted by benign context", rule.name)
                continue
            if benign and rule.severity < 0.9:
                LOG.debug("rule %s suppressed by benign support context", rule.name)
                continue
            verdict = Verdict(
                action=rule.action,
                category=rule.category,
                severity=rule.severity,
                reason=rule.reason,
                matched=[rule.name],
            )
            if best is None or verdict.severity > best.severity:
                best = verdict
        return best or Verdict(reason="clean")


class OutputPolicy:
    """Screens model-generated text only."""

    _LEAK_MARKERS = (
        "you are the official",
        "use only the provided sources",
        "critical rules",
        "rule a:",
        "rule b:",
        "system prompt",
        "sources:",
        "### instruction",
    )
    _PROFANITY = _rx(r"\b(fuck\w*|shit|bitch|asshole|bastard|cunt)\b")
    _EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")

    def __init__(self, canary: str) -> None:
        self.canary = canary

    def evaluate(self, answer: str, sources: Sequence[str]) -> Verdict:
        low = (answer or "").lower()
        if self.canary and self.canary.lower() in low:
            return Verdict(
                Action.BLOCK,
                "prompt_leak",
                1.0,
                "system prompt canary appeared in the output",
                ["canary"],
            )
        for marker in self._LEAK_MARKERS:
            if marker in low:
                return Verdict(
                    Action.BLOCK, "prompt_leak", 0.95, "output echoed system instructions", [marker]
                )
        if self._PROFANITY.search(low):
            return Verdict(
                Action.BLOCK,
                "offensive_output",
                0.9,
                "generated text contained offensive language",
                ["profanity"],
            )
        joined = " ".join(sources).lower()
        for email in self._EMAIL.findall(answer or ""):
            if email.lower() not in joined:
                return Verdict(
                    Action.BLOCK,
                    "pii_leak",
                    0.9,
                    "output contained an email address absent from the sources",
                    ["email"],
                )
        for m in re.finditer(r"(?:\d[ \-]?){13,19}", answer or ""):
            if luhn_valid(re.sub(r"\D", "", m.group(0))):
                return Verdict(
                    Action.BLOCK, "pii_leak", 1.0, "output contained a payment card number", ["pan"]
                )
        return Verdict(reason="clean")
