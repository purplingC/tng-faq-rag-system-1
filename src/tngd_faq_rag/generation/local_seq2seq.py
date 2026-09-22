"""This file writes answers with a local transformers model, when you opt in."""

from __future__ import annotations
from collections.abc import Sequence
from ..config import Config
from ..models import Candidate, Generation
from ..store import MetadataStore
from ..text import clean_text, tokenize
from .base import Generator, PromptBuilder


class LocalSeq2SeqGenerator(Generator):
    """Local abstractive generation through transformers, opt-in."""

    name = "local-seq2seq"

    def __init__(self, cfg: Config, prompt_builder: PromptBuilder) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # pyright: ignore[reportMissingImports]

        self.cfg = cfg
        self.pb = prompt_builder
        self.tok = AutoTokenizer.from_pretrained(cfg.local_model, use_fast=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(cfg.local_model)
        self.model.eval()
        limit = int(getattr(self.tok, "model_max_length", 512) or 512)
        self.max_input = min(limit, 1024)
        self.pb.max_input_tokens = min(self.pb.max_input_tokens, self.max_input - 32)

    def _is_instruction_echo(self, out: str) -> bool:
        instr = self.pb.instructions().lower()
        o = " ".join((out or "").lower().split())
        if not o:
            return True
        instr_tokens = set(tokenize(instr))
        out_tokens = set(tokenize(o))
        if not out_tokens:
            return True
        overlap = len(out_tokens & instr_tokens) / len(out_tokens)
        return overlap >= 0.7 or ("not_in_kb" in o and len(o) > 40)

    def generate(
        self, question: str, candidates: Sequence[Candidate], store: MetadataStore
    ) -> Generation:
        import torch  # pyright: ignore[reportMissingImports]

        prompt, used = self.pb.build(question, candidates)
        enc = self.tok(prompt, return_tensors="pt", truncation=True, max_length=self.max_input)
        with torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=220,
                num_beams=4,
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=3,
            )
        text = self.tok.decode(out[0], skip_special_tokens=True).strip()
        if not text or "NOT_IN_KB" in text.upper() or self._is_instruction_echo(text):
            return Generation(text="", backend=self.name, raw=text, abstained=True)
        return Generation(
            text=clean_text(text), backend=self.name, used_source_indices=used, raw=text
        )
