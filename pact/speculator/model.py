"""
Speculator inference: load the fine-tuned LoRA adapter and score
a thought chunk as ready (1) or keep-thinking (0).
"""

from __future__ import annotations

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from pact.speculator.dataset import build_input_text


class Speculator:
    """
    Wraps the LoRA-adapted sequence classifier.

    Call .score(prompt, thought_so_far, tool_name) -> float
    to get the probability that the thought is ready for tool execution.
    """

    def __init__(
        self,
        base_model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
        adapter_path: str = "checkpoints/speculator/adapter",
        device: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForSequenceClassification.from_pretrained(
            base_model_name,
            num_labels=2,
            torch_dtype=torch.bfloat16,
        )
        self.model = PeftModel.from_pretrained(base, adapter_path)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score(self, prompt: str, thought_so_far: str, tool_name: str) -> float:
        """
        Return P(ready_to_execute | prompt, thought, tool).
        Value in [0, 1].  Compare against tau to decide.
        """
        record = {
            "prompt": prompt,
            "cumulative_thought": thought_so_far,
            "target_tool_call": {"tool_name": tool_name},
        }
        text = build_input_text(record)
        enc = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**enc).logits  # (1, 2)
        probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item()  # P(label=1)

    def is_ready(self, prompt: str, thought_so_far: str, tool_name: str, tau: float) -> bool:
        return self.score(prompt, thought_so_far, tool_name) >= tau
