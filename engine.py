"""
small engine for krave

Usage:
    from engine import KraveEngine

    engine = KraveEngine(
        ckpt_path="/path/to/Krave-2.5",
        config="inference/configs/config_671B.json"
    )

    response = engine.generate("Explain quantum computing.")
    print(response)

    engine.chat()
"""

import json
import os
import sys
from typing import List, Optional

INFERENCE_DIR = os.path.join(os.path.dirname(__file__), "inference")
if INFERENCE_DIR not in sys.path:
    sys.path.insert(0, INFERENCE_DIR)

from generate import generate as generate_tokens
from model import ModelArgs, Transformer
from safetensors.torch import load_model
from transformers import AutoTokenizer


class KraveEngine:
    def __init__(
        self,
        ckpt_path: str,
        config: str = "inference/configs/config_671B.json",
        max_new_tokens: int = 200,
        temperature: float = 0.7,
        device: str = "cuda",
        weights_file: str | None = None,
    ):
        self.ckpt_path = ckpt_path
        self.config_path = config
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device
        self.weights_file = weights_file
        self._model = None
        self._tokenizer = None
        self._args = None
        self._loaded = False

    def _weight_file(self) -> str:
        if self.weights_file:
            return self.weights_file
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        rank = int(os.getenv("RANK", "0"))
        return os.path.join(self.ckpt_path, f"model{rank}-mp{world_size}.safetensors")

    def _load(self):
        if self._loaded:
            return

        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is not installed. Krave Engine requires torch==2.4.1.") from exc

        with open(self.config_path) as f:
            self._args = ModelArgs(**json.load(f))

        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(42)
        with torch.device(self.device):
            self._model = Transformer(self._args)

        self._tokenizer = AutoTokenizer.from_pretrained(self.ckpt_path)

        weight_file = self._weight_file()
        if not os.path.exists(weight_file):
            raise FileNotFoundError(
                f"Missing weight file: {weight_file}. Put your custom weights in the checkpoint directory."
            )

        load_model(self._model, weight_file)
        self._loaded = True

    def generate(self, prompt: str) -> str:
        self._load()
        tokens = self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        output_tokens = generate_tokens(
            self._model,
            [tokens],
            self.max_new_tokens,
            self._tokenizer.eos_token_id,
            self.temperature,
        )
        return self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    def generate_batch(self, prompts: List[str]) -> List[str]:
        self._load()
        assert len(prompts) <= self._args.max_batch_size
        all_tokens = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
            )
            for p in prompts
        ]
        output_tokens = generate_tokens(
            self._model,
            all_tokens,
            self.max_new_tokens,
            self._tokenizer.eos_token_id,
            self.temperature,
        )
        return self._tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    def chat(self, first_message: Optional[str] = None):
        self._load()
        messages = []
        if first_message:
            messages.append({"role": "user", "content": first_message})
            self._respond(messages)

        while True:
            try:
                user_input = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if user_input == "/exit":
                break
            if user_input == "/clear":
                messages.clear()
                continue
            if not user_input:
                continue
            messages.append({"role": "user", "content": user_input})
            reply = self._respond(messages)
            messages.append({"role": "assistant", "content": reply})

    def _respond(self, messages) -> str:
        tokens = self._tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        output_tokens = generate_tokens(
            self._model,
            [tokens],
            self.max_new_tokens,
            self._tokenizer.eos_token_id,
            self.temperature,
        )
        reply = self._tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        print(reply)
        return reply


if __name__ == "__main__":
    print("Krave Prepared")
