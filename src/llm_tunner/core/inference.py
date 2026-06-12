"""Model loading and text generation.

Loads a Hugging Face causal LM (optionally with a LoRA adapter), applies the model's
chat template, and generates. Used both for plain chat and as the generator in the RAG
query path. The model is cached so repeated chats don't reload weights.
"""

from __future__ import annotations

from dataclasses import dataclass

from .device import detect_device
from .rag import RagAnswer, RagIndex


@dataclass
class GenerationConfig:
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


class ChatModel:
    """A loaded causal LM + tokenizer, with optional LoRA adapter."""

    def __init__(self, model_id: str, adapter_path: str | None = None) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self._model = None
        self._tokenizer = None

    def load(self) -> None:
        import torch  # type: ignore
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

        info = detect_device()
        dtype = torch.bfloat16 if info.kind == "cuda" else torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        device_map = "auto" if info.kind == "cuda" else None
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=dtype, device_map=device_map
        )
        if device_map is None:
            self._model = self._model.to("mps" if info.kind == "mps" else "cpu")

        if self.adapter_path:
            from peft import PeftModel  # type: ignore

            self._model = PeftModel.from_pretrained(self._model, self.adapter_path)

        self._model.eval()

    def _ensure(self) -> None:
        if self._model is None:
            self.load()

    def generate(self, messages: list[dict], config: GenerationConfig | None = None) -> str:
        """Generate an assistant reply for a list of chat `messages`."""
        import torch  # type: ignore

        self._ensure()
        config = config or GenerationConfig()
        tok = self._tokenizer
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=tok.eos_token_id,
            )
        generated = out[0][inputs["input_ids"].shape[1] :]
        return tok.decode(generated, skip_special_tokens=True).strip()

    def chat(self, query: str, history: list[dict] | None = None,
             config: GenerationConfig | None = None) -> str:
        messages = list(history or [])
        messages.append({"role": "user", "content": query})
        return self.generate(messages, config)

    def rag_answer(self, query: str, index: RagIndex, top_k: int | None = None,
                   config: GenerationConfig | None = None) -> RagAnswer:
        """Retrieve from a knowledge base and answer with citations."""
        contexts = index.retrieve(query, top_k=top_k)
        if not contexts:
            return RagAnswer(
                answer="I couldn't find anything relevant in the knowledge base.",
                contexts=[],
            )
        messages = index.build_prompt(query, contexts)
        answer = self.generate(messages, config)
        return RagAnswer(answer=answer, contexts=contexts)
