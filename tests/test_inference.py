from __future__ import annotations

from llm_tunner.core.inference import _format_messages


class ChatTokenizer:
    chat_template = "configured"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return f"templated:{messages[0]['content']}"


class BaseTokenizer:
    chat_template = None


def test_format_messages_uses_tokenizer_chat_template():
    prompt = _format_messages(ChatTokenizer(), [{"role": "user", "content": "Hello"}])
    assert prompt == "templated:Hello"


def test_format_messages_falls_back_for_base_tokenizer():
    prompt = _format_messages(
        BaseTokenizer(),
        [
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hello"},
        ],
    )
    assert prompt == "System: Be concise.\nUser: Hello\nAssistant:"
