from __future__ import annotations

from llm_tunner.core.inference import ChatModel, _contextualize_query, _format_messages


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


def test_contextualize_query_prepends_previous_user_turn():
    history = [
        {"role": "user", "content": "Tell me about the Q3 plan."},
        {"role": "assistant", "content": "It covers hiring and budget."},
    ]
    out = _contextualize_query("What about its cost?", history)
    assert "Q3 plan" in out
    assert "its cost" in out


def test_contextualize_query_without_history_is_identity():
    assert _contextualize_query("hello", None) == "hello"
    assert _contextualize_query("hello", []) == "hello"
    assert _contextualize_query("hello", [{"role": "assistant", "content": "x"}]) == "hello"


def test_chat_prepends_system_prompt_then_history_then_query():
    model = ChatModel("x")
    captured: dict = {}

    def fake_generate(messages, config=None):
        captured["messages"] = messages
        captured["config"] = config
        return "ok"

    model.generate = fake_generate
    out = model.chat(
        "hi",
        history=[{"role": "user", "content": "prev"}, {"role": "assistant", "content": "a"}],
        system_prompt="Be terse",
    )
    assert out == "ok"
    messages = captured["messages"]
    assert messages[0] == {"role": "system", "content": "Be terse"}
    assert messages[1] == {"role": "user", "content": "prev"}
    assert messages[-1] == {"role": "user", "content": "hi"}


def test_rag_answer_contextualizes_retrieval_and_forwards_history():
    class FakeCtx:
        citation = "doc.pdf p.1"
        text = "context"
        source = "doc.pdf"
        page_number = 1
        score = 0.9

    class FakeIndex:
        def __init__(self) -> None:
            self.retrieve_query = None
            self.build_kwargs = None

        def retrieve(self, query, top_k=None):
            self.retrieve_query = query
            return [FakeCtx()]

        def build_prompt(self, query, contexts, history=None, system_prefix=""):
            self.build_kwargs = {"history": history, "system_prefix": system_prefix}
            return [{"role": "user", "content": query}]

    model = ChatModel("x")
    model.generate = lambda messages, config=None: "answer"
    index = FakeIndex()
    history = [
        {"role": "user", "content": "the Q3 plan"},
        {"role": "assistant", "content": "..."},
    ]
    result = model.rag_answer("its cost?", index, history=history, system_prompt="terse")

    assert result.answer == "answer"
    assert "the Q3 plan" in index.retrieve_query  # retrieval query was contextualized
    assert index.build_kwargs["history"] == history
    assert index.build_kwargs["system_prefix"] == "terse"
