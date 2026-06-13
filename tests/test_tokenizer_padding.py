from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_tunner.core.inference import ensure_pad_token


def test_ensure_pad_token_keeps_existing_pad_token():
    tokenizer = SimpleNamespace(pad_token="<pad>", eos_token="<eos>", bos_token="<bos>")
    ensure_pad_token(tokenizer)
    assert tokenizer.pad_token == "<pad>"


def test_ensure_pad_token_falls_back_to_eos():
    tokenizer = SimpleNamespace(pad_token=None, eos_token="<eos>", bos_token="<bos>")
    ensure_pad_token(tokenizer)
    assert tokenizer.pad_token == "<eos>"


def test_ensure_pad_token_falls_back_to_bos():
    tokenizer = SimpleNamespace(pad_token=None, eos_token=None, bos_token="<bos>")
    ensure_pad_token(tokenizer)
    assert tokenizer.pad_token == "<bos>"


def test_ensure_pad_token_raises_when_no_token_available():
    tokenizer = SimpleNamespace(pad_token=None, eos_token=None, bos_token=None)
    with pytest.raises(RuntimeError, match="pad/eos/bos"):
        ensure_pad_token(tokenizer)
