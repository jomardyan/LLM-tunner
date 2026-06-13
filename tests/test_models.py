from __future__ import annotations

import sys
from types import ModuleType

import pytest

from llm_tunner.core.models import huggingface_authenticated, login_huggingface


def test_huggingface_authenticated_uses_standard_token(monkeypatch):
    module = ModuleType("huggingface_hub")
    module.get_token = lambda: "hf_test"
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    assert huggingface_authenticated()


def test_huggingface_login_rejects_empty_token():
    with pytest.raises(ValueError, match="access token"):
        login_huggingface("  ")


def test_huggingface_login_uses_provider_store(monkeypatch):
    saved = {}
    module = ModuleType("huggingface_hub")

    def login(*, token, add_to_git_credential):
        saved.update(token=token, add_to_git_credential=add_to_git_credential)

    module.login = login
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    login_huggingface(" hf_test ")
    assert saved == {"token": "hf_test", "add_to_git_credential": False}
