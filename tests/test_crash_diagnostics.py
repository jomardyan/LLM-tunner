from __future__ import annotations


def test_crash_diagnostics_creates_log(monkeypatch, tmp_path):
    import llm_tunner.diagnostics as diagnostics

    monkeypatch.setenv("LLM_TUNNER_HOME", str(tmp_path))
    monkeypatch.setattr(diagnostics, "_CRASH_LOG", None)
    monkeypatch.setattr(diagnostics.faulthandler, "enable", lambda **kwargs: None)

    diagnostics.configure_crash_diagnostics()

    assert (tmp_path / "logs" / "native-crash.log").exists()
    diagnostics._CRASH_LOG.close()
    diagnostics._CRASH_LOG = None
