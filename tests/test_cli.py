"""CLI tests. Torch-free: cli imports core/tasks lazily, so these run in the core suite.

The `isolated_home` autouse fixture (conftest) points LLM_TUNNER_HOME at a temp dir, so
`list`/storage commands read empty folders rather than real user data.
"""

from __future__ import annotations

from llm_tunner.cli import ConsoleSignals, build_parser, main

_REQUIRED = {
    "build-kb": ["--name", "kb", "--pdf", "a.pdf"],
    "gen-dataset": ["--pdf", "a.pdf"],
    "chat": ["--query", "hi"],
    "finetune": ["--dataset", "d.jsonl"],
}


def test_parser_wires_every_subcommand():
    parser = build_parser()
    for command in ("devices", "list", "build-kb", "gen-dataset", "chat", "finetune"):
        args = parser.parse_args([command, *_REQUIRED.get(command, [])])
        assert callable(args.func)


def test_parser_chat_defaults_and_flags():
    args = build_parser().parse_args(["chat", "--query", "hi", "--kb", "docs", "--no-rag"])
    assert args.query == "hi"
    assert args.kb == "docs"
    assert args.no_rag is True
    assert args.top_k == 4  # default
    assert args.provider == "auto"


def test_console_signals_verbose_writes_to_stderr(capsys):
    signals = ConsoleSignals(quiet=False)
    signals.progress.emit(1, 2, "halfway")
    signals.log.emit("loading")

    class _Metric:
        step = 3
        loss = 0.5

    signals.metric.emit(_Metric())
    err = capsys.readouterr().err
    assert "1/2" in err
    assert "loading" in err
    assert "loss=0.5000" in err


def test_console_signals_quiet_suppresses_output(capsys):
    signals = ConsoleSignals(quiet=True)
    signals.log.emit("nope")
    signals.progress.emit(1, 1, "nope")
    assert capsys.readouterr().err == ""


def test_main_without_command_prints_help_and_returns_1(capsys):
    assert main([]) == 1
    assert "usage" in capsys.readouterr().out.lower()


def test_main_devices_runs(capsys):
    assert main(["devices"]) == 0
    assert "Recommended models:" in capsys.readouterr().out


def test_main_list_on_empty_home(capsys):
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "Knowledge bases:" in out
    assert "Datasets:" in out
    assert "Adapters:" in out
    assert "(none)" in out


def test_main_reports_clean_error_not_traceback(capsys):
    # Fine-tuning a non-existent dataset should fail cleanly (exit 1, no traceback).
    assert main(["finetune", "--dataset", "does-not-exist.jsonl"]) == 1
    assert "error:" in capsys.readouterr().err
