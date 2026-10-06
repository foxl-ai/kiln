"""engine/watchdog.py: a blocked device section ends the rank loudly and names its calls, a healthy one never does."""

import time

import pytest

from kiln.engine import watchdog


def _wd(timeout, **kw):
    exits = []
    wd = watchdog.ExecWatchdog(3, timeout, poll=0.01, exit_fn=exits.append, **kw)
    return wd, exits


def test_a_section_longer_than_the_timeout_ends_the_rank_and_names_its_calls(capsys):
    wd, exits = _wd(0.1)
    try:
        wd.launched("decode", (16, 264))
        wd.launched("prefill", (1024, 264))
        with wd.blocking("reading back a decode call"):
            deadline = time.monotonic() + 5
            while not exits and time.monotonic() < deadline:
                time.sleep(0.01)
    finally:
        wd.close()
    assert exits == [watchdog.EXIT_HANG]
    assert wd.fired
    err = capsys.readouterr().err
    assert "KILN EXEC WATCHDOG rank 3: reading back a decode call blocked for" in err
    assert "decode (16, 264)" in err and "prefill (1024, 264)" in err
    assert err.index("decode (16, 264)") < err.index("prefill (1024, 264)")  # oldest first


def test_short_sections_and_idle_time_never_fire():
    wd, exits = _wd(0.2)
    try:
        for _ in range(5):
            with wd.blocking("launching decode"):
                time.sleep(0.02)
        time.sleep(0.5)  # idle (between sections) is not watched, however long
    finally:
        wd.close()
    assert exits == [] and not wd.fired


def test_nested_sections_keep_the_outer_start():
    wd, exits = _wd(0.15)
    try:
        with wd.blocking("outer"):
            time.sleep(0.1)
            with wd.blocking("inner"):
                deadline = time.monotonic() + 5
                while not exits and time.monotonic() < deadline:
                    time.sleep(0.01)
    finally:
        wd.close()
    assert exits == [watchdog.EXIT_HANG]


def test_an_error_inside_a_section_is_reraised_with_the_calls(capsys):
    wd, exits = _wd(10)
    try:
        wd.launched("mixed", (8, 1024))
        with pytest.raises(RuntimeError, match="status=1"):
            with wd.blocking("reading back a prefill call"):
                raise RuntimeError("Failed to schedule neff execution. status=1")
        with wd.blocking("again"):  # the section closed: a later one starts fresh
            pass
    finally:
        wd.close()
    assert exits == []
    err = capsys.readouterr().err
    assert "RuntimeError in reading back a prefill call" in err and "mixed (8, 1024)" in err


def test_make_is_off_without_a_neuron_device_or_with_timeout_zero(monkeypatch):
    monkeypatch.delenv("KILN_EXEC_TIMEOUT_S", raising=False)
    assert watchdog.timeout_s() == watchdog.DEFAULT_TIMEOUT_S
    assert watchdog.make(0, "cpu") is None
    monkeypatch.setenv("KILN_EXEC_TIMEOUT_S", "0")
    assert watchdog.make(0, "neuron") is None
    monkeypatch.setenv("KILN_EXEC_TIMEOUT_S", "45")
    wd = watchdog.make(5, "neuron")
    try:
        assert wd is not None and wd.timeout == 45.0 and wd.rank == 5
    finally:
        wd.close()
