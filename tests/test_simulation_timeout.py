"""The simulators' time limit, with scripts in place of Xyce and VACASK."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest

from cobra.configuration.configuration import ConfigurationError
from cobra.spice_sim.run_command import check_timeout, run_command
from cobra.spice_sim.vacask_simulator import VacaskSimulator
from cobra.spice_sim.xyce_simulator import XyceSimulator
from tests.conftest import netlist_path

SIMULATORS = [(XyceSimulator, "xyce_command", "Xyce"), (VacaskSimulator, "vacask_command", "VACASK")]

#: Long enough for a command below to start [s]; processes start slowly on Windows
LIMIT = 1.0 if os.name == "posix" else 5.0

#: Writes its process id and then runs far beyond the limit
HANG = """\
import os, sys, time
from pathlib import Path
part = Path(sys.argv[1] + ".part")
part.write_text(str(os.getpid()))
part.rename(sys.argv[1])
time.sleep(60)
"""

#: Runs a command in a thread, as COBRA does, and presses Ctrl+C once it has started
CTRL_C = """\
import ctypes, os, signal, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, wait
from pathlib import Path
from cobra.spice_sim.run_command import run_command

def press():
    while not Path(sys.argv[2]).exists():
        time.sleep(0.05)
    if os.name == "posix":
        os.killpg(0, signal.SIGINT)
    else:
        os.kill(0, signal.CTRL_C_EVENT)

if os.name != "posix":
    ctypes.windll.kernel32.SetConsoleCtrlHandler(None, False)
threading.Thread(target=press, daemon=True).start()
with ThreadPoolExecutor() as executor:
    command = [sys.executable, sys.argv[1], sys.argv[2]]
    wait([executor.submit(run_command, command, Path.cwd(), float(sys.argv[3]))])
"""


def _script(tmp_path: Path, code: str, name: str = "script.py") -> str:
    script = tmp_path / name
    script.write_text(code, encoding="utf-8")
    return str(script)


def _wrapper(tmp_path: Path, command: list[str]) -> str:
    """An executable script that runs *command*."""
    line = " ".join(f'"{part}"' for part in command)
    if os.name != "posix":
        return _script(tmp_path, f"@echo off\n{line}\n", "simulator.bat")
    script = Path(_script(tmp_path, f"#!/bin/sh\n{line}\nexit $?\n", "simulator.sh"))
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    return str(script)


def _has_ended(pid_file: Path) -> bool:
    """Whether the process that wrote *pid_file* ends within a few seconds."""
    try:
        process = psutil.Process(int(pid_file.read_text(encoding="utf-8")))
        deadline = time.monotonic() + 5
        while process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)
    except psutil.NoSuchProcess:
        pass
    return True


@pytest.mark.parametrize("timeout", [0.0, 30.0])
def test_a_command_within_the_limit_returns_its_output(tmp_path, timeout):
    code = "import sys; print('out'); print('err', file=sys.stderr); sys.exit(3)"

    proc = run_command([sys.executable, "-c", code], tmp_path, timeout)

    assert (proc.returncode, proc.stdout, proc.stderr) == (3, "out\n", "err\n")


def test_a_large_output_does_not_block_the_command(tmp_path):
    proc = run_command([sys.executable, "-c", "print('x' * 1_000_000)"], tmp_path, 30.0)

    assert len(proc.stdout) == 1_000_001


def test_stopping_reaches_the_process_behind_a_wrapper_script(tmp_path):
    wrapper = _wrapper(tmp_path, [sys.executable, _script(tmp_path, HANG), str(tmp_path / "pid")])

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([wrapper], tmp_path, LIMIT)

    assert time.monotonic() - started < LIMIT + 4
    assert _has_ended(tmp_path / "pid")


@pytest.mark.parametrize("timeout", [0.0, 60.0])
def test_ctrl_c_ends_a_command_run_by_a_thread(tmp_path, timeout):
    pid_file = tmp_path / "pid"
    command = [sys.executable, _script(tmp_path, CTRL_C, "run.py"), _script(tmp_path, HANG), str(pid_file), str(timeout)]
    # A console of its own, so that the Ctrl+C does not reach the test run.
    flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    with (tmp_path / "log").open("w") as log:
        run = subprocess.Popen(
            command, cwd=tmp_path, stdout=log, stderr=log, start_new_session=os.name == "posix", creationflags=flags
        )
        try:
            run.wait(40)
        except subprocess.TimeoutExpired:
            for process in psutil.Process(run.pid).children(recursive=True):
                process.kill()
            run.kill()
            pytest.fail(f"still running: {(tmp_path / 'log').read_text(encoding='utf-8')}")
    assert _has_ended(pid_file)


def test_a_command_that_ignores_the_request_is_killed(tmp_path, monkeypatch):
    import cobra.spice_sim.run_command as module

    monkeypatch.setattr(module, "_GRACE_PERIOD", 0.3)
    script = _script(tmp_path, "import signal\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\n" + HANG)

    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        run_command([sys.executable, script, str(tmp_path / "pid")], tmp_path, LIMIT)

    assert time.monotonic() - started < LIMIT + 4
    assert _has_ended(tmp_path / "pid")


@pytest.mark.parametrize(("simulator", "command", "label"), SIMULATORS)
def test_a_simulation_that_hangs_is_a_failed_simulation(tmp_path, caplog, simulator, command, label):
    netlist = tmp_path / "circuit.cir"
    netlist.write_text(netlist_path("minimal_ac").read_text(encoding="utf-8"), encoding="utf-8")
    parsed = XyceSimulator.netlist_parser.parse_file(netlist)
    wrapper = _wrapper(tmp_path, [sys.executable, _script(tmp_path, HANG), str(tmp_path / "pid")])
    hanging = simulator(**{command: wrapper, "simulation_timeout": LIMIT})

    with caplog.at_level("WARNING"):
        result = hanging.run_simulation(str(netlist), parsed)

    assert result is None
    assert f"{label} stopped after {LIMIT:g} s (simulation_timeout)" in caplog.text
    assert _has_ended(tmp_path / "pid")


@pytest.mark.parametrize(("simulator", "command", "label"), SIMULATORS)
def test_there_is_no_limit_by_default(simulator, command, label):
    assert simulator().timeout == 0.0
    setting = next(s for s in simulator._settings if s.name == "simulation_timeout")
    assert setting.default == 0.0
    assert setting.advanced


@pytest.mark.parametrize("timeout", [-1, float("nan"), float("inf"), True, "60"])
def test_invalid_limits_are_rejected(timeout):
    with pytest.raises(ConfigurationError, match="simulation_timeout"):
        check_timeout(timeout)
