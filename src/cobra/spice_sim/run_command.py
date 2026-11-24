"""Run a simulator command, with an optional time limit."""

import contextlib
import math
import os
import subprocess
from pathlib import Path

import psutil

from cobra.configuration.configuration import ConfigurationError
from cobra.configuration.setting import CobraSetting

#: How long a stopped command may take to end before it is killed [s]
_GRACE_PERIOD = 5.0

#: The time limit every simulator backend offers
TIMEOUT_SETTING = CobraSetting(
    name="simulation_timeout",
    dtype=float,
    default=0.0,
    description=(
        "Seconds one simulation may run before it is stopped and the trial counts as failed.\n"
        "0 sets no limit."
    ),
    advanced=True,
)


def check_timeout(simulation_timeout: float) -> float:
    """*simulation_timeout* as seconds.

    Raises:
        ConfigurationError: It is not a finite number of at least 0.
    """
    if (
        isinstance(simulation_timeout, bool)
        or not isinstance(simulation_timeout, (int, float))
        or not math.isfinite(simulation_timeout)
        or simulation_timeout < 0
    ):
        raise ConfigurationError(
            f"simulation_timeout must be a number of seconds of at least 0, got {simulation_timeout!r}"
        )
    return float(simulation_timeout)


def run_command(
    command: list[str], cwd: str | Path, timeout: float = 0.0, stdin: int | None = None
) -> subprocess.CompletedProcess[str]:
    """Run *command* in *cwd* and return its exit code and output.

    Args:
        command: The command and its arguments.
        cwd: The directory to run it in.
        timeout: Seconds it may run; 0 sets no limit.
        stdin: As for :class:`subprocess.Popen`.

    Raises:
        OSError: The command cannot be started.
        subprocess.TimeoutExpired: It ran into *timeout*. It and the processes it started
            have been stopped.
    """
    if not timeout:
        return subprocess.run(
            command, capture_output=True, text=True, cwd=cwd, check=False, stdin=stdin
        )
    process = subprocess.Popen(
        command,
        stdin=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except BaseException:
        _stop(process)
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def _stop(process: "subprocess.Popen[str]") -> None:
    """End *process* and the processes it started, first by request, then by force."""
    processes = _tree(process.pid)
    for running in processes:
        with contextlib.suppress(psutil.Error):
            running.terminate()
    _, alive = psutil.wait_procs(processes, timeout=_GRACE_PERIOD)
    for running in {started for parent in alive for started in _tree(parent.pid)}:
        with contextlib.suppress(psutil.Error):
            running.kill()
    process.wait()
    # Not read to the end: a process that has left the tree could keep them open. On Windows,
    # communicate() reads them in threads, which close them.
    if os.name == "posix":
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                stream.close()


def _tree(pid: int) -> list[psutil.Process]:
    """Process *pid* and everything it started (MPI ranks, a simulator behind a wrapper script)."""
    try:
        process = psutil.Process(pid)
        return [process, *process.children(recursive=True)]
    except psutil.NoSuchProcess:
        return []
