# DesireeIA
# Copyright (c) Passaro Francesco Paolo. All rights reserved.
# Licensed under the DesireeIA License - see LICENSE and the "License"
# section of README.md for full terms: no modification, no unauthorized
# integration, no AI training/ingestion without explicit written consent
# from the author.

"""Simple start/stop/status/restart control for desireeia-server, run as a
background process - no extra dependency, stdlib only (matches the rest of
this package: everything installs in one shot).

    desireeia-server-ctl start [-- <desireeia-server args>]
    desireeia-server-ctl stop
    desireeia-server-ctl restart [-- <desireeia-server args>]
    desireeia-server-ctl status

The PID and log file live under the same data directory desireeia-server
itself already uses (~/.desireeia by default, or --data-dir), so `start`
without arguments and `stop` never need to agree on anything else.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from .config import DESIREEIA_DATA_DIRNAME

PID_FILE_NAME = "server.pid"
LOG_FILE_NAME = "server.log"


def _data_dir(argv: list[str]) -> Path:
    # Mirrors config.py's own --data-dir handling, without importing the
    # full argparse setup in config.py (that one requires --model/--models-dir
    # context this control script doesn't have).
    for i, arg in enumerate(argv):
        if arg == "--data-dir" and i + 1 < len(argv):
            return Path(argv[i + 1])
        if arg.startswith("--data-dir="):
            return Path(arg.split("=", 1)[1])
    return Path.home() / DESIREEIA_DATA_DIRNAME


def _is_running(pid: int) -> bool:
    if os.name == "nt":
        # os.kill(pid, 0) is not a reliable liveness check on Windows
        # (verified: it reported a still-alive process as dead) - ask the
        # OS directly via tasklist instead.
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            capture_output=True, text=True,
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _read_pid(pid_file: Path) -> Optional[int]:
    if not pid_file.exists():
        return None
    try:
        return int(pid_file.read_text().strip())
    except ValueError:
        return None


def start(extra_args: list[str]) -> int:
    data_dir = _data_dir(extra_args)
    data_dir.mkdir(parents=True, exist_ok=True)
    pid_file = data_dir / PID_FILE_NAME
    log_file = data_dir / LOG_FILE_NAME

    existing = _read_pid(pid_file)
    if existing is not None and _is_running(existing):
        print(f"desireeia-server is already running (pid {existing})")
        return 0

    kwargs: dict = {}
    if os.name == "nt":
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        # Best-effort: lets the child escape a parent job object that would
        # otherwise kill it when the launching process/shell exits (e.g. a
        # CI runner or sandboxed terminal). Some restricted job objects deny
        # this - CreateProcess still succeeds either way, just without the
        # breakaway taking effect.
        flags |= getattr(subprocess, "CREATE_BREAKAWAY_FROM_JOB", 0)
        kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    with open(log_file, "ab") as log:
        proc = subprocess.Popen(
            [sys.executable, "-m", "desireeiaserver", *extra_args],
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            **kwargs,
        )
    pid_file.write_text(str(proc.pid))
    print(f"desireeia-server started (pid {proc.pid})")
    print(f"  log:  {log_file}")
    print(f"  stop: desireeia-server-ctl stop")
    return 0


def stop() -> int:
    data_dir = _data_dir([])
    pid_file = data_dir / PID_FILE_NAME
    pid = _read_pid(pid_file)
    if pid is None:
        print("desireeia-server is not running (no pid file)")
        return 0
    if not _is_running(pid):
        print("desireeia-server is not running (stale pid file, cleaning up)")
        pid_file.unlink(missing_ok=True)
        return 0

    if os.name == "nt":
        # taskkill /T (tree) rather than os.kill(pid): on Windows, python.exe
        # under a venv can be a launcher shim whose own PID differs from the
        # real worker process it spawns (verified: Popen(...).pid and the
        # child's own os.getpid() do not match on this platform) - killing
        # only the tracked PID can leave the actual server running orphaned.
        # /T kills the whole process tree regardless of that indirection.
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True)
    else:
        # start() used start_new_session=True, so pid is also the process
        # group id: killpg reaches any subprocess the server itself might
        # spawn, not just the tracked pid.
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            os.kill(pid, signal.SIGTERM)
        for _ in range(20):
            if not _is_running(pid):
                break
            time.sleep(0.5)
        else:
            print(f"desireeia-server (pid {pid}) did not stop in time, sending SIGKILL")
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                os.kill(pid, signal.SIGKILL)

    pid_file.unlink(missing_ok=True)
    print(f"desireeia-server stopped (pid {pid})")
    return 0


def status() -> int:
    data_dir = _data_dir([])
    pid_file = data_dir / PID_FILE_NAME
    pid = _read_pid(pid_file)
    if pid is not None and _is_running(pid):
        print(f"running (pid {pid})")
        return 0
    print("stopped")
    return 1


def main(argv: Optional[list[str]] = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(
        prog="desireeia-server-ctl",
        description="Start/stop/status/restart desireeia-server as a background process.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p_start = sub.add_parser("start", help="Start desireeia-server in the background.")
    p_start.add_argument("server_args", nargs=argparse.REMAINDER,
                          help="Arguments forwarded to desireeia-server (e.g. -- --model model.gguf --port 8080).")
    sub.add_parser("stop", help="Stop the running desireeia-server.")
    p_restart = sub.add_parser("restart", help="Stop then start desireeia-server.")
    p_restart.add_argument("server_args", nargs=argparse.REMAINDER)
    sub.add_parser("status", help="Report whether desireeia-server is running.")

    args = parser.parse_args(argv)
    server_args = getattr(args, "server_args", [])
    if server_args and server_args[0] == "--":
        server_args = server_args[1:]

    if args.command == "start":
        raise SystemExit(start(server_args))
    if args.command == "stop":
        raise SystemExit(stop())
    if args.command == "restart":
        stop()
        raise SystemExit(start(server_args))
    if args.command == "status":
        raise SystemExit(status())


if __name__ == "__main__":
    main()
