# desireeia-server

OpenAI-compatible API server and self-hosted chat web UI for the
[DesireeIA](../README.md) local LLM inference engine. The inference motor is
**always DesireeIA** — this package is a thin HTTP layer over the existing
[`desireeia`](../python/desireeia) Python wrapper and ships everything it needs.

The external project this surface mirrors (and generalizes past) is referred
to here only as **desireeialmn**; no code or documentation text is copied from
it. Convention of this repository: the words "llama" and "colibri" are not
used anywhere in code, docs or identifiers.

## Quick start

Published on PyPI since 2026-09-16:
https://pypi.org/project/desireeia-server/ (depends on
https://pypi.org/project/desireeia/).

### Install from PyPI

```bash
pip install desireeia-server
```

This also installs its `desireeia` dependency (the inference engine
wrapper, with prebuilt native binaries bundled — see
[desireeia's README](../Build/Python/README.md#bundled-platforms) for the
per-OS prerequisites, e.g. the VC++ Redistributable on Windows).

### Run

Identical on Windows, Linux and macOS — `desireeia-server` is a normal
console command once installed:

```bash
desireeia-server --model path/to/model.gguf --port 8080
# open http://127.0.0.1:8080
```

Without `--model`, the server runs in router mode: it manages whatever
checkpoints it finds in the models folder (default `<data-dir>/models`,
created automatically). Drop a GGUF in there, or upload it from the UI, and
it becomes selectable without a restart.

```bash
desireeia-server --models-dir ./models
```

By default the server runs in the foreground (logs print to the terminal).
To stop it: **Ctrl+C** in that terminal, on any OS.

### Start / stop / restart in the background

`desireeia-server-ctl` (installed alongside `desireeia-server`, identical
command on Windows/Linux/macOS) runs the server as a detached background
process and tracks it for you — no manual PID handling:

```bash
desireeia-server-ctl start -- --model path/to/model.gguf --port 8080
desireeia-server-ctl status     # "running (pid ...)" or "stopped"
desireeia-server-ctl stop
desireeia-server-ctl restart -- --model path/to/model.gguf --port 8080
```

Everything after `--` is forwarded to `desireeia-server` unchanged (same
flags as running it directly). Logs go to `<data-dir>/server.log`
(`~/.desireeia/server.log` by default) and the PID is tracked in
`<data-dir>/server.pid` - `stop` reads it from there, so it always works
even across different terminal sessions.

### Dev install from the repo (instead of PyPI)

```bash
pip install <desireeia_source>/python <desireeia_source>/DesireeIAServer   # Windows: use \ instead of /
```

The engine, backend and sampling surface you can steer:

```bash
desireeia-server --model model.gguf --backend auto --threads 8 \
    --n-gpu-layers 0 --temperature 0.7 --top-k 40 --top-p 0.95 \
    --repeat-penalty 1.1 --n-predict 512
```

Run `desireeia-server --help` for the full surface.

## Layout

- `desireeiaserver/` — the package: `config.py` (CLI/env), `engine.py` (the
  DesireeIA adapter), `app.py` (FastAPI factory), `api/` (routes), `static/`
  (generated UI bundle, copied from `../ui` via `sync_static.py`).
- `../ui/` — the web UI source (single-page chat app).
- `../python/` — the `desireeia` engine wrapper this package depends on.

## Development

```bash
python -m venv .venv                      # repo root, gitignored
.venv/Scripts/pip install -e ./python
.venv/Scripts/pip install -e ./DesireeIAServer[dev]
.venv/Scripts/python -m pytest DesireeIAServer/tests
```

## Roadmap

See `../ProjectsRequirements.md` §10. Bootstrap status: HTTP scaffold,
`/health`, `/props`, `/v1/models` shape, UI placeholder, tests with an
in-memory mock engine. Streaming, router/slots, tools and the full UI land in
the next iterations.