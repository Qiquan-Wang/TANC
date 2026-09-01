#!/usr/bin/env python3
"""Local server for the TANC Visual Builder.

Serves the static UI and (optionally) *runs* the generated Python so you can
see the figures without leaving the browser.

    python web/server.py            # then open http://localhost:8000

Dependency-free (stdlib only). The ``Run`` button POSTs code to ``/run``, which
executes it in a temp working directory and streams back stdout + any
``tda_result_*.png`` figures. This executes arbitrary Python — run it only
locally, on code you generated yourself.
"""
from __future__ import annotations

import base64
import glob
import json
import os
import subprocess
import sys
import tempfile
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)          # …/tanc  (so `import tanc` works)
RUN_TIMEOUT = int(os.environ.get("TDA_RUN_TIMEOUT", "1800"))   # seconds


# ── Python-environment ("kernel") discovery ──────────────────────────────────
# interpreter sub-paths inside an env root — Unix and Windows layouts
_ENV_EXES = ("bin/python", "bin/python3", "python.exe", os.path.join("Scripts", "python.exe"))


def _candidate_pythons() -> list[str]:
    """Find candidate Python interpreters the user might want to run code with.

    Cross-platform: covers Unix (``bin/python``) and Windows (``python.exe`` /
    ``Scripts\\python.exe``), local venvs, and conda/mamba/pyenv environments —
    including Windows miniforge under ``%USERPROFILE%\\AppData\\Local``.
    """
    cands: list[str] = [sys.executable]
    for var in ("VIRTUAL_ENV", "CONDA_PREFIX"):
        root = os.environ.get(var)
        if root:
            cands += [os.path.join(root, e) for e in _ENV_EXES]
    # local venvs near the repo (and a couple of levels up)
    for r in [REPO_ROOT, os.path.dirname(REPO_ROOT), os.path.dirname(os.path.dirname(REPO_ROOT))]:
        for name in (".venv", "venv", "env"):
            cands += [os.path.join(r, name, e) for e in _ENV_EXES]
    # conda / mamba roots — Unix, Windows-home, and Windows AppData\Local
    home = os.path.expanduser("~")
    conda_roots = []
    for base in ("miniconda3", "anaconda3", "miniforge3", "mambaforge"):
        conda_roots += [os.path.join(home, base),
                        os.path.join(home, "AppData", "Local", base),
                        os.path.join("C:" + os.sep, "ProgramData", base)]
    if os.environ.get("CONDA_PREFIX"):                       # base of the active conda install
        conda_roots.append(os.path.dirname(os.path.dirname(os.environ["CONDA_PREFIX"])))
    for root in conda_roots:
        cands += [os.path.join(root, "bin", "python"), os.path.join(root, "python.exe")]  # base env
        for envdir in glob.glob(os.path.join(root, "envs", "*")):                          # named envs
            cands += [os.path.join(envdir, e) for e in _ENV_EXES]
    cands += glob.glob(os.path.join(home, ".pyenv", "versions", "*", "bin", "python"))
    seen, out = set(), []
    for p in cands:
        rp = os.path.realpath(p)
        if rp in seen:
            continue
        seen.add(rp)
        if os.path.exists(p) and os.access(p, os.X_OK):
            out.append(p)
    return out


def _probe_env(python: str) -> dict:
    """Which relevant packages actually *import* under *python* (repo on PYTHONPATH).

    Uses a real import, not just ``find_spec`` — an installed-but-broken package
    (e.g. a torchvision with a missing dependency) reports ``False``, so the
    badge reflects what will really happen when the code runs.
    """
    probe = (
        "import json\n"
        "res = {}\n"
        "for k in ('torch', 'torchvision', 'tensorflow', 'tanc', 'gtda'):\n"
        "    try:\n"
        "        __import__(k); res[k] = True\n"
        "    except Exception:\n"
        "        res[k] = False\n"
        "print(json.dumps(res))"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    try:
        r = subprocess.run([python, "-c", probe], capture_output=True, text=True, timeout=60, env=env)
        line = (r.stdout or "").strip().splitlines()[-1] if r.stdout.strip() else "{}"
        return json.loads(line)
    except Exception:
        return {}


def _kernel_label(python: str) -> str:
    # …/<name>/bin/python  →  "<name>";  else the interpreter's parent-parent dir
    parts = python.split(os.sep)
    if "bin" in parts:
        i = parts.index("bin")
        if i >= 1:
            return parts[i - 1] or python
    return os.path.basename(os.path.dirname(os.path.dirname(python))) or python


_KERNEL_CACHE: list[dict] = []


def _list_kernels() -> list[dict]:
    global _KERNEL_CACHE
    cur = os.path.realpath(sys.executable)
    out = []
    for p in _candidate_pythons():
        info = _probe_env(p)
        out.append({
            "path": p,
            "label": _kernel_label(p),
            "torch": bool(info.get("torch")),
            "torchvision": bool(info.get("torchvision")),
            "tensorflow": bool(info.get("tensorflow")),
            "tanc": bool(info.get("tanc")),
            "gtda": bool(info.get("gtda")),
            "current": os.path.realpath(p) == cur,
        })
    _KERNEL_CACHE = out
    return out


def _best_kernel_path() -> str | None:
    """Best available interpreter: has tanc + a framework, preferring one
    that also has giotto-tda (so Mapper works).  Used as the /run fallback."""
    kernels = _KERNEL_CACHE or _list_kernels()
    usable = [k for k in kernels if k["tanc"] and (k["torch"] or k["tensorflow"])]
    usable.sort(key=lambda k: (k["gtda"], k["current"]), reverse=True)
    return usable[0]["path"] if usable else None


def _upgrade_for_gtda(python: str) -> tuple[str, str]:
    """If *python* lacks giotto-tda, swap in an env that has it **and the same
    framework** (so Mapper works without breaking a torch/tf-specific run).
    Returns (interpreter, note)."""
    kernels = _KERNEL_CACHE or _list_kernels()
    rp = os.path.realpath(python)
    cur = next((k for k in kernels if os.path.realpath(k["path"]) == rp), None)
    if cur and cur["gtda"]:
        return python, ""                      # already Mapper-capable
    want_torch = cur["torch"] if cur else True
    want_tf = cur["tensorflow"] if cur else False
    for k in kernels:
        if not k["gtda"]:
            continue
        if want_torch and not k["torch"]:
            continue
        if want_tf and not k["tensorflow"]:
            continue
        return k["path"], (f"[builder] the selected interpreter has no giotto-tda; "
                           f"running with '{k['path']}' instead (has giotto-tda).\n")
    return python, ""                          # no better option — run as-is


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=HERE, **kw)

    def log_message(self, fmt, *args):        # quieter logging
        sys.stderr.write("· " + (fmt % args) + "\n")

    def end_headers(self):
        # Development server: never let a browser cache the app. index.html
        # carries a ?v= cache-buster that is bumped by hand, and forgetting to
        # bump it after editing app.js silently serves the stale file.
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0].rstrip("/") == "/kernels":
            return self._json({"kernels": _list_kernels()})
        return super().do_GET()

    def do_POST(self):
        if self.path.rstrip("/") != "/run":
            self.send_error(404); return
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            code = payload.get("code", "")
            python = payload.get("python") or ""
        except Exception as exc:
            return self._json({"stderr": f"bad request: {exc}"}, status=400)
        self._run_stream(code, python)

    def _run_stream(self, code: str, python: str = "") -> None:
        """Execute the code, streaming NDJSON: one ``meta`` record, then a
        ``line`` record per stdout/stderr line (so the UI can show a live log
        and parse epoch-progress markers), then a final ``done`` record with
        the figures.  The child runs with ``python -u`` so lines arrive as
        they are printed rather than in 8 kB blocks."""
        if not (python and os.path.exists(python) and os.access(python, os.X_OK)):
            # No/invalid interpreter chosen → pick the best detected env
            # (one with giotto-tda if available), not just the server's python.
            python = _best_kernel_path() or sys.executable
        # If it can't do Mapper, upgrade to a giotto-tda env of the same framework.
        python, upgrade_note = _upgrade_for_gtda(python)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()

        def emit(obj) -> bool:
            try:
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
                self.wfile.flush()
                return True
            except (BrokenPipeError, ConnectionResetError):
                return False           # browser went away — stop politely

        emit({"type": "meta", "python": python, "note": upgrade_note})

        with tempfile.TemporaryDirectory(prefix="tda_run_") as work:
            script = os.path.join(work, "experiment.py")
            with open(script, "w") as fh:
                fh.write(code)
            # Make the repo importable without installing the package.
            env = dict(os.environ)
            env["PYTHONPATH"] = REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")
            env["MPLBACKEND"] = "Agg"
            proc = subprocess.Popen(
                [python, "-u", script],
                cwd=work, env=env, text=True, bufsize=1,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
            watchdog = threading.Timer(RUN_TIMEOUT, proc.kill)
            watchdog.start()
            client_gone = False
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    if not emit({"type": "line", "text": line.rstrip("\n")}):
                        client_gone = True
                        proc.kill()
                        break
                rc = proc.wait()
            finally:
                timed_out = not watchdog.is_alive() and proc.returncode is not None
                watchdog.cancel()
            if client_gone:
                return

            figures = []
            for png in sorted(glob.glob(os.path.join(work, "tda_result_*.png"))):
                with open(png, "rb") as fh:
                    figures.append(base64.b64encode(fh.read()).decode("ascii"))
            htmls = []
            for h in sorted(glob.glob(os.path.join(work, "tda_result_*.html"))):
                with open(h, "r", encoding="utf-8") as fh:
                    htmls.append(fh.read())
            emit({"type": "done", "rc": rc, "figures": figures, "html": htmls,
                  "timeout": bool(timed_out and rc != 0), "python": python})

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    port = int(os.environ.get("PORT", "8000"))
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as exc:
        print(f"Could not bind port {port}: {exc}", flush=True)
        print(f"Another server is probably running there. Try:  PORT=8001 python {os.path.basename(__file__)}", flush=True)
        raise SystemExit(1)

    url = f"http://localhost:{port}"
    print("=" * 60, flush=True)
    print(f"  TANC Visual Builder is running.", flush=True)
    print(f"  Open:  {url}", flush=True)
    print(f"  Stop:  Ctrl-C", flush=True)
    print("=" * 60, flush=True)
    print("(serving quietly — this line is normal; the server waits here for requests)", flush=True)

    # Best-effort: pop the browser open so it's obvious something happened.
    if os.environ.get("TDA_NO_BROWSER") != "1":
        try:
            import threading, webbrowser
            threading.Timer(0.6, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nbye", flush=True)


if __name__ == "__main__":
    main()
