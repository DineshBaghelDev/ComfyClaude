"""ComfyClaude Studio: a local web UI over the same tools Claude uses.

    uv run python ui.py [--port 8199] [--no-browser]

Tool calls go through server.py's functions, so outputs, history (jobs.json) and
settings are shared with the MCP server. Jobs run one at a time in a worker thread;
a websocket listener reports ComfyUI's sampling progress and live previews.
"""
import argparse
import base64
import json
import os
import queue
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import imaging
import server as S

PAGE = Path(__file__).parent / "ui" / "index.html"
UPLOADS = S.OUT / "ui-uploads"
TOOLS = {name: getattr(S, name) for name in (
    "generate_image", "generate_drafts", "generate_with_model", "generate_from_reference",
    "generate_with_control", "inpaint", "outpaint", "recolor_image", "upscale",
    "remove_background", "run_custom_workflow")}
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp")
PENDING = re.compile(r"job_status\('([^']+)'\)")
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")

tasks, work = {}, queue.Queue()
lock = threading.Lock()
thumbs = OrderedDict()
running = {"task": None}  # the task the worker is executing, for progress events


# ---- tasks -------------------------------------------------------------------
def _images(text):
    """Image files named in a tool's result text, with their label and saved metadata."""
    found = []
    for line in text.splitlines():
        if line.startswith("Saved to:"):
            continue
        label, _, rest = line.partition(": ")
        for cand, lab in ((line.strip(), ""), (rest.strip(), label.strip())):
            if cand.lower().endswith(IMAGE_EXT) and Path(cand).is_file() and cand not in [f["path"] for f in found]:
                meta = Path(cand).with_suffix(".json")
                found.append({"path": cand, "label": lab,
                              "meta": json.loads(meta.read_text()) if meta.exists() else {}})
                break
    return found


def _text(result):
    return result if isinstance(result, str) else next((r for r in result if isinstance(r, str)), "")


def _execute(task):
    text = _text(TOOLS[task["tool"]](**task["params"]))
    texts = [text]
    for pid in PENDING.findall(text):  # long jobs: keep waiting until ComfyUI finishes them
        while not task["cancel"]:
            more = _text(S.job_status(pid, 30))
            if not PENDING.search(more):
                texts.append(more)
                break
    images = [img for t in texts for img in _images(t)]
    return "\n".join(texts), images


def _worker():
    while True:
        task = work.get()
        if task["cancel"]:
            continue
        task.update(status="running", started=time.time())
        running["task"] = task
        try:
            task["text"], task["images"] = _execute(task)
            task["status"] = "cancelled" if task["cancel"] else ("done" if task["images"] else "error")
        except Exception as e:  # report to the UI instead of killing the worker
            task.update(status="cancelled" if task["cancel"] else "error", text=f"{type(e).__name__}: {e}")
            traceback.print_exc()
        running["task"] = None
        task.pop("preview", None)
        task["finished"] = time.time()


def _submit(tool, params):
    if tool not in TOOLS:
        raise ValueError(f"Unknown tool {tool}")
    task = {"id": uuid.uuid4().hex[:10], "tool": tool, "params": params, "status": "queued",
            "created": time.time(), "text": "", "images": [], "cancel": False, "progress": None}
    with lock:
        tasks[task["id"]] = task
        for old in sorted(tasks.values(), key=lambda t: t["created"])[:-60]:
            if old["status"] not in ("queued", "running"):
                tasks.pop(old["id"], None)
    work.put(task)
    return task


def _cancel(task_id):
    task = tasks.get(task_id)
    if not task or task["status"] not in ("queued", "running"):
        return
    task["cancel"] = True
    if task["status"] == "queued":
        task["status"] = "cancelled"
    else:
        try:
            S.comfy.interrupt()
        except Exception:
            pass


# ---- ComfyUI progress (websocket) ------------------------------------------------
def _ws_connect(base):
    url = urlparse(base)
    sock = socket.create_connection((url.hostname, url.port or 80), timeout=10)
    key = base64.b64encode(os.urandom(16)).decode()
    sock.sendall((f"GET /ws?clientId={S.comfy.client_id} HTTP/1.1\r\nHost: {url.netloc}\r\n"
                  f"Upgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: {key}\r\n"
                  "Sec-WebSocket-Version: 13\r\n\r\n").encode())
    head = b""
    while b"\r\n\r\n" not in head:
        chunk = sock.recv(1024)
        if not chunk:
            raise ConnectionError("websocket handshake failed")
        head += chunk
    if b" 101 " not in head.split(b"\r\n", 1)[0]:
        raise ConnectionError("websocket upgrade refused")
    sock.settimeout(None)
    return sock, head.split(b"\r\n\r\n", 1)[1]


def _ws_frames(sock, buf):
    """Yield (opcode, payload) from an unmasked server stream."""
    def need(n):
        nonlocal buf
        while len(buf) < n:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("websocket closed")
            buf += chunk
        out, buf = buf[:n], buf[n:]
        return out
    while True:
        b1, b2 = need(2)
        size = b2 & 0x7F
        if size == 126:
            size = struct.unpack(">H", need(2))[0]
        elif size == 127:
            size = struct.unpack(">Q", need(8))[0]
        yield b1 & 0x0F, need(size)


def _on_message(opcode, payload):
    task = running["task"]
    if not task:
        return
    if opcode == 1:
        msg = json.loads(payload)
        data = msg.get("data") or {}
        if msg.get("type") == "progress":
            prog = task["progress"] or {"prompts": []}
            if data.get("prompt_id") and data["prompt_id"] not in prog["prompts"]:
                prog["prompts"].append(data["prompt_id"])
            prog.update(value=data.get("value", 0), max=data.get("max", 1))
            task["progress"] = prog
    elif opcode == 2 and len(payload) > 8:
        kind = struct.unpack(">I", payload[:4])[0]
        if kind == 1:  # PREVIEW_IMAGE: type, image format, bytes
            task["preview"] = payload[8:]
        elif kind == 4:  # PREVIEW_IMAGE_WITH_METADATA: type, metadata length, metadata, bytes
            task["preview"] = payload[8 + struct.unpack(">I", payload[4:8])[0]:]
        task["preview_n"] = task.get("preview_n", 0) + 1


def _progress_listener():
    while True:
        try:
            sock, rest = _ws_connect(S.comfy.connect())
            for opcode, payload in _ws_frames(sock, rest):
                if opcode == 8:
                    break
                try:
                    _on_message(opcode, payload)
                except Exception:
                    pass
            sock.close()
        except Exception:
            pass
        time.sleep(3)


# ---- info / history -------------------------------------------------------------
def _info():
    out = {"styles": S.PRESETS, "default_quality": S.CFG["default_quality"],
           "qualities": {k: {"description": q["description"]} for k, q in S.CFG["qualities"].items()},
           "models": {k: {"description": p.get("description", k), "negative": p.get("negative_prompt", ""),
                          "prefix": p.get("prompt_prefix", ""), "steps": p.get("steps", 30)}
                      for k, p in S.CFG.get("generation_models", {}).items()},
           "edit_models": list(S.CFG.get("editing", {}).get("models", {})),
           "tool_quality": {k: v.get("quality") for k, v in S.CAPS["tools"].items()},
           "workflows": {name: sorted(set(PLACEHOLDER.findall(p.read_text())))
                         for name, p in sorted(S._custom_workflows().items())},
           "output_dir": str(S.OUT), "loras": [], "missing": {}, "timings": {}}
    for m in S._load_jobs().values():
        if m.get("seconds"):
            out["timings"].setdefault(f"{m.get('tool')}|{m.get('quality') or ''}", []).append(m["seconds"])
    out["timings"] = {k: sum(v[-10:]) / len(v[-10:]) for k, v in out["timings"].items()}
    try:
        S.comfy.connect()
        S.comfy.object_info(refresh=True)
    except ConnectionError as e:
        out["comfy"] = {"ok": False, "error": str(e)}
        return out
    out["comfy"] = {"ok": True, "url": S.comfy.base}
    for name, q in S.CFG["qualities"].items():
        out["qualities"][name]["problem"] = S._quality_problem(q)
    for name, p in S.CFG.get("generation_models", {}).items():
        out["models"][name]["ready"] = S._has("CheckpointLoaderSimple", "ckpt_name", p["checkpoint"])
    out["loras"] = S.comfy.options("LoraLoader", "lora_name") or []
    for tool, info in S.CAPS["tools"].items():
        missing = [S.CFG["models"][k] for k in info["needs"]
                   if not S._has(*S.CAPS["model_nodes"][k], S.CFG["models"][k])]
        if missing:
            out["missing"][tool] = missing
    return out


def _history(limit=1000):
    items = []
    for m in S._load_jobs().values():
        for p in S._main_files(m.get("files") or []):
            items.append({"path": p, "meta": m, "time": m.get("submitted", 0)})
    for meta_file in S.OUT.glob("recolor_*.json"):
        try:
            m = json.loads(meta_file.read_text())
        except ValueError:
            continue
        items.append({"path": str(meta_file.with_suffix(".png")), "meta": m, "time": meta_file.stat().st_mtime})
    items = [i for i in items if Path(i["path"]).is_file()]
    items.sort(key=lambda i: i["time"], reverse=True)
    return items[:limit]


def _allowed(path):
    """Only image files inside the output folder are served or revealed."""
    p = Path(path).resolve()
    return p.suffix.lower() in IMAGE_EXT and p.is_file() and p.is_relative_to(S.OUT.resolve())


def _reveal(path):
    target = Path(path) if path else S.OUT
    if path and not _allowed(path):
        raise ValueError("Not an output image")
    if sys.platform == "win32":
        subprocess.Popen(["explorer", f"/select,{target}"] if path else ["explorer", str(target)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(target)] if path else ["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target.parent if path else target)])


def _thumb(path, size):
    key = (path, Path(path).stat().st_mtime, size)
    if key in thumbs:
        thumbs.move_to_end(key)
    else:
        thumbs[key] = imaging.preview_jpeg(path, size, 82)
        while len(thumbs) > 400:
            thumbs.popitem(last=False)
    return thumbs[key]


# ---- HTTP ------------------------------------------------------------------------
def _public(task):
    return {k: v for k, v in task.items() if k not in ("cancel", "preview")}


class Handler(BaseHTTPRequestHandler):
    port = 8199

    def log_message(self, *args):
        pass

    def _send(self, body, ctype="application/json", status=200, cache=False):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        return self.rfile.read(int(self.headers.get("Content-Length") or 0))

    def _trusted(self, post=False):
        """Block DNS rebinding (Host) and cross-site requests (Origin and a custom header)."""
        hosts = {f"127.0.0.1:{self.port}", f"localhost:{self.port}"}
        if self.headers.get("Host", "") not in hosts:
            return False
        origin = self.headers.get("Origin")
        if origin and urlparse(origin).netloc not in hosts:
            return False
        return not post or self.headers.get("X-Studio") == "1"

    def do_GET(self):
        if not self._trusted():
            return self._send({"error": "forbidden"}, status=403)
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/":
                self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif url.path == "/api/info":
                self._send(_info())
            elif url.path == "/api/ping":
                try:
                    self._send({"comfy": bool(S.comfy.connect()), "url": S.comfy.base})
                except ConnectionError:
                    self._send({"comfy": False})
            elif url.path == "/api/tasks":
                with lock:
                    self._send([_public(t) for t in sorted(tasks.values(), key=lambda t: -t["created"])])
            elif url.path == "/api/history":
                self._send(_history())
            elif url.path.startswith("/api/preview/"):
                task = tasks.get(url.path.rsplit("/", 1)[-1])
                if not task or not task.get("preview"):
                    return self._send({"error": "no preview"}, status=404)
                self._send(task["preview"], "image/jpeg")
            elif url.path in ("/api/file", "/api/thumb"):
                path = q.get("path", "")
                if not _allowed(path):
                    return self._send({"error": "not found"}, status=404)
                if url.path == "/api/file":
                    ext = Path(path).suffix.lower().lstrip(".").replace("jpg", "jpeg")
                    return self._send(Path(path).read_bytes(), f"image/{ext}", cache=True)
                self._send(_thumb(path, min(1024, max(64, int(q.get("size", "384"))))), "image/jpeg", cache=True)
            else:
                self._send({"error": "not found"}, status=404)
        except Exception as e:
            self._send({"error": str(e)}, status=500)

    def do_POST(self):
        if not self._trusted(post=True):
            return self._send({"error": "forbidden"}, status=403)
        url = urlparse(self.path)
        try:
            if url.path == "/api/run":
                body = json.loads(self._body())
                self._send(_public(_submit(body.get("tool"), body.get("params") or {})))
            elif url.path.startswith("/api/retry/"):
                old = tasks.get(url.path.rsplit("/", 1)[-1])
                if not old:
                    raise ValueError("Unknown task")
                self._send(_public(_submit(old["tool"], old["params"])))
            elif url.path.startswith("/api/cancel/"):
                _cancel(url.path.rsplit("/", 1)[-1])
                self._send({"ok": True})
            elif url.path == "/api/clear":
                for task in list(tasks.values()):
                    if task["status"] == "queued":
                        _cancel(task["id"])
                self._send({"ok": True})
            elif url.path == "/api/upload":
                name = re.sub(r"[^\w.-]+", "_", self.headers.get("X-Filename") or "image.png")[-80:]
                if not name.lower().endswith(IMAGE_EXT):
                    name += ".png"
                UPLOADS.mkdir(parents=True, exist_ok=True)
                path = UPLOADS / f"{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:6]}_{name}"
                path.write_bytes(self._body())
                self._send({"path": str(path)})
            elif url.path == "/api/start":
                S.ensure_comfy()
                self._send({"ok": True})
            elif url.path == "/api/free":
                S.ensure_comfy()
                S.comfy.free_memory()
                self._send({"ok": True})
            elif url.path == "/api/reveal":
                _reveal(json.loads(self._body() or b"{}").get("path", ""))
                self._send({"ok": True})
            else:
                self._send({"error": "not found"}, status=404)
        except Exception as e:
            self._send({"error": f"{type(e).__name__}: {e}"}, status=400)


def main():
    ap = argparse.ArgumentParser(description="ComfyClaude Studio web UI")
    ap.add_argument("--port", type=int, default=int(os.environ.get("COMFYCLAUDE_UI_PORT", 8199)))
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    Handler.port = args.port
    threading.Thread(target=_worker, daemon=True).start()
    threading.Thread(target=_progress_listener, daemon=True).start()
    httpd = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"ComfyClaude Studio: {url}  (outputs: {S.OUT})", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
