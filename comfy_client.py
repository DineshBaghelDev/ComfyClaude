"""Minimal ComfyUI HTTP API client (stdlib only)."""
import json
import time
import uuid
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class ComfyClient:
    def __init__(self, urls):
        self.urls = [urls] if isinstance(urls, str) else list(urls)
        self.base = None
        self.client_id = str(uuid.uuid4())
        self._object_info = None

    # ---- connection -------------------------------------------------------
    @staticmethod
    def _alive(url):
        try:
            urllib.request.urlopen(f"{url}/system_stats", timeout=3)
            return True
        except Exception:
            return False

    def connect(self):
        """Return a responding ComfyUI URL (re-checked each call, so a restarted
        or crashed server is noticed). Raises ConnectionError if none respond."""
        if self.base and self._alive(self.base):
            return self.base
        for url in self.urls:
            if self._alive(url):
                if url != self.base:
                    self._object_info = None  # different/restarted server: refresh model lists
                self.base = url
                return url
        self.base = None
        self._object_info = None
        raise ConnectionError(
            f"ComfyUI is not reachable at {self.urls}. Start ComfyUI, or set comfy_path so it can be auto-started."
        )

    def _get(self, path):
        # Uses the URL verified by the last connect(); callers run ensure_comfy() first.
        with urllib.request.urlopen(f"{self.base or self.connect()}{path}", timeout=30) as r:
            return r.read()

    def _post_json(self, path, payload):
        req = urllib.request.Request(
            f"{self.base or self.connect()}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"ComfyUI rejected request: {e.read().decode(errors='replace')}")

    # ---- API --------------------------------------------------------------
    def object_info(self, refresh=False):
        """Node definitions, including the model files each loader can see."""
        if self._object_info is None or refresh:
            self._object_info = json.loads(self._get("/object_info"))
        return self._object_info

    def options(self, node, field):
        """Choices of a combo input (e.g. model files for a loader). None if the node is missing."""
        info = self.object_info().get(node)
        if info is None:
            return None
        spec = {**info["input"].get("required", {}), **info["input"].get("optional", {})}.get(field)
        if not spec:
            return []
        if isinstance(spec[0], list):          # old style: [[choices], {...}]
            return spec[0]
        return spec[1].get("options", []) if len(spec) > 1 else []  # new style: ["COMBO", {"options": [...]}]

    def system_stats(self):
        return json.loads(self._get("/system_stats"))

    def queue_prompt(self, workflow):
        res = self._post_json("/prompt", {"prompt": workflow, "client_id": self.client_id})
        return res["prompt_id"]

    def history(self, prompt_id):
        data = json.loads(self._get(f"/history/{prompt_id}"))
        return data.get(prompt_id)

    def queue_position(self, prompt_id):
        q = json.loads(self._get("/queue"))
        if any(item[1] == prompt_id for item in q.get("queue_running", [])):
            return "running"
        pending = [item[1] for item in q.get("queue_pending", [])]
        if prompt_id in pending:
            return f"queued (position {pending.index(prompt_id) + 1})"
        return None

    def wait(self, prompt_id, timeout):
        """Poll until done or timeout. Returns history entry or None."""
        end = time.time() + timeout
        while time.time() < end:
            h = self.history(prompt_id)
            if h and h.get("status", {}).get("completed") is not None:
                return h
            if h and h.get("status", {}).get("status_str") == "error":
                return h
            time.sleep(1.5)
        return None

    def download_outputs(self, hist, out_dir):
        """Save every output image of a finished job into out_dir."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for node_out in hist.get("outputs", {}).values():
            for img in node_out.get("images", []):
                q = urllib.parse.urlencode(
                    {"filename": img["filename"], "subfolder": img.get("subfolder", ""), "type": img.get("type", "output")}
                )
                dest = out_dir / img["filename"]
                dest.write_bytes(self._get(f"/view?{q}"))
                paths.append(str(dest))
        return paths

    def upload_image(self, file_path):
        """Upload a local image to ComfyUI's input folder. Returns its name there."""
        file_path = Path(file_path)
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; filename="{file_path.name}"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode() + file_path.read_bytes() + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="overwrite"\r\n\r\ntrue'
            f"\r\n--{boundary}--\r\n"
        ).encode()
        req = urllib.request.Request(
            f"{self.base or self.connect()}/upload/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            res = json.loads(r.read())
        return f"{res['subfolder']}/{res['name']}" if res.get("subfolder") else res["name"]

    def free_memory(self):
        self._post_json("/free", {"unload_models": True, "free_memory": True})

    def interrupt(self):
        self._post_json("/interrupt", {})
