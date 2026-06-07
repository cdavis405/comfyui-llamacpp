"""
ComfyUI <-> llama.cpp integration nodes.

Unlike the LM Studio node pack (which only talks to an already-running server),
this pack OWNS the ``llama-server`` process. Most of the knobs we want to expose
(context size, image-token fidelity, batch / ubatch size, KV-cache type,
flash-attn) are llama-server *launch* flags rather than per-request parameters,
so the node launches the server itself with the requested flags, runs the
chat request(s), and can tear the server down again to free VRAM for the
diffusion model.

Mirrors the settings in ~/.local/bin/vision-llm.sh.

Lifecycle / VRAM:
  * Before launching llama-server the node evicts ComfyUI's own cached models
    (diffusion / text-encoder / VAE) from VRAM (free_comfy_vram, default on).
    ComfyUI keeps these resident between runs, so on the 2nd+ run there would be
    no room for the LLM otherwise.
  * The llama-server process is kept alive between calls (so folder captioning
    reuses one loaded model across every image) and is only torn down when:
      - unload_after=True on the chat node, or
      - an explicit "Llama.cpp Unload" node runs, or
      - the launch config changes (different model/ctx/etc -> relaunch), or
      - ComfyUI exits (atexit).
    Killing llama-server frees the LLM's VRAM so the diffusion model can load.
"""

import io
import os
import re
import glob
import json
import time
import atexit
import shutil
import hashlib
import threading
import subprocess

import numpy as np
import requests
from PIL import Image

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
# Default model + sampling mirror ~/.local/bin/vision-llm.sh.
DEFAULT_MODEL = "unsloth/gemma-4-31B-it-GGUF:UD-Q6_K_XL"
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff")


# --------------------------------------------------------------------------- #
# llama-server process manager (module-level singleton)
# --------------------------------------------------------------------------- #
def _find_llama_server():
    """Locate the llama-server binary."""
    found = shutil.which("llama-server")
    if found:
        return found
    for cand in (
        "/usr/bin/llama-server",
        "/usr/local/bin/llama-server",
        os.path.expanduser("~/.local/bin/llama-server"),
    ):
        if os.path.isfile(cand):
            return cand
    return None


class _ServerState:
    proc = None          # subprocess.Popen or None
    key = None           # hash of the launch config currently running
    base_url = None      # e.g. http://127.0.0.1:8080
    log_path = None      # captured stdout/stderr


_server = _ServerState()
_server_lock = threading.RLock()


def _health_ok(base_url, timeout=2.0):
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def _config_key(cfg):
    """Stable hash of the *launch-relevant* config. Sampling params are sent
    per-request, so changing only those does NOT relaunch the server."""
    blob = json.dumps(cfg, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def stop_server(reason=""):
    """Terminate the managed llama-server (frees its VRAM). Safe to call when
    nothing is running."""
    with _server_lock:
        proc = _server.proc
        if proc is None:
            return "llama-server not running."
        try:
            proc.terminate()
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        except Exception as e:
            _server.proc = None
            _server.key = None
            _server.base_url = None
            return f"llama-server stop error: {e}"
        _server.proc = None
        _server.key = None
        _server.base_url = None
        return f"llama-server stopped{(' (' + reason + ')') if reason else ''}."


def _build_launch_cmd(binary, cfg):
    """Translate a launch config dict into a llama-server argv list."""
    cmd = [binary]

    # model: local file -> -m, otherwise treat as a HF repo spec for -hf
    model = cfg["model"].strip()
    model_path = os.path.expanduser(model)
    if os.path.exists(model_path) or model_path.lower().endswith(".gguf"):
        cmd += ["-m", model_path]
    else:
        cmd += ["-hf", model]

    cmd += ["--host", cfg["host"], "--port", str(cfg["port"])]
    cmd += ["--ctx-size", str(cfg["ctx"])]

    ngl = str(cfg["ngl"]).strip().lower()
    if ngl and ngl != "auto":
        cmd += ["-ngl", ngl]

    cmd += ["--batch-size", str(cfg["batch_size"])]
    cmd += ["--ubatch-size", str(cfg["ubatch_size"])]
    cmd += ["--image-max-tokens", str(cfg["image_max_tokens"])]
    cmd += ["--image-min-tokens", str(cfg["image_min_tokens"])]

    # flash-attn is always on (required for quantized KV cache, and the best
    # default for this single-GPU image-analysis workload).
    cmd += ["-fa", "on"]

    # KV cache: on -> q8_0 (smaller VRAM, needs flash-attn); off -> f16 default.
    if cfg["kv_cache_q8"]:
        cmd += ["--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]

    cmd += ["-np", "1"]

    cmd += ["--jinja"] if cfg["jinja"] else ["--no-jinja"]

    return cmd


def ensure_server(cfg, startup_timeout=900):
    """Make sure a llama-server matching ``cfg`` is up; (re)launch if needed.
    Returns the base_url."""
    with _server_lock:
        base_url = f"http://{cfg['host']}:{cfg['port']}"
        key = _config_key(cfg)

        # Reuse an already-running server with the same launch config.
        if (
            _server.proc is not None
            and _server.proc.poll() is None
            and _server.key == key
            and _health_ok(base_url)
        ):
            return base_url

        # Config changed, server died, or nothing running -> start fresh.
        if _server.proc is not None:
            stop_server("config changed / restart")

        binary = _find_llama_server()
        if not binary:
            raise RuntimeError(
                "llama-server not found on PATH. Install llama.cpp "
                "(or adjust _find_llama_server)."
            )

        cmd = _build_launch_cmd(binary, cfg)
        log_path = os.path.join(
            os.environ.get("TMPDIR", "/tmp"), "comfyui-llamacpp-server.log"
        )
        log_f = open(log_path, "w", encoding="utf-8")
        log_f.write("==> launch: " + " ".join(cmd) + "\n\n")
        log_f.flush()

        print("[LlamaCpp] launching: " + " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                start_new_session=True,  # own process group -> clean teardown
            )
        except Exception as e:
            log_f.close()
            raise RuntimeError(f"Failed to launch llama-server: {e}")

        _server.proc = proc
        _server.key = key
        _server.base_url = base_url
        _server.log_path = log_path

        # Wait for /health. First-ever run of an -hf model may download GBs, so
        # the timeout is generous; cached loads come up in seconds.
        deadline = time.time() + startup_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                _server.proc = None
                _server.key = None
                tail = _read_log_tail(log_path)
                raise RuntimeError(
                    f"llama-server exited during startup (code {proc.returncode}).\n"
                    f"--- {log_path} (tail) ---\n{tail}"
                )
            if _health_ok(base_url):
                print(f"[LlamaCpp] server ready at {base_url}")
                return base_url
            time.sleep(1.0)

        tail = _read_log_tail(log_path)
        stop_server("startup timeout")
        raise RuntimeError(
            f"llama-server did not become healthy within {startup_timeout}s.\n"
            f"--- {log_path} (tail) ---\n{tail}"
        )


def _read_log_tail(path, n=40):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except Exception:
        return "(no log)"


@atexit.register
def _shutdown_server_atexit():
    stop_server("ComfyUI exit")


# --------------------------------------------------------------------------- #
# image / text helpers
# --------------------------------------------------------------------------- #
def _tensor_to_data_url(image):
    """Convert a single ComfyUI image tensor (H,W,C or 1,H,W,C float 0..1)
    into a PNG base64 data URL."""
    import base64

    arr = image
    if hasattr(arr, "cpu"):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 4:
        arr = arr[0]
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(arr)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def _iter_images(images):
    """Yield individual (H,W,C) frames from a ComfyUI IMAGE batch tensor."""
    if images is None:
        return
    n = images.shape[0] if hasattr(images, "shape") and images.ndim == 4 else 1
    for i in range(n):
        yield images[i] if images.ndim == 4 else images


def _strip_reasoning(text):
    """Remove inline <think>...</think> blocks some chat templates leak into
    the visible content."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S | re.I)
    return text.strip()


def _free_comfy_vram():
    """Evict ComfyUI's cached models (diffusion / text-encoder / VAE) so the
    LLM has room. ComfyUI reloads whatever it needs afterwards."""
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
        mm.soft_empty_cache()
        print("[LlamaCpp] Freed ComfyUI VRAM (unloaded all cached models).")
    except Exception as e:
        print(f"[LlamaCpp] Could not free ComfyUI VRAM: {e}")


# --------------------------------------------------------------------------- #
# Node: llama.cpp Chat
# --------------------------------------------------------------------------- #
class LlamaCppChat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("STRING", {"default": DEFAULT_MODEL}),
                "system_prompt": ("STRING", {
                    "multiline": True,
                    "default": "You are a precise image-captioning assistant.",
                }),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail for use as a training caption.",
                }),
            },
            "optional": {
                "image1": ("IMAGE",),
                "image2": ("IMAGE",),

                # ---- sampling (sent per-request) ----
                "temperature": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 4.0, "step": 0.05}),
                "top_k": ("INT", {"default": 40, "min": 0, "max": 1000}),
                "top_p": ("FLOAT", {"default": 0.95, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_p": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05}),
                "repeat_penalty": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "max_tokens": ("INT", {"default": 1024, "min": -1, "max": 32768}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                # ---- server launch flags (changing any of these relaunches) ----
                "ctx": ("INT", {"default": 5000, "min": 256, "max": 1_000_000}),
                "image_max_tokens": ("INT", {"default": 4480, "min": 0, "max": 100000}),
                "image_min_tokens": ("INT", {"default": 1120, "min": 0, "max": 100000}),
                "batch_size": ("INT", {"default": 4096, "min": 1, "max": 1_000_000}),
                "ubatch_size": ("INT", {"default": 4096, "min": 1, "max": 1_000_000}),
                "kv_cache_q8": ("BOOLEAN", {"default": True}),
                "jinja": ("BOOLEAN", {"default": True}),
                "ngl": ("STRING", {"default": "auto"}),
                "host": ("STRING", {"default": DEFAULT_HOST}),
                "port": ("INT", {"default": DEFAULT_PORT, "min": 1, "max": 65535}),
                "startup_timeout": ("INT", {"default": 900, "min": 10, "max": 7200}),

                # ---- VRAM / lifecycle ----
                "free_comfy_vram": ("BOOLEAN", {"default": True}),
                "unload_after": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "generate"
    CATEGORY = "llama.cpp"

    def generate(self, model, system_prompt, prompt,
                 image1=None, image2=None,
                 temperature=1.5, top_k=40, top_p=0.95, min_p=0.05,
                 presence_penalty=0.0, repeat_penalty=1.0,
                 max_tokens=1024, seed=0,
                 ctx=5000, image_max_tokens=4480, image_min_tokens=1120,
                 batch_size=4096, ubatch_size=4096, kv_cache_q8=True, jinja=True,
                 ngl="auto", host=DEFAULT_HOST, port=DEFAULT_PORT,
                 startup_timeout=900,
                 free_comfy_vram=True, unload_after=False):

        # 1. Make room: evict ComfyUI's diffusion/text-encoder/VAE from VRAM.
        if free_comfy_vram:
            _free_comfy_vram()

        # 2. Ensure llama-server is up with the requested launch config.
        launch_cfg = {
            "model": model,
            "host": host,
            "port": int(port),
            "ctx": int(ctx),
            "ngl": ngl,
            "batch_size": int(batch_size),
            "ubatch_size": int(ubatch_size),
            "image_max_tokens": int(image_max_tokens),
            "image_min_tokens": int(image_min_tokens),
            "kv_cache_q8": bool(kv_cache_q8),
            "jinja": bool(jinja),
        }
        base_url = ensure_server(launch_cfg, startup_timeout=int(startup_timeout))

        # 3. Build the chat request.
        messages = []
        if system_prompt and system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        frames = []
        for imgs in (image1, image2):
            for frame in _iter_images(imgs):
                frames.append(frame)

        if frames:
            content = [{"type": "text", "text": prompt}]
            for frame in frames:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": _tensor_to_data_url(frame)},
                })
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "top_k": int(top_k),
            "top_p": float(top_p),
            "min_p": float(min_p),
            "presence_penalty": float(presence_penalty),
            "repeat_penalty": float(repeat_penalty),
            "stream": False,
        }
        if max_tokens and max_tokens > 0:
            payload["max_tokens"] = int(max_tokens)
        if seed:
            payload["seed"] = int(seed)

        # 4. Run it.
        try:
            r = requests.post(
                f"{base_url}/v1/chat/completions",
                json=payload,
                timeout=600,
            )
            r.raise_for_status()
            data = r.json()
            choice = data["choices"][0]
            text = choice["message"].get("content")
            finish = choice.get("finish_reason")
        except requests.exceptions.RequestException as e:
            if unload_after:
                stop_server("after error")
            raise RuntimeError(f"llama-server request failed ({base_url}): {e}")
        except (KeyError, IndexError, ValueError) as e:
            if unload_after:
                stop_server("after error")
            raise RuntimeError(f"Unexpected llama-server response: {e}")

        text = _strip_reasoning(text)

        if not text:
            print(
                f"[LlamaCppChat] WARNING: empty response (finish_reason={finish!r}). "
                f"If this is a reasoning model, raise max_tokens so it can finish "
                f"thinking and still answer."
            )
        elif finish == "length":
            print(
                f"[LlamaCppChat] WARNING: response truncated (hit max_tokens="
                f"{max_tokens}). Caption may be cut off; raise max_tokens."
            )

        # 5. Optionally tear the server down to free VRAM for diffusion.
        if unload_after:
            print(f"[LlamaCppChat] {stop_server('unload_after')}")

        return (text,)


# --------------------------------------------------------------------------- #
# Node: Load Images From Folder (list output)
# --------------------------------------------------------------------------- #
class LlamaCppLoadImagesFromFolder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": ""}),
            },
            "optional": {
                "pattern": ("STRING", {"default": "*"}),
                "sort": (["name", "name_desc"],),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
                "limit": ("INT", {"default": 0, "min": 0, "max": 1_000_000}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "filename")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load"
    CATEGORY = "llama.cpp"

    def load(self, folder, pattern="*", sort="name", start_index=0, limit=0):
        import torch

        folder = os.path.expanduser((folder or "").strip())
        if not folder or not os.path.isdir(folder):
            raise RuntimeError(f"Folder not found: {folder!r}")

        files = []
        for p in glob.glob(os.path.join(folder, pattern)):
            if os.path.isfile(p) and p.lower().endswith(IMAGE_EXTS):
                files.append(p)
        files.sort(reverse=(sort == "name_desc"))

        if start_index:
            files = files[start_index:]
        if limit and limit > 0:
            files = files[:limit]
        if not files:
            raise RuntimeError(f"No images matched {pattern!r} in {folder}")

        images, names = [], []
        for path in files:
            img = Image.open(path).convert("RGB")
            arr = np.asarray(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(arr).unsqueeze(0)  # (1,H,W,C)
            images.append(tensor)
            names.append(os.path.splitext(os.path.basename(path))[0])

        return (images, names)


# --------------------------------------------------------------------------- #
# Node: Save Text (caption to .txt matching filename)
# --------------------------------------------------------------------------- #
class LlamaCppSaveText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "filename": ("STRING", {"forceInput": True}),
                "output_dir": ("STRING", {"default": ""}),
            },
            "optional": {
                "extension": ("STRING", {"default": "txt"}),
                "overwrite": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    CATEGORY = "llama.cpp"
    OUTPUT_NODE = True

    def save(self, text, filename, output_dir, extension="txt", overwrite=True):
        output_dir = os.path.expanduser((output_dir or "").strip())
        if not output_dir:
            raise RuntimeError("output_dir is empty")
        os.makedirs(output_dir, exist_ok=True)

        ext = extension.strip().lstrip(".") or "txt"
        path = os.path.join(output_dir, f"{filename}.{ext}")
        if os.path.exists(path) and not overwrite:
            print(f"[LlamaCppSaveText] exists, skipped: {path}")
            return (path,)

        with open(path, "w", encoding="utf-8") as f:
            f.write(text or "")
        print(f"[LlamaCppSaveText] wrote {path}")
        return {"ui": {"string": [path]}, "result": (path,)}


# --------------------------------------------------------------------------- #
# Node: Unload (explicit / passthrough) — stops llama-server to free VRAM
# --------------------------------------------------------------------------- #
class AnyType(str):
    def __ne__(self, other):
        return False


ANY = AnyType("*")


class LlamaCppUnload:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "passthrough": (ANY, {}),
            },
        }

    RETURN_TYPES = (ANY,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "run"
    CATEGORY = "llama.cpp"
    OUTPUT_NODE = True

    def run(self, passthrough=None):
        msg = stop_server("Unload node")
        print(f"[LlamaCppUnload] {msg}")
        return (passthrough,)
