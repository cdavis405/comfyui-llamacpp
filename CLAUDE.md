# comfyui-llamacpp

ComfyUI custom node pack that drives a local **llama.cpp** (`llama-server`).
Sibling of `comfyui-lmstudio` — same workflow shapes (folder captioning, prompt
generation) but the node OWNS the server process instead of talking to an
already-running one.

## Layout
- `llamacpp_nodes.py` — all node classes + the llama-server process manager.
- `__init__.py` — `NODE_CLASS_MAPPINGS` / `NODE_DISPLAY_NAME_MAPPINGS`.
- `workflows/caption_folder.json` — folder→caption→`.txt` LoRA captioning graph.
- `workflows/prompt.json` — image→prompt→CLIP encode graph.
- `README.md` — usage.

## Key facts / design
- Lives in `~/Projects/ComfyUI/custom_nodes/comfyui-llamacpp`.
- ComfyUI venv: `~/Projects/ComfyUI/venv/bin/python` (Python 3.14).
- Binary: `llama-server` (PATH; `/usr/bin/llama-server` on this box). Build
  9544+. Settings mirror `~/.local/bin/vision-llm.sh`.
- **The node launches/owns llama-server.** Most knobs (ctx, image-max/min-tokens,
  batch/ubatch, KV-cache type, flash-attn, ngl) are *launch* flags, not request
  params — so the node spawns `llama-server` with them, then chats.
- Chat = `POST /v1/chat/completions` (OpenAI-compatible; vision via base64
  `image_url`; llama.cpp also accepts `top_k`/`min_p`/`repeat_penalty` in the
  body). Readiness = `GET /health`.
- Sampling (temp/top_k/top_p/min_p/presence_penalty/repeat_penalty/seed/
  max_tokens) is sent **per request** → changing it does NOT relaunch the
  server. Launch flags are hashed into a config key; changing any of them
  relaunches.
- `flash-attn` is hard-wired `on` (required for q8 KV cache). `kv_cache_q8`
  toggles `--cache-type-k/v q8_0` (on) vs f16 (off).
- Two optional image inputs (`image1`, `image2`); each may be a batch — all
  frames are attached to the one user message.

## VRAM lifecycle (the whole point)
- `free_comfy_vram` (default on): before launch, calls
  `comfy.model_management.unload_all_models()` + `soft_empty_cache()` so the
  diffusion model / text encoder don't hog VRAM when the LLM loads.
- The llama-server process **persists** across calls (folder captioning reuses
  one loaded model per image). It is torn down (freeing the LLM's VRAM) when:
  `unload_after=True`, the `llama.cpp Unload Server` node runs, the launch
  config changes, or ComfyUI exits (atexit).
- Prompt workflow: set `unload_after=True` so VRAM frees for CLIP+diffusion.
  Folder workflow: leave it False; click the Unload node when the batch is done.
- Server stdout/stderr → `$TMPDIR/comfyui-llamacpp-server.log` (tail is shown
  in the raised error if startup fails).

## Testing without the ComfyUI server
Import by path (dir name has a hyphen):
```
cd custom_nodes/comfyui-llamacpp
~/Projects/ComfyUI/venv/bin/python -c "import importlib.util as u; \
  s=u.spec_from_file_location('n','llamacpp_nodes.py'); \
  m=u.module_from_spec(s); s.loader.exec_module(m); \
  print(m._find_llama_server())"
```
A real chat test launches llama-server (VRAM + possibly a first-run download).
To reload node code in a running ComfyUI, restart ComfyUI.
