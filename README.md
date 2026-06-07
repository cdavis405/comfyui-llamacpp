# ComfyUI ⇄ llama.cpp

Caption images and build prompts inside ComfyUI using a local **llama.cpp**
vision model. The successor to `comfyui-lmstudio` — same idea, but the node
launches and controls `llama-server` itself so you get the launch-flag knobs
(context size, image-token fidelity, batch sizes, KV-cache type, ngl, …) that
the OpenAI/LM-Studio API never exposed. Defaults mirror
`~/.local/bin/vision-llm.sh`.

## Requirements
- `llama-server` on your `PATH` (this box: `/usr/bin/llama-server`).
- A vision GGUF reachable by `-hf <repo>:<quant>` (default
  `unsloth/gemma-4-31B-it-GGUF:UD-Q6_K_XL`) or a local `.gguf` path.
- `requests`, `pillow`, `numpy` (already in any ComfyUI install).

Drop the folder in `custom_nodes/` and restart ComfyUI. Nodes appear under the
**llama.cpp** category.

## Nodes
| Node | Purpose |
|------|---------|
| **llama.cpp Chat** | Launch llama-server, run a (vision) chat completion, return the text. Two optional image inputs. |
| **llama.cpp Load Images From Folder** | List-output loader → one chat call per image. |
| **llama.cpp Save Text** | Write the response to `<filename>.txt` next to / into a folder. |
| **llama.cpp Unload Server** | Stop llama-server to free VRAM (run after a folder batch). |

## llama.cpp Chat inputs
**Required:** `model` (HF repo spec or local `.gguf` path), `system_prompt`,
`prompt`.

**Sampling (sent per request — changing these does NOT relaunch the server):**
`temperature` (default 1.5), `top_k` (40), `top_p` (0.95), `min_p` (0.05),
`presence_penalty` (0.0), `repeat_penalty` (1.0), `max_tokens` (1024, -1 =
unlimited), `seed` (0 = random).

**Server launch flags (changing any of these relaunches llama-server):**
`ctx` (5000), `image_max_tokens` (4480), `image_min_tokens` (1120),
`batch_size` (4096), `ubatch_size` (4096), `kv_cache_q8` (on → q8_0 KV cache,
off → f16), `jinja` (on), `ngl` (`auto` or a layer count), `host`, `port`
(8080), `startup_timeout` (900 s — generous for a first-run model download).

> **Fidelity dial:** raising `image_max_tokens` sharpens image analysis. Image
> tokens decode in one non-causal chunk, so keep `batch_size` and `ubatch_size`
> ≥ `image_max_tokens` or llama.cpp aborts on the image step. `flash-attn` is
> always `on` (required for the q8 KV cache).

**VRAM / lifecycle:**
- `free_comfy_vram` (default **on**) — before launch, evicts ComfyUI's cached
  diffusion model / text encoder / VAE from VRAM so the LLM fits. ComfyUI
  reloads them afterwards.
- `unload_after` (default **off**) — after the response, stop llama-server to
  free its VRAM for the next diffusion run.

## Usage

### Caption a folder (LoRA captions)
`Load Images From Folder` → `Chat` (`image1`) → `Save Text`.
Leave `unload_after` **off** so the model stays loaded across every image. When
the batch finishes, run the **Unload Server** node once to free VRAM. (No
diffusion here, so VRAM handoff doesn't matter mid-run.)

### Build a prompt then generate
`Load Image` → `Chat` (`image1`, `unload_after` **on**) → `CLIPTextEncode` → …
With `free_comfy_vram` on and `unload_after` on, the LLM gets clear VRAM to run,
then releases it so CLIP + the diffusion model load into the freed space.

### Compare two images
Wire one image to `image1` and another to `image2`; both are attached to the
same message ("what changed between these?", "which is sharper?", …).

## Why it owns the server
Context size, image-token limits, batch/ubatch, KV-cache type and ngl are
`llama-server` **startup** flags, not request parameters. To expose them the
node spawns `llama-server` with the chosen flags, waits for `/health`, then
chats. The process is reused across calls (one model load per folder) and is
torn down on `unload_after`, the Unload node, a launch-config change, or ComfyUI
exit. Startup/run logs: `$TMPDIR/comfyui-llamacpp-server.log`.

## Troubleshooting
- **"llama-server not found"** — install llama.cpp / fix `PATH`.
- **Startup timeout / exited during startup** — the raised error includes the
  log tail; a first-run `-hf` download can be many GB (raise `startup_timeout`).
- **Aborts on the image step** — `batch_size`/`ubatch_size` are below
  `image_max_tokens`; raise them.
- **Port already in use** — another `llama-server` (e.g. `vision-llm.sh`) holds
  8080; stop it or change `port`.
