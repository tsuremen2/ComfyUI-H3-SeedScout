# ComfyUI-H3-SeedScout

**MiniMax H3 Seed Scout Sampler** — one node that replaces the
`RandomNoise` + `SamplerCustomAdvanced` pair in MiniMax H3 t2v/i2v workflows.

* **interactive mode (v2, default)** — **one queued prompt**: scouts N seeds through the
  first K steps, shows clickable seed-number buttons + an animated preview window **on
  the node**, blocks the running prompt until you confirm a seed, then finishes *that
  exact seed* through the remaining steps and hands a full-quality latent downstream.
  Requires a connected browser (see "Interactive mode" below).
* **scout mode** — runs N seeds through only the first K steps of the **full** sigma
  schedule (`sigmas[:scout_step + 1]`, a real "step 3 of 20", *not* a fresh 3-step
  schedule), decodes a cheap preview of the x0 (denoised) estimate for each seed, and
  saves one labelled animated WebP per seed into the ComfyUI output folder. Previews
  also appear on the node itself.
* **final mode** — byte-for-byte the stock `SamplerCustomAdvanced` behaviour with the
  full `sigmas` and `selected_seed`, so `VAEDecode` / `VAEDecodeAudio` / `CreateVideo`
  downstream keep working unchanged.

Because scouting uses `comfy.sample.prepare_noise(latent, seed, batch_index)` — exactly
what `Noise_RandomNoise` does — **a scouted seed is the same seed you can type into a
plain `RandomNoise` node** in your original untouched workflow.

## Node

**Category:** `sampling/custom_sampling/minimax_h3`
**Node id:** `MiniMaxH3SeedScoutSampler`

Inputs: `guider` (GUIDER), `sampler` (SAMPLER), `sigmas` (SIGMAS),
`latent_image` (LATENT), `vae` (VAE, *optional* — the H3 **video** VAE).

Widgets: `mode` (`interactive` | `scout` | `final`), `seed_start`, `seed_count`,
`seed_stride`, `scout_step`, `selected_seed`, `preview_mode`, `preview_frames`,
`preview_fps`, `max_preview_side`, `filename_prefix`, `selection_timeout`.

Outputs: `output` (LATENT), `denoised_output` (LATENT), `seed_used` (INT),
`seed_report` (STRING).

## Interactive mode (v2)

Queue **`E:\Stablediffusion\Workflow\video_minimax_h3_t2v_seed_scout_INTERACTIVE_API.json`**
(or set `mode = interactive` on any existing graph) **from the ComfyUI web UI** — drag
the JSON onto the canvas, or queue it while a browser tab is connected.

What happens, in one prompt:

1. Each of `seed_count` seeds is sampled through `sigmas[:scout_step+1]`. Each seed's
   **sampler output** (`samples`, *not* x0) is parked on CPU — that is what the
   continuation needs.
2. Previews are decoded, saved as WebPs, and pushed to the browser
   (`PromptServer.instance.send_sync("h3_seed_scout_preview", …, client_id)`).
3. The node shows one button per seed plus a preview window. **Click** a number to swap
   the preview. **Double-click** (or highlight + **Continue ▶**) → confirmation dialog →
   the choice is POSTed to `/h3_seed_scout/select`. The chosen seed is also written into
   the `selected_seed` widget so it persists in the saved workflow.
4. The backend, blocked on a `threading.Event`, wakes, frees the other seeds' latents,
   and resumes the winner through `sigmas[scout_step:]` with **zero noise**.
5. Downstream `VAEDecode` / `VAEDecodeAudio` / `CreateVideo` / `SaveVideo` run normally.

### Why the continuation is trajectory-exact

`scout_sigmas = sigmas[:k+1]`, `continue_sigmas = sigmas[k:]` — they deliberately share
the boundary sigma, exactly like core `SplitSigmas`
(`nodes_custom_sampler.py:222`, `sigmas[:step+1]` / `sigmas[step:]`).

MiniMax H3 is `ModelType.FLOW` (`comfy/model_base.py:2067`) → `CONST` model sampling
(`comfy/model_sampling.py:94`):

* pass 1 ends with `inverse_noise_scaling(σ_k, s) = s / (1 - σ_k)`
* pass 2 starts with `noise_scaling(σ_k, 0, latent) = σ_k·0 + (1 - σ_k)·latent`

so the two cancel and pass 2 begins on **bit-identical state** to where pass 1 stopped.
Zero noise is built exactly like `Noise_EmptyNoise.generate_noise`
(`nodes_custom_sampler.py:702`), including the NestedTensor branch.

**Caveat:** with a *multistep* sampler (the workflow's `res_multistep`), splitting at
step k resets the sampler's internal history buffer, so step k+1 restarts first-order
instead of reusing the previous derivative. That is inherent to every two-stage
`SplitSigmas` graph in ComfyUI, not something this node adds — the latent handoff itself
is exact.

### Cancel / timeout / no browser

* **Cancel** — the wait loop is `event.wait(0.5)` in a loop that calls
  `comfy.model_management.throw_exception_if_processing_interrupted()` each turn, so the
  UI Cancel button aborts within ~0.5 s. A `h3_seed_scout_done {status:"cancelled"}`
  message unlocks the node UI. Previews already written to `output/` are kept.
* **Interrupt during scouting** — previews scouted so far are still decoded and written
  to disk, then `InterruptProcessingException` is re-raised so nothing downstream runs.
* **`selection_timeout` > 0** — on expiry the node logs loudly
  (`***** SELECTION TIMEOUT *****`), sends `status:"timeout"`, falls back to
  `seeds[0]`, and continues. `0` (default) waits forever.
* **No browser / headless API queue** — `server.client_id is None` (execution.py:737),
  so no preview reaches a client. The node detects that *before* blocking, logs a
  warning, and auto-continues with `seeds[0]` rather than hanging a headless queue.
  For real headless use, prefer the two-file `scout` → `final` flow
  (`..._seed_scout_SCOUT_API.json` then `..._seed_scout_FINAL_API.json`).

## Wiring into `video_minimax_h3_t2v_sage_then_kijai_sol_tau08_API.json`

Remove/bypass **`noise` (RandomNoise)** and **`..._sampler` (SamplerCustomAdvanced)**
and drop this node in their place:

| this node's socket | connect from |
|---|---|
| `guider` | `22_..._guider` (BasicGuider) output 0 |
| `sampler` | `sampler_select` (KSamplerSelect) output 0 |
| `sigmas` | `22_..._scheduler` (BasicScheduler) output 0 |
| `latent_image` | `conditioning` (MiniMaxH3ImageToVideo) **output 1** |
| `vae` | `video_vae` (VAELoader) output 0 |

Downstream, connect this node's `output` (LATENT) to
`22_..._video_decode` (VAEDecode) `samples` **and** `22_..._audio_decode`
(VAEDecodeAudio) `samples` — the nested (video, audio) latent is passed through intact.

Workflow:

1. `mode = scout`, `seed_start` = wherever you like, `seed_count = 6`,
   `scout_step = 3`. Queue. You get 6 WebPs on the node and in `output/`, each named
   `h3_scout_seed_<SEED>_00001_.webp`. Muting the downstream Save/CreateVideo chain
   during scouting is recommended (scout mode's LATENT output is only the *first*
   seed's partial latent and is not meant to be rendered).
2. Pick a seed from the previews / from `seed_report`.
3. `mode = final`, `selected_seed = <that seed>`. Unmute downstream. Queue.

## Audit notes / deviations from the original spec

Everything below was verified against **this** install
(`E:\Stablediffusion\ComfyUI_qwen`), not upstream docs.

1. **`SamplerCustomAdvanced` is `io.ComfyNode` here** (`comfy_extras/nodes_custom_sampler.py:1014`).
   This pack still uses the **classic `INPUT_TYPES` / `RETURN_TYPES` dict API**, which is
   what every other working custom node in this install uses
   (`MiniMaxH3_LatentUpscaler`, `ComfyUI-BFSNodes`, …) and which the installed frontend
   definitely supports. The sampling body mirrors `SamplerCustomAdvanced.execute` exactly:
   `fix_empty_latent_channels(model_patcher, samples, downscale_ratio_spacial,
   downscale_ratio_temporal)` → `latent_preview.prepare_callback(model_patcher,
   sigmas.shape[-1]-1, x0_output)` → `guider.sample(noise, latent, sampler, sigmas,
   denoise_mask=…, callback=…, disable_pbar=…, seed=…)` → pop the two `downscale_ratio_*`
   keys → x0 unpack → `process_latent_out(x0.cpu())`.
2. **x0 arrives already nested — spec assumption partly wrong.** In this install
   `CFGGuider.sample` (`comfy/samplers.py` ~1287) wraps the callback and re-nests both
   `x0` and `x` via `unpack_latents` *before* the callback runs, so
   `x0_output["x0"]` is already a `NestedTensor` for H3. Core's
   `if samples.is_nested and not x0.is_nested:` unpack branch is a legacy fallback that
   never fires here. It is reproduced anyway for parity.
3. **No tiny VAE — confirmed.** `models/vae_approx/` contains only the placeholder file,
   and `MiniMaxH3Video` in `comfy/latent_formats.py:570` defines
   `latent_rgb_factors` + `latent_rgb_factors_bias` but **no `taesd_decoder_name`**.
   So: `preview_mode = "vae"` (real H3 video VAE, default) or
   `preview_mode = "latent2rgb"` (free, blocky, uses those factors). No KJNodes
   dependency, no new packages.
4. **Evenly-spaced temporal slicing is NOT safe — spec deviation.**
   `comfy/ldm/minimax/vae.py` `MiniMaxH3VideoVAE` is a *causal*, temporally chunked
   codec (`clip_length=17`, `vae_ratio_t=4`, `tokens_chunk_size=5`, `token_drop=3`,
   `token_overlap=2`), and `comfy/sd.py` pins
   `upscale_ratio[0] = max(1, (t-2)//5*17 + 5)`. Sampling every Nth latent frame would
   feed the decoder a discontinuous token sequence. Instead the node decodes a
   **contiguous temporal prefix** starting at token 0 — valid token counts 2, 7, 12, …
   → 5, 22, 39, … frames — which is exactly what the decoder sees for a shorter clip.
   `preview_frames` therefore selects the smallest legal prefix that yields at least
   that many frames, then the result is truncated to `preview_frames`.
   Practical consequence: a preview always shows the **opening** of the clip, and the
   cheapest possible preview is 5 frames (2 latent tokens).
5. **Previews are decoded after the seed loop, not inside it — deliberate deviation.**
   Decoding inside the loop would evict the diffusion model for the VAE and reload it on
   every seed. The node samples all seeds (model stays resident, x0 video latents parked
   on CPU), then decodes/saves all previews in one VAE pass. Same output, far less
   thrash. `comfy.model_management.soft_empty_cache()` runs between seeds; no forced
   full unload, per spec.
6. **Progress bar**: `latent_preview.prepare_callback` already creates a
   `comfy.utils.ProgressBar(steps)` per `guider.sample` call, so each seed drives the
   node progress bar and live latent preview. No second competing bar was added.
   `comfy.model_management.throw_exception_if_processing_interrupted()` is called
   between seeds; on interrupt the previews scouted so far are still **written to disk**
   before `InterruptProcessingException` is re-raised (the node UI will not show them,
   since the prompt aborts — check `output/`).
7. **VAE socket is optional** rather than required, so `preview_mode = "latent2rgb"`
   works standalone. With `preview_mode = "vae"` and no VAE connected, the node warns
   and falls back to latent2rgb rather than erroring.
8. **`scout_step` is clamped** to `[1, len(sigmas)-1]`.
9. Node is `OUTPUT_NODE = True` so `{"ui": {"images": …, "animated": …}}` renders on
   the node; the result tuple is still returned via `{"result": …}` so the LATENT
   outputs remain usable.

### v2 audit (interactive mode)

10. **`send_sync` signature confirmed** — `server.py:1392`,
    `def send_sync(self, event, data, sid=None)`, and it is thread-safe by design
    (`self.loop.call_soon_threadsafe(self.messages.put_nowait, …)`), so calling it from
    the execution worker thread is correct. `sid` is `PromptServer.instance.client_id`,
    set per prompt from `extra_data["client_id"]` (`execution.py:737`) and `None` for
    headless API queues (`execution.py:433/494`).
11. **Route registration** — `@PromptServer.instance.routes.post(...)`, identical to
    `custom_nodes/RES4LYF/res4lyf.py:37`, the only other pack here that registers
    routes. Registration happens at *import* time because `custom_nodes` are imported
    before the aiohttp app freezes its route table; it is guarded by a
    `_ROUTES_REGISTERED` flag and a try/except so a module reload cannot raise.
12. **Frontend API version** — `requirements.txt` pins
    `comfyui-frontend-package==1.47.12`. The JS uses `window.comfyAPI.app` /
    `window.comfyAPI.api` (the idiom used by the installed
    `ComfyUI-KJNodes/web/js/fast_preview_batch.js` and `hdr_preview.js`), with a
    dynamic-`import("../../scripts/app.js")` fallback. Widgets are attached with
    `node.addDOMWidget(name, "div", element, {serialize:false})` — also exactly what
    KJNodes uses here (`context_windows_visualizer.js:403`,
    `hdr_preview.js:415`). No frameworks, no bundler, no new deps.
13. **`WEB_DIRECTORY = "./web"`** is now set in `__init__.py`; `nodes.py:2279` picks it
    up into `EXTENSION_WEB_DIRS` and `server.py:363` serves it at
    `/extensions/ComfyUI-H3-SeedScout/`.
14. **`IS_CHANGED` returns NaN in interactive mode** so the execution cache can never
    short-circuit a re-scout. `scout`/`final` keep normal caching.
15. **Previews are pushed as base64 data URIs** rather than by output-path reference.
    The WebPs are still written to `output/` exactly as in v1, but embedding them means
    the node UI works regardless of subfolder/URL-quoting differences.

## Known limitations

* Not executed inside a live ComfyUI process — only `py_compile` + API grep-audit.
  In particular the exact WebP frame timing, the animated-preview rendering on the node,
  and H3 VAE prefix-decode memory behaviour are unverified at runtime.
* Scout mode returns the **first** seed's partial latent as `output` / `denoised_output`.
  It is a partially-denoised latent — don't render it as a final video.
* Previews cover only the first ~5–22 frames of the clip (see note 4).
* Audio is never previewed (video stream only, `unbind()[0]`, matching core `VAEDecode`).
* **Interactive mode holds a queue slot open** for as long as you take to decide. Nothing
  else can run on that ComfyUI instance meanwhile (this is inherent to the Image-Chooser
  blocking pattern).
* Multistep samplers lose their history buffer at the split boundary — see "Why the
  continuation is trajectory-exact".
* Interactive mode assumes **one** scout node per prompt is waiting at a time; sessions
  are keyed by node id, so two waiting nodes are fine, but re-queueing the same node id
  while it is already waiting is not.
* `validate_comfy_workflow.py` only understands the **UI graph** format (`nodes`/`links`);
  on API-format JSON it reports `nodes=0 links=0`. The API-format node-reference check
  was therefore done separately (all 15 nodes, every `["node", slot]` ref resolves).
* Unverified without a live run: the actual DOM widget rendering/sizing on frontend
  1.47.12, animated-WebP playback inside the node, the websocket round-trip, and real
  VRAM behaviour of the scout→block→continue sequence.
