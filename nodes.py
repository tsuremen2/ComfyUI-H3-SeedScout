"""MiniMax H3 Seed Scout Sampler (v2 — interactive select-and-continue).

Drop-in replacement for the `RandomNoise` + `SamplerCustomAdvanced` pair in MiniMax H3
text/image-to-video workflows.

Three modes:

* ``interactive`` (default) -- ONE queued prompt: scout N seeds through
  ``sigmas[:scout_step + 1]``, push previews to the browser, BLOCK on a
  ``threading.Event`` until the user picks a seed on the node, then resume that exact
  seed's saved partial latent through ``sigmas[scout_step:]`` (Image-Chooser pattern).
  Outputs a full-quality latent.
* ``scout`` -- v1 behaviour: partial run of N seeds, save/label one animated WebP each,
  return the first seed's partial latent. API/headless friendly.
* ``final`` -- byte-for-byte stock ``SamplerCustomAdvanced`` on ``selected_seed``.

Everything here is verified against THIS install
(E:\\Stablediffusion\\ComfyUI_qwen); see README.md for the audit notes.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
import time

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers  # noqa: F401  (ensures guider classes are importable/loaded)
import comfy.utils
import folder_paths
import latent_preview

try:  # Pillow ships with ComfyUI
    from PIL import Image
    _PIL_OK = True
except Exception:  # pragma: no cover - Pillow is always present in ComfyUI
    Image = None
    _PIL_OK = False


LOG = logging.getLogger("H3SeedScout")

PREVIEW_MODES = ["vae", "tae", "latent2rgb"]
MODES = ["interactive", "scout", "final"]

EVT_PREVIEW = "h3_seed_scout_preview"
EVT_WAITING = "h3_seed_scout_waiting"
EVT_DONE = "h3_seed_scout_done"
SELECT_ROUTE = "/h3_seed_scout/select"


# ---------------------------------------------------------------------------
# interactive selection plumbing
# ---------------------------------------------------------------------------

_SESSIONS: dict[str, dict] = {}
_SESSIONS_LOCK = threading.Lock()


def _session_open(node_id: str, seeds: list[int]) -> dict:
    session = {"event": threading.Event(), "seeds": list(seeds), "seed": None}
    with _SESSIONS_LOCK:
        _SESSIONS[str(node_id)] = session
    return session


def _session_close(node_id: str) -> None:
    with _SESSIONS_LOCK:
        _SESSIONS.pop(str(node_id), None)


def _session_select(node_id: str, seed: int) -> tuple[bool, str]:
    """Called from the aiohttp route thread. Returns (ok, message)."""
    with _SESSIONS_LOCK:
        session = _SESSIONS.get(str(node_id))
        if session is None:
            return False, "no H3 Seed Scout node {} is waiting".format(node_id)
        if session["event"].is_set():
            return False, "selection already made"
        if int(seed) not in session["seeds"]:
            return False, "seed {} is not one of the scouted seeds {}".format(
                seed, session["seeds"])
        session["seed"] = int(seed)
    session["event"].set()
    return True, "ok"


def _get_server():
    """PromptServer.instance, or None when running headless / before server init."""
    try:
        import server
        return getattr(server.PromptServer, "instance", None)
    except Exception:  # pragma: no cover
        return None


def _send(event: str, payload: dict) -> bool:
    """server.PromptServer.instance.send_sync(event, data, sid) — verified server.py:1392.

    sid is the client_id of the browser that queued the prompt (execution.py:737).
    Returns False when there is no server or no connected client.
    """
    srv = _get_server()
    if srv is None:
        return False
    client_id = getattr(srv, "client_id", None)
    if client_id is None:
        return False
    try:
        srv.send_sync(event, payload, client_id)
        return True
    except Exception:  # pragma: no cover
        LOG.exception("[H3SeedScout] send_sync(%s) failed", event)
        return False


_ROUTES_REGISTERED = False


def _register_routes() -> None:
    """@PromptServer.instance.routes.post(...) — same pattern as RES4LYF/res4lyf.py:37.

    Must run at import time: custom_nodes are imported before the aiohttp app starts
    and its RouteTableDef is frozen.
    """
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    srv = _get_server()
    if srv is None:
        LOG.warning("[H3SeedScout] no PromptServer instance; interactive mode disabled.")
        return
    try:
        from aiohttp import web
    except Exception:  # pragma: no cover
        LOG.warning("[H3SeedScout] aiohttp unavailable; interactive mode disabled.")
        return

    try:
        _decorator = srv.routes.post(SELECT_ROUTE)
    except Exception:  # pragma: no cover - routes already frozen
        LOG.warning("[H3SeedScout] could not register %s; interactive mode disabled.",
                    SELECT_ROUTE)
        return

    @_decorator
    async def h3_seed_scout_select(request):  # noqa: F811
        try:
            data = await request.json()
            node_id = data.get("node_id")
            seed = data.get("seed")
            if node_id is None or seed is None:
                return web.json_response({"ok": False, "error": "node_id and seed required"},
                                         status=400)
            ok, msg = _session_select(str(node_id), int(seed))
            return web.json_response({"ok": ok, "error": None if ok else msg},
                                     status=200 if ok else 409)
        except Exception as exc:  # pragma: no cover
            LOG.exception("[H3SeedScout] /select failed")
            return web.json_response({"ok": False, "error": str(exc)}, status=500)

    _ROUTES_REGISTERED = True
    LOG.info("[H3SeedScout] registered POST %s", SELECT_ROUTE)


_register_routes()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _video_stream(latent):
    """Return the video stream of a (possibly nested) H3 latent tensor.

    Mirrors core `nodes.VAEDecode.decode` (ComfyUI_qwen/nodes.py ~line 332):
    for a NestedTensor the video stream is `unbind()[0]`, audio is `[1]`.
    """
    if getattr(latent, "is_nested", False):
        return latent.unbind()[0]
    return latent


def _zeros_like_latent(latent_samples):
    """Zero noise, exactly like `Noise_EmptyNoise.generate_noise`
    (comfy_extras/nodes_custom_sampler.py:702)."""
    if getattr(latent_samples, "is_nested", False):
        zeros = [
            torch.zeros(t.shape, dtype=t.dtype, layout=t.layout, device="cpu")
            for t in latent_samples.unbind()
        ]
        return comfy.nested_tensor.NestedTensor(zeros)
    return torch.zeros(
        latent_samples.shape, dtype=latent_samples.dtype,
        layout=latent_samples.layout, device="cpu",
    )


def _to_cpu(latent_samples):
    return latent_samples.cpu() if hasattr(latent_samples, "cpu") else latent_samples


def _frames_for_tokens(num_tokens: int) -> int:
    """H3 video VAE temporal expansion, copied from comfy/sd.py `upscale_ratio[0]`.

    frames = max(1, (tokens - 2) // 5 * 17 + 5)
    """
    return max(1, (num_tokens - 2) // 5 * 17 + 5)


def _tokens_for_frames(target_frames: int, available_tokens: int) -> int:
    """Smallest contiguous latent-token PREFIX that decodes to >= target_frames.

    The H3 video VAE (comfy/ldm/minimax/vae.py, MiniMaxH3VideoVAE) is a *causal*,
    temporally chunked codec (clip_length=17, vae_ratio_t=4, tokens_chunk_size=5,
    token_drop=3, token_overlap=2).  Evenly-spaced temporal sampling of latent
    frames is therefore NOT safe -- it would feed the decoder a discontinuous
    token sequence.  A contiguous prefix starting at token 0 *is* safe: it is
    exactly what the decoder sees for a shorter clip.

    Valid token counts are 2, 7, 12, 17, ... (2 + 5k) -> 5, 22, 39, 56, ... frames.
    """
    if available_tokens <= 2:
        return max(1, available_tokens)
    k = 0
    while _frames_for_tokens(2 + 5 * k) < target_frames:
        k += 1
        if 2 + 5 * k >= available_tokens:
            break
    return max(2, min(2 + 5 * k, available_tokens))


def _downscale_images(images: torch.Tensor, max_side: int) -> torch.Tensor:
    """images: [N, H, W, C] float in 0..1. Downscale so max(H, W) <= max_side."""
    if max_side <= 0 or images.ndim != 4:
        return images
    h, w = int(images.shape[1]), int(images.shape[2])
    longest = max(h, w)
    if longest <= max_side:
        return images
    scale = max_side / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    # common_upscale wants [N, C, H, W]
    x = images.movedim(-1, 1)
    x = comfy.utils.common_upscale(x, new_w, new_h, "bilinear", "disabled")
    return x.movedim(1, -1)


def _latent2rgb_images(video_latent: torch.Tensor, latent_format, num_frames: int) -> torch.Tensor:
    """Near-free preview using latent_rgb_factors (comfy/latent_formats.py).

    MiniMaxH3Video defines latent_rgb_factors + latent_rgb_factors_bias and has NO
    taesd_decoder_name, so this is the only zero-VAE preview available here.
    video_latent: [B, C, T, H, W] -> returns [T', H, W, 3] in 0..1.
    """
    factors = getattr(latent_format, "latent_rgb_factors", None)
    if factors is None:
        raise RuntimeError("latent_format has no latent_rgb_factors; use preview_mode='vae'")
    bias = getattr(latent_format, "latent_rgb_factors_bias", None)

    x = video_latent
    if x.ndim == 5:
        x = x[0]  # [C, T, H, W]
    elif x.ndim == 4:
        x = x[0].unsqueeze(1)  # [C, 1, H, W]
    x = x.float().cpu()
    t = x.shape[1]
    if num_frames > 0:
        t = min(t, num_frames)
        x = x[:, :t]

    w = torch.tensor(factors, dtype=torch.float32).transpose(0, 1)  # [3, C]
    b = torch.tensor(bias, dtype=torch.float32) if bias is not None else None
    # [C, T, H, W] -> [T, H, W, C]
    x = x.permute(1, 2, 3, 0)
    out = torch.nn.functional.linear(x, w, bias=b)
    return ((out + 1.0) / 2.0).clamp(0.0, 1.0)


def _vae_preview_images(vae, video_latent: torch.Tensor, preview_frames: int) -> torch.Tensor:
    """Decode a contiguous temporal PREFIX of the video latent with the real H3 VAE."""
    x = video_latent
    if x.ndim == 4:  # [B, C, H, W] -> add temporal dim
        x = x.unsqueeze(2)
    available = int(x.shape[2])
    tokens = _tokens_for_frames(max(1, preview_frames), available)
    x = x[:1, :, :tokens]  # first batch item, contiguous prefix from t=0
    images = vae.decode(x)
    if images.ndim == 5:  # [B, T, H, W, C] -> combine batches, like core VAEDecode
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    if preview_frames > 0 and images.shape[0] > preview_frames:
        images = images[:preview_frames]
    return images.float().clamp(0.0, 1.0).cpu()


def _save_preview(images: torch.Tensor, filename_prefix: str, seed: int, fps: int):
    """Save an animated WebP (PNG fallback) and return (ui_entry, animated, path, mime)."""
    output_dir = folder_paths.get_output_directory()
    height = int(images.shape[1])
    width = int(images.shape[2])
    prefix = "{}_seed_{}".format(filename_prefix, seed)
    full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        prefix, output_dir, width, height
    )

    pil_frames = []
    for frame in images:
        arr = (frame.numpy() * 255.0).clip(0, 255).astype("uint8")
        pil_frames.append(Image.fromarray(arr))

    ext = "webp"
    mime = "image/webp"
    file = "{}_{:05}_.{}".format(filename, counter, ext)
    path = os.path.join(full_output_folder, file)
    try:
        pil_frames[0].save(
            path,
            save_all=True,
            duration=int(1000.0 / max(1, fps)),
            append_images=pil_frames[1:],
            lossless=False,
            quality=85,
            method=4,
        )
        animated = len(pil_frames) > 1
    except Exception as exc:  # pragma: no cover - defensive
        LOG.warning("[H3SeedScout] WebP save failed (%s); falling back to PNG.", exc)
        ext = "png"
        mime = "image/png"
        file = "{}_{:05}_.{}".format(filename, counter, ext)
        path = os.path.join(full_output_folder, file)
        pil_frames[0].save(path, compress_level=4)
        animated = False

    return {"filename": file, "subfolder": subfolder, "type": "output"}, animated, path, mime


_TAE_DECODER = None
_TAE_TRIED = False


def _get_tae_decoder():
    """taeh3.safetensors from models/vae_approx via KJNodes' TinyVAEDecoder.

    Imported by file path so we depend on the module, not on KJNodes' package
    layout/registration. Returns None (once, with a warning) if unavailable.
    """
    global _TAE_DECODER, _TAE_TRIED
    if _TAE_TRIED:
        return _TAE_DECODER
    _TAE_TRIED = True
    try:
        import importlib.util
        kj_tiny = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "ComfyUI-KJNodes", "nodes", "tiny_vae.py",
        )
        spec = importlib.util.spec_from_file_location("h3ss_kj_tiny_vae", kj_tiny)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        dec = mod.load_tiny_vae_decoder("taeh3.safetensors")
        if dec is not None and dec.latent_channels != 24:
            LOG.warning("[H3SeedScout] taeh3 decodes %d-channel latents, H3 video is 24; "
                        "ignoring it.", dec.latent_channels)
            dec = None
        _TAE_DECODER = dec
        if dec is not None:
            LOG.info("[H3SeedScout] taeh3 tiny VAE loaded (upscale %dx).", dec.upscale_ratio)
    except Exception:
        LOG.exception("[H3SeedScout] could not load taeh3 tiny VAE")
        _TAE_DECODER = None
    return _TAE_DECODER


def _tae_preview_images(video_latent: torch.Tensor, num_frames: int) -> torch.Tensor:
    """Per-latent-frame taeh3 decode: [B,C,T,H,W] -> [T',H,W,3] in 0..1.

    2D per-frame decoder — no temporal 4x decompression, so T' is latent frames
    (like latent2rgb). Play at fps/4 for ~real time.
    """
    dec = _get_tae_decoder()
    if dec is None:
        raise RuntimeError("taeh3 tiny VAE unavailable; use preview_mode='vae' or 'latent2rgb'")
    x = video_latent
    if x.ndim == 4:
        x = x.unsqueeze(2)
    t = int(x.shape[2])
    indices = None
    if 0 < num_frames < t:
        indices = [round(i * (t - 1) / (num_frames - 1)) for i in range(num_frames)] \
            if num_frames > 1 else [0]
    return dec.decode_video(x[:1], frame_indices=indices).clamp(0.0, 1.0).cpu()


def _encode_webp_b64(images: torch.Tensor, fps: int) -> str | None:
    """Encode [N,H,W,C] 0..1 frames to an in-memory animated WebP, base64'd."""
    import io as _io
    try:
        frames = [Image.fromarray((f.numpy() * 255.0).clip(0, 255).astype("uint8"))
                  for f in images]
        buf = _io.BytesIO()
        frames[0].save(buf, format="WEBP", save_all=True,
                       duration=int(1000.0 / max(1, fps)),
                       append_images=frames[1:], lossless=False, quality=70, method=2)
        return base64.b64encode(buf.getvalue()).decode("ascii")
    except Exception:  # pragma: no cover
        LOG.exception("[H3SeedScout] provisional webp encode failed")
        return None


def _b64_file(path: str) -> str | None:
    try:
        with open(path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("ascii")
    except Exception:  # pragma: no cover
        LOG.exception("[H3SeedScout] could not base64 %s", path)
        return None


# ---------------------------------------------------------------------------
# node
# ---------------------------------------------------------------------------

class MiniMaxH3SeedScoutSampler:
    """Seed-scouting / interactive sampler for MiniMax H3 (classic ComfyUI node API)."""

    @staticmethod
    def _preview_modes():
        """'tae' is offered only when taeh3.safetensors is actually installed."""
        try:
            if "taeh3.safetensors" in folder_paths.get_filename_list("vae_approx"):
                return list(PREVIEW_MODES)
        except Exception:
            pass
        return [m for m in PREVIEW_MODES if m != "tae"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "mode": (MODES, {"default": "interactive"}),
                "seed_start": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                     "control_after_generate": True},
                ),
                "seed_count": ("INT", {"default": 6, "min": 1, "max": 32}),
                "seed_stride": ("INT", {"default": 1, "min": 1, "max": 1000000}),
                "scout_step": ("INT", {"default": 3, "min": 1, "max": 1000}),
                "selected_seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF,
                     "control_after_generate": True},
                ),
                "preview_mode": (cls._preview_modes(), {"default": "vae"}),
                "preview_frames": ("INT", {"default": 8, "min": 1, "max": 512}),
                "preview_fps": ("INT", {"default": 8, "min": 1, "max": 60}),
                "max_preview_side": ("INT", {"default": 384, "min": 64, "max": 2048, "step": 16}),
                "filename_prefix": ("STRING", {"default": "h3_scout"}),
                "selection_timeout": (
                    "INT",
                    {"default": 0, "min": 0, "max": 86400,
                     "tooltip": "Interactive mode: seconds to wait for a selection. "
                                "0 = wait forever."},
                ),
            },
            "optional": {
                "vae": ("VAE",),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("output", "denoised_output", "seed_used", "seed_report")
    FUNCTION = "run"
    CATEGORY = "sampling/custom_sampling/minimax_h3"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "MiniMax H3 seed scout. interactive: scout N seeds at sigmas[:scout_step+1], "
        "show clickable seed buttons + preview on the node, block until you confirm, "
        "then finish that exact seed through sigmas[scout_step:]. "
        "scout: previews only (headless). final: stock SamplerCustomAdvanced."
    )

    @classmethod
    def IS_CHANGED(cls, mode, **kwargs):
        # interactive mode must never be served from the execution cache — the user
        # has to be able to re-scout and pick again.
        if mode == "interactive":
            return float("nan")
        return False

    # -- core sampling, mirrors comfy_extras/nodes_custom_sampler.SamplerCustomAdvanced --

    @staticmethod
    def _prepare_latent(guider, latent_image):
        latent = latent_image.copy()
        samples = latent["samples"]
        samples = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            samples,
            latent.get("downscale_ratio_spacial", None),
            latent.get("downscale_ratio_temporal", None),
        )
        latent["samples"] = samples
        return latent, samples

    @staticmethod
    def _sample_core(guider, sampler, sigmas, latent, latent_samples, noise, seed):
        """One guider.sample() call. Returns (out_latent, denoised_latent_or_None, x0_raw)."""
        noise_mask = latent.get("noise_mask", None)

        x0_output = {}
        callback = latent_preview.prepare_callback(
            guider.model_patcher, sigmas.shape[-1] - 1, x0_output
        )
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        samples = guider.sample(
            noise,
            latent_samples,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=seed,
        )
        samples = samples.to(comfy.model_management.intermediate_device())

        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = samples

        out_denoised = None
        x0_out = None
        if "x0" in x0_output:
            x0 = x0_output["x0"]
            # In this install CFGGuider.sample already re-nests x0 before the callback
            # (comfy/samplers.py ~1287); this branch is the legacy fallback, kept for parity.
            if getattr(samples, "is_nested", False) and not getattr(x0, "is_nested", False):
                latent_shapes = [x.shape for x in samples.unbind()]
                x0 = comfy.nested_tensor.NestedTensor(
                    comfy.utils.unpack_latents(x0, latent_shapes)
                )
            x0_out = guider.model_patcher.model.process_latent_out(x0.cpu())
            out_denoised = latent.copy()
            out_denoised.pop("downscale_ratio_spacial", None)
            out_denoised.pop("downscale_ratio_temporal", None)
            out_denoised["samples"] = x0_out
        return out, out_denoised, x0_out

    def _sample_seed(self, guider, sampler, sigmas, latent, latent_samples, seed):
        """Fresh-noise pass, exactly what Noise_RandomNoise + SamplerCustomAdvanced do."""
        batch_inds = latent.get("batch_index", None)
        noise = comfy.sample.prepare_noise(latent_samples, seed, batch_inds)
        return self._sample_core(guider, sampler, sigmas, latent, latent_samples, noise, seed)

    def _sample_continue(self, guider, sampler, sigmas, latent, partial_samples, seed):
        """Zero-noise continuation from a partial latent (DisableNoise + SplitSigmas)."""
        noise = _zeros_like_latent(partial_samples)
        return self._sample_core(
            guider, sampler, sigmas, latent, partial_samples, noise, seed
        )

    # ------------------------------------------------------------------ run

    def run(self, guider, sampler, sigmas, latent_image, mode, seed_start, seed_count,
            seed_stride, scout_step, selected_seed, preview_mode, preview_frames,
            preview_fps, max_preview_side, filename_prefix, selection_timeout=0,
            vae=None, unique_id=None):

        if mode == "final":
            return self._run_final(guider, sampler, sigmas, latent_image, selected_seed)

        interactive = (mode == "interactive")
        return self._run_scout(
            guider, sampler, sigmas, latent_image, seed_start, seed_count, seed_stride,
            scout_step, preview_mode, preview_frames, preview_fps, max_preview_side,
            filename_prefix, vae, interactive, int(selection_timeout or 0), unique_id,
        )

    def _run_final(self, guider, sampler, sigmas, latent_image, selected_seed):
        latent, samples = self._prepare_latent(guider, latent_image)
        out, out_denoised, _ = self._sample_seed(
            guider, sampler, sigmas, latent, samples, selected_seed
        )
        if out_denoised is None:
            out_denoised = out
        report = "mode=final  seed={}  steps={}".format(selected_seed, sigmas.shape[-1] - 1)
        return {
            "ui": {"images": []},
            "result": (out, out_denoised, int(selected_seed), report),
        }

    # ------------------------------------------------------------- scout core

    def _run_scout(self, guider, sampler, sigmas, latent_image, seed_start, seed_count,
                   seed_stride, scout_step, preview_mode, preview_frames, preview_fps,
                   max_preview_side, filename_prefix, vae, interactive, selection_timeout,
                   unique_id):
        total_steps = int(sigmas.shape[-1]) - 1
        if total_steps < 1:
            raise ValueError("sigmas must contain at least 2 values")
        k = max(1, min(int(scout_step), total_steps))
        scout_sigmas = sigmas[: k + 1]      # TRUE partial run of the full schedule
        continue_sigmas = sigmas[k:]        # overlaps one boundary, per core SplitSigmas

        if preview_mode == "vae" and vae is None:
            LOG.warning(
                "[H3SeedScout] preview_mode='vae' but no VAE connected; "
                "falling back to tae/latent2rgb."
            )
            preview_mode = "tae"
        if preview_mode == "tae" and _get_tae_decoder() is None:
            LOG.warning("[H3SeedScout] taeh3 not available; falling back to latent2rgb.")
            preview_mode = "latent2rgb"
        if not _PIL_OK:
            raise RuntimeError("Pillow is required for H3 Seed Scout previews")

        latent, samples = self._prepare_latent(guider, latent_image)
        latent_format = guider.model_patcher.model.latent_format

        seeds = [int(seed_start) + i * int(seed_stride) for i in range(int(seed_count))]
        node_id = str(unique_id) if unique_id is not None else "unknown"

        first_out = None
        first_denoised = None
        collected = []   # [seed, preview_video_latent_cpu, elapsed, partial_samples_cpu]
        interrupted = False

        LOG.info(
            "[H3SeedScout] %s: scouting %d seed(s) at step %d/%d (sigmas[:%d])",
            "interactive" if interactive else "scout", len(seeds), k, total_steps, k + 1,
        )

        # ---- 1. sample every seed through the partial schedule -----------------
        try:
            for idx, seed in enumerate(seeds):
                comfy.model_management.throw_exception_if_processing_interrupted()
                t0 = time.time()
                out, out_denoised, x0_out = self._sample_seed(
                    guider, sampler, scout_sigmas, latent, samples, seed
                )
                elapsed = time.time() - t0

                if first_out is None:
                    first_out = out
                    first_denoised = out_denoised if out_denoised is not None else out

                # preview comes from x0 (denoised estimate); continuation needs the
                # actual sampler output `samples`.
                preview_src = x0_out if x0_out is not None else out["samples"]
                preview_video = _video_stream(preview_src).detach().float().cpu()
                partial = _to_cpu(out["samples"]) if interactive else None
                collected.append([seed, preview_video, elapsed, partial])

                LOG.info("[H3SeedScout] seed %d/%d = %d  (%.1fs)",
                         idx + 1, len(seeds), seed, elapsed)

                # Instant provisional preview (free latent2rgb) so the gallery fills
                # seed-by-seed; the proper VAE previews replace these after the loop.
                if interactive:
                    try:
                        # taeh3 tiny VAE if available (real colors, still ~free),
                        # else latent2rgb
                        if _get_tae_decoder() is not None:
                            prov = _tae_preview_images(preview_video, int(preview_frames))
                        else:
                            prov = _latent2rgb_images(
                                preview_video, latent_format, int(preview_frames))
                        prov = _downscale_images(prov, min(int(max_preview_side), 256))
                        # latent frames are 4x temporally compressed vs pixel frames,
                        # so quarter the fps to play the provisional at ~real time
                        b64 = _encode_webp_b64(prov, max(1, int(preview_fps) // 4))
                        if b64:
                            _send(EVT_PREVIEW, {
                                "node_id": node_id, "index": idx, "total": len(seeds),
                                "seed": seed, "elapsed": round(elapsed, 2),
                                "mime": "image/webp", "image_b64": b64,
                                "provisional": True,
                            })
                    except Exception:
                        LOG.exception("[H3SeedScout] provisional preview failed")

                del out, out_denoised, x0_out, preview_src
                comfy.model_management.soft_empty_cache()
        except comfy.model_management.InterruptProcessingException:
            interrupted = True
            LOG.warning("[H3SeedScout] interrupted; saving %d preview(s) already scouted.",
                        len(collected))

        # ---- 2. decode + save + push previews ---------------------------------
        # Done AFTER the sampling loop so the diffusion model loads once and the VAE
        # loads once, instead of thrashing on every seed.
        results = []
        animated_any = False
        lines = [
            "mode={}  scout_step={}/{}  seeds={}  preview={}".format(
                "interactive" if interactive else "scout", k, total_steps,
                len(seeds), preview_mode,
            )
        ]
        pushed = 0
        for idx, entry in enumerate(collected):
            seed, video, elapsed = entry[0], entry[1], entry[2]
            try:
                if preview_mode == "vae":
                    images = _vae_preview_images(vae, video, int(preview_frames))
                    save_fps = int(preview_fps)
                elif preview_mode == "tae":
                    images = _tae_preview_images(video, int(preview_frames))
                    # per-latent-frame decode: quarter fps for ~real-time playback
                    save_fps = max(1, int(preview_fps) // 4)
                else:
                    images = _latent2rgb_images(video, latent_format, int(preview_frames))
                    save_fps = max(1, int(preview_fps) // 4)
                images = _downscale_images(images, int(max_preview_side))
                ui_entry, animated, path, mime = _save_preview(
                    images, filename_prefix, seed, save_fps
                )
                results.append(ui_entry)
                animated_any = animated_any or animated
                lines.append(
                    "  seed {:<22} {:6.1f}s  {}x{}x{}f  {}".format(
                        seed, elapsed, int(images.shape[2]), int(images.shape[1]),
                        int(images.shape[0]), os.path.basename(path),
                    )
                )
                if interactive and not interrupted:
                    b64 = _b64_file(path)
                    if b64 and _send(EVT_PREVIEW, {
                        "node_id": node_id,
                        "index": idx,
                        "total": len(collected),
                        "seed": seed,
                        "elapsed": round(elapsed, 2),
                        "mime": mime,
                        "image_b64": b64,
                        "filename": ui_entry["filename"],
                        "subfolder": ui_entry["subfolder"],
                    }):
                        pushed += 1
            except Exception as exc:
                LOG.exception("[H3SeedScout] preview failed for seed %s", seed)
                lines.append("  seed {:<22} {:6.1f}s  PREVIEW FAILED: {}".format(
                    seed, elapsed, exc))
            # the preview latent is no longer needed
            entry[1] = None
            comfy.model_management.soft_empty_cache()

        if interrupted:
            lines.append("  ** INTERRUPTED — previews above were still written to disk **")
            LOG.info("[H3SeedScout] report:\n%s", "\n".join(lines))
            _send(EVT_DONE, {"node_id": node_id, "seed": None, "status": "interrupted"})
            _session_close(node_id)
            raise comfy.model_management.InterruptProcessingException()

        if first_out is None:
            raise RuntimeError("H3 Seed Scout produced no samples")

        if not interactive:
            report = "\n".join(lines)
            LOG.info("[H3SeedScout] report:\n%s", report)
            ui = {"images": results}
            if animated_any:
                ui["animated"] = (True,) * len(results)
            return {
                "ui": ui,
                "result": (first_out, first_denoised, int(seeds[0]), report),
            }

        # ---- 3. block until the user picks a seed -----------------------------
        # interactive returns the continued result, not the first seed's partial
        first_out = None
        first_denoised = None
        scouted = [e[0] for e in collected]
        chosen, how = self._await_selection(
            node_id, scouted, k, total_steps, selection_timeout, pushed
        )
        lines.append("  selection: seed {} ({})".format(chosen, how))

        # ---- 4. free the losers, continue the winner --------------------------
        partial = None
        for entry in collected:
            if entry[0] == chosen and partial is None:
                partial = entry[3]
            entry[3] = None
        collected = None
        comfy.model_management.soft_empty_cache()

        if partial is None:
            raise RuntimeError(
                "H3 Seed Scout: no stored partial latent for seed {}".format(chosen))

        LOG.info("[H3SeedScout] continuing seed %s through sigmas[%d:] (%d steps)",
                 chosen, k, total_steps - k)
        t0 = time.time()
        out, out_denoised, _ = self._sample_continue(
            guider, sampler, continue_sigmas, latent, partial, chosen
        )
        cont_elapsed = time.time() - t0
        if out_denoised is None:
            out_denoised = out
        lines.append("  continue: seed {} steps {}->{}  {:.1f}s".format(
            chosen, k, total_steps, cont_elapsed))

        report = "\n".join(lines)
        LOG.info("[H3SeedScout] report:\n%s", report)
        _send(EVT_DONE, {"node_id": node_id, "seed": chosen, "status": "continued"})

        # Interactive mode: previews already live in the node's own gallery widget.
        # Returning them in `ui` would stack ComfyUI's default image previews on top.
        return {
            "ui": {"images": []},
            "result": (out, out_denoised, int(chosen), report),
        }

    # --------------------------------------------------------------- blocking

    def _await_selection(self, node_id, seeds, k, total_steps, timeout, pushed):
        """Block on a threading.Event; poll so Cancel in the UI still works."""
        if pushed == 0:
            LOG.warning(
                "[H3SeedScout] no browser client received the previews "
                "(headless / API queue?). Falling back to seed %s without waiting.",
                seeds[0],
            )
            return seeds[0], "no browser client — auto-selected first seed"

        session = _session_open(node_id, seeds)
        _send(EVT_WAITING, {
            "node_id": node_id,
            "seeds": seeds,
            "scout_step": k,
            "total_steps": total_steps,
            "remaining_steps": total_steps - k,
            "timeout": timeout,
        })
        LOG.info("[H3SeedScout] waiting for a seed selection on node %s "
                 "(timeout=%s) — pick one on the node in the browser.",
                 node_id, timeout or "none")

        start = time.time()
        try:
            while not session["event"].wait(0.5):
                comfy.model_management.throw_exception_if_processing_interrupted()
                if timeout > 0 and (time.time() - start) >= timeout:
                    LOG.warning(
                        "[H3SeedScout] ***** SELECTION TIMEOUT after %ss — "
                        "falling back to the FIRST scouted seed %s *****",
                        timeout, seeds[0],
                    )
                    _send(EVT_DONE, {"node_id": node_id, "seed": seeds[0],
                                     "status": "timeout"})
                    return seeds[0], "timeout after {}s — fell back to first seed".format(
                        timeout)
            chosen = session["seed"]
        except comfy.model_management.InterruptProcessingException:
            _send(EVT_DONE, {"node_id": node_id, "seed": None, "status": "cancelled"})
            raise
        finally:
            _session_close(node_id)

        waited = time.time() - start
        LOG.info("[H3SeedScout] user selected seed %s after %.1fs", chosen, waited)
        return int(chosen), "user selected after {:.1f}s".format(waited)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3SeedScoutSampler": MiniMaxH3SeedScoutSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3SeedScoutSampler": "MiniMax H3 Seed Scout Sampler",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
