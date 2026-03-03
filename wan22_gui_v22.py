#!/usr/bin/env python3
"""
Wan2.2 Character Studio — Gradio GUI
Character replacement & animation using Wan2.2-Animate-14B
"""

import gradio as gr
import subprocess
import os
import sys
import tempfile
import shutil
import cv2

# ─── Config ───────────────────────────────────────────────────────────────────
DEFAULT_CKPT_DIR = "./Wan2.2-Animate-14B"
DEFAULT_STEPS    = 20
DEFAULT_FPS      = 30
RESOLUTIONS      = ["Auto (from video)", "1280x720", "960x540", "832x480", "640x480"]

import threading
import time

current_process  = None

# ─── Server Busy Lock ─────────────────────────────────────────────────────────
# Single GPU server: only one generation at a time.
# All clients poll get_server_status() via gr.Timer to see live state.

_generation_lock  = threading.Lock()
_server_busy      = False          # True while a generation is running
_busy_client_info = ""             # short description shown to waiting clients
_busy_start_time  = 0.0            # epoch seconds when job started


def _set_busy(info: str):
    global _server_busy, _busy_client_info, _busy_start_time
    _server_busy      = True
    _busy_client_info = info
    _busy_start_time  = time.time()


def _set_free():
    global _server_busy, _busy_client_info, _busy_start_time
    _server_busy      = False
    _busy_client_info = ""
    _busy_start_time  = 0.0


def get_server_status():
    """Called every ~3s by gr.Timer for ALL connected clients."""
    if not _server_busy:
        return (
            gr.update(visible=False),                       # busy_banner hidden
            gr.update(interactive=True,  value="▶  GENERATE VIDEO"),  # run_btn enabled
        )
    elapsed = int(time.time() - _busy_start_time)
    mins, secs = divmod(elapsed, 60)
    elapsed_str = f"{mins}m {secs:02d}s" if mins else f"{secs}s"
    banner_html = f"""
    <div class="busy-banner">
      <div class="busy-icon">⏳</div>
      <div>
        <div class="busy-title">SERVER ĐANG BẬN</div>
        <div class="busy-desc">{_busy_client_info}</div>
        <div class="busy-timer">⏱ Đã chạy: {elapsed_str} — vui lòng chờ hoặc thử lại sau</div>
      </div>
    </div>"""
    return (
        gr.update(visible=True, value=banner_html),         # busy_banner shown
        gr.update(interactive=False, value="⏳  Server đang bận..."),  # run_btn disabled
    )


# ─── Model Cache (keep models in VRAM between runs) ──────────────────────────
# Models are loaded once and reused across generations.
# Cache key = (model_type, ckpt_dir, device, dtype)
# Call clear_model_cache() to free VRAM manually.

import torch

_model_cache: dict = {}          # key → pipeline object
_model_cache_lock = threading.Lock()
_model_cache_info: dict = {}     # key → {"loaded_at", "use_count", "vram_gb"}


def _cache_key(model_type: str, ckpt_dir: str, cuda_devices: str, fp8: bool) -> str:
    return f"{model_type}|{ckpt_dir}|{cuda_devices}|fp8={fp8}"


def get_cached_model(model_type: str, ckpt_dir: str, cuda_devices: str,
                     fp8: bool, cpu_offload: bool, logs: list, emit):
    """
    Return cached pipeline if available, else load and cache it.
    model_type: "i2v", "t2v", "flf2v_high", "flf2v_low"
    """
    key = _cache_key(model_type, ckpt_dir, cuda_devices, fp8)

    with _model_cache_lock:
        if key in _model_cache:
            _model_cache_info[key]["use_count"] += 1
            count = _model_cache_info[key]["use_count"]
            logs.append(f"  ⚡  Model cache HIT [{model_type}] — use #{count}, skipping load")
            yield None, emit(logs)
            return _model_cache[key]

    # Cache miss — load model
    logs.append(f"  🔄  Loading {model_type} model from {ckpt_dir} ...")
    logs.append(f"  ℹ️  (Will be cached in VRAM for future runs)")
    yield None, emit(logs)

    try:
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline, WanPipeline

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices
        dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16

        t0 = time.time()

        if model_type in ("i2v", "flf2v_high", "flf2v_low"):
            vae = AutoencoderKLWan.from_pretrained(
                ckpt_dir, subfolder="vae", torch_dtype=torch.float32
            )
            pipe = WanImageToVideoPipeline.from_pretrained(
                ckpt_dir, vae=vae, torch_dtype=torch.bfloat16
            )
        elif model_type == "t2v":
            pipe = WanPipeline.from_pretrained(
                ckpt_dir, torch_dtype=torch.bfloat16
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        if cpu_offload:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")

        elapsed = time.time() - t0

        # Estimate VRAM usage
        vram_gb = 0.0
        try:
            vram_gb = torch.cuda.memory_allocated() / 1e9
        except Exception:
            pass

        with _model_cache_lock:
            _model_cache[key] = pipe
            _model_cache_info[key] = {
                "loaded_at":  time.strftime("%H:%M:%S"),
                "use_count":  1,
                "vram_gb":    vram_gb,
                "model_type": model_type,
                "ckpt_dir":   ckpt_dir,
            }

        logs.append(f"  ✅  Model loaded in {elapsed:.1f}s — cached in VRAM ({vram_gb:.1f} GB used)")
        yield None, emit(logs)
        return pipe

    except Exception as e:
        import traceback
        logs += [f"  ❌  Failed to load model: {e}", traceback.format_exc()]
        yield None, emit(logs)
        return None


def clear_model_cache(model_type: str = "all") -> str:
    """Free cached models from VRAM. model_type='all' clears everything."""
    with _model_cache_lock:
        if model_type == "all":
            keys = list(_model_cache.keys())
        else:
            keys = [k for k in _model_cache if k.startswith(model_type + "|")]

        if not keys:
            return "ℹ️  Cache already empty."

        freed = []
        for key in keys:
            pipe = _model_cache.pop(key)
            info = _model_cache_info.pop(key, {})
            del pipe
            freed.append(info.get("model_type", key))

        try:
            torch.cuda.empty_cache()
            import gc; gc.collect()
        except Exception:
            pass

        return f"✅  Freed {len(freed)} model(s): {', '.join(freed)}"


def get_cache_status() -> str:
    """Return human-readable cache status for UI display."""
    with _model_cache_lock:
        if not _model_cache:
            return "📭  No models in VRAM cache"
        lines = ["📦  Models in VRAM cache:"]
        for key, info in _model_cache_info.items():
            lines.append(
                f"  • {info['model_type']:12s}  |  loaded {info['loaded_at']}"
                f"  |  used {info['use_count']}×"
                f"  |  ~{info['vram_gb']:.1f} GB"
            )
        try:
            total = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            lines.append(f"  GPU VRAM: {total:.1f} GB allocated / {reserved:.1f} GB reserved")
        except Exception:
            pass
        return "\n".join(lines)


# ─── GPU Detection ────────────────────────────────────────────────────────────

def detect_gpus():
    """Return list of (index, name, vram_gb) for all available NVIDIA GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                idx  = int(parts[0])
                name = parts[1]
                vram = round(int(parts[2]) / 1024, 1)
                gpus.append((idx, name, vram))
        return gpus
    except Exception:
        return []


def gpu_choices():
    """Return list of checkbox labels for available GPUs."""
    gpus = detect_gpus()
    if not gpus:
        return ["GPU 0 (unknown)"]
    return [f"GPU {idx} — {name}  [{vram} GB]" for idx, name, vram in gpus]


def parse_selected_gpus(selected_labels):
    """Extract GPU indices using regex — handles em-dash, multi-digit IDs, etc."""
    import re
    indices = []
    seen = set()
    for label in (selected_labels or []):
        m = re.search(r"GPU\s+(\d+)", label, re.IGNORECASE)
        if m:
            val = m.group(1)
            if val not in seen:
                seen.add(val)
                indices.append(val)
    return ",".join(indices) if indices else "0"


def merge_audio(source_video, output_video, logs):
    """Merge audio from source_video into output_video using ffmpeg. Returns new path."""
    merged = output_video.replace(".mp4", "_audio.mp4")
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=codec_type", "-of", "default=noprint_wrappers=1",
             source_video],
            capture_output=True, text=True, timeout=15
        )
        if "audio" not in probe.stdout:
            logs.append("  ℹ️  Source video has no audio track — skipping audio merge.")
            return output_video, logs

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", output_video,
                "-i", source_video,
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-shortest",
                merged,
            ],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode == 0 and os.path.exists(merged):
            logs.append("  🔊  Audio merged from source video.")
            return merged, logs
        else:
            logs.append(f"  ⚠️  Audio merge failed: {result.stderr[-200:]}")
            return output_video, logs
    except FileNotFoundError:
        logs.append("  ⚠️  ffmpeg not found — skipping audio merge. Install with: apt install ffmpeg")
        return output_video, logs
    except Exception as e:
        logs.append(f"  ⚠️  Audio merge error: {e}")
        return output_video, logs


# ─── Face Swap Pipeline ──────────────────────────────────────────────────────

def check_insightface():
    """Check if insightface is available."""
    try:
        import insightface
        return True
    except ImportError:
        return False


def run_face_swap(video_path, face_image_path, output_path, logs,
                  blend_alpha=0.92, face_enhance=True):
    """
    Standalone face-only swap using InsightFace Buffalo_L.
    Steps:
      1. Detect face in reference image → get face embedding
      2. For each video frame: detect face → swap using inswapper_128
      3. Optionally enhance with GFPGAN/CodeFormer
      4. Write output video
    Returns (output_path, logs)
    """
    try:
        import insightface
        from insightface.app import FaceAnalysis
        import numpy as np
    except ImportError:
        logs.append("  ❌  insightface not installed. Run:")
        logs.append("      pip install insightface onnxruntime-gpu")
        return None, logs

    # ── Load models ──
    logs.append("  🔄  Loading InsightFace Buffalo_L...")
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Load inswapper
    import os, urllib.request
    swapper_path = os.path.expanduser("~/.insightface/models/inswapper_128.onnx")
    if not os.path.exists(swapper_path):
        os.makedirs(os.path.dirname(swapper_path), exist_ok=True)
        # Try multiple sources — HuggingFace requires login, use public mirrors
        download_urls = [
            "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx",
            "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx",
        ]
        downloaded = False
        for url in download_urls:
            try:
                logs.append(f"  📥  Downloading inswapper_128.onnx from {url.split('/')[2]}...")
                tmp_path = swapper_path + ".tmp"
                urllib.request.urlretrieve(url, tmp_path)
                os.rename(tmp_path, swapper_path)
                logs.append("  ✅  inswapper_128.onnx downloaded.")
                downloaded = True
                break
            except Exception as e:
                logs.append(f"  ⚠️  Failed from {url.split('/')[2]}: {e}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        if not downloaded:
            logs.append("  ❌  Auto-download failed. Please download manually:")
            logs.append("      wget -O ~/.insightface/models/inswapper_128.onnx" +
                        " https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx")
            return None, logs

    swapper = insightface.model_zoo.get_model(swapper_path,
                download=False, download_zip=False)

    # ── Get reference face ──
    ref_img = cv2.imread(face_image_path)
    if ref_img is None:
        logs.append("  ❌  Cannot read reference image.")
        return None, logs

    ref_faces = app.get(ref_img)
    if not ref_faces:
        logs.append("  ❌  No face detected in reference image. Use a clear frontal face photo.")
        return None, logs

    ref_face = sorted(ref_faces, key=lambda f: f.bbox[2] - f.bbox[0], reverse=True)[0]
    logs.append(f"  ✅  Reference face detected (det_score={ref_face.det_score:.2f})")

    # ── Process video ──
    cap = cv2.VideoCapture(video_path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Write raw frames to a temp file first (OpenCV can't write H264 directly)
    # Then re-encode with ffmpeg to produce browser-compatible H264/AAC mp4
    raw_path = output_path.replace(".mp4", "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")   # MJPG → fast lossless intermediate
    out = cv2.VideoWriter(raw_path, fourcc, fps, (w, h))

    swapped_count = 0
    skipped_count = 0
    frame_idx = 0
    log_interval = max(1, total_frames // 10)   # log every 10%

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        if faces:
            target_faces = sorted(faces, key=lambda f: f.bbox[2] - f.bbox[0], reverse=True)
            for tgt_face in target_faces[:1]:   # only largest face
                frame = swapper.get(frame, tgt_face, ref_face, paste_back=True)
            swapped_count += 1
        else:
            skipped_count += 1

        out.write(frame)
        frame_idx += 1

        if frame_idx % log_interval == 0:
            pct = int(frame_idx / total_frames * 100)
            logs.append(f"  🖼  Progress: {pct}%  ({frame_idx}/{total_frames} frames)")

    cap.release()
    out.release()

    logs.append(f"  ✅  Face swap done: {swapped_count} frames swapped, {skipped_count} frames skipped (no face detected)")

    # ── Re-encode to browser-compatible H264 mp4 with ffmpeg ──
    logs.append("  🎞️  Re-encoding to H264/mp4 (browser-compatible)...")
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-i", raw_path,
        "-c:v", "libx264",       # H264 — universally supported
        "-preset", "fast",        # encode speed vs compression tradeoff
        "-crf", "18",             # quality: 0=lossless, 23=default, 18=high quality
        "-pix_fmt", "yuv420p",    # required for browser compatibility
        "-movflags", "+faststart", # moov atom at front → instant browser playback
        output_path,
    ]
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=600)
    if os.path.exists(raw_path):
        os.remove(raw_path)   # cleanup intermediate file

    if result.returncode != 0:
        logs.append(f"  ⚠️  ffmpeg re-encode failed: {result.stderr[-300:]}")
        logs.append("  ℹ️  Falling back to raw output (may trigger browser warning)")
        # rename raw as output fallback
        if os.path.exists(raw_path):
            os.rename(raw_path, output_path)
    else:
        logs.append("  ✅  Re-encode complete — H264 mp4 ready.")

    return output_path, logs


# ─── I2V Valid Sizes ─────────────────────────────────────────────────────────
# generate.py only accepts these exact sizes for i2v/t2v tasks
I2V_VALID_SIZES = [
    (1280, 720), (720, 1280),
    (832,  480), (480,  832),
    (1280, 704), (704, 1280),
    (1024, 704), (704, 1024),
]

def find_closest_i2v_size(img_w, img_h):
    """Return the I2V valid size string closest to the input image dimensions."""
    import math
    img_ratio = img_w / img_h
    best_size = None
    best_score = float("inf")
    for w, h in I2V_VALID_SIZES:
        # Score = ratio difference + area scale factor (prefer same orientation)
        ratio_diff = abs((w / h) - img_ratio)
        # Penalize orientation flip
        same_orient = (img_w >= img_h) == (w >= h)
        orient_penalty = 0 if same_orient else 2.0
        score = ratio_diff + orient_penalty
        if score < best_score:
            best_score = score
            best_size = (w, h)
    return f"{best_size[0]}x{best_size[1]}"


def on_i2v_image_upload(image_path):
    """Called when I2V image is uploaded — detect size and pick closest valid size."""
    if image_path is None:
        return gr.update(), "—"
    try:
        import cv2
        img = cv2.imread(image_path)
        if img is not None:
            h, w = img.shape[:2]
            best = find_closest_i2v_size(w, h)
            bw, bh = best.split("x")
            info = f"📐 {w}×{h} → closest valid: {best}"
            return gr.update(value=best), info
    except Exception:
        pass
    return gr.update(), "—"


# ─── FLF2V Pipeline (First-Last Frame to Video) ──────────────────────────────

def process_flf2v(first_image, last_image, prompt, size, steps, seed,
                  duration_sec, ckpt_dir_high, ckpt_dir_low,
                  selected_gpus, cpu_offload, fp8_quant, guide_scale):
    """
    First-Last Frame to Video using Wan2.2 dual-model pipeline via diffusers.
    Requires:
      - wan2.2-I2V-A14B-High-Noise  (conditions on first frame)
      - wan2.2-I2V-A14B-Low-Noise   (conditions on last frame)
    Steps:
      1. Load high-noise model → encode first frame → get latent_start
      2. Load low-noise model  → encode last frame  → get latent_end
      3. Interpolate + denoise → full video
    """
    if first_image is None:
        yield None, "❌  Vui lòng upload ảnh frame đầu."
        return
    if last_image is None:
        yield None, "❌  Vui lòng upload ảnh frame cuối."
        return
    if not prompt.strip():
        yield None, "❌  Vui lòng nhập prompt."
        return

    if not _generation_lock.acquire(blocking=False):
        yield None, f"🚫  SERVER ĐANG BẬN\n    {_busy_client_info}\n    Vui lòng chờ rồi thử lại."
        return

    cuda_devices = parse_selected_gpus(selected_gpus) if selected_gpus else "0"
    gpu_count    = len(selected_gpus) if selected_gpus else 1
    _set_busy(f"FLF2V | GPU:{cuda_devices} | {size} | {steps}steps")
    work_dir    = tempfile.mkdtemp(prefix="wan22_flf2v_")
    output_path = os.path.join(work_dir, "flf2v_output.mp4")

    out_fps   = 16
    frame_num = max(1, round(duration_sec * out_fps))
    if (frame_num - 1) % 4 != 0:
        frame_num = ((frame_num - 1) // 4 + 1) * 4 + 1

    size_str  = size.replace("x", "*")
    w, h      = [int(x) for x in size.split("x")]

    def emit(lines): return "\n".join(lines)
    logs = [
        "╔" + "═"*50 + "╗",
        "║   🎞️  FIRST-LAST FRAME TO VIDEO" + " "*17 + "║",
        "╠" + "═"*50 + "╣",
        f"║  Size        {size:<35}║",
        f"║  Steps       {steps:<35}║",
        f"║  Duration    {duration_sec:.1f}s  →  frame_num: {frame_num:<20}║",
        f"║  GPU(s)      {cuda_devices:<35}║",
        "╚" + "═"*50 + "╝", "",
    ]
    yield None, emit(logs)

    try:
        from diffusers.utils import export_to_video, load_image
        from PIL import Image

        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_devices

        # ── Get or load cached models ──
        pipe_high = None
        for update in get_cached_model("flf2v_high", ckpt_dir_high, cuda_devices,
                                        fp8_quant, cpu_offload, logs, emit):
            if isinstance(update, tuple):
                yield update
            else:
                pipe_high = update
        if pipe_high is None:
            return

        pipe_low = None
        for update in get_cached_model("flf2v_low", ckpt_dir_low, cuda_devices,
                                        fp8_quant, cpu_offload, logs, emit):
            if isinstance(update, tuple):
                yield update
            else:
                pipe_low = update
        if pipe_low is None:
            return

        # ── Load images ──
        img_first = load_image(first_image).resize((w, h))
        img_last  = load_image(last_image).resize((w, h))
        logs.append(f"  ✅  Images loaded: {w}×{h}")
        yield None, emit(logs)

        # ── Generate with high-noise (first frame conditioning) ──
        logs.append("  🎨  Step 1/2 — Generating from first frame (high-noise)...")
        yield None, emit(logs)

        seed_val = int(seed) if seed and int(seed) > 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed_val)

        output_high = pipe_high(
            image          = img_first,
            prompt         = prompt,
            height         = h,
            width          = w,
            num_frames     = frame_num,
            num_inference_steps = steps,
            guidance_scale = guide_scale,
            generator      = generator,
            output_type    = "latent",
        )
        latent_high = output_high.frames   # latent space, not decoded yet
        logs.append("  ✅  High-noise pass complete.")
        yield None, emit(logs)

        # ── Generate with low-noise (last frame conditioning) ──
        logs.append("  🎨  Step 2/2 — Generating from last frame (low-noise)...")
        yield None, emit(logs)

        output_low = pipe_low(
            image          = img_last,
            prompt         = prompt,
            height         = h,
            width          = w,
            num_frames     = frame_num,
            num_inference_steps = steps,
            guidance_scale = guide_scale,
            generator      = generator,
            latents        = latent_high,   # pass latent from first pass
            output_type    = "np",
        )
        frames = output_low.frames[0]
        logs.append("  ✅  Low-noise pass complete.")
        yield None, emit(logs)

        # ── Export video ──
        logs.append("  🎞️  Exporting video...")
        yield None, emit(logs)
        raw_path = output_path.replace(".mp4", "_raw.mp4")
        export_to_video(frames, raw_path, fps=out_fps)

        # Re-encode browser-compatible
        subprocess.run([
            "ffmpeg", "-y", "-i", raw_path,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            output_path,
        ], capture_output=True, timeout=300)
        if os.path.exists(raw_path):
            os.remove(raw_path)

        final = os.path.join(tempfile.gettempdir(), "wan22_flf2v_output.mp4")
        if os.path.exists(output_path):
            shutil.copy2(output_path, final)
            logs += ["", "╔"+"═"*50+"╗",
                     "║   ✅  SUCCESS  —  FLF2V Ready!           ║",
                     "╚"+"═"*50+"╝"]
            yield final, emit(logs)
        else:
            logs.append("❌  Output file not found.")
            yield None, emit(logs)

    except ImportError as e:
        logs += [f"\n❌  Missing dependency: {e}",
                 "  Install: pip install diffusers transformers accelerate Pillow"]
        yield None, emit(logs)
    except Exception as e:
        import traceback
        logs += [f"\n❌  Exception: {e}", traceback.format_exc()]
        yield None, emit(logs)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _set_free()
        _generation_lock.release()


# ─── I2V / T2V Commands ──────────────────────────────────────────────────────

def build_i2v_cmd(image_path, prompt, output_path, size, steps, seed,
                  ckpt_dir, gpu_count=1, cpu_offload=True, fp8_quant=False,
                  frame_num=81, guide_scale=5.0, sample_shift=3.0):
    """Build Image-to-Video generate command."""
    size_str = size.replace("x", "*")   # convert "1280x720" → "1280*720"
    if gpu_count > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", "1", "--nproc_per_node", str(gpu_count),
            "generate.py",
            "--task",         "i2v-A14B",
            "--ckpt_dir",     ckpt_dir,
            "--image",        image_path,
            "--prompt",       prompt,
            "--size",         size_str,
            "--sample_steps", str(steps),
            "--frame_num",    str(frame_num),
            "--sample_guide_scale", str(guide_scale),
            "--sample_shift", str(sample_shift),
            "--save_file",    output_path,
            "--dit_fsdp", "--t5_fsdp",
            "--ulysses_size", str(gpu_count),
        ]
    else:
        cmd = [
            sys.executable, "generate.py",
            "--task",         "i2v-A14B",
            "--ckpt_dir",     ckpt_dir,
            "--image",        image_path,
            "--prompt",       prompt,
            "--size",         size_str,
            "--sample_steps", str(steps),
            "--frame_num",    str(frame_num),
            "--sample_guide_scale", str(guide_scale),
            "--sample_shift", str(sample_shift),
            "--save_file",    output_path,
        ]
    if seed and int(seed) > 0:
        cmd += ["--base_seed", str(seed)]
    if not cpu_offload:
        cmd += ["--offload_model", "False"]
    if fp8_quant:
        cmd.append("--convert_model_dtype")
    return cmd


def build_t2v_cmd(prompt, output_path, size, steps, seed,
                  ckpt_dir, gpu_count=1, cpu_offload=True, fp8_quant=False,
                  frame_num=81, guide_scale=5.0, sample_shift=8.0):
    """Build Text-to-Video generate command."""
    size_str = size.replace("x", "*")   # convert "1280x720" → "1280*720"
    if gpu_count > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes", "1", "--nproc_per_node", str(gpu_count),
            "generate.py",
            "--task",         "t2v-A14B",
            "--ckpt_dir",     ckpt_dir,
            "--prompt",       prompt,
            "--size",         size_str,
            "--sample_steps", str(steps),
            "--frame_num",    str(frame_num),
            "--sample_guide_scale", str(guide_scale),
            "--sample_shift", str(sample_shift),
            "--save_file",    output_path,
            "--dit_fsdp", "--t5_fsdp",
            "--ulysses_size", str(gpu_count),
        ]
    else:
        cmd = [
            sys.executable, "generate.py",
            "--task",         "t2v-A14B",
            "--ckpt_dir",     ckpt_dir,
            "--prompt",       prompt,
            "--size",         size_str,
            "--sample_steps", str(steps),
            "--frame_num",    str(frame_num),
            "--sample_guide_scale", str(guide_scale),
            "--sample_shift", str(sample_shift),
            "--save_file",    output_path,
        ]
    if seed and int(seed) > 0:
        cmd += ["--base_seed", str(seed)]
    if not cpu_offload:
        cmd += ["--offload_model", "False"]
    if fp8_quant:
        cmd.append("--convert_model_dtype")
    return cmd


def process_i2v(image_input, prompt, size, steps, seed, duration_sec, fps,
                ckpt_dir, selected_gpus, cpu_offload, fp8_quant, guide_scale, sample_shift):
    """Image-to-Video generation pipeline."""
    if image_input is None:
        yield None, "❌  Vui lòng upload ảnh đầu vào."
        return
    if not prompt.strip():
        yield None, "❌  Vui lòng nhập prompt mô tả video."
        return
    if not os.path.exists(ckpt_dir):
        yield None, f"❌  Checkpoint directory not found: {ckpt_dir}"
        return

    if not _generation_lock.acquire(blocking=False):
        yield None, f"🚫  SERVER ĐANG BẬN\n    {_busy_client_info}\n    Vui lòng chờ rồi thử lại."
        return

    cuda_devices = parse_selected_gpus(selected_gpus) if selected_gpus else "0"
    gpu_count    = len(selected_gpus) if selected_gpus else 1
    _set_busy(f"I2V | GPU:{cuda_devices} | {size} | {steps}steps")
    work_dir    = tempfile.mkdtemp(prefix="wan22_i2v_")
    output_path = os.path.join(work_dir, "i2v_output.mp4")

    # frame_num from duration × user fps
    out_fps   = int(fps)
    frame_num = max(1, round(duration_sec * out_fps))
    if (frame_num - 1) % 4 != 0:
        frame_num = ((frame_num - 1) // 4 + 1) * 4 + 1

    def emit(lines): return "\n".join(lines)
    logs = [
        "╔" + "═"*50 + "╗",
        "║   🖼️  IMAGE-TO-VIDEO Generation" + " "*17 + "║",
        "╠" + "═"*50 + "╣",
        f"║  Size       {size:<36}║",
        f"║  Steps      {steps:<36}║",
        f"║  Duration   {duration_sec:.1f}s → frame_num: {frame_num:<24}║",
        f"║  GPU(s)     {cuda_devices:<36}║",
        f"║  CFG Scale  {guide_scale:<36}║",
        "╚" + "═"*50 + "╝", "",
    ]
    yield None, emit(logs)

    proc_env = os.environ.copy()
    proc_env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    proc_env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    proc_env["LD_LIBRARY_PATH"] = (
        proc_env.get("LD_LIBRARY_PATH", "") + ":" +
        str(sys.executable).replace("/bin/python", "/lib/python" +
        f"{sys.version_info.major}.{sys.version_info.minor}/site-packages/torch/lib")
    )

    try:
        from diffusers.utils import export_to_video, load_image
        from PIL import Image

        # ── Get or load cached model ──
        pipe = None
        for update in get_cached_model("i2v", ckpt_dir, cuda_devices, fp8_quant,
                                        cpu_offload, logs, emit):
            if isinstance(update, tuple):
                yield update
            else:
                pipe = update
        if pipe is None:
            return

        # ── Run inference ──
        logs += [f"{'─'*52}", "  🎨  Generating I2V...", f"{'─'*52}", ""]
        yield None, emit(logs)

        w, h = [int(x) for x in size.split("x")]
        img = load_image(image_input).resize((w, h))
        seed_val = int(seed) if seed and int(seed) > 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed_val)

        result = pipe(
            image=img, prompt=prompt,
            height=h, width=w,
            num_frames=frame_num,
            num_inference_steps=steps,
            guidance_scale=guide_scale,
            generator=generator,
        )
        frames = result.frames[0]

        # ── Export + re-encode ──
        raw = output_path.replace(".mp4", "_raw.mp4")
        export_to_video(frames, raw, fps=out_fps)
        final = os.path.join(tempfile.gettempdir(), "wan22_i2v_output.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", raw,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", final,
        ], capture_output=True, timeout=300)
        if os.path.exists(raw):
            os.remove(raw)

        logs += ["", "╔"+"═"*50+"╗",
                 "║   ✅  SUCCESS  —  I2V Ready!             ║",
                 "╚"+"═"*50+"╝"]
        yield final if os.path.exists(final) else None, emit(logs)

    except Exception as e:
        import traceback
        logs += [f"\n❌  Exception: {e}", traceback.format_exc()]
        yield None, emit(logs)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _set_free()
        _generation_lock.release()


def process_t2v(prompt, size, steps, seed, duration_sec,
                ckpt_dir, selected_gpus, cpu_offload, fp8_quant, guide_scale, sample_shift):
    """Text-to-Video generation pipeline."""
    if not prompt.strip():
        yield None, "❌  Vui lòng nhập prompt mô tả video."
        return
    if not os.path.exists(ckpt_dir):
        yield None, f"❌  Checkpoint directory not found: {ckpt_dir}"
        return

    if not _generation_lock.acquire(blocking=False):
        yield None, f"🚫  SERVER ĐANG BẬN\n    {_busy_client_info}\n    Vui lòng chờ rồi thử lại."
        return

    cuda_devices = parse_selected_gpus(selected_gpus) if selected_gpus else "0"
    gpu_count    = len(selected_gpus) if selected_gpus else 1
    _set_busy(f"T2V | GPU:{cuda_devices} | {size} | {steps}steps")
    work_dir    = tempfile.mkdtemp(prefix="wan22_t2v_")
    output_path = os.path.join(work_dir, "t2v_output.mp4")

    out_fps   = 16
    frame_num = max(1, round(duration_sec * out_fps))
    if (frame_num - 1) % 4 != 0:
        frame_num = ((frame_num - 1) // 4 + 1) * 4 + 1

    def emit(lines): return "\n".join(lines)
    logs = [
        "╔" + "═"*50 + "╗",
        "║   📝  TEXT-TO-VIDEO Generation" + " "*19 + "║",
        "╠" + "═"*50 + "╣",
        f"║  Size       {size:<36}║",
        f"║  Steps      {steps:<36}║",
        f"║  Duration   {duration_sec:.1f}s → frame_num: {frame_num:<24}║",
        f"║  GPU(s)     {cuda_devices:<36}║",
        "╚" + "═"*50 + "╝", "",
    ]
    yield None, emit(logs)

    proc_env = os.environ.copy()
    proc_env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    proc_env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    try:
        cmd = build_t2v_cmd(
            prompt, output_path, size, steps, seed,
            ckpt_dir, gpu_count=gpu_count, cpu_offload=cpu_offload,
            fp8_quant=fp8_quant, frame_num=frame_num,
            guide_scale=guide_scale, sample_shift=sample_shift,
        )
        logs += [f"{'─'*52}", "  🎬  Generating T2V...", f"{'─'*52}", "",
                 f"  ⚙️  CMD: {' '.join(cmd)}", ""]
        yield None, emit(logs)

        from diffusers.utils import export_to_video

        # ── Get or load cached model ──
        pipe = None
        for update in get_cached_model("t2v", ckpt_dir, cuda_devices, fp8_quant,
                                        cpu_offload, logs, emit):
            if isinstance(update, tuple):
                yield update
            else:
                pipe = update
        if pipe is None:
            return

        # ── Run inference ──
        logs += [f"{'─'*52}", "  🎬  Generating T2V...", f"{'─'*52}", ""]
        yield None, emit(logs)

        w, h = [int(x) for x in size.split("x")]
        seed_val = int(seed) if seed and int(seed) > 0 else torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator("cuda").manual_seed(seed_val)

        result = pipe(
            prompt=prompt,
            height=h, width=w,
            num_frames=frame_num,
            num_inference_steps=steps,
            guidance_scale=guide_scale,
            generator=generator,
        )
        frames = result.frames[0]

        raw = output_path.replace(".mp4", "_raw.mp4")
        export_to_video(frames, raw, fps=16)
        final = os.path.join(tempfile.gettempdir(), "wan22_t2v_output.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-i", raw,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", final,
        ], capture_output=True, timeout=300)
        if os.path.exists(raw):
            os.remove(raw)

        logs += ["", "╔"+"═"*50+"╗",
                 "║   ✅  SUCCESS  —  T2V Ready!             ║",
                 "╚"+"═"*50+"╝"]
        yield final if os.path.exists(final) else None, emit(logs)

    except Exception as e:
        import traceback
        logs += [f"\n❌  Exception: {e}", traceback.format_exc()]
        yield None, emit(logs)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _set_free()
        _generation_lock.release()


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_video_info(video_path):
    """Return (width, height, fps, duration_sec, total_frames) from video file."""
    try:
        cap          = cv2.VideoCapture(video_path)
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps          = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if w > 0 and h > 0 and fps > 0:
            duration = total_frames / fps
            return w, h, round(fps), round(duration, 3), total_frames
    except Exception:
        pass
    return 1280, 720, DEFAULT_FPS, 0.0, 0


def get_video_duration(video_path):
    """Return duration in seconds."""
    try:
        cap   = cv2.VideoCapture(video_path)
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if fps > 0 and total > 0:
            return total / fps
    except Exception:
        pass
    return 0.0


def on_video_upload(video_path):
    """Called when video is uploaded — return info string."""
    if video_path is None:
        return "—"
    w, h, fps, duration, total_frames = get_video_info(video_path)
    return f"📐 {w}×{h}  •  🎞 {fps} fps  •  ⏱ {duration:.1f}s  ({total_frames} frames)"


def resolve_resolution(resolution_str, video_path=None):
    if resolution_str == "Auto (from video)" and video_path:
        w, h, *_ = get_video_info(video_path)
        return w, h
    elif resolution_str == "Auto (from video)":
        return 1280, 720
    w, h = resolution_str.split("x")
    return int(w), int(h)


# ─── Build commands ───────────────────────────────────────────────────────────

def build_preprocess_cmd(video_path, character_path, save_path,
                         w, h, fps, use_flux, mode, ckpt_dir):
    cmd = [
        sys.executable,
        "./wan/modules/animate/preprocess/preprocess_data.py",
        "--ckpt_path",      f"{ckpt_dir}/process_checkpoint",
        "--video_path",     video_path,
        "--refer_path",     character_path,
        "--save_path",      save_path,
        "--resolution_area", str(w), str(h),
        "--fps",            str(fps),
    ]
    cmd.append("--replace_flag" if mode == "Replace" else "--retarget_flag")
    if use_flux:
        cmd.append("--use_flux")
    return cmd


def build_generate_cmd(save_path, output_path, steps, seed, mode,
                       relighting, ckpt_dir, gpu_count=1,
                       cpu_offload=True, fp8_quant=False,
                       t5_cpu=False, sample_shift=5.0, guide_scale=1.0,
                       frame_num=77):
    """
    Build generate command.
    - Single GPU : python generate.py ...
    - Multi-GPU  : python -m torch.distributed.run
                     --nnodes 1 --nproc_per_node N
                     generate.py --dit_fsdp --t5_fsdp --ulysses_size N ...
    Speed options:
      cpu_offload      : --offload_model True/False (default True saves VRAM)
      fp8_quant        : --convert_model_dtype  (FP8, +20-30% speed)
      t5_cpu           : --t5_cpu (T5 encoder on CPU, saves ~8GB VRAM)
      sample_shift     : --sample_shift (timestep shift, default 5.0)
      guide_scale      : --sample_guide_scale (CFG scale, default 1.0)
    """
    if gpu_count > 1:
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            "--nnodes",         "1",
            "--nproc_per_node", str(gpu_count),
            "generate.py",
            "--task",           "animate-14B",
            "--ckpt_dir",       ckpt_dir,
            "--src_root_path",  save_path,
            "--refert_num",     "1",
            "--sample_steps",   str(steps),
            "--frame_num",      str(frame_num),
            "--save_file",      output_path,
            "--dit_fsdp",
            "--t5_fsdp",
            "--ulysses_size",   str(gpu_count),
        ]
    else:
        cmd = [
            sys.executable, "generate.py",
            "--task",           "animate-14B",
            "--ckpt_dir",       ckpt_dir,
            "--src_root_path",  save_path,
            "--refert_num",     "1",
            "--sample_steps",   str(steps),
            "--frame_num",      str(frame_num),
            "--save_file",      output_path,
        ]

    # ── Speed / memory options ──
    if not cpu_offload:
        cmd += ["--offload_model", "False"]
    if fp8_quant:
        cmd.append("--convert_model_dtype")
    if t5_cpu:
        cmd.append("--t5_cpu")
    if abs(sample_shift - 5.0) > 0.01:
        cmd += ["--sample_shift", str(sample_shift)]
    if abs(guide_scale - 1.0) > 0.01:
        cmd += ["--sample_guide_scale", str(guide_scale)]

    if mode == "Replace":
        cmd.append("--replace_flag")
    if relighting and mode == "Replace":
        cmd.append("--use_relighting_lora")
    if seed and int(seed) > 0:
        cmd += ["--base_seed", str(int(seed))]
    return cmd


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_video(video_input, character_input, mode, resolution, fps,
                  steps, seed, use_flux, relighting, ckpt_dir,
                  selected_gpus, merge_audio_flag,
                  cpu_offload, fp8_quant, t5_cpu, sample_shift, guide_scale):
    global current_process

    # ── Validate ──
    if video_input is None:
        yield None, "❌  Please upload a source video."
        return
    if character_input is None:
        yield None, "❌  Please upload a character image."
        return
    if not os.path.exists(ckpt_dir):
        yield None, f"❌  Checkpoint directory not found:\n    {ckpt_dir}"
        return

    # ── Server busy check ──
    if not _generation_lock.acquire(blocking=False):
        yield None, (
            f"🚫  SERVER ĐANG BẬN\n"
            f"    {_busy_client_info}\n"
            f"    Vui lòng chờ tiến trình hiện tại hoàn thành rồi thử lại."
        )
        return

    # Lock acquired — mark server as busy
    gpu_ids = parse_selected_gpus(selected_gpus) if selected_gpus else "0"
    _set_busy(f"Mode: {mode} | GPU: {gpu_ids} | FPS: {fps} | Steps: {steps}"[:60])
    try:
        # ── GPU setup ──
        cuda_devices = parse_selected_gpus(selected_gpus) if selected_gpus else "0"
        gpu_label    = f"CUDA {cuda_devices}"

        w, h = resolve_resolution(resolution, video_input)
        res_label = f"{w}×{h}" + ("  (auto)" if resolution == "Auto (from video)" else "")

        work_dir    = tempfile.mkdtemp(prefix="wan22_")
        save_path   = os.path.join(work_dir, "process_results")
        output_path = os.path.join(work_dir, "output.mp4")
        os.makedirs(save_path, exist_ok=True)

        def emit(lines):
            return "\n".join(lines)

        offload_label  = "Yes (save VRAM)" if cpu_offload  else "No  (faster, needs VRAM)"
        fp8_label      = "Yes (+20-30% speed)" if fp8_quant else "No"
        t5cpu_label    = "Yes (save ~8 GB VRAM)" if t5_cpu   else "No"
        logs = [
            "╔" + "═" * 50 + "╗",
            "║   🎬  WAN 2.2  —  Processing Started" + " " * 13 + "║",
            "╠" + "═" * 50 + "╣",
            f"║  Mode        {mode:<36}║",
            f"║  Resolution  {res_label:<36}║",
            f"║  FPS         {fps:<36}║",
            f"║  Steps       {steps:<36}║",
            f"║  Seed        {(seed if seed and int(seed)>0 else 'random'):<36}║",
            f"║  Relighting  {'Yes' if relighting else 'No':<36}║",
            f"║  FLUX        {'Yes' if use_flux else 'No':<36}║",
            f"║  GPU(s)      {gpu_label:<36}║",
            f"║  Audio Merge {'Yes' if merge_audio_flag else 'No':<36}║",
            "╠" + "═" * 50 + "╣",
            f"║  CPU Offload {offload_label:<36}║",
            f"║  FP8 Quant   {fp8_label:<36}║",
            f"║  T5 on CPU   {t5cpu_label:<36}║",
            f"║  Shift       {sample_shift:<36}║",
            f"║  CFG Scale   {guide_scale:<36}║",
            "╚" + "═" * 50 + "╝",
            "",
        ]
        yield None, emit(logs)

        # ── Face Swap mode (standalone pipeline, no generate.py needed) ──
        if mode == "Face Swap":
            try:
                logs += [f"{'─'*52}", "  💫  Face Swap Mode (InsightFace)", f"{'─'*52}", ""]
                yield None, emit(logs)

                face_out = os.path.join(work_dir, "face_swapped.mp4")
                face_out, logs = run_face_swap(
                    video_path=video_input,
                    face_image_path=character_input,
                    output_path=face_out,
                    logs=logs,
                )
                yield None, emit(logs)

                if face_out is None or not os.path.exists(face_out):
                    logs += ["", "❌  Face swap FAILED. See logs above."]
                    yield None, emit(logs)
                    return

                # Merge audio if requested
                final_path = face_out
                if merge_audio_flag:
                    logs += [f"{'─'*52}", "  🔊  Merging Audio", f"{'─'*52}", ""]
                    yield None, emit(logs)
                    final_path, logs = merge_audio(video_input, face_out, logs)
                    yield None, emit(logs)

                logs += [
                    "",
                    "╔" + "═" * 50 + "╗",
                    "║   ✅  SUCCESS  —  Face Swap Ready!      " + " " * 9 + "║",
                    "╚" + "═" * 50 + "╝",
                ]
                if os.path.exists(final_path):
                    final = os.path.join(tempfile.gettempdir(), "wan22_faceswap_output.mp4")
                    shutil.copy2(final_path, final)
                    logs.append(f"  📁  {final}")
                    yield final, emit(logs)
                else:
                    logs.append("  ❌  Output file not found.")
                    yield None, emit(logs)
            except Exception as e:
                import traceback
                logs += [f"\n❌  Exception: {e}", traceback.format_exc()]
                yield None, emit(logs)
            return   # lock released by outer finally

        step_rc = [0]

        # Shared env with CUDA_VISIBLE_DEVICES set
        proc_env = os.environ.copy()
        proc_env["CUDA_VISIBLE_DEVICES"] = cuda_devices
        proc_env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

        def run_step(label, cmd):
            nonlocal logs
            global current_process
            logs += [f"{'─'*52}", f"  {label}", f"{'─'*52}", ""]
            yield None, emit(logs)

            current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True, bufsize=1,
                env=proc_env,
            )
            for line in current_process.stdout:
                line = line.rstrip()
                if line:
                    logs.append(f"  {line}")
                    yield None, emit(logs)

            current_process.wait()
            rc = current_process.returncode
            current_process = None
            step_rc[0] = rc

        # ── Step 1: Preprocess ──
        preprocess_cmd = build_preprocess_cmd(
            video_input, character_input, save_path,
            w, h, fps, use_flux, mode, ckpt_dir
        )
        for result in run_step("📦  STEP 1 / 2  —  Preprocessing", preprocess_cmd):
            yield result

        if step_rc[0] != 0:
            logs += ["", f"❌  Preprocessing FAILED (exit code {step_rc[0]}). See logs above."]
            yield None, emit(logs)
            shutil.rmtree(work_dir, ignore_errors=True)
            return

        logs += ["", "  ✅  Preprocessing complete!", ""]
        yield None, emit(logs)

        # ── Step 2: Generate ──
        # Clear VRAM fragment cache before generation to avoid OOM on longer videos
        try:
            import torch, gc
            torch.cuda.empty_cache()
            gc.collect()
            logs.append("  🧹  VRAM cache cleared before generation")
            yield None, emit(logs)
        except Exception:
            pass

        gpu_count = len(selected_gpus) if selected_gpus else 1

        # ── Compute frame_num to preserve full video duration ──
        # frame_num is calculated from user_fps so that:
        #   output_duration = frame_num / user_fps ≈ source_duration
        # Wan2.2 internally renders at 30fps but we re-stamp the container
        # via ffmpeg to user_fps after generation — no re-encode needed.
        # Wan2.2 constraint: frame_num must satisfy (frame_num - 1) % 4 == 0
        _duration = get_video_duration(video_input)
        if _duration > 0:
            _raw_frames = _duration * fps          # fps = user-chosen output fps
            _frame_num  = max(1, round(_raw_frames))
            # Round up to nearest 4k+1 (1, 5, 9, ..., 77, ...)
            if (_frame_num - 1) % 4 != 0:
                _frame_num = ((_frame_num - 1) // 4 + 1) * 4 + 1
        else:
            _frame_num = 77   # fallback default
        _expected_dur = _frame_num / fps
        logs.append(f"  📐  Source: {_duration:.2f}s | frame_num: {_frame_num} @ {fps}fps → expected output: {_expected_dur:.2f}s")
        yield None, emit(logs)

        generate_cmd = build_generate_cmd(
            save_path, output_path, steps, seed,
            mode, relighting, ckpt_dir, gpu_count=gpu_count,
            cpu_offload=cpu_offload, fp8_quant=fp8_quant,
            t5_cpu=t5_cpu, sample_shift=sample_shift, guide_scale=guide_scale,
            frame_num=_frame_num,
        )
        mode_label = f"torchrun x{gpu_count} GPU (FSDP+Ulysses)" if gpu_count > 1 else "single GPU"
        # VRAM estimate: base ~30GB model + ~150MB per frame per GPU
        _vram_per_frame = (w * h) / (960 * 540) * 0.15   # GB per frame
        _vram_extra = _frame_num * _vram_per_frame / gpu_count
        _vram_total_est = 30 + _vram_extra
        logs += [
            f"  ⚙️  Launch mode: {mode_label}",
            f"  ⚙️  VRAM estimate: ~{_vram_total_est:.0f} GB total ({30:.0f} GB model + {_vram_extra:.1f} GB frames / {gpu_count} GPU)",
            f"  ⚙️  CMD: {' '.join(generate_cmd)}", ""
        ]
        yield None, emit(logs)
        for result in run_step("🎨  STEP 2 / 2  —  Generating Video", generate_cmd):
            yield result

        if step_rc[0] != 0:
            logs += ["", f"❌  Generation FAILED (exit code {step_rc[0]}). See logs above."]
            yield None, emit(logs)
            shutil.rmtree(work_dir, ignore_errors=True)
            return

        # ── Step 3: FPS re-stamp ──
        # Wan2.2 always writes output at 30fps container.
        # We re-stamp the pts so container fps = user fps (no re-encode, instant).
        # Because frame_num = round(duration * user_fps), this preserves duration.
        final_path = output_path
        if os.path.exists(output_path):
            logs += [f"{'─'*52}", f"  🎞️  STEP 3 — Re-stamping container to {fps}fps", f"{'─'*52}", ""]
            yield None, emit(logs)
            restamped = output_path.replace(".mp4", f"_{fps}fps.mp4")
            # Re-encode with libx264 at target fps.
            # -c:v copy cannot change fps reliably — must re-encode.
            # -vf "fps=USER_FPS" resamples frame timing correctly.
            r = subprocess.run([
                "ffmpeg", "-y",
                "-i", output_path,
                "-vf", f"fps={fps}",       # resample to target fps (drop/dup frames)
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "17",              # high quality, near-lossless
                "-pix_fmt", "yuv420p",
                "-an",                     # drop audio (merged in next step)
                "-movflags", "+faststart",
                restamped,
            ], capture_output=True, text=True, timeout=600)
            if r.returncode == 0 and os.path.exists(restamped):
                final_path = restamped
                actual = _frame_num / fps
                logs.append(f"  ✅  Re-stamp OK: {_frame_num} frames @ {fps}fps = {actual:.2f}s")
            else:
                logs.append(f"  ⚠️  Re-stamp failed (keeping 30fps): {r.stderr[-200:]}")
            yield None, emit(logs)

        # ── Step 4: Audio merge (optional) ──
        if merge_audio_flag and os.path.exists(final_path):
            logs += [f"{'─'*52}", "  🔊  STEP 4 — Merging Audio", f"{'─'*52}", ""]
            yield None, emit(logs)
            final_path, logs = merge_audio(video_input, final_path, logs)
            yield None, emit(logs)

        # ── Done ──
        logs += [
            "",
            "╔" + "═" * 50 + "╗",
            "║   ✅  SUCCESS  —  Video Ready!          " + " " * 9 + "║",
            "╚" + "═" * 50 + "╝",
        ]

        if os.path.exists(final_path):
            final = os.path.join(tempfile.gettempdir(), "wan22_output.mp4")
            shutil.copy2(final_path, final)
            logs.append(f"  📁  {final}")
            yield final, emit(logs)
        else:
            logs.append("  ❌  Output file not found.")
            yield None, emit(logs)

    except Exception as e:
        import traceback
        logs += [f"\n❌  Exception: {e}", traceback.format_exc()]
        yield None, emit(logs)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)
        _set_free()
        _generation_lock.release()


def cancel_process():
    global current_process
    if current_process and current_process.poll() is None:
        current_process.terminate()
        return "⚠️  Process cancelled."
    return "ℹ️  No active process."


# ─── CSS ──────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

/* ── Base ── */
*, *::before, *::after { box-sizing: border-box; }

body { background: #0d0d12 !important; }

.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    background: #0d0d12 !important;
    font-family: 'Inter', sans-serif !important;
    color: #d4d4e8 !important;
    padding: 24px !important;
}

/* ── Header ── */
.wan-header {
    display: flex;
    align-items: center;
    gap: 20px;
    background: #131318;
    border: 1px solid #252530;
    border-radius: 14px;
    padding: 22px 28px;
    margin-bottom: 22px;
}

.wan-logo {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #6c5ce7, #e84393);
    border-radius: 12px;
    display: flex; align-items: center; justify-content: center;
    font-size: 24px;
    flex-shrink: 0;
    box-shadow: 0 4px 20px rgba(108,92,231,0.35);
}

.wan-title { font-family: 'IBM Plex Mono', monospace; font-size: 20px; font-weight: 600;
    background: linear-gradient(135deg, #a29bfe, #fd79a8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    margin: 0; letter-spacing: -0.3px; }

.wan-sub { color: #6b6b85; font-size: 13px; margin: 3px 0 0; font-weight: 300; }

.wan-badges { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }

.wan-badge {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    background: rgba(162,155,254,0.1); border: 1px solid rgba(162,155,254,0.25);
    color: #a29bfe; padding: 2px 10px; border-radius: 20px; letter-spacing: 0.3px;
}

/* ── Cards ── */
.card {
    background: #131318;
    border: 1px solid #252530;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 14px;
}

.card-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 10px; text-transform: uppercase; letter-spacing: 2px;
    color: #6c5ce7; margin: 0 0 14px; display: flex; align-items: center; gap: 6px;
}

.card-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, #252530, transparent);
}

/* ── Mode pills ── */
.mode-pills .wrap { gap: 8px !important; }
.mode-pills label {
    background: #1a1a22 !important;
    border: 1px solid #2e2e3e !important;
    border-radius: 8px !important;
    padding: 9px 16px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #9494b0 !important;
}
.mode-pills label:has(input:checked) {
    background: rgba(108,92,231,0.12) !important;
    border-color: #6c5ce7 !important;
    color: #a29bfe !important;
}

/* ── Video info pill ── */
.video-info input {
    background: rgba(108,92,231,0.08) !important;
    border: 1px solid rgba(108,92,231,0.2) !important;
    border-radius: 6px !important;
    color: #a29bfe !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 12px !important;
    padding: 6px 12px !important;
}

/* ── Inputs ── */
input[type=text], input[type=number], select, textarea:not(.log-area) {
    background: #1a1a22 !important;
    border: 1px solid #2e2e3e !important;
    border-radius: 8px !important;
    color: #d4d4e8 !important;
    font-family: 'Inter', sans-serif !important;
}

input:focus, select:focus {
    border-color: #6c5ce7 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(108,92,231,0.12) !important;
}

/* ── Sliders ── */
input[type=range] { accent-color: #6c5ce7 !important; }

/* ── Buttons ── */
.btn-generate {
    background: linear-gradient(135deg, #6c5ce7, #8b5cf6) !important;
    border: none !important; border-radius: 10px !important;
    color: #fff !important; font-weight: 600 !important;
    font-size: 14px !important; letter-spacing: 0.3px !important;
    padding: 12px 0 !important;
    box-shadow: 0 4px 16px rgba(108,92,231,0.3) !important;
    transition: all 0.2s ease !important;
}
.btn-generate:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(108,92,231,0.45) !important;
}

.btn-cancel {
    background: #1a1a22 !important;
    border: 1px solid #2e2e3e !important;
    border-radius: 10px !important;
    color: #6b6b85 !important;
    font-size: 13px !important;
}
.btn-cancel:hover { border-color: #e84393 !important; color: #e84393 !important; }

/* ── Log ── */
#log-wrap textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 11.5px !important;
    line-height: 1.75 !important;
    background: #07070c !important;
    color: #50fa7b !important;
    border: 1px solid #1a2b1a !important;
    border-radius: 10px !important;
    padding: 14px !important;
    resize: none !important;
}

/* ── Accordion ── */
.gr-accordion {
    background: #131318 !important;
    border: 1px solid #252530 !important;
    border-radius: 10px !important;
}

/* ── Divider ── */
.divider { height: 1px; background: #252530; margin: 6px 0 14px; }

/* ── Face Swap Note ── */
.faceswap-note {
    font-size: 12px; color: #8888a8; line-height: 1.9;
    background: rgba(253,121,168,0.05);
    border: 1px solid rgba(253,121,168,0.2);
    border-left: 3px solid #fd79a8;
    border-radius: 8px; padding: 10px 14px; margin-top: 10px;
}
.faceswap-note code {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px; color: #fd79a8;
    background: rgba(253,121,168,0.1);
    padding: 1px 6px; border-radius: 4px;
}

/* ── Busy Banner ── */
#busy-banner {
    margin-bottom: 16px;
}
.busy-banner {
    display: flex;
    align-items: center;
    gap: 18px;
    background: linear-gradient(135deg, rgba(232,67,147,0.12), rgba(214,48,49,0.08));
    border: 1px solid rgba(232,67,147,0.4);
    border-left: 4px solid #e84393;
    border-radius: 12px;
    padding: 16px 22px;
    animation: pulse-border 2s ease-in-out infinite;
}
@keyframes pulse-border {
    0%, 100% { border-left-color: #e84393; box-shadow: 0 0 0 0 rgba(232,67,147,0); }
    50%       { border-left-color: #fd79a8; box-shadow: 0 0 16px 2px rgba(232,67,147,0.2); }
}
.busy-icon {
    font-size: 32px;
    animation: spin-slow 3s linear infinite;
    flex-shrink: 0;
}
@keyframes spin-slow {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
.busy-title {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; font-weight: 600;
    color: #fd79a8; letter-spacing: 1.5px;
    text-transform: uppercase;
}
.busy-desc {
    font-size: 12px; color: #9494b0; margin: 4px 0 2px;
    font-family: 'IBM Plex Mono', monospace;
}
.busy-timer {
    font-size: 11px; color: #6b6b85;
    font-family: 'IBM Plex Mono', monospace;
}

/* ── Speed Hint ── */
.speed-hint {
    font-size: 11.5px; color: #8888a8; line-height: 2.2;
    background: rgba(80,250,123,0.04);
    border: 1px solid #1a2b1a;
    border-radius: 8px; padding: 10px 14px; margin-top: 10px;
}
.speed-tag {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    padding: 2px 8px; border-radius: 4px; margin-right: 6px;
}
.speed-safe { background: rgba(80,250,123,0.15); color: #50fa7b; }
.speed-warn { background: rgba(255,184,0,0.12);  color: #f1c40f; }
.speed-info { background: rgba(108,92,231,0.12); color: #a29bfe; }

/* ── GPU Selector ── */
.gpu-selector .wrap { gap: 6px !important; flex-direction: column !important; }
.gpu-selector label {
    background: #1a1a22 !important;
    border: 1px solid #2e2e3e !important;
    border-radius: 8px !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    transition: all 0.15s ease !important;
    font-size: 12px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    color: #9494b0 !important;
}
.gpu-selector label:has(input:checked) {
    background: rgba(80,250,123,0.08) !important;
    border-color: #50fa7b !important;
    color: #50fa7b !important;
}

.gpu-info-bar {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #6b6b85;
    background: rgba(108,92,231,0.06);
    border: 1px solid #252530;
    border-radius: 6px;
    padding: 6px 12px;
    margin-top: 8px;
}

/* ── Tips ── */
.tips {
    background: #131318; border: 1px solid #252530;
    border-left: 3px solid #6c5ce7;
    border-radius: 10px; padding: 14px 18px;
    margin-top: 18px; font-size: 12.5px; color: #8888a8; line-height: 1.9;
}
.tips-head {
    font-family: 'IBM Plex Mono', monospace; font-size: 10px;
    text-transform: uppercase; letter-spacing: 2px; color: #6c5ce7; margin-bottom: 8px;
}
"""

AUTO_SCROLL_JS = """
function() {
    function setupScrolling() {
        const wrap = document.querySelector('#log-wrap');
        if (!wrap) { setTimeout(setupScrolling, 500); return; }

        const observer = new MutationObserver(() => {
            const ta = wrap.querySelector('textarea');
            if (ta && !ta._pauseScroll) {
                ta.scrollTop = ta.scrollHeight;
            }
        });
        observer.observe(wrap, { childList: true, subtree: true, characterData: true });

        setInterval(() => {
            const ta = wrap.querySelector('textarea');
            if (!ta) return;
            // Resume auto-scroll when user reaches bottom
            const atBottom = ta.scrollHeight - ta.scrollTop - ta.clientHeight < 40;
            if (atBottom) ta._pauseScroll = false;
            // Auto-scroll if not paused
            if (!ta._pauseScroll) ta.scrollTop = ta.scrollHeight;
        }, 250);

        wrap.addEventListener('wheel', () => {
            const ta = wrap.querySelector('textarea');
            if (ta) ta._pauseScroll = true;
        }, { passive: true });
    }
    setupScrolling();
}
"""


# ─── UI ───────────────────────────────────────────────────────────────────────

def build_ui(ckpt_dir=DEFAULT_CKPT_DIR):
    with gr.Blocks(title="Wan2.2 Character Studio", js=AUTO_SCROLL_JS) as demo:

        # ── Header ──────────────────────────────────────────
        gr.HTML("""
        <div class="wan-header">
          <div class="wan-logo">⬡</div>
          <div>
            <div class="wan-title">WAN 2.2 CHARACTER STUDIO</div>
            <div class="wan-sub">AI-powered character replacement &amp; animation</div>
            <div class="wan-badges">
              <span class="wan-badge">Animate-14B</span>
              <span class="wan-badge">I2V-14B</span>
              <span class="wan-badge">T2V-14B</span>
              <span class="wan-badge">GPU Accelerated</span>
            </div>
          </div>
        </div>
        """)

        # ── Server Status Banner ──
        busy_banner = gr.HTML(value="", visible=False, elem_id="busy-banner")
        status_timer = gr.Timer(value=3)

        # ── VRAM Cache Controls ──────────────────────────────
        with gr.Accordion("🧠  VRAM Model Cache", open=False):
            with gr.Row():
                cache_status_box = gr.Textbox(
                    label="Cache Status",
                    value="📭  No models cached yet — first generation will load model into VRAM",
                    interactive=False, lines=4,
                )
            with gr.Row():
                cache_refresh_btn = gr.Button("🔄 Refresh Status", scale=2)
                cache_clear_i2v   = gr.Button("🗑 Clear I2V",      scale=1, variant="secondary")
                cache_clear_t2v   = gr.Button("🗑 Clear T2V",      scale=1, variant="secondary")
                cache_clear_flf   = gr.Button("🗑 Clear FLF2V",    scale=1, variant="secondary")
                cache_clear_all   = gr.Button("⚠️ Clear ALL",      scale=1, variant="stop")

        with gr.Tabs():

          # ══════════════════════════════════════════════════
          # TAB 1: Animate (existing)
          # ══════════════════════════════════════════════════
          with gr.TabItem("🎭  Animate / Replace / Face Swap"):

            with gr.Row(equal_height=False, variant="panel"):

                # ══ LEFT COLUMN ════════════════════════════════════
                with gr.Column(scale=5, min_width=380):

                    # ── Media inputs ──
                    gr.HTML('<div class="card"><div class="card-title">📹 Input Media</div>')

                    video_input = gr.Video(label="Source Video", height=190)

                    res_info = gr.Textbox(
                        label="", value="Upload a video to detect resolution & fps",
                        interactive=False, container=False,
                        elem_classes="video-info",
                    )

                    character_input = gr.Image(
                        label="Character Image", type="filepath", height=190,
                    )

                    gr.HTML('</div>')

                    # ── Mode ──
                    gr.HTML('<div class="card"><div class="card-title">🎭 Mode</div>')

                    mode = gr.Radio(
                        choices=["Replace", "Animate", "Face Swap"],
                        value="Replace",
                        label="",
                        info="Replace: swap full character  |  Animate: mimic motion  |  Face Swap: chỉ thay khuôn mặt (InsightFace, không cần generate.py)",
                        elem_classes="mode-pills",
                    )

                    # Face Swap info box (shown conditionally via CSS/note)
                    gr.HTML('''
                    <div class="faceswap-note" id="faceswap-note">
                      <span style="color:#fd79a8;font-family:IBM Plex Mono,monospace;font-size:10px;text-transform:uppercase;letter-spacing:1px;">💄 Face Swap Mode</span><br>
                      Dùng <b>InsightFace + inswapper_128</b> — không cần Wan2.2 model, nhanh hơn nhiều.<br>
                      Ảnh tham chiếu: <b>ảnh chân dung rõ mặt, nhìn thẳng</b> cho kết quả tốt nhất.<br>
                      Cần cài: <code>pip install insightface onnxruntime-gpu</code>
                    </div>
                    ''')

                    gr.HTML('</div>')

                    # ── Output settings ──
                    gr.HTML('<div class="card"><div class="card-title">🎛 Output Settings</div>')

                    with gr.Row():
                        resolution = gr.Dropdown(
                            choices=RESOLUTIONS,
                            value="Auto (from video)",
                            label="Resolution",
                            scale=2,
                        )
                        fps = gr.Slider(
                            minimum=8, maximum=60, value=DEFAULT_FPS, step=1,
                            label="FPS (frames per second)",
                            scale=3,
                        )

                    with gr.Row():
                        steps = gr.Slider(
                            minimum=10, maximum=50, value=DEFAULT_STEPS, step=5,
                            label="Inference Steps",
                            scale=3,
                        )
                        seed = gr.Number(
                            label="Seed  (0 = random)",
                            value=0, precision=0, minimum=0,
                            scale=2,
                        )

                    gr.HTML('</div>')

                    # ── GPU Selection ──
                    gr.HTML('<div class="card"><div class="card-title">🖥️ GPU Selection</div>')

                    _gpu_choices = gpu_choices()
                    gpu_selector = gr.CheckboxGroup(
                        choices=_gpu_choices,
                        value=[_gpu_choices[0]],
                        label="Select GPU(s) to use",
                        info="Multi-select: hold Ctrl/Cmd or click multiple. Selected GPUs share the workload via CUDA_VISIBLE_DEVICES.",
                        elem_classes="gpu-selector",
                    )

                    gpu_info = gr.HTML(
                        value=f'<div class="gpu-info-bar">🔍  {len(_gpu_choices)} GPU(s) detected on this server</div>'
                    )

                    gr.HTML('</div>')

                    # ── Audio Options ──
                    gr.HTML('<div class="card"><div class="card-title">🔊 Audio</div>')

                    merge_audio_flag = gr.Checkbox(
                        label="Merge original audio from source video into output",
                        value=True,
                        info="Requires ffmpeg installed. Audio is trimmed to output video length.",
                    )

                    gr.HTML('</div>')

                    # ── Speed Optimization ──
                    gr.HTML('''<div class="card"><div class="card-title">⚡ Speed Optimization</div>''')

                    with gr.Row():
                        cpu_offload = gr.Checkbox(
                            label="CPU Offload",
                            value=True,
                            info="Tắt để tăng tốc 30-50% nếu VRAM ≥ 80GB",
                            scale=1,
                        )
                        fp8_quant = gr.Checkbox(
                            label="FP8 Quantization",
                            value=False,
                            info="Tăng tốc +20-30%, giảm VRAM. Cần GPU Hopper/Blackwell",
                            scale=1,
                        )
                        t5_cpu = gr.Checkbox(
                            label="T5 Encoder → CPU",
                            value=False,
                            info="Chuyển T5 lên CPU, tiết kiệm ~8GB VRAM GPU",
                            scale=1,
                        )

                    with gr.Row():
                        sample_shift = gr.Slider(
                            minimum=1.0, maximum=10.0, value=5.0, step=0.5,
                            label="Sample Shift  (default 5.0)",
                            info="Tăng → nhấn mạnh chi tiết fine. Giảm → smooth hơn",
                            scale=3,
                        )
                        guide_scale = gr.Slider(
                            minimum=1.0, maximum=10.0, value=1.0, step=0.5,
                            label="CFG Guide Scale  (default 1.0)",
                            info="Tăng → bám prompt hơn nhưng có thể overfit",
                            scale=3,
                        )

                    gr.HTML('''
                    <div class="speed-hint">
                      <span class="speed-tag speed-safe">✅ An toàn</span> CPU Offload OFF + FP8 ON = nhanh nhất, chất lượng gần như không đổi<br>
                      <span class="speed-tag speed-warn">⚠️ Thận trọng</span> T5 CPU làm preprocessing chậm hơn một chút<br>
                      <span class="speed-tag speed-info">💡 Mẹo</span> Steps 10-15 thay vì 20 = nhanh gấp đôi, chất lượng giảm nhẹ
                    </div>
                    ''')

                    gr.HTML('</div>')

                    # ── Advanced ──
                    with gr.Accordion("⚙️  Advanced Options", open=False):
                        ckpt_dir_input = gr.Textbox(
                            label="Checkpoint Directory", value=ckpt_dir,
                        )
                        with gr.Row():
                            use_flux = gr.Checkbox(
                                label="FLUX reference quality",
                                value=False,
                                info="Requires FLUX.1-Kontext-dev (~24 GB)",
                                scale=1,
                            )
                            relighting = gr.Checkbox(
                                label="Relighting LoRA",
                                value=True,
                                info="Match character lighting (Replace only)",
                                scale=1,
                            )

                    gr.HTML('<div style="height:12px"></div>')

                    # ── Buttons ──
                    with gr.Row():
                        run_btn = gr.Button(
                            "▶  GENERATE VIDEO",
                            variant="primary",
                            elem_classes="btn-generate",
                            scale=4,
                        )
                        cancel_btn = gr.Button(
                            "✕",
                            variant="secondary",
                            elem_classes="btn-cancel",
                            scale=1,
                        )

                # ══ RIGHT COLUMN ═══════════════════════════════════
                with gr.Column(scale=5, min_width=380):

                    # ── Video output ──
                    gr.HTML('<div class="card"><div class="card-title">🎬 Output Video</div>')
                    video_output = gr.Video(label="", height=300)
                    gr.HTML('</div>')

                    # ── Log ──
                    gr.HTML('<div class="card"><div class="card-title">📋 Process Log</div>')
                    gr.HTML('<div id="log-wrap">')
                    log_output = gr.Textbox(
                        label="",
                        lines=20, max_lines=30,
                        interactive=False,
                        placeholder="Logs will stream here in real-time...",
                    )
                    gr.HTML('</div></div>')

            # ── Tips ────────────────────────────────────────────
            gr.HTML("""
            <div class="tips">
              <div class="tips-head">💡 Tips</div>
              <b>Replace</b>: keeps original background &amp; lighting — best for face/body swap<br>
              <b>Animate</b>: character mimics video motion with new background from character image<br>
              Character image: plain background · full body visible · clear face for best results<br>
              FPS: match source video FPS for smooth output · lower FPS = faster generation<br>
              <b>Multi-GPU</b>: selecting multiple GPUs sets <code>CUDA_VISIBLE_DEVICES</code> — model must support multi-GPU (check generate.py flags)<br>
              <b>Audio</b>: original audio is trimmed to match output video length via ffmpeg<br>
              <b>Tối ưu tốc độ</b>: Tắt CPU Offload (nếu VRAM ≥ 80GB) + bật FP8 = combo nhanh nhất không ảnh hưởng chất lượng<br>
              <b>Steps</b>: giảm từ 20 → 10 sẽ nhanh gấp đôi, chất lượng giảm nhẹ — thử trước khi render full<br>
              <b>Face Swap</b>: mode nhanh nhất — không dùng Wan2.2, chỉ cần insightface. Ảnh mặt rõ, nhìn thẳng, không đeo kính tốt nhất
            </div>
            """)

            # ── Event handlers (Tab 1) ───────────────────────────────────
            video_input.change(fn=on_video_upload, inputs=[video_input], outputs=[res_info])
            run_btn.click(
                fn=process_video,
                inputs=[video_input, character_input, mode, resolution, fps, steps, seed,
                        use_flux, relighting, ckpt_dir_input, gpu_selector, merge_audio_flag,
                        cpu_offload, fp8_quant, t5_cpu, sample_shift, guide_scale],
                outputs=[video_output, log_output],
            )
            cancel_btn.click(fn=cancel_process, outputs=[log_output])

              # ══════════════════════════════════════════════════
              # TAB 2: Image-to-Video
              # ══════════════════════════════════════════════════
          with gr.TabItem("🖼️  Image to Video"):
            gr.HTML('''<div class="tips" style="margin-bottom:12px">
              <b>Image-to-Video</b>: Tạo video từ 1 ảnh + prompt mô tả chuyển động.<br>
              Checkpoint: <code>./Wan2.2-I2V-A14B</code> — tải về từ HuggingFace nếu chưa có.
            </div>''')

            with gr.Row(equal_height=False):
              with gr.Column(scale=5):
                gr.HTML('<div class="card"><div class="card-title">🖼️ Input</div>')
                i2v_image = gr.Image(label="Input Image", type="filepath", height=220)
                i2v_img_info = gr.Textbox(
                    label="", value="Upload ảnh để tự động chọn size phù hợp",
                    interactive=False, container=False, elem_classes="video-info",
                )
                i2v_prompt = gr.Textbox(
                    label="Prompt", lines=4,
                    placeholder="Mô tả chuyển động bạn muốn trong video...",
                )
                gr.HTML('</div>')

                gr.HTML('<div class="card"><div class="card-title">⚙️ Settings</div>')
                with gr.Row():
                    i2v_size = gr.Dropdown(
                        choices=["1280x720", "720x1280", "832x480", "480x832",
                                 "1280x704", "704x1280", "1024x704", "704x1024"],
                        value="832x480", label="Resolution",
                    )
                    i2v_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                with gr.Row():
                    i2v_duration = gr.Slider(1.0, 10.0, value=5.0, step=0.5,
                                             label="Duration (seconds)")
                    i2v_fps = gr.Slider(8, 24, value=16, step=1, label="FPS")
                    i2v_seed = gr.Textbox(label="Seed (0=random)", value="0")
                i2v_frame_info = gr.Textbox(
                    label="", value="frame_num: 81  (5.0s × 16fps)",
                    interactive=False, container=False, elem_classes="video-info",
                )
                with gr.Row():
                    i2v_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="CFG Scale")
                    i2v_shift = gr.Slider(0.0, 10.0, value=3.0, step=0.5, label="Sample Shift")
                with gr.Row():
                    i2v_cpu_offload = gr.Checkbox(label="CPU Offload", value=True)
                    i2v_fp8 = gr.Checkbox(label="FP8 Quant (Hopper+)", value=False)
                with gr.Accordion("⚙️ Advanced", open=False):
                    i2v_ckpt = gr.Textbox(label="I2V Checkpoint Dir",
                                          value="./Wan2.2-I2V-A14B")
                    i2v_gpus = gr.CheckboxGroup(
                        choices=[f"GPU {g[0]} — {g[1]} [{g[2]:.1f} GB]"
                                 for g in detect_gpus()],
                        label="GPU Selection",
                        value=[f"GPU {detect_gpus()[0][0]} — {detect_gpus()[0][1]} [{detect_gpus()[0][2]:.1f} GB]"]
                        if detect_gpus() else [],
                        elem_classes="gpu-selector",
                    )
                gr.HTML('</div>')

                with gr.Row():
                    i2v_run_btn = gr.Button("▶  GENERATE I2V", variant="primary",
                                            elem_classes="btn-generate", scale=4)
                    i2v_cancel_btn = gr.Button("✕", variant="secondary",
                                               elem_classes="btn-cancel", scale=1)

              with gr.Column(scale=5):
                gr.HTML('<div class="card"><div class="card-title">🎬 Output</div>')
                i2v_output = gr.Video(label="", height=300)
                gr.HTML('</div>')
                gr.HTML('<div class="card"><div class="card-title">📋 Log</div>')
                gr.HTML('<div id="log-wrap-i2v">')
                i2v_log = gr.Textbox(label="", lines=20, max_lines=30,
                                     interactive=False,
                                     placeholder="I2V logs will appear here...")
                gr.HTML('</div></div>')

            # Auto-select closest valid size when image is uploaded
            i2v_image.change(
                fn=on_i2v_image_upload,
                inputs=[i2v_image],
                outputs=[i2v_size, i2v_img_info],
            )

            # Live frame_num preview
            def update_frame_info(duration, fps):
                fn = max(1, round(duration * fps))
                if (fn - 1) % 4 != 0:
                    fn = ((fn - 1) // 4 + 1) * 4 + 1
                vram = round(fn * 0.15, 1)
                return f"frame_num: {fn}  ({duration:.1f}s × {fps}fps)  •  ~{vram:.1f} GB extra VRAM"

            i2v_duration.change(fn=update_frame_info, inputs=[i2v_duration, i2v_fps], outputs=[i2v_frame_info])
            i2v_fps.change(fn=update_frame_info,      inputs=[i2v_duration, i2v_fps], outputs=[i2v_frame_info])

            i2v_run_btn.click(
                fn=process_i2v,
                inputs=[i2v_image, i2v_prompt, i2v_size, i2v_steps, i2v_seed,
                        i2v_duration, i2v_fps, i2v_ckpt, i2v_gpus, i2v_cpu_offload,
                        i2v_fp8, i2v_cfg, i2v_shift],
                outputs=[i2v_output, i2v_log],
            )
            i2v_cancel_btn.click(fn=cancel_process, outputs=[i2v_log])

          # ══════════════════════════════════════════════════
          # TAB 3: Text-to-Video
          # ══════════════════════════════════════════════════
          with gr.TabItem("📝  Text to Video"):
            gr.HTML('''<div class="tips" style="margin-bottom:12px">
              <b>Text-to-Video</b>: Tạo video hoàn toàn từ prompt văn bản.<br>
              Checkpoint: <code>./Wan2.2-T2V-A14B</code> — tải về từ HuggingFace nếu chưa có.
            </div>''')

            with gr.Row(equal_height=False):
              with gr.Column(scale=5):
                gr.HTML('<div class="card"><div class="card-title">📝 Prompt</div>')
                t2v_prompt = gr.Textbox(
                    label="Prompt", lines=6,
                    placeholder="Mô tả video bạn muốn tạo, càng chi tiết càng tốt...",
                )
                t2v_neg_prompt = gr.Textbox(
                    label="Negative Prompt (optional)", lines=2,
                    placeholder="Những gì bạn không muốn xuất hiện trong video...",
                    value="",
                )
                gr.HTML('</div>')

                gr.HTML('<div class="card"><div class="card-title">⚙️ Settings</div>')
                with gr.Row():
                    t2v_size = gr.Dropdown(
                        choices=["1280x720", "720x1280", "832x480", "480x832",
                                 "1280x704", "704x1280", "1024x704", "704x1024"],
                        value="832x480", label="Resolution",
                    )
                    t2v_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                with gr.Row():
                    t2v_duration = gr.Slider(1.0, 10.0, value=5.0, step=0.5,
                                             label="Duration (seconds)")
                    t2v_seed = gr.Textbox(label="Seed (0=random)", value="0")
                with gr.Row():
                    t2v_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="CFG Scale")
                    t2v_shift = gr.Slider(0.0, 10.0, value=8.0, step=0.5, label="Sample Shift")
                with gr.Row():
                    t2v_cpu_offload = gr.Checkbox(label="CPU Offload", value=True)
                    t2v_fp8 = gr.Checkbox(label="FP8 Quant (Hopper+)", value=False)
                with gr.Accordion("⚙️ Advanced", open=False):
                    t2v_ckpt = gr.Textbox(label="T2V Checkpoint Dir",
                                          value="./Wan2.2-T2V-A14B")
                    t2v_gpus = gr.CheckboxGroup(
                        choices=[f"GPU {g[0]} — {g[1]} [{g[2]:.1f} GB]"
                                 for g in detect_gpus()],
                        label="GPU Selection",
                        value=[f"GPU {detect_gpus()[0][0]} — {detect_gpus()[0][1]} [{detect_gpus()[0][2]:.1f} GB]"]
                        if detect_gpus() else [],
                        elem_classes="gpu-selector",
                    )
                gr.HTML('</div>')

                with gr.Row():
                    t2v_run_btn = gr.Button("▶  GENERATE T2V", variant="primary",
                                            elem_classes="btn-generate", scale=4)
                    t2v_cancel_btn = gr.Button("✕", variant="secondary",
                                               elem_classes="btn-cancel", scale=1)

              with gr.Column(scale=5):
                gr.HTML('<div class="card"><div class="card-title">🎬 Output</div>')
                t2v_output = gr.Video(label="", height=300)
                gr.HTML('</div>')
                gr.HTML('<div class="card"><div class="card-title">📋 Log</div>')
                gr.HTML('<div id="log-wrap-t2v">')
                t2v_log = gr.Textbox(label="", lines=20, max_lines=30,
                                     interactive=False,
                                     placeholder="T2V logs will appear here...")
                gr.HTML('</div></div>')

            t2v_run_btn.click(
                fn=process_t2v,
                inputs=[t2v_prompt, t2v_size, t2v_steps, t2v_seed,
                        t2v_duration, t2v_ckpt, t2v_gpus, t2v_cpu_offload,
                        t2v_fp8, t2v_cfg, t2v_shift],
                outputs=[t2v_output, t2v_log],
            )
            t2v_cancel_btn.click(fn=cancel_process, outputs=[t2v_log])

          # ══════════════════════════════════════════════════
          # TAB 4: First-Last Frame to Video
          # ══════════════════════════════════════════════════
          with gr.TabItem("🎞️  First ↔ Last Frame"):
            gr.HTML('''<div class="tips" style="margin-bottom:12px">
              <b>First-Last Frame to Video</b>: Chọn ảnh frame đầu + frame cuối,
              model tự động tạo video chuyển động liền mạch giữa 2 ảnh.<br>
              Cần 2 checkpoint: <code>Wan2.2-I2V-A14B-High-Noise</code>
              và <code>Wan2.2-I2V-A14B-Low-Noise</code>
            </div>''')

            with gr.Row(equal_height=False):
              with gr.Column(scale=5):
                gr.HTML('<div class="card"><div class="card-title">🖼️ Frame Đầu & Cuối</div>')
                with gr.Row():
                    flf_first = gr.Image(label="🟢 Frame Đầu (Start)", type="filepath", height=200)
                    flf_last  = gr.Image(label="🔴 Frame Cuối (End)",  type="filepath", height=200)
                flf_img_info = gr.Textbox(
                    label="", value="Upload 2 ảnh để tự động chọn size phù hợp",
                    interactive=False, container=False, elem_classes="video-info",
                )
                flf_prompt = gr.Textbox(
                    label="Prompt", lines=3,
                    placeholder="Mô tả chuyển động giữa 2 frame, ví dụ: người đứng dậy và bước đi...",
                )
                gr.HTML('</div>')

                gr.HTML('<div class="card"><div class="card-title">⚙️ Settings</div>')
                with gr.Row():
                    flf_size = gr.Dropdown(
                        choices=["1280x720","720x1280","832x480","480x832",
                                 "1280x704","704x1280","1024x704","704x1024"],
                        value="832x480", label="Resolution",
                    )
                    flf_steps = gr.Slider(1, 50, value=20, step=1, label="Steps")
                with gr.Row():
                    flf_duration = gr.Slider(1.0, 10.0, value=5.0, step=0.5,
                                             label="Duration (seconds)")
                    flf_seed = gr.Textbox(label="Seed (0=random)", value="0")
                with gr.Row():
                    flf_cfg = gr.Slider(1.0, 10.0, value=5.0, step=0.5, label="CFG Scale")
                    flf_cpu_offload = gr.Checkbox(label="CPU Offload", value=True)
                    flf_fp8 = gr.Checkbox(label="FP8 Quant", value=False)
                with gr.Accordion("⚙️ Checkpoint Dirs", open=False):
                    flf_ckpt_high = gr.Textbox(
                        label="High-Noise Model (First Frame)",
                        value="./Wan2.2-I2V-A14B-High-Noise",
                    )
                    flf_ckpt_low = gr.Textbox(
                        label="Low-Noise Model (Last Frame)",
                        value="./Wan2.2-I2V-A14B-Low-Noise",
                    )
                    flf_gpus = gr.CheckboxGroup(
                        choices=[f"GPU {g[0]} — {g[1]} [{g[2]:.1f} GB]" for g in detect_gpus()],
                        label="GPU Selection",
                        value=[f"GPU {detect_gpus()[0][0]} — {detect_gpus()[0][1]} [{detect_gpus()[0][2]:.1f} GB]"]
                        if detect_gpus() else [],
                        elem_classes="gpu-selector",
                    )
                gr.HTML('''<div class="tips" style="margin-top:10px;font-size:11px">
                  📥 Download models:<br>
                  <code>huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-High-Noise --local-dir ./Wan2.2-I2V-A14B-High-Noise</code><br>
                  <code>huggingface-cli download Wan-AI/Wan2.2-I2V-A14B-Low-Noise --local-dir ./Wan2.2-I2V-A14B-Low-Noise</code>
                </div>''')
                gr.HTML('</div>')

                with gr.Row():
                    flf_run_btn = gr.Button("▶  GENERATE FLF2V", variant="primary",
                                            elem_classes="btn-generate", scale=4)
                    flf_cancel_btn = gr.Button("✕", variant="secondary",
                                               elem_classes="btn-cancel", scale=1)

              with gr.Column(scale=5):
                gr.HTML('<div class="card"><div class="card-title">🎬 Output</div>')
                flf_output = gr.Video(label="", height=300)
                gr.HTML('</div>')
                gr.HTML('<div class="card"><div class="card-title">📋 Log</div>')
                gr.HTML('<div id="log-wrap-flf">')
                flf_log = gr.Textbox(label="", lines=20, max_lines=30,
                                     interactive=False,
                                     placeholder="FLF2V logs will appear here...")
                gr.HTML('</div></div>')

            flf_first.change(
                fn=on_i2v_image_upload,
                inputs=[flf_first],
                outputs=[flf_size, flf_img_info],
            )
            flf_run_btn.click(
                fn=process_flf2v,
                inputs=[flf_first, flf_last, flf_prompt, flf_size, flf_steps,
                        flf_seed, flf_duration, flf_ckpt_high, flf_ckpt_low,
                        flf_gpus, flf_cpu_offload, flf_fp8, flf_cfg],
                outputs=[flf_output, flf_log],
            )
            flf_cancel_btn.click(fn=cancel_process, outputs=[flf_log])

          # ── Timer polls ALL tabs ────────────────────────────
        status_timer.tick(
          fn=get_server_status,
          inputs=[],
          outputs=[busy_banner, run_btn],
        )

        # ── Cache control events ──────────────────────────────
        cache_refresh_btn.click(fn=get_cache_status, outputs=[cache_status_box])
        def _clear_and_status(mtype):
            clear_model_cache(mtype)
            return get_cache_status()

        cache_clear_i2v.click(fn=lambda: _clear_and_status("i2v"),   outputs=[cache_status_box])
        cache_clear_t2v.click(fn=lambda: _clear_and_status("t2v"),   outputs=[cache_status_box])
        cache_clear_flf.click(fn=lambda: _clear_and_status("flf2v"), outputs=[cache_status_box])
        cache_clear_all.click(fn=lambda: _clear_and_status("all"),   outputs=[cache_status_box])

        # Auto-refresh cache status after each generation
        def refresh_cache_after(*args):
            return get_cache_status()

    return demo


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Wan2.2 Character Studio")
    parser.add_argument("--host",     default="0.0.0.0")
    parser.add_argument("--port",     type=int, default=7861)
    parser.add_argument("--share",    action="store_true")
    parser.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════╗
║     Wan2.2 Character Studio              ║
║     http://{args.host}:{args.port}       ║
╚══════════════════════════════════════════╝
    """)

    demo = build_ui(ckpt_dir=args.ckpt_dir)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CSS,
    )
