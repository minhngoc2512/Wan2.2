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

current_process = None


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
    """Extract GPU indices from selected checkbox labels, e.g. 'GPU 2 — ...' → '2'."""
    indices = []
    for label in (selected_labels or []):
        try:
            idx = int(label.split("GPU")[1].strip().split()[0].rstrip("—").strip())
            indices.append(str(idx))
        except Exception:
            pass
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


# ─── Helpers ──────────────────────────────────────────────────────────────────

def get_video_info(video_path):
    """Return (width, height, fps) from video file."""
    try:
        cap = cv2.VideoCapture(video_path)
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if w > 0 and h > 0:
            return w, h, round(fps) if fps > 0 else DEFAULT_FPS
    except Exception:
        pass
    return 1280, 720, DEFAULT_FPS


def on_video_upload(video_path):
    """Called when video is uploaded — return info string."""
    if video_path is None:
        return "—"
    w, h, fps = get_video_info(video_path)
    return f"📐 {w}×{h}  •  🎞 {fps} fps"


def resolve_resolution(resolution_str, video_path=None):
    if resolution_str == "Auto (from video)" and video_path:
        w, h, _ = get_video_info(video_path)
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
                       relighting, ckpt_dir):
    cmd = [
        sys.executable, "generate.py",
        "--task",        "animate-14B",
        "--ckpt_dir",    ckpt_dir,
        "--src_root_path", save_path,
        "--refert_num",  "1",
        "--sample_steps", str(steps),
        "--save_file",   output_path,
    ]
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
                  selected_gpus, merge_audio_flag):
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
        "╚" + "═" * 50 + "╝",
        "",
    ]
    yield None, emit(logs)

    step_rc = [0]

    # Shared env with CUDA_VISIBLE_DEVICES set
    proc_env = os.environ.copy()
    proc_env["CUDA_VISIBLE_DEVICES"] = cuda_devices

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

    try:
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
        generate_cmd = build_generate_cmd(
            save_path, output_path, steps, seed,
            mode, relighting, ckpt_dir
        )
        for result in run_step("🎨  STEP 2 / 2  —  Generating Video", generate_cmd):
            yield result

        if step_rc[0] != 0:
            logs += ["", f"❌  Generation FAILED (exit code {step_rc[0]}). See logs above."]
            yield None, emit(logs)
            shutil.rmtree(work_dir, ignore_errors=True)
            return

        # ── Step 3: Audio merge (optional) ──
        final_path = output_path
        if merge_audio_flag and os.path.exists(output_path):
            logs += [f"{'─'*52}", "  🔊  STEP 3 / 3  —  Merging Audio", f"{'─'*52}", ""]
            yield None, emit(logs)
            final_path, logs = merge_audio(video_input, output_path, logs)
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
              <span class="wan-badge">GPU Accelerated</span>
              <span class="wan-badge">Replace / Animate</span>
            </div>
          </div>
        </div>
        """)

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
                    choices=["Replace", "Animate"],
                    value="Replace",
                    label="",
                    info="Replace: swap character, keep original background  |  Animate: character mimics video motion",
                    elem_classes="mode-pills",
                )

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
          <b>Audio</b>: original audio is trimmed to match output video length via ffmpeg
        </div>
        """)

        # ── Event handlers ───────────────────────────────────
        video_input.change(
            fn=on_video_upload,
            inputs=[video_input],
            outputs=[res_info],
        )

        run_btn.click(
            fn=process_video,
            inputs=[
                video_input, character_input,
                mode, resolution, fps, steps, seed,
                use_flux, relighting, ckpt_dir_input,
                gpu_selector, merge_audio_flag,
            ],
            outputs=[video_output, log_output],
        )

        cancel_btn.click(
            fn=cancel_process,
            outputs=[log_output],
        )

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
