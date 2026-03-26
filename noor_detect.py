#!/usr/bin/env python3
"""
noor_detect.py — Inference script for the Noor Music Detector
Built by OpenNoorIlm | Bismillah!

Usage examples
--------------
# Basic detection report:
  python noor_detect.py --audio lecture.mp3 --output ./out

# Use a custom model path:
  python noor_detect.py --audio lecture.mp3 --model /path/to/noor_music_detector.keras

# Detect music in a video:
  python noor_detect.py --video talk.mp4 --output ./out

# Flag (timestamp) every music segment and save JSON + CSV report:
  python noor_detect.py --audio lecture.mp3 --flag --output ./out

# Remove music from audio (replace with silence):
  python noor_detect.py --audio lecture.mp3 --remove --output ./out

# Remove music from video (mute music segments, keep video):
  python noor_detect.py --video talk.mp4 --remove --output ./out

# Full pipeline — flag + remove + keep background audio intact:
  python noor_detect.py --video talk.mp4 --flag --remove --keep-background --output ./out

# Tune confidence threshold:
  python noor_detect.py --audio file.mp3 --threshold 0.80 --output ./out
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except Exception:  # pragma: no cover
    TQDM_AVAILABLE = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from rich.console import Console
    from rich.status import Status
    RICH_AVAILABLE = True
    _console = Console()
except Exception:  # pragma: no cover
    RICH_AVAILABLE = False
    _console = None


class SavedModelPredictor:
    def __init__(self, model_dir, tf):
        self.model_dir = model_dir
        with status("Loading SavedModel..."):
            self._loaded = tf.saved_model.load(model_dir)
        # Pick signature
        sigs = getattr(self._loaded, "signatures", {})
        if "serving_default" in sigs:
            self._fn = sigs["serving_default"]
        elif sigs:
            self._fn = list(sigs.values())[0]
        else:
            raise ValueError("SavedModel has no callable signatures.")

        _, input_spec = self._fn.structured_input_signature
        if not input_spec:
            raise ValueError("SavedModel signature has no named inputs.")
        self._input_key = list(input_spec.keys())[0]

        outputs = self._fn.structured_outputs
        if not outputs:
            raise ValueError("SavedModel signature has no outputs.")
        self._output_key = list(outputs.keys())[0]

        info(
            f"SavedModel signature: input='{self._input_key}' "
            f"output='{self._output_key}'"
        )

    def predict(self, batch, verbose=0):
        import tensorflow as tf
        x = tf.convert_to_tensor(batch)
        out = self._fn(**{self._input_key: x})
        y = out[self._output_key]
        return y.numpy()

# ──────────────────────────────────────────────────────────────────────────────
# Default config (matches training notebook exactly)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_RATE   = 22050
DEFAULT_CHUNK_SECONDS = 2
DEFAULT_N_MELS        = 128
DEFAULT_HOP_LENGTH    = 512
DEFAULT_MIN_VOLUME    = 0.005
DEFAULT_CATEGORIES    = ["music", "not_music", "background"]


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

class NoorArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Provide a helpful hint for Windows CMD where '&' splits commands.
        err(message)
        if os.name == "nt":
            warn(
                "Windows CMD note: if your file path contains '&', wrap it in double quotes "
                "(or escape it as '^&'). Example:\n"
                '  --audio "C:\\path\\FREE R^&B Type Beat 2024.mp3"'
            )
        self.print_help()
        self.exit(2)


def parse_args():
    p = NoorArgumentParser(
        description="Noor Music Detector — detect, flag, and remove music from audio/video.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # ── Input (mutually exclusive) ─────────────────────────────────────────────
    inp = p.add_mutually_exclusive_group(required=True)
    inp.add_argument("--audio", metavar="PATH",
                     help="Input audio file (mp3, wav, flac, ogg, m4a, opus, ...)")
    inp.add_argument("--video", metavar="PATH",
                     help="Input video file (mp4, mkv, avi, mov, ...)")

    # ── Output ─────────────────────────────────────────────────────────────────
    p.add_argument("--output", "-o", metavar="DIR", default="./noor_output",
                   help="Directory to write all output files (default: ./noor_output)")

    # ── Actions ────────────────────────────────────────────────────────────────
    p.add_argument("--flag", action="store_true",
                   help="Flag music segments and save a timestamped report (JSON + CSV)")
    p.add_argument("--remove", action="store_true",
                   help="Remove / mute detected music segments in the output file")

    # ── Model ──────────────────────────────────────────────────────────────────
    p.add_argument(
        "--model", metavar="PATH", default=None,
        help=(
            "Path to the trained model file (.keras or .h5). "
            "If omitted, the script searches in this order: "
            "(1) noor_music_detector.keras next to this script, "
            "(2) noor_music_detector.h5 next to this script, "
            "(3) ./noor_music_model/noor_music_detector.keras, "
            "(4) ./noor_music_model/noor_music_detector.h5"
        ),
    )
    p.add_argument(
        "--saved-model", metavar="DIR", default=None,
        help=(
            "Path to a TensorFlow SavedModel directory "
            "(must contain saved_model.pb or saved_model.pbtxt)."
        ),
    )
    p.add_argument(
        "--auto-find", action="store_true",
        help=(
            "When a directory is given for --model, auto-search inside it for "
            "noor_music_detector.keras/.h5 or a 'saved_model' subfolder."
        ),
    )
    p.add_argument(
        "--metadata", metavar="PATH", default=None,
        help=(
            "Path to metadata.json saved during training. "
            "Auto-detected next to the model file if not given. "
            "Sets sample_rate, n_mels, chunk_seconds, categories, etc."
        ),
    )

    # ── Detection tuning ───────────────────────────────────────────────────────
    p.add_argument("--threshold", type=float, default=0.70,
                   help="Min confidence to label a chunk as 'music' (0-1, default: 0.70)")
    p.add_argument("--chunk-seconds", type=float, default=None,
                   help="Override chunk length in seconds (default: from metadata or 2.0)")
    p.add_argument("--merge-gap", type=float, default=2.0,
                   help="Merge music segments separated by <= this many seconds (default: 2.0)")
    p.add_argument("--min-segment", type=float, default=1.0,
                   help="Drop flagged segments shorter than this many seconds (default: 1.0)")

    # ── Removal options ────────────────────────────────────────────────────────
    p.add_argument("--keep-background", action="store_true",
                   help="When removing music, keep 'background' class audio "
                        "(default: mute both music and background)")
    p.add_argument("--fade", type=float, default=0.3,
                   help="Fade-in/out duration in seconds around removed segments (default: 0.3)")
    p.add_argument("--audio-format", default="wav",
                   choices=["wav", "mp3", "flac", "aac", "ogg"],
                   help="Output audio format when input is audio (default: wav)")

    # ── Report ─────────────────────────────────────────────────────────────────
    p.add_argument("--report-format", default="both",
                   choices=["json", "csv", "both", "none"],
                   help="Format for the segment report (default: both)")

    # ── Misc ───────────────────────────────────────────────────────────────────
    p.add_argument("--batch-size", type=int, default=64,
                   help="Batch size for model inference (default: 64)")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "gpu"],
                   help="TensorFlow device — auto / cpu / gpu (default: auto)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print per-chunk predictions during inference")
    p.add_argument("--no-color", action="store_true",
                   help="Disable ANSI colour output in the terminal")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Terminal colour helpers
# ──────────────────────────────────────────────────────────────────────────────

class C:
    RED    = "\033[91m"
    GREEN  = "\033[92m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

NO_COLOR = False   # overwritten in main()

def col(text, color):
    return text if NO_COLOR else f"{color}{text}{C.RESET}"

def _print(msg):
    if RICH_AVAILABLE and _console is not None:
        _console.print(msg)
    else:
        print(msg)

def info(msg): _print(col("  [INFO] ", C.CYAN)   + msg)
def ok(msg):   _print(col("  [ OK ] ", C.GREEN)  + msg)
def warn(msg): _print(col("  [WARN] ", C.YELLOW) + msg)
def err(msg):  _print(col("  [ERR ] ", C.RED)    + msg)

@contextmanager
def status(msg: str):
    if RICH_AVAILABLE and _console is not None:
        with _console.status(msg, spinner="dots"):
            yield
    else:
        info(msg)
        yield

def fmt_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ──────────────────────────────────────────────────────────────────────────────
# Model & metadata discovery
# ──────────────────────────────────────────────────────────────────────────────

def _find(candidates):
    for c in candidates:
        if c and os.path.exists(str(c)):
            return str(c)
    return None


def _looks_like_saved_model(dir_path: str) -> bool:
    return (
        os.path.exists(os.path.join(dir_path, "saved_model.pb")) or
        os.path.exists(os.path.join(dir_path, "saved_model.pbtxt"))
    )


def _auto_find_in_dir(base_dir: str):
    """Search a directory for model files or a SavedModel subfolder."""
    if not base_dir or not os.path.isdir(base_dir):
        return None
    saved_sub = os.path.join(base_dir, "saved_model")
    if os.path.isdir(saved_sub) and _looks_like_saved_model(saved_sub):
        return saved_sub
    if _looks_like_saved_model(base_dir):
        return base_dir
    candidates = [
        os.path.join(base_dir, "noor_music_detector.keras"),
        os.path.join(base_dir, "noor_music_detector.h5"),
    ]
    found = _find(candidates)
    if found:
        return found
    return None


def load_model_and_meta(args):
    """
    Load the Keras model and training metadata.

    Model search order when --model is NOT supplied:
      1. <script_dir>/noor_music_detector.keras
      2. <script_dir>/noor_music_detector.h5
      3. <cwd>/noor_music_model/noor_music_detector.keras
      4. <cwd>/noor_music_model/noor_music_detector.h5
    """
    info(f"Device mode: {args.device}")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    with status("Importing TensorFlow..."):
        import tensorflow as tf

    if args.device == "cpu":
        info("TensorFlow device forced to CPU.")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            ok(f"GPU detected: {gpus[0].name}")
        else:
            warn("No GPU found — running on CPU.")

    # ── Resolve model path ─────────────────────────────────────────────────────
    script_dir = Path(__file__).parent
    info("Resolving model path...")
    model_path = None

    if args.saved_model:
        info(f"--saved-model provided: {args.saved_model}")
        model_path = args.saved_model
    elif args.model:
        if os.path.isdir(args.model):
            if args.auto_find:
                info("Auto-find enabled: scanning model directory...")
                model_path = _auto_find_in_dir(args.model)
            else:
                model_path = args.model
        else:
            model_path = args.model

    if model_path is None:
        model_path = _find([
            script_dir / "noor_music_detector.keras",                       # next to script
            script_dir / "noor_music_detector.h5",
            Path("noor_music_model") / "noor_music_detector.keras",         # default training output dir
            Path("noor_music_model") / "noor_music_detector.h5",
        ])

    if model_path is None:
        err(
            "Model file not found.\n"
            "  Pass --model /path/to/noor_music_detector.keras (.h5)\n"
            "  or a SavedModel directory, or place the model file next to this script.\n"
            "  Tip: use --auto-find if your model is inside a folder."
        )
        sys.exit(1)

    if args.model and not os.path.exists(args.model):
        err(f"--model path does not exist: {args.model}")
        sys.exit(1)
    if args.saved_model and not os.path.exists(args.saved_model):
        err(f"--saved-model path does not exist: {args.saved_model}")
        sys.exit(1)

    info(f"Loading model from: {model_path}")
    if os.path.isdir(model_path):
        if not _looks_like_saved_model(model_path):
            err(
                "SavedModel file does not exist at:\n"
                f"{model_path}\\{{saved_model.pbtxt|saved_model.pb}}"
            )
            if args.auto_find:
                warn("Auto-find was enabled, but no SavedModel found in this directory.")
            sys.exit(1)
        info("Detected SavedModel directory — using SavedModel inference wrapper.")
        model = SavedModelPredictor(model_path, tf)
        ok("SavedModel loaded successfully.")
    else:
        with status("Loading model into TensorFlow..."):
            try:
                model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
                ok("Model loaded successfully.")
            except Exception as e:
                warn(f"Standard load failed: {e}")
                # Try SavedModel fallback in same folder if available
                parent = str(Path(model_path).parent)
                fallback = _auto_find_in_dir(parent) if args.auto_find else None
                if fallback and os.path.isdir(fallback):
                    warn("Falling back to SavedModel in the same folder.")
                    model = SavedModelPredictor(fallback, tf)
                    ok("SavedModel loaded successfully.")
                else:
                    err(
                        "Model load failed. If you have a SavedModel directory, "
                        "use --saved-model or --auto-find on the model folder."
                    )
                    raise

    # ── Resolve metadata path ──────────────────────────────────────────────────
    info("Resolving metadata path...")
    meta_path = _find([
        args.metadata,
        str(Path(model_path).parent / "metadata.json"),     # next to the model file
        str(script_dir / "metadata.json"),
        "noor_music_model/metadata.json",
    ])

    meta = {}
    if meta_path:
        with status("Loading metadata..."):
            with open(meta_path) as f:
                meta = json.load(f)
        ok(f"Metadata loaded from: {meta_path}")
    else:
        warn("metadata.json not found — using default config values from the training notebook.")

    cfg = {
        "sample_rate":   meta.get("sample_rate",   DEFAULT_SAMPLE_RATE),
        "chunk_seconds": meta.get("chunk_seconds", DEFAULT_CHUNK_SECONDS),
        "n_mels":        meta.get("n_mels",        DEFAULT_N_MELS),
        "hop_length":    meta.get("hop_length",    DEFAULT_HOP_LENGTH),
        "categories":    meta.get("categories",    DEFAULT_CATEGORIES),
    }

    # CLI overrides
    if args.chunk_seconds is not None:
        cfg["chunk_seconds"] = args.chunk_seconds
        info(f"Override: chunk_seconds = {cfg['chunk_seconds']}")

    info(
        f"Config: SR={cfg['sample_rate']} Hz | "
        f"chunk={cfg['chunk_seconds']}s | "
        f"n_mels={cfg['n_mels']} | "
        f"classes={cfg['categories']}"
    )
    return model, cfg


# ──────────────────────────────────────────────────────────────────────────────
# Audio extraction (video -> wav via ffmpeg)
# ──────────────────────────────────────────────────────────────────────────────

def _require_ffmpeg():
    if shutil.which("ffmpeg") is None:
        err("ffmpeg not found. Install it: https://ffmpeg.org/download.html")
        sys.exit(1)


def extract_audio_from_video(video_path: str, tmp_dir: str, sr: int) -> str:
    _require_ffmpeg()
    out = os.path.join(tmp_dir, "extracted_audio.wav")
    info(f"Extracting audio from video: {video_path}")
    info(f"Target sample rate: {sr} Hz | Temp dir: {tmp_dir}")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ar", str(sr),
        "-ac", "1", "-f", "wav", out,
        "-loglevel", "error",
    ]
    with status("Running ffmpeg audio extraction..."):
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err(f"ffmpeg failed:\n{r.stderr}")
        sys.exit(1)
    ok("Audio extracted from video.")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Mel spectrogram helper (matches training notebook exactly)
# ──────────────────────────────────────────────────────────────────────────────

def audio_to_mel(chunk, sr, n_mels, hop_length):
    with status("Importing librosa..."):
        import librosa
    mel    = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_db


# ──────────────────────────────────────────────────────────────────────────────
# Inference — chunk the audio and run the model
# ──────────────────────────────────────────────────────────────────────────────

def run_inference(audio_path: str, model, cfg: dict, args) -> list:
    """
    Returns a list of chunk dicts:
        { start, end, label, confidence, probs }
    """
    import librosa

    sr            = cfg["sample_rate"]
    chunk_seconds = cfg["chunk_seconds"]
    n_mels        = cfg["n_mels"]
    hop_length    = cfg["hop_length"]
    categories    = cfg["categories"]
    chunk_size    = int(sr * chunk_seconds)

    info(f"Loading audio: {audio_path}")
    with status("Loading audio into memory..."):
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    duration  = len(audio) / sr
    ok(f"Audio duration: {fmt_time(duration)}")
    info(f"Samples: {len(audio)} | Sample rate: {sr} Hz")

    chunk_records = []
    mel_slots     = []   # (slot_index, mel_array)
    silent_chunks = 0

    total_chunks = max(0, (len(audio) - chunk_size) // chunk_size)
    info(f"Chunking audio: chunk_size={chunk_size} samples | expected_chunks={total_chunks}")
    for start_sample in tqdm(
        range(0, len(audio) - chunk_size, chunk_size),
        desc="Chunking audio",
        unit="chunk",
    ):
        chunk   = audio[start_sample : start_sample + chunk_size]
        t_start = start_sample / sr
        t_end   = t_start + chunk_seconds

        if np.max(np.abs(chunk)) <= DEFAULT_MIN_VOLUME:
            chunk_records.append(
                dict(start=t_start, end=t_end, label="silent",
                     confidence=1.0, probs={c: 0.0 for c in categories})
            )
            silent_chunks += 1
            continue

        mel = audio_to_mel(chunk, sr, n_mels, hop_length)
        mel_slots.append((len(chunk_records), mel))
        chunk_records.append(
            dict(start=t_start, end=t_end, label=None, confidence=0.0, probs={})
        )

    if args.verbose:
        info(f"Silent chunks: {silent_chunks} | Non-silent chunks: {len(mel_slots)}")

    if not mel_slots:
        warn("No usable audio chunks found (all silent or too short).")
        return chunk_records

    info(f"Running inference on {len(mel_slots)} chunks (batch size {args.batch_size})...")

    indices    = [s[0] for s in mel_slots]
    X          = np.array([s[1] for s in mel_slots])[..., np.newaxis]  # (N, n_mels, T, 1)
    all_preds  = []

    for b in tqdm(
        range(0, len(X), args.batch_size),
        desc="Model inference",
        unit="batch",
    ):
        preds = model.predict(X[b : b + args.batch_size], verbose=0)
        all_preds.extend(preds)

    for slot, pred in zip(indices, all_preds):
        pred_idx   = int(np.argmax(pred))
        label      = categories[pred_idx]
        confidence = float(pred[pred_idx])
        probs      = {cat: float(pred[j]) for j, cat in enumerate(categories)}

        chunk_records[slot].update(label=label, confidence=confidence, probs=probs)

        if args.verbose:
            bar   = col("█" * int(confidence * 20), C.RED if label == "music" else C.GREEN)
            lbl   = col(f"{label:<12}", C.RED if label == "music" else C.GREEN)
            print(f"    {fmt_time(chunk_records[slot]['start'])}  {lbl}  {confidence:.1%}  {bar}")

    ok("Inference complete.")
    return chunk_records


# ──────────────────────────────────────────────────────────────────────────────
# Segment merging & filtering
# ──────────────────────────────────────────────────────────────────────────────

def build_music_segments(
    chunk_records: list,
    threshold: float,
    merge_gap: float,
    min_segment: float,
    keep_background: bool,
    categories: list,
) -> list:
    """
    Convert per-chunk predictions into clean [start, end] music segments.
    A chunk is treated as "music" when:
      - label == 'music'      AND confidence >= threshold
      - label == 'background' AND NOT keep_background AND confidence >= threshold
    """
    info(
        f"Building segments with threshold={threshold}, merge_gap={merge_gap}, "
        f"min_segment={min_segment}, keep_background={keep_background}"
    )
    music_ranges = []
    for r in chunk_records:
        if r["label"] in (None, "silent"):
            continue
        is_music = (r["label"] == "music"      and r["confidence"] >= threshold)
        is_bg    = (r["label"] == "background" and not keep_background
                    and r["confidence"] >= threshold)
        if is_music or is_bg:
            music_ranges.append((r["start"], r["end"]))

    if not music_ranges:
        return []

    # Merge nearby segments
    merged = [list(music_ranges[0])]
    for start, end in music_ranges[1:]:
        if start - merged[-1][1] <= merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # Drop segments that are too short
    merged = [s for s in merged if (s[1] - s[0]) >= min_segment]

    return [{"start": s[0], "end": s[1], "duration": round(s[1] - s[0], 3)} for s in merged]


# ──────────────────────────────────────────────────────────────────────────────
# Report generation
# ──────────────────────────────────────────────────────────────────────────────

def write_reports(
    chunk_records: list,
    music_segments: list,
    out_dir: str,
    stem: str,
    fmt: str,
    total_duration: float,
    categories: list,
):
    info(f"Writing reports to: {out_dir} (format: {fmt})")
    os.makedirs(out_dir, exist_ok=True)
    music_time = sum(s["duration"] for s in music_segments)
    pct        = music_time / total_duration * 100 if total_duration else 0

    report_data = {
        "file":           stem,
        "total_duration": round(total_duration, 3),
        "music_duration": round(music_time, 3),
        "music_percent":  round(pct, 2),
        "segment_count":  len(music_segments),
        "categories":     categories,
        "music_segments": [
            {
                "start":    s["start"],
                "end":      s["end"],
                "duration": s["duration"],
                "start_ts": fmt_time(s["start"]),
                "end_ts":   fmt_time(s["end"]),
            }
            for s in music_segments
        ],
        "chunks": [
            {
                "start":      round(r["start"], 3),
                "end":        round(r["end"], 3),
                "label":      r.get("label", "silent"),
                "confidence": round(r.get("confidence", 0), 4),
            }
            for r in chunk_records
        ],
    }

    if fmt in ("json", "both"):
        json_path = os.path.join(out_dir, f"{stem}_report.json")
        with open(json_path, "w") as f:
            json.dump(report_data, f, indent=2)
        ok(f"JSON report  -> {json_path}")

    if fmt in ("csv", "both"):
        csv_path = os.path.join(out_dir, f"{stem}_segments.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["#", "start_s", "end_s", "duration_s", "start_ts", "end_ts"])
            for i, s in enumerate(music_segments, 1):
                w.writerow([i, s["start"], s["end"], s["duration"],
                             fmt_time(s["start"]), fmt_time(s["end"])])
        ok(f"CSV report   -> {csv_path}")

    return report_data


# ──────────────────────────────────────────────────────────────────────────────
# Music removal
# ──────────────────────────────────────────────────────────────────────────────

def remove_music_from_audio(
    audio_path: str,
    music_segments: list,
    out_dir: str,
    stem: str,
    fmt: str,
    fade: float,
    sr: int,
) -> str:
    """Replace music segments with silence (with optional fade-in/out)."""
    with status("Importing librosa + soundfile..."):
        import librosa
        import soundfile as sf

    info("Loading audio for music removal...")
    with status("Loading audio for removal..."):
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    fade_n    = int(fade * sr)
    info(f"Removal settings: fade={fade}s ({fade_n} samples), format={fmt}")

    for seg in tqdm(music_segments, desc="Removing music", unit="segment"):
        s = int(seg["start"] * sr)
        e = int(seg["end"]   * sr)

        if fade_n > 0 and s >= fade_n:
            audio[s - fade_n : s] *= np.linspace(1.0, 0.0, fade_n)

        audio[s:e] = 0.0

        if fade_n > 0 and e + fade_n <= len(audio):
            audio[e : e + fade_n] *= np.linspace(0.0, 1.0, fade_n)

    out_path = os.path.join(out_dir, f"{stem}_cleaned.{fmt}")
    sf.write(out_path, audio, sr)
    ok(f"Cleaned audio -> {out_path}")
    return out_path


def remove_music_from_video(
    video_path: str,
    audio_path: str,
    out_dir: str,
    stem: str,
) -> str:
    """Mux cleaned audio back into the original video using ffmpeg."""
    _require_ffmpeg()
    out_path = os.path.join(out_dir, f"{stem}_cleaned.mp4")
    info("Muxing cleaned audio back into video...")
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", audio_path,
        "-c:v", "copy",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        out_path,
        "-loglevel", "error",
    ]
    with status("Running ffmpeg mux..."):
        r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        err(f"ffmpeg mux failed:\n{r.stderr}")
        sys.exit(1)
    ok(f"Cleaned video -> {out_path}")
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Summary print
# ──────────────────────────────────────────────────────────────────────────────

def print_summary(report_data: dict, music_segments: list):
    print()
    print(col("=" * 60, C.BOLD))
    print(col("  Noor Music Detector — Results", C.BOLD))
    print(col("=" * 60, C.BOLD))
    print(f"  Total duration : {fmt_time(report_data['total_duration'])}")
    print(f"  Music found    : {fmt_time(report_data['music_duration'])}"
          f"  ({report_data['music_percent']:.1f}%)")
    print(f"  Segments       : {report_data['segment_count']}")

    if music_segments:
        print()
        print(col("  Flagged segments:", C.YELLOW))
        for i, s in enumerate(music_segments, 1):
            print(f"    {i:3}. {fmt_time(s['start'])}  ->  {fmt_time(s['end'])}"
                  f"  ({s['duration']:.1f}s)")
    else:
        print()
        print(col("  No music detected above threshold.", C.GREEN))

    print(col("=" * 60, C.BOLD))
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    global NO_COLOR
    args     = parse_args()
    NO_COLOR = args.no_color

    try:
        print()
        print(col("  +----------------------------------+", C.CYAN))
        print(col("  |   Noor Music Detector v1.0       |", C.CYAN))
        print(col("  |   Built by OpenNoorIlm           |", C.CYAN))
        print(col("  +----------------------------------+", C.CYAN))
        print()

        if not RICH_AVAILABLE:
            warn("rich not installed — falling back to plain text progress.")
        if not TQDM_AVAILABLE:
            warn("tqdm not installed — progress bars disabled.")

        info("Parsing arguments and initializing...")
        info(f"Output directory: {args.output}")
        info(f"Report format: {args.report_format}")
        info(f"Actions: flag={args.flag} remove={args.remove} keep_background={args.keep_background}")
        info(f"Threshold={args.threshold} | merge_gap={args.merge_gap} | min_segment={args.min_segment}")
        if args.verbose:
            info(f"Audio format: {args.audio_format} | Batch size: {args.batch_size}")
        info(f"Auto-find: {args.auto_find} | Saved-model: {args.saved_model}")

        # ── Load model ─────────────────────────────────────────────────────────
        model, cfg = load_model_and_meta(args)
        categories  = cfg["categories"]

        # ── Validate input ─────────────────────────────────────────────────────
        is_video   = args.video is not None
        input_path = args.video if is_video else args.audio
        stem       = Path(input_path).stem

        info(f"Input path: {input_path}")
        if not os.path.exists(input_path):
            err(f"Input file not found: {input_path}")
            sys.exit(1)

        os.makedirs(args.output, exist_ok=True)
        info("Output directory is ready.")

        # ── Extract audio track if input is a video ────────────────────────────
        tmp_dir = None
        if is_video:
            tmp_dir    = tempfile.mkdtemp(prefix="noor_")
            info(f"Created temp directory: {tmp_dir}")
            audio_path = extract_audio_from_video(input_path, tmp_dir, cfg["sample_rate"])
        else:
            audio_path = input_path
            info("Audio input provided directly (no extraction).")

        # ── Run inference ──────────────────────────────────────────────────────
        chunk_records = run_inference(audio_path, model, cfg, args)

        with status("Computing total duration..."):
            import librosa
            total_duration = librosa.get_duration(path=audio_path)
        info(f"Total duration computed: {fmt_time(total_duration)}")

        # ── Build clean music segments ─────────────────────────────────────────
        with status("Building music segments..."):
            music_segments = build_music_segments(
                chunk_records,
                threshold       = args.threshold,
                merge_gap       = args.merge_gap,
                min_segment     = args.min_segment,
                keep_background = args.keep_background,
                categories      = categories,
            )

        # ── Compute report data ────────────────────────────────────────────────
        music_time  = sum(s["duration"] for s in music_segments)
        pct         = music_time / total_duration * 100 if total_duration else 0
        report_data = {
            "file":           stem,
            "total_duration": round(total_duration, 3),
            "music_duration": round(music_time, 3),
            "music_percent":  round(pct, 2),
            "segment_count":  len(music_segments),
            "categories":     categories,
            "music_segments": music_segments,
        }

        # ── Print summary ──────────────────────────────────────────────────────
        with status("Preparing summary..."):
            print_summary(report_data, music_segments)

        # ── Write report (--flag or --report-format != none) ───────────────────
        if args.report_format != "none":
            with status("Writing reports..."):
                write_reports(
                    chunk_records, music_segments,
                    out_dir        = args.output,
                    stem           = stem,
                    fmt            = args.report_format,
                    total_duration = total_duration,
                    categories     = categories,
                )

        # ── Remove / mute music (--remove) ─────────────────────────────────────
        if args.remove:
            if not music_segments:
                warn("No music segments to remove — output will be identical to input.")

            cleaned_audio = remove_music_from_audio(
                audio_path     = audio_path,
                music_segments = music_segments,
                out_dir        = args.output,
                stem           = stem,
                fmt            = args.audio_format,
                fade           = args.fade,
                sr             = cfg["sample_rate"],
            )

            if is_video:
                remove_music_from_video(
                    video_path = input_path,
                    audio_path = cleaned_audio,
                    out_dir    = args.output,
                    stem       = stem,
                )

        # ── Cleanup ────────────────────────────────────────────────────────────
        if tmp_dir and os.path.exists(tmp_dir):
            with status("Cleaning up temporary files..."):
                shutil.rmtree(tmp_dir)

        ok("All done! Alhamdulillah!")
        info(f"Outputs saved to: {os.path.abspath(args.output)}")
        print()
    except Exception as e:
        err(f"Unhandled error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
