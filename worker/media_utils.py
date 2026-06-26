import os
import signal
import subprocess

RECORDING_CRF = int(os.getenv("RECORDING_CRF", "28"))
RECORDING_MAX_WIDTH = int(os.getenv("RECORDING_MAX_WIDTH", "1280"))
RECORDING_ENCODE_PRESET = os.getenv("RECORDING_ENCODE_PRESET", "fast")


def stop_ffmpeg_gracefully(proc: subprocess.Popen, timeout: int = 20) -> int:
    if proc.poll() is not None:
        return proc.returncode or 0

    proc.send_signal(signal.SIGINT)
    try:
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.send_signal(signal.SIGTERM)
        try:
            return proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()


def is_valid_media(path: str) -> bool:
    if not path or not os.path.isfile(path) or os.path.getsize(path) == 0:
        return False
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            path,
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        return False
    duration = result.stdout.strip()
    try:
        return bool(duration) and float(duration) > 0
    except ValueError:
        return False


def transcode_recording(raw_path: str, output_path: str) -> bool:
    scale = f"scale='min({RECORDING_MAX_WIDTH},iw)':-2"
    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "warning",
        "-i",
        raw_path,
        "-an",
        "-vf",
        scale,
        "-c:v",
        "libx264",
        "-preset",
        RECORDING_ENCODE_PRESET,
        "-crf",
        str(RECORDING_CRF),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    if result.returncode != 0:
        print(f"Recording transcode failed: {result.stderr.strip()}")
        return False
    return is_valid_media(output_path)


def remux_with_faststart(input_path: str, output_path: str) -> bool:
    cmd = [
        "ffmpeg",
        "-y",
        "-nostdin",
        "-loglevel",
        "warning",
        "-i",
        input_path,
        "-an",
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, check=False)
    if result.returncode != 0:
        print(f"Recording remux failed: {result.stderr.strip()}")
        return False
    return is_valid_media(output_path)
