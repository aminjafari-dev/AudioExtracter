"""
Video compression engine.

This module contains all ffmpeg interaction for video compression. No UI or
Qt code lives here — only subprocess calls and plain data models.

The core ffmpeg command mirrors what you would run manually:

    ffmpeg -i input.mov -vcodec libx264 -crf 23 -preset medium output.mp4

Usage example:
    compressor = VideoCompressor()

    job = CompressionJob(
        input_path=Path("raw.mov"),
        output_path=Path("raw_compressed.mp4"),
        codec=VideoCodec.H264,
        crf=23,
        preset=CompressionPreset.MEDIUM,
    )

    result = compressor.compress(job)
    if result.success:
        saved = 100 * (1 - result.compressed_size / result.original_size)
        print(f"Saved {saved:.0f}% — output at {result.output_path}")
    else:
        print(f"Error: {result.error_message}")
"""

import shutil
import subprocess
from pathlib import Path

from src.core.models import CompressionJob, CompressionResult


# ---------------------------------------------------------------------------
# VideoCompressor service
# ---------------------------------------------------------------------------

class VideoCompressor:
    """
    Wraps ffmpeg to compress video files using H.264 or H.265 codecs.

    This class is stateless — instantiate once and call compress() for each
    job. All calls are synchronous and blocking; run them on a background
    thread (e.g. QThreadPool / QRunnable) to keep the UI responsive.

    The generated command follows the pattern:
        ffmpeg -y -i <input>
               -vcodec <codec>  (libx264 or libx265)
               -crf    <crf>    (quality level, 0–51)
               -preset <preset> (speed/size tradeoff)
               -acodec aac -b:a 192k   (re-encode audio to widely-supported AAC)
               <output.mp4>

    Example:
        compressor = VideoCompressor()
        result = compressor.compress(job)
    """

    def __init__(self) -> None:
        # Locate ffmpeg early so callers get a clear EnvironmentError on
        # construction rather than a cryptic failure during compression.
        self._ffmpeg_path = self._find_ffmpeg()
        # ffprobe lives alongside ffmpeg; fall back to bare name if not found.
        self._ffprobe_path: str = shutil.which("ffprobe") or "ffprobe"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def ffmpeg_path(self) -> str:
        """
        Return the resolved absolute path to the ffmpeg binary.

        Exposed so background workers can build their own Popen commands
        without going through the compress() wrapper.

        Example:
            cmd = [compressor.ffmpeg_path, "-y", "-i", "in.mov", "out.mp4"]
        """
        return self._ffmpeg_path

    def get_duration(self, input_path: Path) -> float:
        """
        Return the total duration of a video/audio file in seconds.

        Uses ffprobe to read the container-level ``format.duration`` value.
        Returns 0.0 if ffprobe is unavailable or the file cannot be probed,
        so callers should treat 0 as "duration unknown" and fall back to an
        indeterminate progress indicator rather than raising an error.

        Args:
            input_path: Path to the media file to probe.

        Returns:
            Duration in seconds as a float, or 0.0 on failure.

        Example:
            duration = compressor.get_duration(Path("video.mov"))
            if duration > 0:
                pct = current_position_s / duration * 100
        """
        try:
            result = subprocess.run(
                [
                    self._ffprobe_path,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(input_path),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            return float(result.stdout.strip())
        except Exception:  # noqa: BLE001 — ffprobe missing, timeout, bad output, etc.
            return 0.0

    def compress(self, job: CompressionJob) -> CompressionResult:
        """
        Run video compression for a single CompressionJob.

        Blocks until ffmpeg exits. Call from a background thread to avoid
        freezing the UI.

        Args:
            job: The CompressionJob describing input path, output path,
                 codec, CRF quality level, and encoding preset.

        Returns:
            CompressionResult with success=True and size info on success,
            or success=False with an error_message on failure.
        """
        # Create the output directory if it does not exist yet.
        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Record original file size before encoding starts.
        # If stat fails we record 0 and continue — it is non-fatal.
        try:
            original_size = job.input_path.stat().st_size
        except OSError:
            original_size = 0

        # Build the complete ffmpeg command.
        # -y            : overwrite output without prompting
        # -i <in>       : input file
        # -vcodec       : video codec (libx264 / libx265)
        # -crf          : constant rate factor — lower = higher quality + larger file
        # -preset       : encoding speed vs. compression tradeoff
        # -acodec aac   : re-encode audio to AAC for universal compatibility
        # -b:a 192k     : audio bitrate
        # <out>         : output .mp4 path
        command = [
            self._ffmpeg_path,
            "-y",
            "-i", str(job.input_path),
            "-vcodec", job.codec.value,
            "-crf", str(job.crf),
            "-preset", job.preset.value,
            "-acodec", "aac",
            "-b:a", "192k",
            str(job.output_path),
        ]

        try:
            proc = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            # ffmpeg signals failure with a non-zero exit code.
            if proc.returncode != 0:
                error_detail = _parse_ffmpeg_error(proc.stderr)
                return CompressionResult(
                    success=False,
                    error_message=f"ffmpeg error: {error_detail}",
                    original_size=original_size,
                )

            # Measure the resulting file size to report space savings.
            try:
                compressed_size = job.output_path.stat().st_size
            except OSError:
                compressed_size = 0

            return CompressionResult(
                success=True,
                output_path=job.output_path,
                original_size=original_size,
                compressed_size=compressed_size,
            )

        except FileNotFoundError:
            # ffmpeg was found at __init__ time but is now missing — unlikely
            # but handled gracefully just in case.
            return CompressionResult(
                success=False,
                error_message=(
                    "ffmpeg executable not found. "
                    "Please install ffmpeg and ensure it is on your PATH."
                ),
                original_size=original_size,
            )
        except Exception as exc:  # noqa: BLE001
            return CompressionResult(
                success=False,
                error_message=str(exc),
                original_size=original_size,
            )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ffmpeg() -> str:
        """
        Locate the ffmpeg binary on PATH.

        Raises:
            EnvironmentError: if ffmpeg is not installed or not on PATH.
        """
        path = shutil.which("ffmpeg")
        if path is None:
            raise EnvironmentError(
                "ffmpeg not found. Install it via:\n"
                "  macOS:   brew install ffmpeg\n"
                "  Ubuntu:  sudo apt install ffmpeg\n"
                "  Windows: https://ffmpeg.org/download.html"
            )
        return path


# ---------------------------------------------------------------------------
# Module-level helper (mirrors the one in extractor.py)
# ---------------------------------------------------------------------------

def _parse_ffmpeg_error(stderr: str) -> str:
    """
    Extract a short, human-readable error message from ffmpeg's stderr output.

    ffmpeg emits verbose info even on success; we only surface the last line
    that starts with a known error prefix.

    Args:
        stderr: Full stderr string from the ffmpeg subprocess.

    Returns:
        A trimmed error string, or the last non-empty stderr line as fallback.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]

    # Common prefixes that appear on actual error lines.
    error_prefixes = ("Error", "Invalid", "Unable", "No such", "Permission")

    for line in reversed(lines):
        if any(line.startswith(prefix) for prefix in error_prefixes):
            return line

    return lines[-1] if lines else "Unknown error"
