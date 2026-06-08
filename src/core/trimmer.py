"""
Audio trimming engine.

This module wraps ffmpeg to cut a region from an audio file. It is the
domain-layer counterpart to AudioExtractor — both are pure services with
no knowledge of Qt or UI.

Architecture note:
    AudioTrimmer is stateless. Instantiate once, call trim() for each job.
    Always run from a background thread (QRunnable / ThreadPoolExecutor).

Usage example:
    trimmer = AudioTrimmer()

    job = TrimJob(
        input_path=Path("music.mp3"),
        output_path=Path("clip.mp3"),
        output_format=AudioFormat.MP3,
        start_time=30.0,
        end_time=90.0,
    )

    result = trimmer.trim(job)
    if result.success:
        print(f"Saved to {result.output_path}")
    else:
        print(f"Error: {result.error_message}")
"""

import shutil
import subprocess
from pathlib import Path

from src.core.models import AudioFormat, TrimJob, TrimResult


# ---------------------------------------------------------------------------
# ffmpeg codec flags (reused from extractor pattern)
# ---------------------------------------------------------------------------

# For trimming we re-encode to ensure the output is a clean, valid file
# at the requested format. Stream copy (-c copy) would be faster but can
# leave keyframe misalignment artefacts at the cut points.
_FORMAT_FLAGS: dict[AudioFormat, list[str]] = {
    AudioFormat.MP3:  ["-acodec", "libmp3lame", "-ab", "192k"],
    AudioFormat.AAC:  ["-acodec", "aac", "-b:a", "192k"],
    AudioFormat.WAV:  ["-acodec", "pcm_s16le"],
    AudioFormat.FLAC: ["-acodec", "flac"],
    AudioFormat.OGG:  ["-acodec", "libvorbis", "-b:a", "192k"],
}


# ---------------------------------------------------------------------------
# AudioTrimmer service
# ---------------------------------------------------------------------------

class AudioTrimmer:
    """
    Wraps ffmpeg to trim (cut) a time region from an audio file.

    The trimmer uses ffmpeg's -ss (seek) and -to (end time) flags to
    perform a sample-accurate cut. Output is always re-encoded so the
    result is a standalone, playable file with no keyframe artefacts.

    Example:
        trimmer = AudioTrimmer()
        result = trimmer.trim(job)
    """

    def __init__(self) -> None:
        self._ffmpeg_path = self._find_ffmpeg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trim(self, job: TrimJob) -> TrimResult:
        """
        Execute the trim and return a TrimResult.

        The selected region [job.start_time, job.end_time] is re-encoded
        to the requested output format. Blocking — call from a thread.

        Args:
            job: TrimJob describing input file, output file, time region.

        Returns:
            TrimResult with success=True on success, or error_message set.
        """
        # Ensure the output directory exists.
        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        codec_flags = _FORMAT_FLAGS.get(job.output_format, [])

        # -ss before -i uses input seeking (faster and accurate for re-encode).
        # -to is relative to the *original* file when -ss precedes -i.
        command = [
            self._ffmpeg_path,
            "-y",
            "-ss", f"{job.start_time:.6f}",
            "-to", f"{job.end_time:.6f}",
            "-i", str(job.input_path),
            "-vn",               # drop video stream if present
            *codec_flags,
            str(job.output_path),
        ]

        try:
            result = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if result.returncode != 0:
                detail = _parse_ffmpeg_error(result.stderr)
                return TrimResult(success=False, error_message=f"ffmpeg error: {detail}")

            return TrimResult(success=True, output_path=job.output_path)

        except FileNotFoundError:
            return TrimResult(
                success=False,
                error_message="ffmpeg not found. Install it and ensure it is on your PATH.",
            )
        except Exception as exc:  # noqa: BLE001
            return TrimResult(success=False, error_message=str(exc))

    @staticmethod
    def is_available() -> bool:
        """Return True if ffmpeg is installed and reachable on PATH."""
        return shutil.which("ffmpeg") is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ffmpeg() -> str:
        """
        Locate ffmpeg on PATH or raise EnvironmentError.

        Raises:
            EnvironmentError: if ffmpeg cannot be found.
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


def _parse_ffmpeg_error(stderr: str) -> str:
    """
    Extract a short error message from ffmpeg's verbose stderr output.

    ffmpeg prepends lots of version/library info; we want the actual
    error line. We scan backwards for a line starting with a known
    error prefix, falling back to the last non-empty line.

    Args:
        stderr: Full stderr string from subprocess.

    Returns:
        A trimmed, human-readable error string.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    error_prefixes = ("Error", "Invalid", "Unable", "No such", "Permission", "Cannot")

    for line in reversed(lines):
        if any(line.startswith(p) for p in error_prefixes):
            return line

    return lines[-1] if lines else "Unknown error"
