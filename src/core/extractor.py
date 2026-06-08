"""
Audio extraction engine.

This module contains all ffmpeg interaction. The rest of the application
never calls ffmpeg directly — it always goes through AudioExtractor.

Architecture note:
    AudioExtractor is a pure domain-layer service. It accepts and returns
    only models from core.models, and has zero knowledge of Qt or any UI.

Usage example:
    extractor = AudioExtractor()
    result = extractor.extract(job)          # synchronous, call from a thread
    if result.success:
        print(f"Audio saved to {result.output_path}")
    else:
        print(f"Error: {result.error_message}")
"""

import subprocess
import shutil
from pathlib import Path

from src.core.models import AudioFormat, ExtractionJob, ExtractionResult


# ---------------------------------------------------------------------------
# ffmpeg codec / format configuration
# ---------------------------------------------------------------------------

# Maps each AudioFormat to the ffmpeg arguments needed to produce it.
# Each entry is a tuple of extra CLI flags appended after the input flag.
#
# Why a dict here instead of a big if/elif?  It keeps the mapping in one
# place and makes adding a new format a one-liner.
_FORMAT_FLAGS: dict[AudioFormat, list[str]] = {
    AudioFormat.MP3:  ["-vn", "-acodec", "libmp3lame", "-ab", "192k"],
    AudioFormat.AAC:  ["-vn", "-acodec", "aac", "-b:a", "192k"],
    AudioFormat.WAV:  ["-vn", "-acodec", "pcm_s16le"],
    AudioFormat.FLAC: ["-vn", "-acodec", "flac"],
    AudioFormat.OGG:  ["-vn", "-acodec", "libvorbis", "-b:a", "192k"],
}


# ---------------------------------------------------------------------------
# AudioExtractor service
# ---------------------------------------------------------------------------

class AudioExtractor:
    """
    Wraps ffmpeg to extract audio tracks from video files.

    This class is stateless — instantiate once and call extract() for each
    job. All calls are synchronous and blocking; run them on a worker thread
    (e.g. QThread or concurrent.futures.ThreadPoolExecutor) to keep the UI
    responsive.

    Example:
        extractor = AudioExtractor()

        job = ExtractionJob(
            input_path=Path("video.mp4"),
            output_path=Path("audio.mp3"),
            output_format=AudioFormat.MP3,
        )

        result = extractor.extract(job)
    """

    def __init__(self) -> None:
        # Verify ffmpeg is available at construction time so callers get a
        # clear error early rather than a cryptic failure during extraction.
        self._ffmpeg_path = self._find_ffmpeg()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, job: ExtractionJob) -> ExtractionResult:
        """
        Run audio extraction for a single ExtractionJob.

        This method blocks until ffmpeg exits. Call it from a background
        thread to avoid freezing the UI.

        Args:
            job: The ExtractionJob describing input, output, and format.

        Returns:
            ExtractionResult with success=True and output_path set on success,
            or success=False and error_message set on failure.
        """
        # Ensure the output directory exists before ffmpeg tries to write.
        job.output_path.parent.mkdir(parents=True, exist_ok=True)

        codec_flags = _FORMAT_FLAGS.get(job.output_format, [])

        # Build the full ffmpeg command.
        # -y       : overwrite existing output without prompting
        # -i <in>  : input file
        # <flags>  : format-specific audio codec flags
        # <out>    : output file path
        command = [
            self._ffmpeg_path,
            "-y",
            "-i", str(job.input_path),
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

            # ffmpeg returns non-zero on failure.
            if result.returncode != 0:
                # Extract a concise error from stderr (last non-empty line).
                error_detail = _parse_ffmpeg_error(result.stderr)
                return ExtractionResult(
                    success=False,
                    error_message=f"ffmpeg error: {error_detail}",
                )

            return ExtractionResult(success=True, output_path=job.output_path)

        except FileNotFoundError:
            # This happens if _ffmpeg_path was found at init but then removed,
            # or on Windows when PATH changes mid-session.
            return ExtractionResult(
                success=False,
                error_message=(
                    "ffmpeg executable not found. "
                    "Please install ffmpeg and ensure it is on your PATH."
                ),
            )
        except Exception as exc:  # noqa: BLE001 — catch-all for unexpected errors
            return ExtractionResult(success=False, error_message=str(exc))

    @staticmethod
    def is_available() -> bool:
        """
        Returns True if ffmpeg is installed and reachable on PATH.

        Useful for showing a warning dialog before the user tries to extract.
        """
        return shutil.which("ffmpeg") is not None

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_ffmpeg() -> str:
        """
        Locate the ffmpeg binary.

        Raises:
            EnvironmentError: if ffmpeg cannot be found on PATH.
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
    Extract a short, human-readable error message from ffmpeg's stderr output.

    ffmpeg dumps a lot of info on stderr even on success; we only want the
    last meaningful error line that starts with a known prefix.

    Args:
        stderr: The full stderr string captured from the ffmpeg process.

    Returns:
        A trimmed error string, or the last non-empty line as a fallback.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]

    # ffmpeg error lines often start with these prefixes.
    error_prefixes = ("Error", "Invalid", "Unable", "No such", "Permission")

    for line in reversed(lines):
        if any(line.startswith(prefix) for prefix in error_prefixes):
            return line

    # Fallback: return the last non-empty line.
    return lines[-1] if lines else "Unknown error"
