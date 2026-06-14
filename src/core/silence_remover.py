"""
Silence detection and removal engine.

This module wraps ffmpeg to detect silent regions in an audio file and
produce a new file with those regions stripped out, leaving only the
audible content stitched seamlessly together.

Architecture note:
    SilenceRemover is stateless — instantiate once, call remove_silence()
    for each job.  All operations are blocking; always run from a
    background thread (QRunnable / ThreadPoolExecutor).

How silence removal works (two-step approach):
    1. Run ffmpeg's ``silencedetect`` filter to find every interval
       where audio is quieter than ``threshold_db`` for at least
       ``min_silence_duration`` seconds.  ffmpeg writes these timestamps
       to stderr; we parse them with a regex.
    2. Invert the silence list into a "keep" list: the audible segments
       between (and slightly overlapping, via ``padding``) the silences.
    3. If there is only one keep segment, a simple ``-ss``/``-to`` trim
       is used.  For multiple segments the ``atrim`` + ``asetpts`` +
       ``concat`` filter graph stitches them gaplessly.

Usage example:
    remover = SilenceRemover()

    job = SilenceRemovalJob(
        input_path=Path("podcast.mp3"),
        output_path=Path("podcast_clean.mp3"),
        output_format=AudioFormat.MP3,
        threshold_db=-30.0,
        min_silence_duration=0.5,
        padding=0.1,
    )

    result = remover.remove_silence(job)
    if result.success:
        print(f"Saved to {result.output_path}. "
              f"Removed {result.segments_removed} silence segment(s).")
    else:
        print(f"Error: {result.error_message}")
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

from src.core.models import AudioFormat, SilenceRemovalJob, SilenceRemovalResult


# ---------------------------------------------------------------------------
# ffmpeg codec flags (mirrors the mapping in trimmer.py)
# ---------------------------------------------------------------------------

# Maps each AudioFormat to the ffmpeg codec arguments needed to encode it.
# Used both for simple trims and for the complex-filter concatenation command.
_FORMAT_FLAGS: dict[AudioFormat, list[str]] = {
    AudioFormat.MP3:  ["-acodec", "libmp3lame", "-ab", "192k"],
    AudioFormat.AAC:  ["-acodec", "aac", "-b:a", "192k"],
    AudioFormat.WAV:  ["-acodec", "pcm_s16le"],
    AudioFormat.FLAC: ["-acodec", "flac"],
    AudioFormat.OGG:  ["-acodec", "libvorbis", "-b:a", "192k"],
}


# ---------------------------------------------------------------------------
# SilenceRemover service
# ---------------------------------------------------------------------------

class SilenceRemover:
    """
    Detects and removes silent regions from an audio file using ffmpeg.

    The class is intentionally stateless so a single instance can safely
    process multiple files sequentially (or you can create one per job).

    Internally the two-step pipeline is:
        detect_silence() → remove_silence()

    ``remove_silence()`` calls ``detect_silence()`` automatically, but
    you can also call ``detect_silence()`` separately if you want to
    inspect which segments would be removed before committing.

    Example:
        remover = SilenceRemover()
        intervals = remover.detect_silence(Path("talk.mp3"))
        print(f"Found {len(intervals)} silence segment(s).")
        result = remover.remove_silence(job)
    """

    def __init__(self) -> None:
        # Resolve the ffmpeg binary once at construction time.
        self._ffmpeg_path: str = _find_ffmpeg()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_silence(
        self,
        path: Path,
        threshold_db: float = -30.0,
        min_duration: float = 0.5,
    ) -> list[tuple[float, float]]:
        """
        Run the ffmpeg ``silencedetect`` filter and return silence intervals.

        The filter writes lines like the following to stderr:
            [silencedetect] silence_start: 1.234
            [silencedetect] silence_end: 3.456 | silence_duration: 2.222

        Both values are parsed; unpaired starts (file ends in silence) are
        included with ``duration`` as the implicit end so the tail is
        still removed.

        Args:
            path:          Audio file to scan.
            threshold_db:  dB level below which audio counts as silence
                           (e.g. -30.0 means anything quieter than -30 dB).
            min_duration:  Minimum gap length in seconds to be classified
                           as silence. Short pauses below this are kept.

        Returns:
            Sorted list of ``(start_seconds, end_seconds)`` tuples, one per
            detected silence interval.  Empty list means no silence found.
        """
        noise_arg  = f"{threshold_db:.1f}dB"
        filter_arg = f"silencedetect=noise={noise_arg}:d={min_duration:.3f}"

        cmd = [
            self._ffmpeg_path,
            "-i",  str(path),
            "-af", filter_arg,
            "-f",  "null",
            "-",                  # output to /dev/null
        ]

        # ffmpeg always writes its own output (including filter logs) to stderr.
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        return _parse_silence_intervals(proc.stderr)

    def remove_silence(self, job: SilenceRemovalJob) -> SilenceRemovalResult:
        """
        Detect silence in ``job.input_path`` and write a cleaned file to
        ``job.output_path`` with all silent regions removed.

        Steps performed internally:
            1. detect_silence() to find all silence intervals.
            2. _get_duration() to know the total file length.
            3. _compute_keep_intervals() to invert silence → audible keep list.
            4. Build ffmpeg command (simple trim or complex concat filter).
            5. Run the command and return the result.

        If no silence is detected, the file is re-encoded as-is (no cuts).

        Args:
            job: SilenceRemovalJob with all processing parameters.

        Returns:
            SilenceRemovalResult with ``success=True`` and ``output_path``
            set on success, or ``error_message`` set on failure.
        """
        # Step 1: scan for silence intervals.
        try:
            silence_intervals = self.detect_silence(
                job.input_path,
                threshold_db=job.threshold_db,
                min_duration=job.min_silence_duration,
            )
        except Exception as exc:  # noqa: BLE001
            return SilenceRemovalResult(success=False, error_message=str(exc))

        # Step 2: probe total duration so we can close the last keep segment.
        try:
            duration = self._get_duration(job.input_path)
        except Exception as exc:  # noqa: BLE001
            return SilenceRemovalResult(success=False, error_message=str(exc))

        # Step 3: convert silence list → keep segments (audible regions).
        keep = _compute_keep_intervals(silence_intervals, duration, job.padding)

        # Guard: if nothing survives (e.g. entire file is silence), keep it all.
        if not keep:
            keep = [(0.0, duration)]

        job.output_path.parent.mkdir(parents=True, exist_ok=True)
        codec_flags = _FORMAT_FLAGS.get(job.output_format, [])

        # Step 4 & 5: build the ffmpeg command and run it.
        try:
            if len(keep) == 1:
                # Simple case — one contiguous segment, use seek flags.
                s, e = keep[0]
                cmd = [
                    self._ffmpeg_path,
                    "-y",
                    "-ss", f"{s:.6f}",
                    "-to", f"{e:.6f}",
                    "-i",  str(job.input_path),
                    "-vn",
                    *codec_flags,
                    str(job.output_path),
                ]
            else:
                # Multiple segments — build a filter_complex that trims and
                # concatenates each keep segment into one seamless stream.
                # Each segment: [0:a]atrim=start=S:end=E,asetpts=N/SR/TB[aI]
                # Final node: [a0][a1]...[aN-1]concat=n=N:v=0:a=1[out]
                filter_parts: list[str] = []
                labels: list[str] = []

                for i, (s, e) in enumerate(keep):
                    lbl = f"a{i}"
                    # atrim extracts the time slice; asetpts resets the
                    # presentation timestamps so the concat node sees
                    # a gapless sequence starting from 0.
                    filter_parts.append(
                        f"[0:a]atrim=start={s:.6f}:end={e:.6f},"
                        f"asetpts=N/SR/TB[{lbl}]"
                    )
                    labels.append(f"[{lbl}]")

                n = len(keep)
                filter_parts.append(
                    f"{''.join(labels)}concat=n={n}:v=0:a=1[out]"
                )
                filter_complex = ";".join(filter_parts)

                cmd = [
                    self._ffmpeg_path,
                    "-y",
                    "-i",              str(job.input_path),
                    "-filter_complex", filter_complex,
                    "-map",            "[out]",
                    *codec_flags,
                    str(job.output_path),
                ]

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

            if proc.returncode != 0:
                detail = _parse_ffmpeg_error(proc.stderr)
                return SilenceRemovalResult(
                    success=False,
                    error_message=f"ffmpeg error: {detail}",
                )

            return SilenceRemovalResult(
                success=True,
                output_path=job.output_path,
                segments_removed=len(silence_intervals),
            )

        except FileNotFoundError:
            return SilenceRemovalResult(
                success=False,
                error_message=(
                    "ffmpeg not found. Install it and ensure it is on your PATH."
                ),
            )
        except Exception as exc:  # noqa: BLE001
            return SilenceRemovalResult(success=False, error_message=str(exc))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_duration(self, path: Path) -> float:
        """
        Query the total duration of an audio file in seconds via ffprobe.

        Falls back to parsing ffmpeg's own stderr if ffprobe is not separately
        available (they typically ship together).

        Args:
            path: Audio file path.

        Returns:
            Duration in seconds as a float.

        Raises:
            RuntimeError: if duration cannot be determined.
        """
        ffprobe = shutil.which("ffprobe")
        if ffprobe is None:
            # Derive ffprobe path from the resolved ffmpeg path as a fallback.
            ffprobe = self._ffmpeg_path.replace("ffmpeg", "ffprobe")

        cmd = [
            ffprobe,
            "-v",            "error",
            "-show_entries", "format=duration",
            "-of",           "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        if proc.returncode == 0 and proc.stdout.strip():
            try:
                return float(proc.stdout.strip())
            except ValueError:
                pass

        raise RuntimeError(
            f"Could not determine duration of '{path.name}'. "
            "Ensure ffprobe is installed alongside ffmpeg."
        )

    # ------------------------------------------------------------------
    # Class-level utilities
    # ------------------------------------------------------------------

    @staticmethod
    def is_available() -> bool:
        """Return True if ffmpeg is installed and reachable on PATH."""
        return shutil.which("ffmpeg") is not None


# ---------------------------------------------------------------------------
# Module-level helpers (private)
# ---------------------------------------------------------------------------

def _find_ffmpeg() -> str:
    """
    Locate the ffmpeg binary on the system PATH.

    Raises:
        EnvironmentError: if ffmpeg is not found, with install instructions.
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


def _parse_silence_intervals(stderr: str) -> list[tuple[float, float]]:
    """
    Extract silence intervals from ffmpeg's silencedetect stderr output.

    ffmpeg writes filter results to stderr in the form:
        [silencedetect @ 0x...] silence_start: 1.234
        [silencedetect @ 0x...] silence_end: 3.456 | silence_duration: 2.222

    We collect all starts and all ends and zip them.  If the file ends
    mid-silence (no trailing silence_end), the unpaired start is dropped
    because the concat step will naturally truncate to actual EOF.

    Args:
        stderr: Raw stderr string from the ffmpeg silencedetect run.

    Returns:
        List of ``(start, end)`` float pairs, sorted by start time.
    """
    starts = [
        float(m)
        for m in re.findall(r"silence_start:\s*([\d.]+)", stderr)
    ]
    ends = [
        float(m)
        for m in re.findall(r"silence_end:\s*([\d.]+)", stderr)
    ]

    # Pair up in order; silences that trail past EOF have no end → skip.
    intervals = [
        (start, ends[i])
        for i, start in enumerate(starts)
        if i < len(ends)
    ]

    return sorted(intervals, key=lambda t: t[0])


def _compute_keep_intervals(
    silence: list[tuple[float, float]],
    duration: float,
    padding: float,
) -> list[tuple[float, float]]:
    """
    Convert silence intervals into audible "keep" intervals.

    Each silence interval is *shrunk* by ``padding`` seconds on both sides
    before removal.  This preserves a brief buffer at the boundary of each
    silent segment so the first/last syllable of speech is not clipped.

    Algorithm:
        1. For each (s_start, s_end) silence, compute the effective cut:
               cut = (s_start + padding, s_end - padding)
           If cut_end <= cut_start, the silence is too short to cut after
           padding — skip it.
        2. The keep intervals are the gaps between these effective cuts
           (plus the portions before the first cut and after the last cut).
        3. Segments shorter than 10 ms are discarded.

    Example:
        silence  = [(2.0, 5.0), (8.0, 10.0)], duration=12.0, padding=0.1
        cuts     = [(2.1, 4.9), (8.1, 9.9)]
        keep     = [(0.0, 2.1), (4.9, 8.1), (9.9, 12.0)]

    Args:
        silence: Sorted list of (start, end) silence intervals.
        duration: Total file duration in seconds.
        padding:  Seconds to preserve at each edge of a silence interval.

    Returns:
        List of (start, end) keep intervals with positive duration.
    """
    if not silence:
        # No silence detected — keep the entire file unchanged.
        return [(0.0, duration)]

    # Build the list of effective cuts (silence intervals after padding shrink).
    cuts: list[tuple[float, float]] = []
    for (s_start, s_end) in silence:
        cut_start = s_start + padding
        cut_end   = s_end   - padding
        # Only cut if there is still meaningful silence after shrinking.
        if cut_end > cut_start + 0.01:
            cuts.append((cut_start, cut_end))

    if not cuts:
        # All silence intervals were shorter than 2 × padding → nothing to cut.
        return [(0.0, duration)]

    # Compute keep intervals as the complement of the cuts within [0, duration].
    keeps: list[tuple[float, float]] = []
    prev_end = 0.0

    for (cut_start, cut_end) in sorted(cuts):
        # The segment from prev_end up to this cut's start is audible.
        if cut_start > prev_end + 0.01:
            keeps.append((prev_end, cut_start))
        prev_end = cut_end

    # Append the final segment after the last cut.
    if prev_end < duration - 0.01:
        keeps.append((prev_end, duration))

    # Drop any keep segment shorter than 10 ms (avoids zero-length atrim nodes).
    return [(s, e) for (s, e) in keeps if e - s >= 0.01]


def _parse_ffmpeg_error(stderr: str) -> str:
    """
    Extract the most relevant error line from ffmpeg's verbose stderr.

    Scans backwards through the output looking for a line that starts with
    a known error prefix.  Falls back to the last non-empty line.

    Args:
        stderr: Full stderr string from a subprocess run.

    Returns:
        A short, trimmed error string suitable for display to the user.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    error_prefixes = ("Error", "Invalid", "Unable", "No such", "Permission", "Cannot")

    for line in reversed(lines):
        if any(line.startswith(p) for p in error_prefixes):
            return line

    return lines[-1] if lines else "Unknown error"
