"""
Data models for the AudioExtracter application.

This module defines the core domain objects used across all layers.
No UI or I/O logic lives here — only plain data structures.

Usage example:
    job = ExtractionJob(
        input_path="/videos/movie.mp4",
        output_path="/audio/movie.mp3",
        output_format=AudioFormat.MP3,
    )
    result = ExtractionResult(success=True, output_path=job.output_path)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# Audio output format enum
# ---------------------------------------------------------------------------

class AudioFormat(Enum):
    """
    Supported audio output formats.

    Each value is the file extension used when writing the output file
    and is also passed directly to ffmpeg's format/codec selection.
    """

    MP3 = "mp3"
    AAC = "aac"
    WAV = "wav"
    FLAC = "flac"
    OGG = "ogg"

    @property
    def label(self) -> str:
        """Human-readable label shown in the UI dropdown."""
        labels = {
            AudioFormat.MP3: "MP3 (192 kbps)",
            AudioFormat.AAC: "AAC (High Quality)",
            AudioFormat.WAV: "WAV (Lossless)",
            AudioFormat.FLAC: "FLAC (Lossless Compressed)",
            AudioFormat.OGG: "OGG Vorbis",
        }
        return labels[self]


# ---------------------------------------------------------------------------
# Job status enum
# ---------------------------------------------------------------------------

class JobStatus(Enum):
    """
    Lifecycle states of a single ExtractionJob.

    Transitions:
        PENDING -> RUNNING -> DONE
                           -> FAILED
    """

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Core data models
# ---------------------------------------------------------------------------

@dataclass
class ExtractionJob:
    """
    Represents one audio-extraction task for a single video file.

    Create one instance per file added to the queue. The UI layer holds
    a list of these and updates `status` / `error` as the extractor runs.

    Example:
        job = ExtractionJob(
            input_path=Path("/videos/movie.mp4"),
            output_path=Path("/audio/movie.mp3"),
            output_format=AudioFormat.MP3,
        )
    """

    input_path: Path
    output_path: Path
    output_format: AudioFormat
    status: JobStatus = field(default=JobStatus.PENDING)
    # Human-readable error message populated when status == FAILED
    error: str = field(default="")

    @property
    def filename(self) -> str:
        """Short display name — just the file stem + extension."""
        return self.input_path.name

    def mark_running(self) -> None:
        """Transition job to the RUNNING state."""
        self.status = JobStatus.RUNNING
        self.error = ""

    def mark_done(self) -> None:
        """Transition job to the DONE state."""
        self.status = JobStatus.DONE

    def mark_failed(self, message: str) -> None:
        """
        Transition job to the FAILED state.

        Args:
            message: Human-readable description of what went wrong.
        """
        self.status = JobStatus.FAILED
        self.error = message


@dataclass
class ExtractionResult:
    """
    The outcome of a single ExtractionJob after the extractor has run.

    Returned by AudioExtractor.extract() and consumed by the UI layer
    to update the matching ExtractionJob in the queue.

    Example:
        result = ExtractionResult(success=True, output_path=Path("/audio/movie.mp3"))
        if result.success:
            print(f"Saved to {result.output_path}")
        else:
            print(f"Failed: {result.error_message}")
    """

    success: bool
    output_path: Path | None = None
    error_message: str = ""


# ---------------------------------------------------------------------------
# Audio trim models
# ---------------------------------------------------------------------------

@dataclass
class TrimJob:
    """
    Represents one audio-trim task: cut a region from an audio file.

    The selected region [start_time, end_time] (both in seconds) is extracted
    from input_path and written to output_path using ffmpeg.

    Example:
        job = TrimJob(
            input_path=Path("/audio/song.mp3"),
            output_path=Path("/audio/song_trimmed.mp3"),
            output_format=AudioFormat.MP3,
            start_time=10.5,
            end_time=45.0,
        )
    """

    input_path: Path
    output_path: Path
    output_format: AudioFormat
    # Trim region boundaries in seconds (inclusive)
    start_time: float
    end_time: float


@dataclass
class TrimResult:
    """
    Outcome of a single TrimJob returned by AudioTrimmer.trim().

    Example:
        result = TrimResult(success=True, output_path=Path("/audio/out.mp3"))
        if result.success:
            print(f"Trimmed file saved to {result.output_path}")
        else:
            print(f"Error: {result.error_message}")
    """

    success: bool
    output_path: Path | None = None
    error_message: str = ""


# ---------------------------------------------------------------------------
# Silence removal models
# ---------------------------------------------------------------------------

@dataclass
class SilenceRemovalJob:
    """
    Parameters for a single silence-removal task.

    Passed to SilenceRemover.remove_silence() which detects all silent
    regions using ffmpeg's ``silencedetect`` filter and stitches together
    only the audible segments.

    ``padding`` adds a small buffer around each silence boundary so that
    the very first/last syllable of speech is never clipped.

    Example:
        job = SilenceRemovalJob(
            input_path=Path("/audio/podcast.mp3"),
            output_path=Path("/audio/podcast_clean.mp3"),
            output_format=AudioFormat.MP3,
            threshold_db=-30.0,
            min_silence_duration=0.5,
            padding=0.1,
        )
    """

    input_path: Path
    output_path: Path
    output_format: AudioFormat
    # Audio level below which content is classified as silence (negative dB).
    threshold_db: float = -30.0
    # Minimum gap length in seconds that qualifies as silence.
    min_silence_duration: float = 0.5
    # Seconds of silence-edge buffer preserved around each kept segment.
    padding: float = 0.1


@dataclass
class SilenceRemovalResult:
    """
    Outcome of a SilenceRemovalJob returned by SilenceRemover.remove_silence().

    ``segments_removed`` tells the caller how many distinct silence intervals
    were detected and stripped; 0 means the file had no detectable silence.

    Example:
        result = SilenceRemovalResult(
            success=True,
            output_path=Path("/audio/podcast_clean.mp3"),
            segments_removed=5,
        )
        if result.success:
            print(f"Removed {result.segments_removed} silence(s). "
                  f"Saved to {result.output_path}")
        else:
            print(f"Failed: {result.error_message}")
    """

    success: bool
    output_path: Path | None = None
    error_message: str = ""
    # Number of silence intervals that were detected and removed.
    segments_removed: int = 0


# ---------------------------------------------------------------------------
# Video compression models
# ---------------------------------------------------------------------------

class VideoCodec(Enum):
    """
    Supported video codecs for compression.

    Each value is passed directly to ffmpeg's -vcodec flag.

    Example:
        cmd = ["-vcodec", VideoCodec.H264.value]  # => "-vcodec libx264"
    """

    H264 = "libx264"
    H265 = "libx265"

    @property
    def label(self) -> str:
        """Human-readable label shown in the UI dropdown."""
        labels = {
            VideoCodec.H264: "H.264 — Best Compatibility",
            VideoCodec.H265: "H.265 — Better Compression",
        }
        return labels[self]


class CompressionPreset(Enum):
    """
    ffmpeg encoding speed/compression trade-off presets.

    Slower presets produce smaller files at the same CRF quality level but
    take longer to encode. "medium" is the recommended default.

    Example:
        "-preset", CompressionPreset.MEDIUM.value  # => "-preset medium"
    """

    ULTRAFAST = "ultrafast"
    FAST      = "fast"
    MEDIUM    = "medium"
    SLOW      = "slow"
    VERYSLOW  = "veryslow"

    @property
    def label(self) -> str:
        """Human-readable label shown in the UI dropdown."""
        labels = {
            CompressionPreset.ULTRAFAST: "Ultra Fast (largest file)",
            CompressionPreset.FAST:      "Fast",
            CompressionPreset.MEDIUM:    "Medium (recommended)",
            CompressionPreset.SLOW:      "Slow (better compression)",
            CompressionPreset.VERYSLOW:  "Very Slow (smallest file)",
        }
        return labels[self]


@dataclass
class CompressionJob:
    """
    Parameters for a single video compression task.

    Passed to VideoCompressor.compress() which invokes ffmpeg with
    the chosen codec, CRF quality level, and encoding speed preset.

    CRF (Constant Rate Factor) controls quality vs. size:
        - 0  = lossless (very large)
        - 18 = visually near-lossless
        - 23 = default, good balance (recommended)
        - 28 = noticeably compressed
        - 51 = worst quality (smallest file)

    Example:
        job = CompressionJob(
            input_path=Path("/videos/raw.mov"),
            output_path=Path("/videos/raw_compressed.mp4"),
            codec=VideoCodec.H264,
            crf=23,
            preset=CompressionPreset.MEDIUM,
        )
    """

    input_path: Path
    output_path: Path
    codec: VideoCodec = field(default_factory=lambda: VideoCodec.H264)
    crf: int = field(default=23)
    preset: CompressionPreset = field(default_factory=lambda: CompressionPreset.MEDIUM)

    @property
    def filename(self) -> str:
        """Short display name — just the file name."""
        return self.input_path.name


@dataclass
class CompressionResult:
    """
    Outcome of a single CompressionJob returned by VideoCompressor.compress().

    ``original_size`` and ``compressed_size`` are in bytes; the UI uses them
    to show how much space was saved as a percentage.

    Example:
        result = CompressionResult(
            success=True,
            output_path=Path("/videos/raw_compressed.mp4"),
            original_size=150_000_000,
            compressed_size=42_000_000,
        )
        if result.success:
            saved_pct = 100 * (1 - result.compressed_size / result.original_size)
            print(f"Saved {saved_pct:.0f}%")
        else:
            print(f"Failed: {result.error_message}")
    """

    success: bool
    output_path: Path | None = None
    error_message: str = ""
    # File sizes in bytes; 0 if the file could not be stat'd.
    original_size: int = 0
    compressed_size: int = 0


@dataclass
class AudioFileInfo:
    """
    Metadata about a loaded audio file, populated by AudioLoader.get_info().

    Used by the waveform panel to display duration, format, etc.

    Example:
        info = AudioFileInfo(
            path=Path("song.mp3"),
            duration=210.5,
            sample_rate=44100,
            channels=2,
            fmt="mp3",
        )
    """

    path: Path
    duration: float       # Total duration in seconds
    sample_rate: int      # Native sample rate in Hz
    channels: int         # 1 = mono, 2 = stereo
    fmt: str              # Format name e.g. "mp3", "wav", "flac"

    @property
    def duration_str(self) -> str:
        """Return duration formatted as M:SS.mmm for display."""
        total_ms = int(self.duration * 1000)
        minutes = total_ms // 60000
        seconds = (total_ms % 60000) / 1000
        return f"{minutes}:{seconds:06.3f}"
