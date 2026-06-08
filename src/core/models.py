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
