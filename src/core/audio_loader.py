"""
Audio file loader for waveform visualization.

This module uses ffmpeg to decode any audio format into raw PCM samples
so the UI can render a waveform without needing format-specific libraries.
All audio processing (decode → downsample → peak-fold) is done here so
the waveform widget only receives a ready-to-draw numpy array.

Architecture note:
    AudioLoader is a pure domain-layer service. It knows nothing about Qt
    or any UI. Results are plain Python objects (numpy arrays + dataclasses).

Usage example:
    loader = AudioLoader()

    # Get file metadata (fast — uses ffprobe only)
    info = loader.get_info(Path("song.mp3"))
    print(f"Duration: {info.duration:.1f}s  SR: {info.sample_rate} Hz")

    # Load downsampled waveform for display
    peaks = loader.load_peaks(Path("song.mp3"), num_bins=1200)
    # peaks.shape == (1200,), values in [0, 1]
"""

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

from src.core.models import AudioFileInfo


# ---------------------------------------------------------------------------
# Supported audio extensions
# ---------------------------------------------------------------------------

# All formats that ffmpeg can decode (used for file filtering in the UI).
SUPPORTED_AUDIO_EXTENSIONS: frozenset[str] = frozenset(
    {".mp3", ".aac", ".m4a", ".wav", ".flac", ".ogg", ".opus", ".aiff", ".aif", ".wma"}
)

AUDIO_FILE_FILTER = (
    "Audio Files (*.mp3 *.aac *.m4a *.wav *.flac *.ogg *.opus *.aiff *.aif *.wma);;"
    "All Files (*)"
)


def is_supported_audio(path: Path) -> bool:
    """Return True if the file extension is a supported audio format."""
    return path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS


# ---------------------------------------------------------------------------
# AudioLoader service
# ---------------------------------------------------------------------------

class AudioLoader:
    """
    Loads audio files via ffmpeg for waveform visualization.

    Two-step API:
      1. get_info()  — fast metadata probe (no decoding)
      2. load_peaks() — decode + downsample to N peak bins

    Both methods are synchronous and blocking. Call them from a QRunnable /
    background thread to keep the UI responsive.

    Example:
        loader = AudioLoader()
        info = loader.get_info(path)
        peaks = loader.load_peaks(path, num_bins=1000)
    """

    def __init__(self) -> None:
        # Verify ffmpeg/ffprobe are on PATH at construction time.
        self._ffmpeg  = self._require_binary("ffmpeg")
        self._ffprobe = self._require_binary("ffprobe")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_info(self, path: Path) -> AudioFileInfo:
        """
        Probe an audio file and return its metadata using ffprobe.

        This is very fast (no decoding) and is suitable to call before
        committing to a full load_peaks() call.

        Args:
            path: Path to the audio file.

        Returns:
            AudioFileInfo with duration, sample_rate, channels, and format.

        Raises:
            RuntimeError: if ffprobe fails or the file is not a valid audio file.
        """
        cmd = [
            self._ffprobe,
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffprobe failed for {path.name}: {result.stderr.strip()}"
            )

        data = json.loads(result.stdout)

        # Find the first audio stream.
        audio_stream = next(
            (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
            None,
        )
        if audio_stream is None:
            raise RuntimeError(f"No audio stream found in {path.name}")

        # Duration — prefer stream duration, fall back to container duration.
        duration_str = audio_stream.get("duration") or data.get("format", {}).get("duration", "0")
        try:
            duration = float(duration_str)
        except (ValueError, TypeError):
            duration = 0.0

        sample_rate = int(audio_stream.get("sample_rate", 44100))
        channels    = int(audio_stream.get("channels", 2))
        fmt         = data.get("format", {}).get("format_name", path.suffix.lstrip("."))

        return AudioFileInfo(
            path=path,
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            fmt=fmt,
        )

    def load_peaks(self, path: Path, num_bins: int = 1200) -> np.ndarray:
        """
        Decode an audio file and return a peak-amplitude array for display.

        The entire file is decoded at a reduced sample rate (4 000 Hz),
        then folded into `num_bins` peak bins. Each bin value is in [0, 1].

        Using a low decode rate (4 kHz) keeps memory and CPU low while still
        capturing amplitude envelope detail suitable for waveform display.

        Args:
            path:     Path to the audio file.
            num_bins: Number of horizontal bins to return (matches widget width).

        Returns:
            1-D float32 numpy array of shape (num_bins,) with values in [0, 1].

        Raises:
            RuntimeError: if ffmpeg fails to decode the file.
        """
        # Decode to mono float32 PCM at a low sample rate for efficiency.
        # -f f32le  → raw 32-bit float little-endian PCM
        # -ac 1     → mix down to mono
        # -ar 4000  → 4 kHz is plenty for amplitude envelope visualization
        cmd = [
            self._ffmpeg,
            "-v", "quiet",
            "-i", str(path),
            "-f", "f32le",
            "-ac", "1",
            "-ar", "4000",
            "pipe:1",
        ]

        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed for {path.name}: {result.stderr.decode(errors='replace').strip()}"
            )

        if not result.stdout:
            raise RuntimeError(f"ffmpeg produced no output for {path.name}")

        # Parse raw bytes → float32 numpy array.
        samples = np.frombuffer(result.stdout, dtype=np.float32)

        if len(samples) == 0:
            return np.zeros(num_bins, dtype=np.float32)

        # Fold into num_bins peak bins.
        # Each bin covers (len(samples) / num_bins) raw samples.
        peaks = _compute_peaks(samples, num_bins)

        # Normalize to [0, 1] so the waveform fills the canvas height.
        max_val = peaks.max()
        if max_val > 0:
            peaks /= max_val

        return peaks

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _require_binary(name: str) -> str:
        """
        Locate a binary on PATH or raise EnvironmentError.

        Args:
            name: Executable name (e.g. "ffmpeg").

        Returns:
            Full path string to the binary.
        """
        path = shutil.which(name)
        if path is None:
            raise EnvironmentError(
                f"{name} not found. Install ffmpeg:\n"
                "  macOS:   brew install ffmpeg\n"
                "  Ubuntu:  sudo apt install ffmpeg"
            )
        return path

    @staticmethod
    def is_available() -> bool:
        """Return True if both ffmpeg and ffprobe are installed."""
        return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ---------------------------------------------------------------------------
# Pure helper — peak folding
# ---------------------------------------------------------------------------

def _compute_peaks(samples: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Fold a 1-D samples array into `num_bins` peak-amplitude bins.

    Each bin receives the maximum absolute amplitude within its slice of
    the input array. Using max(abs) rather than RMS gives a waveform that
    looks like a standard audio editor (DAW) overview.

    Args:
        samples:  1-D float32 array of PCM samples (any length).
        num_bins: Desired number of output bins.

    Returns:
        1-D float32 array of shape (num_bins,).

    Example:
        peaks = _compute_peaks(samples, 800)
        # peaks[i] = max absolute sample in the i-th slice
    """
    n = len(samples)

    if n <= num_bins:
        # Fewer samples than bins — pad with zeros.
        out = np.zeros(num_bins, dtype=np.float32)
        out[:n] = np.abs(samples)
        return out

    # Use reshape to avoid a Python loop: split into num_bins even slices.
    # We trim the tail so the length is exactly divisible.
    trim = n - (n % num_bins)
    trimmed = samples[:trim]
    reshaped = trimmed.reshape(num_bins, -1)
    return np.max(np.abs(reshaped), axis=1).astype(np.float32)
