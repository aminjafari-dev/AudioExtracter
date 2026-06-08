"""
File path utilities for the AudioExtracter application.

Handles output path generation so that every other module can stay
focused on its own responsibility rather than doing string manipulation.

Usage example:
    from src.utils.file_utils import build_output_path

    output = build_output_path(
        input_path=Path("/videos/holiday.mp4"),
        output_format=AudioFormat.MP3,
        output_dir=Path("/audio"),
    )
    # => Path("/audio/holiday.mp3")
"""

from pathlib import Path

from src.core.models import AudioFormat


def build_output_path(
    input_path: Path,
    output_format: AudioFormat,
    output_dir: Path | None = None,
    suffix: str = "",
) -> Path:
    """
    Compute the output file path for an extraction or trim job.

    By default the output sits next to the input file (same directory),
    with the extension replaced by the chosen audio format. Supply
    *output_dir* to redirect to a different folder. Supply *suffix* to
    append an extra string to the stem (e.g. "_trimmed").

    Collision handling: if the target path already exists, a numeric
    suffix is appended (e.g. "movie_1.mp3", "movie_2.mp3") so existing
    files are never silently overwritten.

    Args:
        input_path:    Source file path.
        output_format: Desired audio output format (determines extension).
        output_dir:    Optional directory for the output file.
                       Defaults to input_path's parent directory.
        suffix:        Optional string appended to the file stem before
                       the extension (e.g. "_trimmed").

    Returns:
        A Path object that does not currently exist on disk.

    Example:
        >>> build_output_path(Path("/videos/clip.mp4"), AudioFormat.MP3)
        PosixPath('/videos/clip.mp3')

        >>> build_output_path(
        ...     Path("/audio/song.mp3"),
        ...     AudioFormat.WAV,
        ...     suffix="_trimmed",
        ... )
        PosixPath('/audio/song_trimmed.wav')
    """
    directory = output_dir if output_dir is not None else input_path.parent
    stem = input_path.stem + suffix
    extension = output_format.value  # e.g. "mp3"

    candidate = directory / f"{stem}.{extension}"

    # Avoid silently overwriting an existing file by appending a counter.
    counter = 1
    while candidate.exists():
        candidate = directory / f"{stem}_{counter}.{extension}"
        counter += 1

    return candidate


def human_readable_size(path: Path) -> str:
    """
    Return a human-readable file size string for a given path.

    Used in the file-list widget to show how large a video file is.
    Returns "—" if the file cannot be stat'd (e.g. does not exist yet).

    Args:
        path: File path to measure.

    Returns:
        A string like "12.4 MB" or "850 KB".

    Example:
        >>> human_readable_size(Path("/videos/big_movie.mkv"))
        '1.2 GB'
    """
    try:
        size = path.stat().st_size
    except OSError:
        return "—"

    # Step through unit thresholds from largest to smallest.
    for unit in ("GB", "MB", "KB"):
        threshold = {"GB": 1024 ** 3, "MB": 1024 ** 2, "KB": 1024}[unit]
        if size >= threshold:
            return f"{size / threshold:.1f} {unit}"

    return f"{size} B"
