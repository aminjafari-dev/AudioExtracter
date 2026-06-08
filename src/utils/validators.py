"""
File validation utilities.

Provides a single, authoritative check for whether a given file path
is a supported video format. All other parts of the application (drop
zone, file dialog filter) import from here so the list stays in one place.

Usage example:
    from src.utils.validators import is_supported_video

    if is_supported_video(Path("movie.mp4")):
        queue.append(path)
    else:
        show_error("Unsupported file format")
"""

from pathlib import Path


# ---------------------------------------------------------------------------
# Supported video extensions
# ---------------------------------------------------------------------------

# Lower-case extensions (without the dot) that ffmpeg can reliably demux.
# Extend this set if you need additional container formats.
SUPPORTED_VIDEO_EXTENSIONS: frozenset[str] = frozenset(
    {
        "mp4", "mkv", "avi", "mov", "wmv",
        "flv", "webm", "m4v", "3gp", "ts",
        "mpg", "mpeg", "vob", "ogv", "rm",
        "rmvb", "divx",
    }
)

# File-dialog filter string — shows only supported files in the browser.
# Qt uses the format "Description (*.ext1 *.ext2 ...)"
VIDEO_FILE_FILTER: str = (
    "Video Files ("
    + " ".join(f"*.{ext}" for ext in sorted(SUPPORTED_VIDEO_EXTENSIONS))
    + ");;All Files (*)"
)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------

def is_supported_video(path: Path) -> bool:
    """
    Return True if *path* has a supported video file extension.

    The check is case-insensitive so that ".MP4" and ".mp4" both pass.
    This does NOT read the file — it only checks the extension.
    For a deeper check the caller would need to run ffprobe, but for a
    desktop drag-and-drop tool the extension check is sufficient.

    Args:
        path: The file path to validate.

    Returns:
        True if the extension is in SUPPORTED_VIDEO_EXTENSIONS, else False.

    Example:
        >>> is_supported_video(Path("clip.MP4"))
        True
        >>> is_supported_video(Path("document.pdf"))
        False
    """
    return path.suffix.lstrip(".").lower() in SUPPORTED_VIDEO_EXTENSIONS


def filter_supported_videos(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """
    Split a list of paths into supported and unsupported groups.

    Useful when the user drops a mixed batch of files — the caller can
    add the supported ones to the queue and warn about the rest.

    Args:
        paths: Mixed list of file paths.

    Returns:
        A tuple (supported, unsupported) of path lists.

    Example:
        good, bad = filter_supported_videos([Path("a.mp4"), Path("b.txt")])
        # good == [Path("a.mp4")]
        # bad  == [Path("b.txt")]
    """
    supported = [p for p in paths if is_supported_video(p)]
    unsupported = [p for p in paths if not is_supported_video(p)]
    return supported, unsupported
