# AudioExtracter

A clean, modern GUI application for extracting audio from video files.

## Features

- **Drag & drop** video files directly onto the window
- **Browse** to select one or multiple video files
- **Batch processing** — queue multiple files and extract all at once
- **Format selection** — export as MP3, AAC, WAV, FLAC, or OGG
- **Real-time progress** per file with status badges
- **Non-blocking UI** — extraction runs on background threads

## Requirements

- Python 3.10+
- [FFmpeg](https://ffmpeg.org/download.html) installed and available on your `$PATH`

## Installation

```bash
pip install -r requirements.txt
```

Make sure `ffmpeg` is installed:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org and add to PATH
```

## Running

```bash
python main.py
```

## Project Structure

```
AudioExtracter/
├── src/
│   ├── core/
│   │   ├── extractor.py        # Audio extraction engine (ffmpeg wrapper)
│   │   └── models.py           # Data models: ExtractionJob, ExtractionResult
│   ├── ui/
│   │   ├── main_window.py      # Main application window
│   │   ├── theme.py            # Colors, fonts, stylesheet constants
│   │   └── widgets/
│   │       ├── drop_zone.py    # Drag & drop target widget
│   │       ├── file_list.py    # Scrollable file queue with status
│   │       └── toolbar.py      # Format selector + action buttons
│   └── utils/
│       ├── file_utils.py       # Path helpers, output naming
│       └── validators.py       # Supported format validation
├── main.py
├── requirements.txt
└── README.md
```

## Supported Input Formats

MP4, MKV, AVI, MOV, WMV, FLV, WEBM, M4V, 3GP, TS, MPG, MPEG

## Output Formats

| Format | Notes                         |
|--------|-------------------------------|
| MP3    | 192 kbps, widely compatible   |
| AAC    | High quality, small size      |
| WAV    | Lossless, large files         |
| FLAC   | Lossless compressed           |
| OGG    | Open source, good quality     |
