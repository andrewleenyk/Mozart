# Music Downloader

A Python script that downloads songs from a text file using yt-dlp. It organizes downloads by genre and saves them as MP3 files.

## Features

- 📁 **Genre-based organization**: Downloads are organized by genre in separate folders
- 🎵 **MP3 format**: Downloads audio in high-quality MP3 format
- 🏷️ **Metadata embedding**: Automatically adds title, artist, and genre metadata
- 🖼️ **Thumbnail embedding**: Embeds video thumbnails in audio files
- 📊 **Progress tracking**: Shows download progress and summary
- 🔍 **YouTube search**: Uses YouTube search to find songs

## Installation

1. Install yt-dlp:

```bash
pip install yt-dlp
```

Or install from requirements:

```bash
pip install -r requirements.txt
```

## Available Scripts

### Main Downloaders

- **`download_songs.py`**: Full-featured downloader with metadata and error handling
- **`download_songs_simple.py`**: Simplified version for basic downloads
- **`download_songs_fixed.py`**: Fixed version with improved search functionality

### Test Scripts

- **`test_download.py`**: Test script to verify single song download functionality

## Usage

### Basic Usage

1. Create a songs text file with the following format:

```txt
# Rock/Alternative
Hotel California - Eagles
Black - Pearl Jam
Take Me Out - Franz Ferdinand

# Instrumentals & Cinematic
Adagio for Strings - Samuel Barber
Time - Hans Zimmer
```

2. Run the downloader:

```bash
# Try the fixed version first (recommended)
python download_songs_fixed.py

# Or try the simple version
python download_songs_simple.py

# Or the full-featured version
python download_songs.py
```

### Advanced Usage

```bash
# Specify custom songs file and output directory
python download_songs_fixed.py my_songs.txt my_downloads

# Test single download first
python test_download.py
```

## File Format

The songs text file should follow this format:

```txt
# Genre Name
Song Title - Artist Name
Another Song - Another Artist

# Another Genre
Song Title - Artist Name
```

- Lines starting with `#` are genre headers
- Other lines should be in "Song Title - Artist Name" format
- Empty lines are ignored

## Output Structure

Downloads are organized by genre:

```
downloads/
├── Rock/Alternative/
│   ├── Hotel California.mp3
│   ├── Black.mp3
│   └── Take Me Out.mp3
└── Instrumentals & Cinematic/
    ├── Adagio for Strings.mp3
    └── Time.mp3
```

## Features

- **High Quality**: Downloads in best available audio quality
- **Metadata**: Automatically adds title, artist, and genre tags
- **Thumbnails**: Embeds video thumbnails in audio files
- **Error Handling**: Continues downloading even if some songs fail
- **Progress Tracking**: Shows detailed progress and final summary
- **Timeout Protection**: Prevents hanging on problematic downloads

## Troubleshooting

### Common Issues

1. **yt-dlp not found**: Install it with `pip install yt-dlp`
2. **Download failures**:
   - Try the `download_songs_fixed.py` version first
   - Check internet connection
   - Some songs may not be available on YouTube
3. **File permissions**: Ensure write permissions in output directory
4. **Search issues**: The fixed version uses quoted search for better results

### Testing

Before downloading all songs, test with a single download:

```bash
python test_download.py
```

This will download one test song to verify everything works.

## Notes

- This script uses YouTube search to find songs
- Downloads are for testing purposes only
- Respect copyright and licensing when downloading content
- Some songs may not be found or may have different versions
- The fixed version (`download_songs_fixed.py`) is recommended for best results

## Script Differences

| Feature        | Simple | Fixed  | Full     |
| -------------- | ------ | ------ | -------- |
| Basic Download | ✅     | ✅     | ✅       |
| Error Handling | Basic  | Good   | Advanced |
| Metadata       | ❌     | ❌     | ✅       |
| Search Method  | Basic  | Quoted | Advanced |
| Recommended    | ❌     | ✅     | ✅       |
