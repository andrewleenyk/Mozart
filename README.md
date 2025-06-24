# Mozart Audio Analysis App

A dual-mode Python application for extracting advanced audio features from music files using sophisticated offline analysis techniques.

## 🎵 What It Does

Extracts Spotify-style features from audio files:

- **Valence**: Mood/positivity (0-1)
- **Danceability**: Rhythmic quality (0-1)
- **Instrumentalness**: Instrumental vs vocal content (0-1)
- **Acousticness**: Acoustic vs electronic sound (0-1)
- **Speechiness**: Speech vs music content (0-1)

Plus confidence scores for each feature.

## 🚀 Quick Start

1. **Install dependencies:**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Setup production directory:**

```bash
python mozart_app.py --setup
```

3. **Run analysis:**

```bash
# Testing mode - analyze test samples
python mozart_app.py --mode testing

# Production mode - analyze your music
python mozart_app.py --mode production
```

## 📁 Dual Mode System

### 🧪 Testing Mode

- **Purpose**: Test and validate the analysis system
- **Directory**: `test_music_samples/` (contains 38 sample files)
- **Output**: `features_test_mode.csv`
- **Use when**: Developing, testing, or validating the system

### 🏭 Production Mode

- **Purpose**: Analyze your personal music collection
- **Directory**: `songs/` (place your MP3/WAV files here)
- **Output**: `features_production_mode.csv`
- **Use when**: Analyzing your own music library

## 📊 Output

Each mode produces a single CSV file with:

- File path and filename
- 5 enhanced audio features (0-1 scale)
- Confidence scores for each feature
- One row per analyzed song

## 🎯 Usage Examples

```bash
# Test the system with sample files
python mozart_app.py --mode testing

# Add your music to songs/ folder, then analyze
python mozart_app.py --mode production

# Create production directory if it doesn't exist
python mozart_app.py --setup
```

## 🔧 Technical Details

- **Supported formats**: MP3, WAV, FLAC, M4A, OGG
- **Processing speed**: ~8x real-time (5-minute song ≈ 40 seconds)
- **Offline processing**: No internet required for analysis
- **Python 3.13 compatible**: Uses librosa for advanced audio analysis

## 📈 Feature Analysis

The app uses sophisticated algorithms to analyze:

- **Spectral characteristics** (brightness, timbre)
- **Rhythmic patterns** (tempo, beat strength, regularity)
- **Harmonic content** (major/minor scales, chord progressions)
- **Energy dynamics** (loudness, energy changes)
- **Vocal vs instrumental** separation

## 🎵 Music Downloader

Also includes a music downloader in `music_downloader/`:

- Download songs from YouTube using `yt-dlp`
- Organize by genre automatically
- Batch processing capabilities

## 📝 Requirements

- Python 3.13+
- librosa, pandas, numpy
- yt-dlp (for music downloading)

See `requirements.txt` for complete dependencies.
