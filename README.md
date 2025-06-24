# Mozart Audio Analysis App

A Python application for extracting advanced audio features from music files using sophisticated offline analysis techniques.

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

2. **Setup your music directory:**

```bash
python mozart_app.py --setup
```

3. **Add your music files:**

   - Place your MP3, WAV, FLAC, M4A, or OGG files in the `songs/` folder
   - You can organize them in subfolders (e.g., `songs/rock/`, `songs/jazz/`)

4. **Run analysis:**

```bash
# Test the system with sample files
python mozart_app.py --mode testing

# Analyze your music collection
python mozart_app.py --mode production
```

## 📁 Two Analysis Modes

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

## 📊 CSV Output Data

Each analysis produces a single CSV file with the following data:

### File Information

- `filepath`: Full path to the audio file
- `filename`: Just the filename

### Audio Features (0-1 scale)

- `valence`: How positive/happy the song sounds (0 = sad, 1 = happy)
- `danceability`: How suitable for dancing (0 = not danceable, 1 = very danceable)
- `instrumentalness`: How much instrumental vs vocal content (0 = mostly vocals, 1 = mostly instrumental)
- `acousticness`: How acoustic vs electronic (0 = electronic, 1 = acoustic)
- `speechiness`: How much speech vs music (0 = instrumental, 1 = spoken word)

### Confidence Scores (0-1 scale)

- `valence_confidence`: How reliable the valence score is
- `danceability_confidence`: How reliable the danceability score is
- `instrumentalness_confidence`: How reliable the instrumentalness score is
- `acousticness_confidence`: How reliable the acousticness score is
- `speechiness_confidence`: How reliable the speechiness score is

### Example CSV Output:

```csv
filepath,filename,valence,valence_confidence,danceability,danceability_confidence,instrumentalness,instrumentalness_confidence,acousticness,acousticness_confidence,speechiness,speechiness_confidence
songs/rock/song1.mp3,song1.mp3,0.721,0.749,0.519,0.266,0.746,0.864,0.220,0.559,0.355,0.566
songs/jazz/song2.mp3,song2.mp3,0.596,0.662,0.572,0.198,0.647,0.716,0.320,0.451,0.322,0.737
```

## 🎯 Usage Examples

```bash
# Test the system with sample files
python mozart_app.py --mode testing

# Add your music to songs/ folder, then analyze
python mozart_app.py --mode production

# Create songs directory if it doesn't exist
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
