# Advanced Audio Features Extraction

A comprehensive Python module for extracting advanced audio features from local audio files using sophisticated analysis techniques. Compatible with Python 3.13 and uses only offline tools.

## Features

### 🎵 Advanced Audio Analysis

- **Harmonic Analysis**: Harmonic-percussive separation, harmonic complexity, harmonic entropy
- **Spectral Features**: Advanced spectral analysis including flux, irregularity, and statistical measures
- **Rhythm Analysis**: Tempo detection, beat strength, rhythm regularity, syncopation analysis
- **Musical Structure**: Key detection, mode analysis (major/minor), chroma features
- **Energy & Dynamics**: RMS energy, loudness, dynamic range, energy flux
- **Spotify-style Features**: Valence, danceability, instrumentalness, acousticness, speechiness

### 🔧 Technical Capabilities

- **Multiprocessing Support**: Parallel processing for large audio collections
- **Multiple Output Formats**: CSV, JSON, YAML
- **Confidence Scoring**: Quality metrics for each analysis
- **Comprehensive Feature Set**: 40+ advanced audio features
- **Python 3.13 Compatible**: No dependency on TensorFlow or other problematic libraries

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd Mozart
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from audio_features_advanced import AdvancedAudioAnalyzer

# Initialize analyzer
analyzer = AdvancedAudioAnalyzer()

# Analyze a folder of audio files
df = analyzer.analyze_folder('path/to/audio/files', output_file='results.csv')
```

### Command Line Usage

```bash
# Analyze a folder with default settings
python audio_features_advanced.py music_samples

# Specify output format and file
python audio_features_advanced.py music_samples --output my_results.json --format json

# Disable multiprocessing for debugging
python audio_features_advanced.py music_samples --no-multiprocessing

# Use specific number of parallel jobs
python audio_features_advanced.py music_samples --jobs 4
```

### Advanced Example

Run the comprehensive example with visualizations:

```bash
python example_advanced.py
```

This will:

- Analyze all audio files in the `music_samples` folder
- Display detailed feature breakdowns
- Generate comprehensive visualizations
- Create a summary report

## Feature Categories

### 🥁 Rhythm Features

- **Tempo**: Beats per minute (BPM)
- **Beat Strength**: Average onset strength at detected beats
- **Rhythm Regularity**: Consistency of beat intervals
- **Rhythmic Complexity**: Entropy of onset strength
- **Syncopation**: Off-beat emphasis measure
- **Beat Count**: Total number of detected beats

### 🎹 Harmonic Features

- **Harmonic Ratio**: Proportion of harmonic vs percussive content
- **Harmonic Energy**: Energy in harmonic component
- **Percussive Energy**: Energy in percussive component
- **Harmonic Entropy**: Complexity of harmonic spectrum
- **Harmonic Centroid**: Spectral centroid of harmonic component

### 🌊 Spectral Features

- **Spectral Centroid**: Brightness measure (Hz)
- **Spectral Rolloff**: Frequency below which 85% of energy is contained
- **Spectral Bandwidth**: Width of the spectral distribution
- **Spectral Contrast**: Difference between peaks and valleys
- **Spectral Flatness**: Measure of noisiness
- **Spectral Flux**: Rate of spectral change
- **Spectral Irregularity**: Variation in spectral shape

### 🎼 Musical Structure

- **Key Strength**: Strength of detected key
- **Key Confidence**: Confidence in key detection
- **Mode**: Major or minor mode
- **Mode Confidence**: Confidence in mode detection
- **Chroma Entropy**: Complexity of pitch distribution

### ⚡ Energy & Dynamics

- **Energy Mean**: Average RMS energy
- **Energy Standard Deviation**: Variation in energy
- **Loudness**: Approximated loudness in dB
- **Dynamic Range**: Difference between max and min energy
- **Energy Flux**: Rate of energy change
- **Zero Crossing Rate**: Measure of noisiness

### 🎵 Spotify-style Features

- **Valence**: Positive emotion measure (0-1)
- **Danceability**: Dance-friendly rhythm measure (0-1)
- **Instrumentalness**: Instrumental vs vocal content (0-1)
- **Acousticness**: Acoustic vs electronic sound (0-1)
- **Speechiness**: Speech vs music content (0-1)

## Output Formats

### CSV Format

Default format with all features in columns:

```csv
filename,tempo,harmonic_ratio,valence,danceability,...
Bohemian Rhapsody.mp3,143.6,0.762,1.000,1.000,...
```

### JSON Format

Structured data with nested features:

```json
[
  {
    "filename": "Bohemian Rhapsody.mp3",
    "rhythm_features": {
      "tempo": 143.6,
      "beat_strength": 3.485
    },
    "harmonic_features": {
      "harmonic_ratio": 0.762
    }
  }
]
```

### YAML Format

Human-readable structured format:

```yaml
- filename: Bohemian Rhapsody.mp3
  tempo: 143.6
  harmonic_ratio: 0.762
  valence: 1.000
```

## Confidence Scoring

The analyzer provides confidence scores for different aspects:

- **Harmonic Confidence**: Based on harmonic ratio
- **Rhythm Confidence**: Based on rhythm regularity
- **Key Confidence**: Based on key detection strength
- **Mode Confidence**: Based on mode detection correlation
- **Overall Confidence**: Average of all confidence scores

## Performance

- **Single File**: ~2-5 seconds depending on file length
- **Multiprocessing**: Scales with CPU cores
- **Memory Usage**: ~100-200MB per analysis
- **Supported Formats**: MP3, WAV, FLAC, M4A, OGG

## Advanced Configuration

```python
# Customize analyzer parameters
analyzer = AdvancedAudioAnalyzer(
    sample_rate=22050,    # Audio sample rate
    hop_length=512,       # STFT hop length
    n_mels=128,          # Number of mel bands
    n_mfcc=20            # Number of MFCC coefficients
)

# Analyze with custom settings
df = analyzer.analyze_folder(
    folder_path='music_samples',
    output_format='json',
    output_file='custom_results.json',
    use_multiprocessing=True,
    n_jobs=4
)
```

## Comparison with Other Modules

| Feature                | Basic | Enhanced | Advanced |
| ---------------------- | ----- | -------- | -------- |
| Basic Features         | ✅    | ✅       | ✅       |
| Spotify-style Features | ❌    | ✅       | ✅       |
| Harmonic Analysis      | ❌    | ❌       | ✅       |
| Spectral Analysis      | Basic | Basic    | Advanced |
| Rhythm Analysis        | Basic | Basic    | Advanced |
| Confidence Scoring     | ❌    | ✅       | ✅       |
| Multiprocessing        | ✅    | ✅       | ✅       |
| Multiple Formats       | ✅    | ✅       | ✅       |

## Troubleshooting

### Common Issues

1. **No audio files found**: Ensure files have supported extensions (.mp3, .wav, .flac, .m4a, .ogg)
2. **Memory errors**: Reduce `n_jobs` or disable multiprocessing
3. **Import errors**: Ensure all dependencies are installed in virtual environment

### Performance Tips

- Use multiprocessing for large collections (>10 files)
- Adjust `sample_rate` for faster processing (lower = faster)
- Use `hop_length=1024` for faster analysis with slight accuracy loss

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **librosa**: Audio and music signal analysis
- **scipy**: Scientific computing and signal processing
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation and analysis
