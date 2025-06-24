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
- **CSV Output**: Clean, structured data output
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
from audio_features_simple import AudioFeatureExtractor

# Initialize extractor
extractor = AudioFeatureExtractor()

# Analyze a folder of audio files
df = extractor.process_directory('path/to/audio/files', output_file='results.csv')
```

### Enhanced Usage

```python
from audio_features_enhanced import EnhancedAudioFeatureExtractor

# Initialize enhanced extractor
enhanced_extractor = EnhancedAudioFeatureExtractor()

# Process existing features and add enhanced analysis
df = enhanced_extractor.process_existing_features(
    features_csv='basic_features.csv',
    output_file='enhanced_results.csv'
)
```

### Advanced Usage

```python
from audio_features_advanced import AdvancedAudioAnalyzer

# Initialize advanced analyzer
analyzer = AdvancedAudioAnalyzer()

# Analyze a folder with advanced features
df = analyzer.analyze_folder('path/to/audio/files', output_file='advanced_results.csv')
```

### Command Line Usage

```bash
# Simple analysis
python analyze_audio.py

# Enhanced analysis
python analyze_audio_enhanced.py

# Run examples
python example.py
python example_advanced.py
```

### Example Scripts

Run the comprehensive examples:

```bash
# Basic example with detailed feature breakdown
python example.py

# Advanced example with confidence scores
python example_advanced.py
```

These will:

- Analyze all audio files in the `music_samples` folder
- Display detailed feature breakdowns
- Generate CSV output files
- Create summary reports

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

## Output Format

### CSV Format

Default format with all features in columns:

```csv
filename,tempo,harmonic_ratio,valence,danceability,instrumentalness,acousticness,speechiness,...
Bohemian Rhapsody.mp3,143.6,0.762,0.779,0.406,0.691,0.442,0.220,...
```

## Confidence Scoring

The analyzers provide confidence scores for different aspects:

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
# Customize extractor parameters
extractor = AudioFeatureExtractor(
    sample_rate=22050    # Audio sample rate
)

# Process with custom settings
df = extractor.process_directory(
    input_dir='music_samples',
    output_file='custom_results.csv',
    use_sqlite=False,
    n_jobs=1
)
```

## Module Comparison

| Feature                | Simple | Enhanced | Advanced |
| ---------------------- | ------ | -------- | -------- |
| Basic Features         | ✅     | ✅       | ✅       |
| Spotify-style Features | ✅     | ✅       | ✅       |
| Harmonic Analysis      | ❌     | ❌       | ✅       |
| Spectral Analysis      | Basic  | Basic    | Advanced |
| Rhythm Analysis        | Basic  | Basic    | Advanced |
| Confidence Scoring     | ❌     | ✅       | ✅       |
| Multiprocessing        | ✅     | ✅       | ✅       |
| CSV Output             | ✅     | ✅       | ✅       |

## Available Scripts

- **`analyze_audio.py`**: Simple analysis with basic features
- **`analyze_audio_enhanced.py`**: Two-step enhanced analysis
- **`example.py`**: Basic example with detailed output
- **`example_advanced.py`**: Advanced example with confidence scores
- **`audio_features_simple.py`**: Core simple extractor
- **`audio_features_enhanced.py`**: Enhanced feature extractor
- **`audio_features_advanced.py`**: Advanced analyzer with harmonic analysis

## Troubleshooting

### Common Issues

1. **No audio files found**: Ensure files have supported extensions (.mp3, .wav, .flac, .m4a, .ogg)
2. **Memory errors**: Reduce `n_jobs` or disable multiprocessing
3. **Import errors**: Ensure all dependencies are installed in virtual environment
4. **Duplicate files**: The system automatically deduplicates files found by glob patterns

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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **librosa**: Audio and music signal analysis
- **scipy**: Scientific computing and signal processing
- **scikit-learn**: Machine learning utilities
- **pandas**: Data manipulation and analysis
