"""
Advanced Audio Features Extraction Module

This module provides comprehensive audio feature extraction using advanced techniques
including harmonic analysis, spectral features, and sophisticated rhythm analysis.
Compatible with Python 3.13 and uses only offline tools.
"""

import os
import glob
import numpy as np
import pandas as pd
import librosa
import librosa.display
from scipy import signal
from scipy.stats import entropy, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import multiprocessing as mp
from joblib import Parallel, delayed
import yaml
import warnings
warnings.filterwarnings('ignore')

class AdvancedAudioAnalyzer:
    """Advanced audio analyzer with sophisticated feature extraction."""
    
    def __init__(self, sample_rate=22050, hop_length=512, n_mels=128, n_mfcc=20):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
    def extract_harmonic_features(self, y, sr):
        """Extract advanced harmonic features."""
        # Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # Harmonic features
        harmonic_energy = np.sum(y_harmonic**2) / len(y_harmonic)
        percussive_energy = np.sum(y_percussive**2) / len(y_percussive)
        harmonic_ratio = harmonic_energy / (harmonic_energy + percussive_energy + 1e-8)
        
        # Harmonic complexity
        harmonic_spectrum = np.abs(librosa.stft(y_harmonic))
        harmonic_entropy = entropy(np.mean(harmonic_spectrum, axis=1) + 1e-8)
        
        # Spectral centroid of harmonic component
        harmonic_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
        
        return {
            'harmonic_energy': harmonic_energy,
            'percussive_energy': percussive_energy,
            'harmonic_ratio': harmonic_ratio,
            'harmonic_entropy': harmonic_entropy,
            'harmonic_centroid_mean': np.mean(harmonic_centroid),
            'harmonic_centroid_std': np.std(harmonic_centroid)
        }
    
    def extract_spectral_features(self, y, sr):
        """Extract advanced spectral features."""
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        # Advanced spectral features
        spectral_skewness = skew(spectral_centroids)
        spectral_kurtosis = kurtosis(spectral_centroids)
        
        # Spectral flux (rate of change)
        spectral_flux = np.mean(np.diff(spectral_centroids))
        
        # Spectral irregularity
        spectral_irregularity = np.std(np.diff(spectral_centroids))
        
        return {
            'spectral_centroid_mean': np.mean(spectral_centroids),
            'spectral_centroid_std': np.std(spectral_centroids),
            'spectral_centroid_skewness': spectral_skewness,
            'spectral_centroid_kurtosis': spectral_kurtosis,
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_contrast_mean': np.mean(spectral_contrast),
            'spectral_flatness_mean': np.mean(spectral_flatness),
            'spectral_flux': spectral_flux,
            'spectral_irregularity': spectral_irregularity
        }
    
    def extract_rhythm_features(self, y, sr):
        """Extract advanced rhythm features."""
        # Tempo and beat tracking
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
        
        # Beat strength analysis
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        beat_strength = np.mean(onset_env[beats]) if len(beats) > 0 else 0
        
        # Rhythm regularity
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            rhythm_regularity = 1 / (np.std(beat_intervals) + 1e-8)
        else:
            rhythm_regularity = 0
        
        # Rhythmic complexity
        rhythmic_complexity = entropy(onset_env + 1e-8)
        
        # Syncopation (off-beat emphasis)
        if len(beats) > 0:
            off_beat_strength = np.mean(onset_env[beats[::2]]) if len(beats) > 1 else 0
            syncopation = off_beat_strength / (beat_strength + 1e-8)
        else:
            syncopation = 0
        
        return {
            'tempo': float(tempo),
            'beat_strength': beat_strength,
            'rhythm_regularity': rhythm_regularity,
            'rhythmic_complexity': rhythmic_complexity,
            'syncopation': syncopation,
            'beat_count': len(beats)
        }
    
    def extract_musical_structure_features(self, y, sr):
        """Extract musical structure features."""
        # Chroma features for key detection
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        
        # Key detection (simplified)
        chroma_mean = np.mean(chroma, axis=1)
        key_strength = np.max(chroma_mean)
        key_confidence = key_strength / (np.sum(chroma_mean) + 1e-8)
        
        # Mode detection (major vs minor)
        major_profile = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
        minor_profile = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
        
        major_correlation = np.corrcoef(chroma_mean, major_profile)[0, 1]
        minor_correlation = np.corrcoef(chroma_mean, minor_profile)[0, 1]
        
        mode = 'major' if major_correlation > minor_correlation else 'minor'
        mode_confidence = max(major_correlation, minor_correlation)
        
        return {
            'key_strength': key_strength,
            'key_confidence': key_confidence,
            'mode': mode,
            'mode_confidence': mode_confidence,
            'chroma_entropy': entropy(chroma_mean + 1e-8)
        }
    
    def extract_energy_dynamics_features(self, y, sr):
        """Extract energy and dynamics features."""
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Loudness (approximated)
        loudness = 20 * np.log10(np.mean(rms) + 1e-8)
        
        # Energy dynamics
        energy_mean = np.mean(rms)
        energy_std = np.std(rms)
        energy_skewness = skew(rms)
        
        # Dynamic range
        dynamic_range = np.max(rms) - np.min(rms)
        
        # Energy flux
        energy_flux = np.mean(np.diff(rms))
        
        # Zero crossing rate (related to noisiness)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        zcr_mean = np.mean(zcr)
        
        return {
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'energy_skewness': energy_skewness,
            'loudness': loudness,
            'dynamic_range': dynamic_range,
            'energy_flux': energy_flux,
            'zero_crossing_rate': zcr_mean
        }
    
    def extract_spotify_style_features(self, y, sr):
        """Extract Spotify-style emotional and character features."""
        # Valence (positive emotion) - based on harmonic content and energy
        harmonic_features = self.extract_harmonic_features(y, sr)
        energy_features = self.extract_energy_dynamics_features(y, sr)
        
        # Normalize energy_mean to 0-1 range (typical range is 0-0.5)
        normalized_energy = np.clip(energy_features['energy_mean'] / 0.5, 0, 1)
        
        # Normalize harmonic entropy to 0-1 range (typical range is 0-10)
        normalized_entropy = np.clip(harmonic_features['harmonic_entropy'] / 10.0, 0, 1)
        
        # Valence: higher for harmonic, lower energy, higher entropy
        valence = (harmonic_features['harmonic_ratio'] * 0.4 + 
                  (1 - normalized_energy) * 0.3 + 
                  normalized_entropy * 0.3)
        valence = np.clip(valence, 0, 1)
        
        # Danceability: based on rhythm regularity and energy
        rhythm_features = self.extract_rhythm_features(y, sr)
        
        # Normalize rhythm regularity to 0-1 range (typical range is 0-2)
        normalized_regularity = np.clip(rhythm_features['rhythm_regularity'] / 2.0, 0, 1)
        
        # Normalize beat strength to 0-1 range (typical range is 0-10)
        normalized_beat_strength = np.clip(rhythm_features['beat_strength'] / 10.0, 0, 1)
        
        danceability = (normalized_regularity * 0.4 + 
                       normalized_energy * 0.3 + 
                       normalized_beat_strength * 0.3)
        danceability = np.clip(danceability, 0, 1)
        
        # Instrumentalness: based on harmonic content and spectral features
        spectral_features = self.extract_spectral_features(y, sr)
        instrumentalness = (harmonic_features['harmonic_ratio'] * 0.5 + 
                           (1 - spectral_features['spectral_flatness_mean']) * 0.5)
        instrumentalness = np.clip(instrumentalness, 0, 1)
        
        # Acousticness: based on spectral characteristics
        # Normalize spectral irregularity to 0-1 range (typical range is 0-1000)
        normalized_irregularity = np.clip(spectral_features['spectral_irregularity'] / 1000.0, 0, 1)
        
        # Normalize zero crossing rate to 0-1 range (typical range is 0-0.5)
        normalized_zcr = np.clip(energy_features['zero_crossing_rate'] / 0.5, 0, 1)
        
        acousticness = ((1 - normalized_irregularity) * 0.6 + 
                       (1 - normalized_zcr) * 0.4)
        acousticness = np.clip(acousticness, 0, 1)
        
        # Speechiness: based on spectral flatness and zero crossing rate
        speechiness = (spectral_features['spectral_flatness_mean'] * 0.6 + 
                      normalized_zcr * 0.4)
        speechiness = np.clip(speechiness, 0, 1)
        
        return {
            'valence': valence,
            'danceability': danceability,
            'instrumentalness': instrumentalness,
            'acousticness': acousticness,
            'speechiness': speechiness
        }
    
    def extract_advanced_features(self, y, sr):
        """Extract all advanced features."""
        # Basic features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        
        # Advanced feature sets
        harmonic_features = self.extract_harmonic_features(y, sr)
        spectral_features = self.extract_spectral_features(y, sr)
        rhythm_features = self.extract_rhythm_features(y, sr)
        structure_features = self.extract_musical_structure_features(y, sr)
        energy_features = self.extract_energy_dynamics_features(y, sr)
        spotify_features = self.extract_spotify_style_features(y, sr)
        
        # MFCC statistics
        mfcc_features = {
            'mfcc_mean': np.mean(mfcc, axis=1).tolist(),
            'mfcc_std': np.std(mfcc, axis=1).tolist(),
            'mfcc_skewness': skew(mfcc, axis=1).tolist(),
            'mfcc_kurtosis': kurtosis(mfcc, axis=1).tolist()
        }
        
        # Mel spectrogram statistics
        mel_features = {
            'mel_energy_mean': np.mean(mel_spectrogram),
            'mel_energy_std': np.std(mel_spectrogram),
            'mel_energy_skewness': skew(mel_spectrogram.flatten()),
            'mel_energy_kurtosis': kurtosis(mel_spectrogram.flatten())
        }
        
        # Combine all features
        all_features = {
            **harmonic_features,
            **spectral_features,
            **rhythm_features,
            **structure_features,
            **energy_features,
            **spotify_features,
            **mel_features
        }
        
        # Add confidence scores
        confidence_scores = {
            'harmonic_confidence': harmonic_features['harmonic_ratio'],
            'rhythm_confidence': rhythm_features['rhythm_regularity'],
            'key_confidence': structure_features['key_confidence'],
            'mode_confidence': structure_features['mode_confidence'],
            'overall_confidence': np.mean([
                harmonic_features['harmonic_ratio'],
                rhythm_features['rhythm_regularity'],
                structure_features['key_confidence']
            ])
        }
        
        all_features.update(confidence_scores)
        
        return all_features, mfcc_features
    
    def analyze_file(self, file_path):
        """Analyze a single audio file."""
        try:
            print(f"Analyzing: {os.path.basename(file_path)}")
            
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Extract features
            features, mfcc_features = self.extract_advanced_features(y, sr)
            
            # Add file information
            features['filename'] = os.path.basename(file_path)
            features['file_path'] = file_path
            features['duration'] = len(y) / sr
            features['sample_rate'] = sr
            
            return features
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            return None
    
    def analyze_folder(self, folder_path, output_format='csv', output_file=None, 
                      use_multiprocessing=True, n_jobs=None):
        """Analyze all audio files in a folder."""
        # Find audio files
        audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(folder_path, ext)))
            audio_files.extend(glob.glob(os.path.join(folder_path, '**', ext), recursive=True))
        
        # Remove duplicates
        audio_files = list(set(audio_files))
        
        if not audio_files:
            print(f"No audio files found in {folder_path}")
            return None
        
        print(f"Found {len(audio_files)} audio files")
        
        # Analyze files
        if use_multiprocessing and len(audio_files) > 1:
            n_jobs = n_jobs or min(mp.cpu_count(), len(audio_files))
            print(f"Using multiprocessing with {n_jobs} jobs")
            results = Parallel(n_jobs=n_jobs)(
                delayed(self.analyze_file)(file_path) for file_path in audio_files
            )
        else:
            results = [self.analyze_file(file_path) for file_path in audio_files]
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        if not results:
            print("No successful analyses")
            return None
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        if output_file is None:
            output_file = f"features_advanced_{output_format}"
        
        if output_format.lower() == 'csv':
            df.to_csv(output_file, index=False)
            print(f"Results saved to {output_file}")
        elif output_format.lower() == 'json':
            df.to_json(output_file, orient='records', indent=2)
            print(f"Results saved to {output_file}")
        elif output_format.lower() == 'yaml':
            with open(output_file, 'w') as f:
                yaml.dump(df.to_dict('records'), f, default_flow_style=False)
            print(f"Results saved to {output_file}")
        
        return df

def main():
    """Main function for command line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Audio Features Extraction')
    parser.add_argument('folder', help='Folder containing audio files')
    parser.add_argument('--output', '-o', help='Output file name')
    parser.add_argument('--format', '-f', choices=['csv', 'json', 'yaml'], 
                       default='csv', help='Output format')
    parser.add_argument('--no-multiprocessing', action='store_true', 
                       help='Disable multiprocessing')
    parser.add_argument('--jobs', '-j', type=int, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    analyzer = AdvancedAudioAnalyzer()
    analyzer.analyze_folder(
        args.folder,
        output_format=args.format,
        output_file=args.output,
        use_multiprocessing=not args.no_multiprocessing,
        n_jobs=args.jobs
    )

if __name__ == "__main__":
    main() 