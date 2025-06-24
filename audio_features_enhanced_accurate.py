"""
Enhanced Audio Features Extraction with Improved Accuracy

This module provides more accurate audio feature extraction focusing on:
- Instrumentalness: Advanced vocal detection and harmonic analysis
- Valence: Sophisticated emotional analysis using spectral and harmonic features
- Danceability: Advanced rhythm and groove analysis

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

class EnhancedAccurateAudioAnalyzer:
    """Enhanced audio analyzer with improved accuracy for key features."""
    
    def __init__(self, sample_rate=22050, hop_length=512, n_mels=128, n_mfcc=20):
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        
    def detect_vocals(self, y, sr):
        """Advanced vocal detection using multiple techniques."""
        # 1. Harmonic-percussive separation
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        
        # 2. Spectral features for vocal detection
        # Vocal frequencies are typically in 85-255 Hz (fundamental) and 255-2000 Hz (harmonics)
        vocal_low_freq = 85
        vocal_high_freq = 2000
        
        # Get spectrogram
        D = librosa.stft(y_harmonic)
        frequencies = librosa.fft_frequencies(sr=sr)
        
        # Find vocal frequency range
        vocal_mask = (frequencies >= vocal_low_freq) & (frequencies <= vocal_high_freq)
        vocal_energy = np.mean(np.abs(D[vocal_mask, :]))
        total_energy = np.mean(np.abs(D))
        
        vocal_ratio = vocal_energy / (total_energy + 1e-8)
        
        # 3. MFCC analysis for vocal characteristics
        mfcc = librosa.feature.mfcc(y=y_harmonic, sr=sr, n_mfcc=13)
        
        # Vocal MFCC patterns (based on research)
        # MFCC 1-3 are most important for vocal detection
        vocal_mfcc_score = np.mean(np.abs(mfcc[1:4, :]))
        # Normalize MFCC score to 0-1 range (typical range is 0-50)
        vocal_mfcc_score = np.clip(vocal_mfcc_score / 50.0, 0, 1)
        
        # 4. Spectral centroid analysis
        spectral_centroid = librosa.feature.spectral_centroid(y=y_harmonic, sr=sr)[0]
        vocal_centroid_score = np.mean(spectral_centroid) / (sr/2)  # Normalize to 0-1
        vocal_centroid_score = np.clip(vocal_centroid_score, 0, 1)
        
        # 5. Zero crossing rate (vocals have lower ZCR than instruments)
        zcr = librosa.feature.zero_crossing_rate(y_harmonic)[0]
        zcr_score = 1 - np.mean(zcr)  # Invert since vocals have lower ZCR
        zcr_score = np.clip(zcr_score, 0, 1)
        
        # 6. Spectral rolloff (vocals have lower rolloff)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y_harmonic, sr=sr)[0]
        rolloff_score = 1 - (np.mean(spectral_rolloff) / (sr/2))
        rolloff_score = np.clip(rolloff_score, 0, 1)
        
        # Normalize vocal ratio to 0-1 range (typical range is 0-5)
        vocal_ratio = np.clip(vocal_ratio / 5.0, 0, 1)
        
        # Combine all vocal detection scores with more balanced weights
        vocal_scores = [
            vocal_ratio * 0.25,
            vocal_mfcc_score * 0.25,
            vocal_centroid_score * 0.2,
            zcr_score * 0.15,
            rolloff_score * 0.15
        ]
        
        vocal_probability = np.mean(vocal_scores)
        # Apply a more conservative scaling to avoid over-detection
        vocal_probability = vocal_probability * 0.7  # Scale down to be more conservative
        vocal_probability = np.clip(vocal_probability, 0, 1)  # Ensure 0-1 range
        
        return {
            'vocal_probability': vocal_probability,
            'vocal_ratio': vocal_ratio,
            'vocal_mfcc_score': vocal_mfcc_score,
            'vocal_centroid_score': vocal_centroid_score,
            'zcr_score': zcr_score,
            'rolloff_score': rolloff_score
        }
    
    def extract_advanced_valence(self, y, sr):
        """Advanced valence detection using multiple emotional indicators."""
        # 1. Harmonic analysis
        harmonic_features = self.extract_harmonic_features(y, sr)
        
        # 2. Spectral features for emotional content
        spectral_features = self.extract_spectral_features(y, sr)
        
        # 3. Tempo and rhythm analysis
        rhythm_features = self.extract_rhythm_features(y, sr)
        
        # 4. Key and mode analysis
        structure_features = self.extract_musical_structure_features(y, sr)
        
        # Valence indicators (based on music psychology research):
        
        # A. Harmonic complexity (more complex = more positive)
        harmonic_complexity = harmonic_features['harmonic_entropy'] / 10.0
        harmonic_complexity = np.clip(harmonic_complexity, 0, 1)
        
        # B. Spectral brightness (brighter = more positive)
        spectral_brightness = spectral_features['spectral_centroid_mean'] / (sr/2)
        spectral_brightness = np.clip(spectral_brightness, 0, 1)
        
        # C. Tempo factor (faster = more positive, but with diminishing returns)
        tempo_factor = np.clip(rhythm_features['tempo'] / 200.0, 0, 1)
        tempo_factor = tempo_factor ** 0.5  # Diminishing returns
        
        # D. Mode factor (major = more positive)
        mode_factor = 0.7 if structure_features['mode'] == 'major' else 0.3
        
        # E. Energy factor (moderate energy = more positive)
        energy_features = self.extract_energy_dynamics_features(y, sr)
        energy_factor = np.clip(energy_features['energy_mean'] / 0.3, 0, 1)
        energy_factor = 1 - abs(energy_factor - 0.5) * 2  # Peak at moderate energy
        
        # F. Spectral flux (more variation = more positive)
        spectral_flux_factor = np.clip(spectral_features['spectral_flux'] / 100.0, 0, 1)
        
        # Combine all factors with weights
        valence_factors = [
            harmonic_complexity * 0.2,
            spectral_brightness * 0.2,
            tempo_factor * 0.15,
            mode_factor * 0.15,
            energy_factor * 0.15,
            spectral_flux_factor * 0.15
        ]
        
        valence = np.mean(valence_factors)
        valence = np.clip(valence, 0, 1)
        
        return {
            'valence': valence,
            'harmonic_complexity': harmonic_complexity,
            'spectral_brightness': spectral_brightness,
            'tempo_factor': tempo_factor,
            'mode_factor': mode_factor,
            'energy_factor': energy_factor,
            'spectral_flux_factor': spectral_flux_factor
        }
    
    def extract_advanced_danceability(self, y, sr):
        """Advanced danceability detection using rhythm and groove analysis."""
        # 1. Basic rhythm features
        rhythm_features = self.extract_rhythm_features(y, sr)
        
        # 2. Energy analysis
        energy_features = self.extract_energy_dynamics_features(y, sr)
        
        # 3. Spectral analysis
        spectral_features = self.extract_spectral_features(y, sr)
        
        # Danceability indicators:
        
        # A. Beat strength and regularity
        beat_strength = np.clip(rhythm_features['beat_strength'] / 10.0, 0, 1)
        rhythm_regularity = np.clip(rhythm_features['rhythm_regularity'] / 2.0, 0, 1)
        
        # B. Tempo suitability (optimal range 90-150 BPM)
        tempo = rhythm_features['tempo']
        if tempo < 60:
            tempo_suitability = 0.1
        elif tempo < 90:
            tempo_suitability = 0.3 + (tempo - 60) / 30 * 0.4
        elif tempo <= 150:
            tempo_suitability = 0.7 + (tempo - 90) / 60 * 0.3
        else:
            tempo_suitability = 1.0 - (tempo - 150) / 100 * 0.3
        tempo_suitability = np.clip(tempo_suitability, 0, 1)
        
        # C. Energy consistency (steady energy = more danceable)
        energy_consistency = 1 - energy_features['energy_std'] / (energy_features['energy_mean'] + 1e-8)
        energy_consistency = np.clip(energy_consistency, 0, 1)
        
        # D. Spectral stability (stable spectrum = more danceable)
        spectral_stability = 1 - spectral_features['spectral_irregularity'] / 1000.0
        spectral_stability = np.clip(spectral_stability, 0, 1)
        
        # E. Syncopation (moderate syncopation = more danceable)
        syncopation = rhythm_features['syncopation']
        syncopation_factor = 1 - abs(syncopation - 0.5) * 2  # Peak at moderate syncopation
        syncopation_factor = np.clip(syncopation_factor, 0, 1)
        
        # F. Bass presence (important for dance music)
        # Analyze low frequency content
        D = librosa.stft(y)
        frequencies = librosa.fft_frequencies(sr=sr)
        bass_mask = frequencies <= 250  # Bass frequencies
        bass_energy = np.mean(np.abs(D[bass_mask, :]))
        total_energy = np.mean(np.abs(D))
        bass_factor = np.clip(bass_energy / (total_energy + 1e-8) * 5, 0, 1)  # Amplify bass importance
        
        # Combine all factors
        danceability_factors = [
            beat_strength * 0.2,
            rhythm_regularity * 0.2,
            tempo_suitability * 0.2,
            energy_consistency * 0.15,
            spectral_stability * 0.1,
            syncopation_factor * 0.1,
            bass_factor * 0.05
        ]
        
        danceability = np.mean(danceability_factors)
        danceability = np.clip(danceability, 0, 1)
        
        return {
            'danceability': danceability,
            'beat_strength': beat_strength,
            'rhythm_regularity': rhythm_regularity,
            'tempo_suitability': tempo_suitability,
            'energy_consistency': energy_consistency,
            'spectral_stability': spectral_stability,
            'syncopation_factor': syncopation_factor,
            'bass_factor': bass_factor
        }
    
    def extract_harmonic_features(self, y, sr):
        """Extract harmonic features."""
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
        """Extract spectral features."""
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
        """Extract rhythm features."""
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
    
    def extract_enhanced_features(self, y, sr):
        """Extract all enhanced features with improved accuracy."""
        # Basic features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=self.n_mels)
        
        # Advanced feature sets
        harmonic_features = self.extract_harmonic_features(y, sr)
        spectral_features = self.extract_spectral_features(y, sr)
        rhythm_features = self.extract_rhythm_features(y, sr)
        structure_features = self.extract_musical_structure_features(y, sr)
        energy_features = self.extract_energy_dynamics_features(y, sr)
        
        # Enhanced accuracy features
        vocal_features = self.detect_vocals(y, sr)
        valence_features = self.extract_advanced_valence(y, sr)
        danceability_features = self.extract_advanced_danceability(y, sr)
        
        # Calculate instrumentalness based on vocal detection
        instrumentalness = 1 - vocal_features['vocal_probability']
        instrumentalness = np.clip(instrumentalness, 0, 1)
        
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
            **vocal_features,
            **valence_features,
            **danceability_features,
            **mel_features,
            'instrumentalness': instrumentalness
        }
        
        # Add confidence scores
        confidence_scores = {
            'harmonic_confidence': harmonic_features['harmonic_ratio'],
            'rhythm_confidence': rhythm_features['rhythm_regularity'],
            'key_confidence': structure_features['key_confidence'],
            'mode_confidence': structure_features['mode_confidence'],
            'vocal_confidence': vocal_features['vocal_probability'],
            'overall_confidence': np.mean([
                harmonic_features['harmonic_ratio'],
                rhythm_features['rhythm_regularity'],
                structure_features['key_confidence']
            ])
        }
        
        all_features.update(confidence_scores)
        
        return all_features
    
    def analyze_file(self, file_path):
        """Analyze a single audio file."""
        try:
            print(f"Analyzing: {os.path.basename(file_path)}")
            
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            # Extract features
            features = self.extract_enhanced_features(y, sr)
            
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
            output_file = f"features_enhanced_accurate_{output_format}"
        
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
    
    parser = argparse.ArgumentParser(description='Enhanced Accurate Audio Features Extraction')
    parser.add_argument('folder', help='Folder containing audio files')
    parser.add_argument('--output', '-o', help='Output file name')
    parser.add_argument('--format', '-f', choices=['csv', 'json', 'yaml'], 
                       default='csv', help='Output format')
    parser.add_argument('--no-multiprocessing', action='store_true', 
                       help='Disable multiprocessing')
    parser.add_argument('--jobs', '-j', type=int, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    analyzer = EnhancedAccurateAudioAnalyzer()
    analyzer.analyze_folder(
        args.folder,
        output_format=args.format,
        output_file=args.output,
        use_multiprocessing=not args.no_multiprocessing,
        n_jobs=args.jobs
    )

if __name__ == "__main__":
    main() 