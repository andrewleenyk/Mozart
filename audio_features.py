"""
Audio Feature Extraction Module

Extracts comprehensive audio features from local .mp3 and .wav files using offline tools.
Features include rhythmic analysis, musical structure, energy dynamics, and emotional characteristics.
"""

import os
import glob
import argparse
import sqlite3
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import tensorflow as tf
from scipy import stats
from scipy.signal import correlate
import joblib

# Try to import musicnn (may need to be installed separately)
try:
    import musicnn
    from musicnn.extractor import extractor
    MUSICNN_AVAILABLE = True
except ImportError:
    MUSICNN_AVAILABLE = False
    print("Warning: musicnn not available. Install with: pip install musicnn")

# Try to import YAMNet (may need to be installed separately)
try:
    import yamnet
    YAMNET_AVAILABLE = True
except ImportError:
    YAMNET_AVAILABLE = False
    print("Warning: YAMNet not available. Install with: pip install yamnet")


class AudioFeatureExtractor:
    """Extract comprehensive audio features from audio files."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        self.yamnet_model = None
        
        # Initialize YAMNet if available
        if YAMNET_AVAILABLE:
            try:
                self.yamnet_model = yamnet.yamnet_frames_model(sample_rate)
                self.yamnet_class_names = yamnet.yamnet_class_names()
            except Exception as e:
                print(f"Warning: Could not load YAMNet model: {e}")
                self.yamnet_model = None
    
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and convert to mono.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio with librosa
            audio, sr = librosa.load(file_path, mono=True, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_rhythmic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract rhythmic features from audio.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary of rhythmic features
        """
        features = {}
        
        try:
            # Tempo
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            features['tempo'] = float(tempo)
            
            # Beat strength (mean of onset strength)
            onset_strength = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            features['beat_strength'] = float(np.mean(onset_strength))
            
            # Beat intervals and rhythmic stability
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                features['rhythmic_stability'] = float(np.std(beat_intervals))
                
                # Regularity (autocorrelation of beat intervals)
                if len(beat_intervals) > 10:
                    autocorr = correlate(beat_intervals, beat_intervals, mode='full')
                    autocorr = autocorr[len(beat_intervals)-1:]
                    # Normalize and take the first peak after lag 0
                    autocorr = autocorr / autocorr[0]
                    if len(autocorr) > 1:
                        features['regularity'] = float(np.max(autocorr[1:min(10, len(autocorr))]))
                    else:
                        features['regularity'] = 0.0
                else:
                    features['regularity'] = 0.0
            else:
                features['rhythmic_stability'] = 0.0
                features['regularity'] = 0.0
                
        except Exception as e:
            print(f"Error extracting rhythmic features: {e}")
            features.update({
                'tempo': 0.0,
                'beat_strength': 0.0,
                'rhythmic_stability': 0.0,
                'regularity': 0.0
            })
        
        return features
    
    def extract_musical_structure(self, audio: np.ndarray) -> Dict[str, str]:
        """
        Extract musical structure features (key and mode).
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary of musical structure features
        """
        features = {}
        
        try:
            # Chroma features
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
            
            # Key detection (simplified - find dominant pitch class)
            chroma_mean = np.mean(chroma, axis=1)
            dominant_pitch = np.argmax(chroma_mean)
            
            # Map pitch class to key names
            key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            features['key'] = key_names[dominant_pitch]
            
            # Mode detection (simplified heuristic)
            # Look at the distribution of major vs minor chord characteristics
            # This is a simplified approach - in practice, you'd want more sophisticated analysis
            
            # Calculate major and minor chord profiles
            major_profile = np.array([1, 0, 0.5, 0, 0.8, 0.3, 0, 1, 0, 0.5, 0, 0.8])
            minor_profile = np.array([1, 0, 0.5, 0.8, 0, 0.3, 0, 1, 0, 0.5, 0.8, 0])
            
            # Rotate profiles to match detected key
            major_profile_rotated = np.roll(major_profile, dominant_pitch)
            minor_profile_rotated = np.roll(minor_profile, dominant_pitch)
            
            # Calculate correlation with actual chroma
            major_corr = np.corrcoef(chroma_mean, major_profile_rotated)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_profile_rotated)[0, 1]
            
            features['mode'] = 'major' if major_corr > minor_corr else 'minor'
            
        except Exception as e:
            print(f"Error extracting musical structure: {e}")
            features.update({
                'key': 'unknown',
                'mode': 'unknown'
            })
        
        return features
    
    def extract_energy_dynamics(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract energy and dynamics features.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary of energy features
        """
        features = {}
        
        try:
            # RMS energy
            rms = librosa.feature.rms(y=audio)[0]
            features['energy'] = float(np.mean(rms))
            
            # Loudness (RMS to dB)
            rms_db = librosa.amplitude_to_db(rms, ref=np.max)
            features['loudness'] = float(np.mean(rms_db))
            
        except Exception as e:
            print(f"Error extracting energy features: {e}")
            features.update({
                'energy': 0.0,
                'loudness': 0.0
            })
        
        return features
    
    def extract_musicnn_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract MusicNN features (valence, danceability, instrumentalness, acousticness).
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary of MusicNN features
        """
        features = {}
        
        if not MUSICNN_AVAILABLE:
            features.update({
                'valence': 0.0,
                'danceability': 0.0,
                'instrumentalness': 0.0,
                'acousticness': 0.0
            })
            return features
        
        try:
            # Save audio temporarily for musicnn
            temp_file = "temp_audio.wav"
            sf.write(temp_file, audio, self.sample_rate)
            
            # Extract features with musicnn
            taggram, tags, features_musicnn = extractor(temp_file, model='MTT_musicnn', 
                                                       input_repr='melspectrogram', 
                                                       aggregation='mean')
            
            # Map tags to our desired features
            tag_mapping = {
                'valence': ['happy', 'energetic', 'positive'],
                'danceability': ['dance', 'rhythmic', 'groovy'],
                'instrumentalness': ['instrumental', 'no vocals'],
                'acousticness': ['acoustic', 'unplugged']
            }
            
            for feature_name, related_tags in tag_mapping.items():
                # Find the best matching tag
                best_score = 0.0
                for tag in related_tags:
                    if tag in tags:
                        tag_idx = list(tags).index(tag)
                        score = float(features_musicnn[tag_idx])
                        best_score = max(best_score, score)
                
                features[feature_name] = best_score
            
            # Clean up temp file
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
        except Exception as e:
            print(f"Error extracting MusicNN features: {e}")
            features.update({
                'valence': 0.0,
                'danceability': 0.0,
                'instrumentalness': 0.0,
                'acousticness': 0.0
            })
        
        return features
    
    def extract_yamnet_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Extract YAMNet features for speechiness detection.
        
        Args:
            audio: Audio data
            
        Returns:
            Dictionary of YAMNet features
        """
        features = {}
        
        if not YAMNET_AVAILABLE or self.yamnet_model is None:
            features['speechiness'] = 0.0
            return features
        
        try:
            # Prepare audio for YAMNet (needs to be float32)
            audio_float32 = audio.astype(np.float32)
            
            # Run YAMNet
            scores, embeddings, spectrogram = self.yamnet_model(audio_float32)
            
            # Find speech-related classes
            speech_classes = ['Speech', 'Male speech, man speaking', 'Female speech, woman speaking',
                            'Child speech, kid speaking', 'Conversation', 'Narration, monologue',
                            'Babbling', 'Speech synthesizer', 'Shout', 'Bellow', 'Whoop', 'Yell',
                            'Children shouting', 'Screaming', 'Whispering', 'Laughter', 'Baby laughter',
                            'Giggle', 'Snicker', 'Belly laugh', 'Chuckle, chortle']
            
            speech_scores = []
            for class_name in speech_classes:
                if class_name in self.yamnet_class_names:
                    class_idx = self.yamnet_class_names.index(class_name)
                    speech_scores.append(np.mean(scores[:, class_idx]))
            
            # Calculate speechiness as the maximum speech-related score
            features['speechiness'] = float(np.max(speech_scores)) if speech_scores else 0.0
            
        except Exception as e:
            print(f"Error extracting YAMNet features: {e}")
            features['speechiness'] = 0.0
        
        return features
    
    def extract_all_features(self, file_path: str) -> Dict[str, any]:
        """
        Extract all features from a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing all extracted features
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        # Initialize features dictionary
        features = {
            'filename': os.path.basename(file_path),
            'filepath': file_path,
            'duration': len(audio) / sr
        }
        
        # Extract all feature categories
        features.update(self.extract_rhythmic_features(audio))
        features.update(self.extract_musical_structure(audio))
        features.update(self.extract_energy_dynamics(audio))
        features.update(self.extract_musicnn_features(audio))
        features.update(self.extract_yamnet_features(audio))
        
        return features
    
    def process_directory(self, input_dir: str, output_file: str = None, 
                         use_sqlite: bool = False, db_path: str = None,
                         n_jobs: int = 1) -> pd.DataFrame:
        """
        Process all audio files in a directory.
        
        Args:
            input_dir: Directory containing audio files
            output_file: Path to save CSV output
            use_sqlite: Whether to save to SQLite database
            db_path: Path to SQLite database
            n_jobs: Number of parallel jobs
            
        Returns:
            DataFrame with extracted features
        """
        # Find all audio files
        audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']
        audio_files = []
        
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(input_dir, ext)))
            audio_files.extend(glob.glob(os.path.join(input_dir, '**', ext), recursive=True))
        
        if not audio_files:
            print(f"No audio files found in {input_dir}")
            return pd.DataFrame()
        
        print(f"Found {len(audio_files)} audio files")
        
        # Process files
        if n_jobs > 1:
            # Parallel processing
            with mp.Pool(n_jobs) as pool:
                results = pool.map(self.extract_all_features, audio_files)
        else:
            # Sequential processing
            results = []
            for i, file_path in enumerate(audio_files):
                print(f"Processing {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
                result = self.extract_all_features(file_path)
                results.append(result)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        if not results:
            print("No features extracted successfully")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        if output_file:
            df.to_csv(output_file, index=False)
            print(f"Features saved to {output_file}")
        
        if use_sqlite and db_path:
            self.save_to_sqlite(df, db_path)
        
        return df
    
    def save_to_sqlite(self, df: pd.DataFrame, db_path: str):
        """
        Save features to SQLite database.
        
        Args:
            df: DataFrame with features
            db_path: Path to SQLite database
        """
        try:
            conn = sqlite3.connect(db_path)
            df.to_sql('audio_features', conn, if_exists='replace', index=False)
            conn.close()
            print(f"Features saved to SQLite database: {db_path}")
        except Exception as e:
            print(f"Error saving to SQLite: {e}")


def main():
    """Command-line interface for the audio feature extractor."""
    parser = argparse.ArgumentParser(description='Extract audio features from local files')
    parser.add_argument('--input-dir', required=True, help='Directory containing audio files')
    parser.add_argument('--output', help='Output CSV file path')
    parser.add_argument('--db', help='SQLite database path')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel jobs')
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = AudioFeatureExtractor()
    
    # Process directory
    df = extractor.process_directory(
        input_dir=args.input_dir,
        output_file=args.output,
        use_sqlite=bool(args.db),
        db_path=args.db,
        n_jobs=args.jobs
    )
    
    if not df.empty:
        print(f"\nExtracted features for {len(df)} files")
        print("\nFeature summary:")
        print(df.describe())


if __name__ == "__main__":
    main() 