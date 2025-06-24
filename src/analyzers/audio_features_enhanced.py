"""
Enhanced Audio Feature Extraction Module

Extracts comprehensive audio features from local .mp3 and .wav files using offline tools.
Includes high-level Spotify-style features using alternative approaches.
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
from scipy import stats
from scipy.signal import correlate
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


class EnhancedAudioFeatureExtractor:
    """Extract comprehensive audio features including high-level Spotify-style features."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the enhanced feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
        print("✅ Enhanced Audio Feature Extractor initialized")
    
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
    
    def extract_enhanced_valence(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract valence using advanced spectral and temporal analysis.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (valence_score, confidence)
        """
        try:
            # Spectral features for valence
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            # Zero crossing rate (noisiness)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            
            # Chroma features for harmonic content
            chroma = librosa.feature.chroma_cqt(y=audio, sr=self.sample_rate)
            
            # Calculate valence indicators
            # Higher spectral centroid often correlates with positive valence
            brightness_score = np.mean(spectral_centroid) / 4000.0
            
            # Lower zero crossing rate often indicates less noise (more positive)
            noise_score = 1.0 - np.clip(np.mean(zcr), 0.0, 0.5) / 0.5
            
            # Harmonic content (major vs minor characteristics)
            chroma_mean = np.mean(chroma, axis=1)
            major_profile = np.array([1, 0, 0.5, 0, 0.8, 0.3, 0, 1, 0, 0.5, 0, 0.8])
            minor_profile = np.array([1, 0, 0.5, 0.8, 0, 0.3, 0, 1, 0, 0.5, 0.8, 0])
            
            major_corr = np.corrcoef(chroma_mean, major_profile)[0, 1]
            minor_corr = np.corrcoef(chroma_mean, minor_profile)[0, 1]
            harmonic_score = max(0, major_corr - minor_corr)
            
            # Combine scores
            valence_score = (brightness_score + noise_score + harmonic_score) / 3.0
            valence_score = np.clip(valence_score, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([brightness_score, noise_score, harmonic_score])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(valence_score), float(confidence)
            
        except Exception as e:
            print(f"Error extracting valence: {e}")
            return 0.5, 0.0
    
    def extract_enhanced_danceability(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract danceability using advanced rhythmic analysis.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (danceability_score, confidence)
        """
        try:
            # Tempo analysis
            tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Onset strength
            onset_strength = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
            
            # Beat tracking
            tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)
            
            # Calculate danceability indicators
            # Tempo score (optimal range 120-140 BPM)
            tempo_score = 0.0
            if 120 <= tempo <= 140:
                tempo_score = 1.0
            elif 100 <= tempo <= 160:
                tempo_score = 0.8
            elif 80 <= tempo <= 180:
                tempo_score = 0.6
            else:
                tempo_score = 0.3
            
            # Beat strength
            beat_strength = np.mean(onset_strength)
            strength_score = np.clip(beat_strength / 10.0, 0.0, 1.0)
            
            # Beat regularity
            if len(beats) > 1:
                beat_intervals = np.diff(beats)
                regularity = 1.0 - np.clip(np.std(beat_intervals) / np.mean(beat_intervals), 0.0, 1.0)
            else:
                regularity = 0.0
            
            # Spectral flux (energy changes)
            spectral_flux = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=512)
            flux_score = np.clip(np.mean(spectral_flux) / 5.0, 0.0, 1.0)
            
            # Combine scores
            danceability = (tempo_score + strength_score + regularity + flux_score) / 4.0
            danceability = np.clip(danceability, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([tempo_score, strength_score, regularity, flux_score])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(danceability), float(confidence)
            
        except Exception as e:
            print(f"Error extracting danceability: {e}")
            return 0.5, 0.0
    
    def extract_enhanced_instrumentalness(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract instrumentalness using harmonic/percussive separation.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (instrumentalness_score, confidence)
        """
        try:
            # Harmonic/percussive separation
            harmonic, percussive = librosa.effects.hpss(audio)
            
            # Calculate energy ratios
            harmonic_energy = np.sum(harmonic ** 2)
            percussive_energy = np.sum(percussive ** 2)
            total_energy = harmonic_energy + percussive_energy
            
            if total_energy > 0:
                harmonic_ratio = harmonic_energy / total_energy
            else:
                harmonic_ratio = 0.5
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # MFCC features for timbral analysis
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            # Zero crossing rate (vocal content often has higher ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_score = 1.0 - np.clip(np.mean(zcr), 0.0, 0.3) / 0.3
            
            # Spectral features for instrumental detection
            # Instruments often have more consistent spectral characteristics
            spectral_consistency = 1.0 - np.clip(np.std(spectral_centroid) / np.mean(spectral_centroid), 0.0, 1.0)
            
            # Combine scores
            instrumentalness = (harmonic_ratio + zcr_score + spectral_consistency) / 3.0
            instrumentalness = np.clip(instrumentalness, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([harmonic_ratio, zcr_score, spectral_consistency])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(instrumentalness), float(confidence)
            
        except Exception as e:
            print(f"Error extracting instrumentalness: {e}")
            return 0.5, 0.0
    
    def extract_enhanced_acousticness(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract acousticness using spectral characteristics.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (acousticness_score, confidence)
        """
        try:
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sample_rate)[0]
            
            # MFCC features
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            # Calculate acousticness indicators
            # Lower spectral centroid often indicates acoustic instruments
            centroid_score = 1.0 - np.clip(np.mean(spectral_centroid) / 4000.0, 0.0, 1.0)
            
            # Lower spectral rolloff often indicates acoustic instruments
            rolloff_score = 1.0 - np.clip(np.mean(spectral_rolloff) / 8000.0, 0.0, 1.0)
            
            # Spectral bandwidth (acoustic instruments often have narrower bandwidth)
            bandwidth_score = 1.0 - np.clip(np.mean(spectral_bandwidth) / 2000.0, 0.0, 1.0)
            
            # MFCC characteristics (acoustic instruments have different timbral characteristics)
            mfcc_variance = np.var(mfcc, axis=1)
            mfcc_score = 1.0 - np.clip(np.mean(mfcc_variance) / 100.0, 0.0, 1.0)
            
            # Combine scores
            acousticness = (centroid_score + rolloff_score + bandwidth_score + mfcc_score) / 4.0
            acousticness = np.clip(acousticness, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([centroid_score, rolloff_score, bandwidth_score, mfcc_score])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(acousticness), float(confidence)
            
        except Exception as e:
            print(f"Error extracting acousticness: {e}")
            return 0.5, 0.0
    
    def extract_enhanced_speechiness(self, audio: np.ndarray) -> Tuple[float, float]:
        """
        Extract speechiness using spectral and temporal characteristics.
        
        Args:
            audio: Audio data
            
        Returns:
            Tuple of (speechiness_score, confidence)
        """
        try:
            # Zero crossing rate (speech has higher ZCR)
            zcr = librosa.feature.zero_crossing_rate(audio)[0]
            zcr_score = np.clip(np.mean(zcr) / 0.3, 0.0, 1.0)
            
            # Spectral features
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sample_rate)[0]
            
            # MFCC features (speech has characteristic MFCC patterns)
            mfcc = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
            
            # Spectral flux (speech has more rapid spectral changes)
            spectral_flux = librosa.onset.onset_strength(y=audio, sr=self.sample_rate, hop_length=512)
            flux_score = np.clip(np.mean(spectral_flux) / 10.0, 0.0, 1.0)
            
            # Spectral characteristics for speech detection
            # Speech often has lower spectral centroid than music
            centroid_score = 1.0 - np.clip(np.mean(spectral_centroid) / 3000.0, 0.0, 1.0)
            
            # Speech often has more consistent spectral rolloff
            rolloff_consistency = 1.0 - np.clip(np.std(spectral_rolloff) / np.mean(spectral_rolloff), 0.0, 1.0)
            
            # Combine scores
            speechiness = (zcr_score + flux_score + centroid_score + rolloff_consistency) / 4.0
            speechiness = np.clip(speechiness, 0.0, 1.0)
            
            # Confidence based on feature consistency
            confidence = np.std([zcr_score, flux_score, centroid_score, rolloff_consistency])
            confidence = 1.0 - np.clip(confidence, 0.0, 0.5) / 0.5
            
            return float(speechiness), float(confidence)
            
        except Exception as e:
            print(f"Error extracting speechiness: {e}")
            return 0.0, 0.0
    
    def extract_enhanced_features(self, file_path: str) -> Dict[str, any]:
        """
        Extract enhanced features from a single audio file.
        
        Args:
            file_path: Path to audio file
            
        Returns:
            Dictionary containing enhanced features
        """
        # Load audio
        audio, sr = self.load_audio(file_path)
        if audio is None:
            return None
        
        # Initialize features dictionary
        features = {
            'filepath': file_path,
            'filename': os.path.basename(file_path)
        }
        
        # Extract enhanced features
        valence, valence_conf = self.extract_enhanced_valence(audio)
        danceability, dance_conf = self.extract_enhanced_danceability(audio)
        instrumentalness, inst_conf = self.extract_enhanced_instrumentalness(audio)
        acousticness, acous_conf = self.extract_enhanced_acousticness(audio)
        speechiness, speech_conf = self.extract_enhanced_speechiness(audio)
        
        # Add features with confidence scores
        features.update({
            'valence': valence,
            'valence_confidence': valence_conf,
            'danceability': danceability,
            'danceability_confidence': dance_conf,
            'instrumentalness': instrumentalness,
            'instrumentalness_confidence': inst_conf,
            'acousticness': acousticness,
            'acousticness_confidence': acous_conf,
            'speechiness': speechiness,
            'speechiness_confidence': speech_conf
        })
        
        return features
    
    def process_existing_features(self, features_csv: str, output_file: str = None,
                                 use_sqlite: bool = False, db_path: str = None,
                                 n_jobs: int = 1, replace_basic: bool = True) -> pd.DataFrame:
        """
        Process existing features CSV and add high-level features.
        
        Args:
            features_csv: Path to existing features CSV file
            output_file: Path to save enhanced CSV output
            use_sqlite: Whether to save to SQLite database
            db_path: Path to SQLite database
            n_jobs: Number of parallel jobs
            replace_basic: If True, replace basic features with enhanced versions
            
        Returns:
            DataFrame with enhanced features
        """
        # Load existing features
        if not os.path.exists(features_csv):
            print(f"Features file {features_csv} not found!")
            return pd.DataFrame()
        
        print(f"Loading existing features from {features_csv}")
        df = pd.read_csv(features_csv)
        
        if 'filepath' not in df.columns:
            print("Error: 'filepath' column not found in features CSV")
            return pd.DataFrame()
        
        print(f"Found {len(df)} tracks to process")
        
        # Process files
        if n_jobs > 1:
            # Parallel processing
            with mp.Pool(n_jobs) as pool:
                results = pool.map(self.extract_enhanced_features, df['filepath'].tolist())
        else:
            # Sequential processing
            results = []
            for i, file_path in enumerate(df['filepath']):
                print(f"Processing {i+1}/{len(df)}: {os.path.basename(file_path)}")
                result = self.extract_enhanced_features(file_path)
                results.append(result)
        
        # Filter out None results
        results = [r for r in results if r is not None]
        
        if not results:
            print("No enhanced features extracted successfully")
            return df
        
        # Create enhanced features DataFrame
        enhanced_df = pd.DataFrame(results)
        
        if replace_basic:
            # Remove basic features that will be replaced by enhanced versions
            basic_features_to_replace = ['valence', 'danceability', 'instrumentalness', 'acousticness', 'speechiness']
            for feature in basic_features_to_replace:
                if feature in df.columns:
                    df = df.drop(feature, axis=1)
                    print(f"Removed basic {feature} feature (will use enhanced version)")
            
            # Also remove any existing confidence columns to avoid duplicates
            confidence_columns_to_remove = ['valence_confidence', 'danceability_confidence', 'instrumentalness_confidence', 'acousticness_confidence', 'speechiness_confidence']
            for col in confidence_columns_to_remove:
                if col in df.columns:
                    df = df.drop(col, axis=1)
                    print(f"Removed existing {col} column (will use enhanced version)")
            
            # Merge enhanced features, replacing basic ones
            merged_df = pd.merge(df, enhanced_df, on='filepath', how='left')
            
            # Remove duplicate filename columns
            if 'filename_enhanced' in merged_df.columns:
                merged_df = merged_df.drop('filename_enhanced', axis=1)
        else:
            # Keep both basic and enhanced features (original behavior)
            merged_df = pd.merge(df, enhanced_df, on='filepath', how='left', suffixes=('', '_enhanced'))
            
            # Remove duplicate filename columns
            if 'filename_enhanced' in merged_df.columns:
                merged_df = merged_df.drop('filename_enhanced', axis=1)
        
        # Save results
        if output_file:
            merged_df.to_csv(output_file, index=False)
            print(f"Enhanced features saved to {output_file}")
        
        if use_sqlite and db_path:
            self.save_to_sqlite(merged_df, db_path)
        
        return merged_df
    
    def save_to_sqlite(self, df: pd.DataFrame, db_path: str):
        """
        Save features to SQLite database.
        
        Args:
            df: DataFrame with features
            db_path: Path to SQLite database
        """
        try:
            conn = sqlite3.connect(db_path)
            df.to_sql('audio_features_enhanced', conn, if_exists='replace', index=False)
            conn.close()
            print(f"Enhanced features saved to SQLite database: {db_path}")
        except Exception as e:
            print(f"Error saving to SQLite: {e}")


def main():
    """Command-line interface for the enhanced audio feature extractor."""
    parser = argparse.ArgumentParser(description='Extract enhanced audio features from existing features CSV')
    parser.add_argument('--input-csv', required=True, help='Path to existing features CSV file')
    parser.add_argument('--output', help='Output CSV file path (default: features_enhanced.csv)')
    parser.add_argument('--db', help='SQLite database path')
    parser.add_argument('--jobs', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--keep-basic', action='store_true', help='Keep basic features alongside enhanced ones (default: replace basic with enhanced)')
    
    args = parser.parse_args()
    
    # Set default output file if not specified
    if not args.output:
        args.output = 'features_enhanced.csv'
    
    # Initialize extractor
    extractor = EnhancedAudioFeatureExtractor()
    
    # Process existing features
    df = extractor.process_existing_features(
        features_csv=args.input_csv,
        output_file=args.output,
        use_sqlite=bool(args.db),
        db_path=args.db,
        n_jobs=args.jobs,
        replace_basic=not args.keep_basic
    )
    
    if not df.empty:
        print(f"\nEnhanced features for {len(df)} files")
        
        # Show summary of enhanced features
        enhanced_columns = ['valence', 'danceability', 'instrumentalness', 'acousticness', 'speechiness']
        available_columns = [col for col in enhanced_columns if col in df.columns]
        
        if available_columns:
            print("\nEnhanced features summary:")
            print(df[available_columns].describe())
        
        # Show confidence scores if available
        confidence_columns = [col for col in df.columns if 'confidence' in col]
        if confidence_columns:
            print(f"\nConfidence scores available for: {confidence_columns}")
        
        # Show which features were replaced if applicable
        if not args.keep_basic:
            print("\nNote: Basic features (valence, danceability, instrumentalness, acousticness, speechiness) were replaced with enhanced versions")
        else:
            print("\nNote: Both basic and enhanced features are included in the output")


if __name__ == "__main__":
    main() 