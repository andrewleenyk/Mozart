#!/usr/bin/env python3
"""
Audio Features Example

This script demonstrates basic audio feature extraction capabilities
with CSV output and detailed feature explanations.
"""

import pandas as pd
from audio_features_simple import AudioFeatureExtractor

def main():
    """Demonstrate basic audio analysis."""
    print("🎵 Audio Features Analysis")
    print("=" * 40)
    
    # Initialize extractor
    extractor = AudioFeatureExtractor()
    
    # Analyze the music samples folder
    print("\n📊 Analyzing audio files...")
    df = extractor.process_directory(
        input_dir='music_samples',
        output_file='features_demo.csv',
        use_sqlite=False
    )
    
    if df is None or df.empty:
        print("❌ No results to analyze")
        return
    
    print(f"\n✅ Successfully analyzed {len(df)} files")
    
    # Display detailed results
    print("\n📈 Detailed Feature Analysis:")
    print("-" * 30)
    
    for idx, row in df.iterrows():
        print(f"\n🎼 File: {row['filename']}")
        print(f"   Duration: {row['duration']:.1f} seconds")
        
        # Rhythm Analysis
        print(f"\n🥁 Rhythm Features:")
        print(f"   Tempo: {row['tempo']:.1f} BPM")
        print(f"   Beat Strength: {row['beat_strength']:.3f}")
        print(f"   Rhythmic Stability: {row['rhythmic_stability']:.3f}")
        print(f"   Regularity: {row['regularity']:.3f}")
        
        # Musical Structure
        print(f"\n🎼 Musical Structure:")
        print(f"   Key: {row['key']}")
        print(f"   Mode: {row['mode']}")
        
        # Energy & Dynamics
        print(f"\n⚡ Energy & Dynamics:")
        print(f"   Energy: {row['energy']:.6f}")
        print(f"   Loudness: {row['loudness']:.1f} dB")
        print(f"   Brightness: {row['brightness']:.1f} Hz")
        print(f"   High Freq Content: {row['high_freq_content']:.1f} Hz")
        
        # Spotify-style Features
        print(f"\n🎵 Spotify-style Features:")
        print(f"   Valence: {row['valence']:.3f} (positive emotion)")
        print(f"   Danceability: {row['danceability']:.3f}")
        print(f"   Instrumentalness: {row['instrumentalness']:.3f}")
        print(f"   Acousticness: {row['acousticness']:.3f}")
        print(f"   Speechiness: {row['speechiness']:.3f}")
        
        # Spectral Features
        print(f"\n🌊 Spectral Features:")
        print(f"   Spectral Centroid Mean: {row['spectral_centroid_mean']:.1f} Hz")
        print(f"   Spectral Centroid Std: {row['spectral_centroid_std']:.1f} Hz")
        print(f"   Spectral Rolloff Mean: {row['spectral_rolloff_mean']:.1f} Hz")
        print(f"   Spectral Rolloff Std: {row['spectral_rolloff_std']:.1f} Hz")
        print(f"   Spectral Bandwidth Mean: {row['spectral_bandwidth_mean']:.1f} Hz")
        print(f"   Spectral Bandwidth Std: {row['spectral_bandwidth_std']:.1f} Hz")
        
        # MFCC Features
        print(f"\n🎛️ MFCC Features:")
        print(f"   MFCC Mean: {row['mfcc_mean']:.3f}")
        print(f"   MFCC Std: {row['mfcc_std']:.3f}")
        print(f"   Mel Energy Mean: {row['mel_energy_mean']:.6f}")
    
    # Create a summary table
    print("\n📋 Feature Summary Table:")
    print("=" * 80)
    
    summary_data = {
        'Category': ['Rhythm', 'Structure', 'Energy', 'Spotify-style'],
        'Tempo (BPM)': [f"{df['tempo'].iloc[0]:.1f}", '', '', ''],
        'Key': ['', df['key'].iloc[0], '', ''],
        'Energy': ['', '', f"{df['energy'].iloc[0]:.6f}", ''],
        'Valence': ['', '', '', f"{df['valence'].iloc[0]:.3f}"],
        'Danceability': ['', '', '', f"{df['danceability'].iloc[0]:.3f}"]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print("\n✅ Analysis complete! Check the generated CSV file.")

if __name__ == "__main__":
    main() 