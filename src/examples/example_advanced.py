#!/usr/bin/env python3
"""
Advanced Audio Features Example

This script demonstrates the advanced audio analysis capabilities
with detailed feature explanations and CSV output.
"""

import pandas as pd
from audio_features_advanced import AdvancedAudioAnalyzer

def main():
    """Demonstrate advanced audio analysis."""
    print("🎵 Advanced Audio Features Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = AdvancedAudioAnalyzer()
    
    # Analyze the music samples folder
    print("\n📊 Analyzing audio files...")
    df = analyzer.analyze_folder('music_samples', output_file='features_advanced_demo.csv')
    
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
        print(f"   Sample Rate: {row['sample_rate']} Hz")
        
        # Rhythm Analysis
        print(f"\n🥁 Rhythm Features:")
        print(f"   Tempo: {row['tempo']:.1f} BPM")
        print(f"   Beat Strength: {row['beat_strength']:.3f}")
        print(f"   Rhythm Regularity: {row['rhythm_regularity']:.3f}")
        print(f"   Rhythmic Complexity: {row['rhythmic_complexity']:.3f}")
        print(f"   Syncopation: {row['syncopation']:.3f}")
        print(f"   Beat Count: {row['beat_count']}")
        
        # Harmonic Analysis
        print(f"\n🎹 Harmonic Features:")
        print(f"   Harmonic Ratio: {row['harmonic_ratio']:.3f}")
        print(f"   Harmonic Energy: {row['harmonic_energy']:.6f}")
        print(f"   Percussive Energy: {row['percussive_energy']:.6f}")
        print(f"   Harmonic Entropy: {row['harmonic_entropy']:.3f}")
        
        # Musical Structure
        print(f"\n🎼 Musical Structure:")
        print(f"   Key Strength: {row['key_strength']:.3f}")
        print(f"   Key Confidence: {row['key_confidence']:.3f}")
        print(f"   Mode: {row['mode']}")
        print(f"   Mode Confidence: {row['mode_confidence']:.3f}")
        
        # Energy & Dynamics
        print(f"\n⚡ Energy & Dynamics:")
        print(f"   Energy Mean: {row['energy_mean']:.6f}")
        print(f"   Loudness: {row['loudness']:.1f} dB")
        print(f"   Dynamic Range: {row['dynamic_range']:.6f}")
        print(f"   Energy Flux: {row['energy_flux']:.2e}")
        
        # Spotify-style Features
        print(f"\n🎵 Spotify-style Features:")
        print(f"   Valence: {row['valence']:.3f} (positive emotion)")
        print(f"   Danceability: {row['danceability']:.3f}")
        print(f"   Instrumentalness: {row['instrumentalness']:.3f}")
        print(f"   Acousticness: {row['acousticness']:.3f}")
        print(f"   Speechiness: {row['speechiness']:.3f}")
        
        # Spectral Features
        print(f"\n🌊 Spectral Features:")
        print(f"   Spectral Centroid: {row['spectral_centroid_mean']:.1f} Hz")
        print(f"   Spectral Rolloff: {row['spectral_rolloff_mean']:.1f} Hz")
        print(f"   Spectral Bandwidth: {row['spectral_bandwidth_mean']:.1f} Hz")
        print(f"   Spectral Flatness: {row['spectral_flatness_mean']:.3f}")
        print(f"   Spectral Flux: {row['spectral_flux']:.3f}")
        
        # Confidence Scores
        print(f"\n🎯 Confidence Scores:")
        print(f"   Harmonic Confidence: {row['harmonic_confidence']:.3f}")
        print(f"   Rhythm Confidence: {row['rhythm_confidence']:.3f}")
        print(f"   Overall Confidence: {row['overall_confidence']:.3f}")
    
    # Create a summary table
    print("\n📋 Feature Summary Table:")
    print("=" * 80)
    
    summary_data = {
        'Category': ['Rhythm', 'Harmony', 'Energy', 'Structure', 'Spotify-style'],
        'Tempo (BPM)': [f"{df['tempo'].iloc[0]:.1f}", '', '', '', ''],
        'Harmonic Ratio': ['', f"{df['harmonic_ratio'].iloc[0]:.3f}", '', '', ''],
        'Energy Mean': ['', '', f"{df['energy_mean'].iloc[0]:.6f}", '', ''],
        'Key Confidence': ['', '', '', f"{df['key_confidence'].iloc[0]:.3f}", ''],
        'Valence': ['', '', '', '', f"{df['valence'].iloc[0]:.3f}"],
        'Danceability': ['', '', '', '', f"{df['danceability'].iloc[0]:.3f}"],
        'Overall Confidence': [f"{df['overall_confidence'].iloc[0]:.3f}", '', '', '', '']
    }
    
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    
    print("\n✅ Analysis complete! Check the generated CSV file.")

if __name__ == "__main__":
    main() 