#!/usr/bin/env python3
"""
Advanced Audio Features Example

This script demonstrates the advanced audio analysis capabilities
with detailed feature explanations and visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from audio_features_advanced import AdvancedAudioAnalyzer
import numpy as np

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
    
    # Create visualizations
    print("\n📊 Creating visualizations...")
    create_visualizations(df)
    
    print("\n✅ Analysis complete! Check the generated plots and CSV file.")

def create_visualizations(df):
    """Create comprehensive visualizations of the audio features."""
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Advanced Audio Features Analysis', fontsize=16, fontweight='bold')
    
    # 1. Spotify-style Features Radar Chart
    spotify_features = ['valence', 'danceability', 'instrumentalness', 'acousticness', 'speechiness']
    spotify_values = [df[feature].iloc[0] for feature in spotify_features]
    
    angles = np.linspace(0, 2 * np.pi, len(spotify_features), endpoint=False).tolist()
    spotify_values += spotify_values[:1]  # Complete the circle
    angles += angles[:1]
    
    ax1 = plt.subplot(2, 3, 1, projection='polar')
    ax1.plot(angles, spotify_values, 'o-', linewidth=2, label='Spotify Features')
    ax1.fill(angles, spotify_values, alpha=0.25)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(spotify_features)
    ax1.set_ylim(0, 1)
    ax1.set_title('Spotify-style Features', pad=20)
    ax1.grid(True)
    
    # 2. Rhythm Features
    rhythm_features = ['tempo', 'beat_strength', 'rhythm_regularity', 'rhythmic_complexity']
    rhythm_values = [df[feature].iloc[0] for feature in rhythm_features]
    
    ax2 = plt.subplot(2, 3, 2)
    bars = ax2.bar(rhythm_features, rhythm_values, color=sns.color_palette("husl", 4))
    ax2.set_title('Rhythm Features')
    ax2.set_ylabel('Value')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, rhythm_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 3. Harmonic vs Percussive Energy
    ax3 = plt.subplot(2, 3, 3)
    harmonic_energy = df['harmonic_energy'].iloc[0]
    percussive_energy = df['percussive_energy'].iloc[0]
    
    ax3.pie([harmonic_energy, percussive_energy], 
            labels=['Harmonic', 'Percussive'], 
            autopct='%1.1f%%', startangle=90)
    ax3.set_title('Energy Distribution')
    
    # 4. Spectral Features
    spectral_features = ['spectral_centroid_mean', 'spectral_rolloff_mean', 
                        'spectral_bandwidth_mean', 'spectral_flatness_mean']
    spectral_values = [df[feature].iloc[0] for feature in spectral_features]
    
    ax4 = plt.subplot(2, 3, 4)
    bars = ax4.bar(spectral_features, spectral_values, color=sns.color_palette("husl", 4))
    ax4.set_title('Spectral Features')
    ax4.set_ylabel('Value')
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, spectral_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(spectral_values)*0.01,
                f'{value:.1f}', ha='center', va='bottom')
    
    # 5. Energy & Dynamics
    energy_features = ['energy_mean', 'loudness', 'dynamic_range', 'zero_crossing_rate']
    energy_values = [df[feature].iloc[0] for feature in energy_features]
    
    ax5 = plt.subplot(2, 3, 5)
    bars = ax5.bar(energy_features, energy_values, color=sns.color_palette("husl", 4))
    ax5.set_title('Energy & Dynamics')
    ax5.set_ylabel('Value')
    plt.setp(ax5.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, energy_values):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(energy_values)*0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    # 6. Confidence Scores
    confidence_features = ['harmonic_confidence', 'rhythm_confidence', 'key_confidence', 'overall_confidence']
    confidence_values = [df[feature].iloc[0] for feature in confidence_features]
    
    ax6 = plt.subplot(2, 3, 6)
    bars = ax6.bar(confidence_features, confidence_values, color=sns.color_palette("husl", 4))
    ax6.set_title('Confidence Scores')
    ax6.set_ylabel('Confidence')
    ax6.set_ylim(0, 1)
    plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, confidence_values):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('advanced_audio_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
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

if __name__ == "__main__":
    main() 