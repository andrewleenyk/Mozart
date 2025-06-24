#!/usr/bin/env python3
"""
Example script demonstrating the Audio Feature Extraction Module

This script shows how to:
1. Extract features from a single audio file
2. Process a directory of audio files
3. Analyze and visualize the results
4. Save results to different formats
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from audio_features_simple import AudioFeatureExtractor

def main():
    """Main example function."""
    
    print("🎵 Audio Feature Extraction Example")
    print("=" * 50)
    
    # Initialize the feature extractor
    extractor = AudioFeatureExtractor(sample_rate=22050)
    
    # Example 1: Extract features from a single file
    print("\n1. Single File Analysis")
    print("-" * 30)
    
    # Replace with path to your audio file
    sample_file = "sample_audio.mp3"  # Change this to your audio file
    
    if os.path.exists(sample_file):
        print(f"Analyzing: {sample_file}")
        features = extractor.extract_all_features(sample_file)
        
        if features:
            print("\nExtracted Features:")
            for key, value in features.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("Failed to extract features from file")
    else:
        print(f"Sample file '{sample_file}' not found. Skipping single file analysis.")
    
    # Example 2: Process a directory
    print("\n2. Directory Processing")
    print("-" * 30)
    
    # Replace with path to your music directory
    music_dir = "music_samples"  # Change this to your music directory
    
    if os.path.exists(music_dir):
        print(f"Processing directory: {music_dir}")
        
        # Process with parallel processing
        df = extractor.process_directory(
            input_dir=music_dir,
            output_file="extracted_features.csv",
            use_sqlite=True,
            db_path="audio_features.db",
            n_jobs=2  # Use 2 parallel processes
        )
        
        if not df.empty:
            print(f"\nSuccessfully processed {len(df)} files!")
            
            # Example 3: Basic analysis
            print("\n3. Basic Analysis")
            print("-" * 30)
            
            # Show summary statistics
            print("\nSummary Statistics:")
            print(df.describe())
            
            # Show musical key distribution
            print(f"\nKey Distribution:")
            key_counts = df['key'].value_counts()
            print(key_counts)
            
            # Show mode distribution
            print(f"\nMode Distribution:")
            mode_counts = df['mode'].value_counts()
            print(mode_counts)
            
            # Find highest and lowest valence songs
            if 'valence' in df.columns:
                max_valence_idx = df['valence'].idxmax()
                min_valence_idx = df['valence'].idxmin()
                
                print(f"\nHappiest song: {df.loc[max_valence_idx, 'filename']} (valence: {df.loc[max_valence_idx, 'valence']:.3f})")
                print(f"Most melancholic song: {df.loc[min_valence_idx, 'filename']} (valence: {df.loc[min_valence_idx, 'valence']:.3f})")
            
            # Find most danceable song
            if 'danceability' in df.columns:
                max_dance_idx = df['danceability'].idxmax()
                print(f"Most danceable song: {df.loc[max_dance_idx, 'filename']} (danceability: {df.loc[max_dance_idx, 'danceability']:.3f})")
            
            # Example 4: Visualization (if matplotlib is available)
            try:
                print("\n4. Creating Visualizations")
                print("-" * 30)
                
                # Set up the plotting style
                plt.style.use('default')
                sns.set_palette("husl")
                
                # Create a figure with multiple subplots
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                fig.suptitle('Audio Feature Analysis', fontsize=16)
                
                # 1. Tempo distribution
                if 'tempo' in df.columns:
                    axes[0, 0].hist(df['tempo'], bins=20, alpha=0.7, edgecolor='black')
                    axes[0, 0].set_title('Tempo Distribution')
                    axes[0, 0].set_xlabel('BPM')
                    axes[0, 0].set_ylabel('Count')
                
                # 2. Key distribution
                if 'key' in df.columns:
                    key_counts.plot(kind='bar', ax=axes[0, 1])
                    axes[0, 1].set_title('Key Distribution')
                    axes[0, 1].set_xlabel('Key')
                    axes[0, 1].set_ylabel('Count')
                    axes[0, 1].tick_params(axis='x', rotation=45)
                
                # 3. Valence vs Danceability scatter plot
                if 'valence' in df.columns and 'danceability' in df.columns:
                    axes[0, 2].scatter(df['valence'], df['danceability'], alpha=0.6)
                    axes[0, 2].set_title('Valence vs Danceability')
                    axes[0, 2].set_xlabel('Valence')
                    axes[0, 2].set_ylabel('Danceability')
                
                # 4. Energy vs Loudness
                if 'energy' in df.columns and 'loudness' in df.columns:
                    axes[1, 0].scatter(df['energy'], df['loudness'], alpha=0.6)
                    axes[1, 0].set_title('Energy vs Loudness')
                    axes[1, 0].set_xlabel('Energy')
                    axes[1, 0].set_ylabel('Loudness (dB)')
                
                # 5. Feature correlation heatmap
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 1:
                    correlation_matrix = df[numeric_cols].corr()
                    sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', ax=axes[1, 1])
                    axes[1, 1].set_title('Feature Correlations')
                
                # 6. Duration distribution
                if 'duration' in df.columns:
                    axes[1, 2].hist(df['duration'], bins=20, alpha=0.7, edgecolor='black')
                    axes[1, 2].set_title('Duration Distribution')
                    axes[1, 2].set_xlabel('Duration (seconds)')
                    axes[1, 2].set_ylabel('Count')
                
                plt.tight_layout()
                plt.savefig('audio_features_analysis.png', dpi=300, bbox_inches='tight')
                print("Visualization saved as 'audio_features_analysis.png'")
                
                # Show the plot
                plt.show()
                
            except ImportError:
                print("Matplotlib/Seaborn not available. Skipping visualizations.")
            except Exception as e:
                print(f"Error creating visualizations: {e}")
            
            # Example 5: Advanced queries
            print("\n5. Advanced Analysis")
            print("-" * 30)
            
            # Find songs in a specific key
            if 'key' in df.columns:
                c_major_songs = df[df['key'] == 'C']
                if not c_major_songs.empty:
                    print(f"\nSongs in C major: {len(c_major_songs)}")
                    for _, song in c_major_songs.iterrows():
                        print(f"  - {song['filename']}")
            
            # Find high-energy songs
            if 'energy' in df.columns:
                high_energy_threshold = df['energy'].quantile(0.8)
                high_energy_songs = df[df['energy'] > high_energy_threshold]
                print(f"\nHigh-energy songs (top 20%): {len(high_energy_songs)}")
                for _, song in high_energy_songs.iterrows():
                    print(f"  - {song['filename']} (energy: {song['energy']:.3f})")
            
            # Find instrumental songs
            if 'instrumentalness' in df.columns:
                instrumental_threshold = 0.7
                instrumental_songs = df[df['instrumentalness'] > instrumental_threshold]
                print(f"\nInstrumental songs (instrumentalness > 0.7): {len(instrumental_songs)}")
                for _, song in instrumental_songs.iterrows():
                    print(f"  - {song['filename']} (instrumentalness: {song['instrumentalness']:.3f})")
            
        else:
            print("No features were extracted successfully.")
    else:
        print(f"Music directory '{music_dir}' not found. Skipping directory processing.")
    
    print("\n" + "=" * 50)
    print("Example completed! Check the generated files:")
    print("- extracted_features.csv (CSV output)")
    print("- audio_features.db (SQLite database)")
    print("- audio_features_analysis.png (visualizations)")


if __name__ == "__main__":
    main() 