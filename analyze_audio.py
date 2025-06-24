#!/usr/bin/env python3
"""
Simple Audio Analysis Script

This script provides a clean, simple interface for analyzing audio files
and outputting results to CSV format only.
"""

from audio_features_simple import AudioFeatureExtractor

def main():
    """Analyze audio files and output to CSV."""
    print("🎵 Audio Analysis - CSV Output Only")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = AudioFeatureExtractor()
    
    # Analyze the music samples folder
    print("\n📊 Analyzing audio files...")
    df = analyzer.process_directory('music_samples', output_file='features_simple.csv')
    
    if df is None or df.empty:
        print("❌ No results to analyze")
        return
    
    print(f"\n✅ Successfully analyzed {len(df)} files")
    print(f"📄 Results saved to: features_simple.csv")
    
    # Show a quick summary
    print("\n📋 Quick Summary:")
    print("-" * 20)
    for idx, row in df.iterrows():
        print(f"🎼 {row['filename']}")
        print(f"   Tempo: {row['tempo']:.1f} BPM")
        print(f"   Valence: {row['valence']:.3f}")
        print(f"   Danceability: {row['danceability']:.3f}")
        print(f"   Duration: {row['duration']:.1f}s")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main() 