#!/usr/bin/env python3
"""
Enhanced Audio Analysis Script

This script provides enhanced audio analysis capabilities
with CSV output only - no visualizations.
"""

from audio_features_enhanced import EnhancedAudioAnalyzer

def main():
    """Analyze audio files with enhanced features and output to CSV."""
    print("🎵 Enhanced Audio Analysis - CSV Output Only")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = EnhancedAudioAnalyzer()
    
    # Analyze the music samples folder
    print("\n📊 Analyzing audio files with enhanced features...")
    df = analyzer.analyze_folder('music_samples', output_file='features_enhanced.csv')
    
    if df is None or df.empty:
        print("❌ No results to analyze")
        return
    
    print(f"\n✅ Successfully analyzed {len(df)} files")
    print(f"📄 Results saved to: features_enhanced.csv")
    
    # Show a quick summary
    print("\n📋 Quick Summary:")
    print("-" * 20)
    for idx, row in df.iterrows():
        print(f"🎼 {row['filename']}")
        print(f"   Tempo: {row['tempo']:.1f} BPM")
        print(f"   Valence: {row['valence']:.3f}")
        print(f"   Danceability: {row['danceability']:.3f}")
        print(f"   Instrumentalness: {row['instrumentalness']:.3f}")
        print(f"   Acousticness: {row['acousticness']:.3f}")
        print(f"   Speechiness: {row['speechiness']:.3f}")
        print(f"   Duration: {row['duration']:.1f}s")
    
    print("\n✅ Enhanced analysis complete!")

if __name__ == "__main__":
    main() 