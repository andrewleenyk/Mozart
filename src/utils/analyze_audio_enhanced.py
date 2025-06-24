#!/usr/bin/env python3
"""
Enhanced Audio Analysis Script

This script provides enhanced audio analysis capabilities
with CSV output only - no visualizations.
"""

from audio_features_simple import AudioFeatureExtractor
from audio_features_enhanced import EnhancedAudioFeatureExtractor

def main():
    """Analyze audio files with enhanced features and output to CSV."""
    print("🎵 Enhanced Audio Analysis - CSV Output Only")
    print("=" * 50)
    
    # Step 1: Get basic features
    print("\n📊 Step 1: Extracting basic audio features...")
    basic_analyzer = AudioFeatureExtractor()
    basic_df = basic_analyzer.process_directory('music_samples', output_file='features_basic.csv')
    
    if basic_df is None or basic_df.empty:
        print("❌ No basic features extracted")
        return
    
    print(f"✅ Basic features extracted for {len(basic_df)} files")
    
    # Step 2: Enhance with advanced features
    print("\n📊 Step 2: Adding enhanced features...")
    enhanced_analyzer = EnhancedAudioFeatureExtractor()
    enhanced_df = enhanced_analyzer.process_existing_features(
        features_csv='features_basic.csv',
        output_file='features_enhanced.csv'
    )
    
    if enhanced_df is None or enhanced_df.empty:
        print("❌ No enhanced features extracted")
        return
    
    print(f"\n✅ Successfully analyzed {len(enhanced_df)} files")
    print(f"📄 Results saved to: features_enhanced.csv")
    
    # Show a quick summary
    print("\n📋 Quick Summary:")
    print("-" * 20)
    for idx, row in enhanced_df.iterrows():
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