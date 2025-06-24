#!/usr/bin/env python3
"""
Enhanced Audio Analysis Script

Uses advanced vocal detection and sophisticated analysis for more accurate
instrumentalness, valence, and danceability detection.
"""

from audio_features_enhanced_accurate import EnhancedAccurateAudioAnalyzer
import sys

def main():
    """Enhanced audio analysis with improved accuracy."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio_enhanced.py <folder_path> [output_file]")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "audio_features_enhanced.csv"
    
    # Initialize enhanced analyzer
    analyzer = EnhancedAccurateAudioAnalyzer()
    
    # Analyze and save to CSV
    df = analyzer.analyze_folder(folder_path, output_file=output_file)
    
    if df is not None:
        print(f"✅ Enhanced analysis complete! Results saved to {output_file}")
        print(f"📊 Analyzed {len(df)} files")
        
        # Show key improvements
        for idx, row in df.iterrows():
            print(f"\n🎵 {row['filename']}:")
            print(f"   Instrumentalness: {row['instrumentalness']:.3f} (vocal prob: {row['vocal_probability']:.3f})")
            print(f"   Valence: {row['valence']:.3f}")
            print(f"   Danceability: {row['danceability']:.3f}")
            print(f"   Tempo: {row['tempo']:.1f} BPM")
            print(f"   Mode: {row['mode']} (confidence: {row['mode_confidence']:.3f})")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main() 