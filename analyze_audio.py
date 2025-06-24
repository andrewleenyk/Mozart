#!/usr/bin/env python3
"""
Simple Audio Analysis Script

Just analyzes audio files and outputs CSV - no visualizations or detailed printing.
"""

from audio_features_advanced import AdvancedAudioAnalyzer
import sys

def main():
    """Simple audio analysis with CSV output."""
    if len(sys.argv) < 2:
        print("Usage: python analyze_audio.py <folder_path> [output_file]")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "audio_features.csv"
    
    # Initialize analyzer
    analyzer = AdvancedAudioAnalyzer()
    
    # Analyze and save to CSV
    df = analyzer.analyze_folder(folder_path, output_file=output_file)
    
    if df is not None:
        print(f"✅ Analysis complete! Results saved to {output_file}")
        print(f"📊 Analyzed {len(df)} files")
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main() 