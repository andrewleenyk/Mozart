#!/usr/bin/env python3
"""
Mozart Audio Analysis Application

A dual-mode application for analyzing audio features:
- TESTING MODE: Analyzes all files in test_music_samples directory
- PRODUCTION MODE: Analyzes files in songs directory at main level

Usage:
    python mozart_app.py --mode testing
    python mozart_app.py --mode production
"""

import os
import sys
import glob
import argparse
from pathlib import Path
from src.analyzers.audio_features_enhanced import EnhancedAudioFeatureExtractor
import pandas as pd

class MozartApp:
    def __init__(self):
        """Initialize the Mozart application."""
        self.extractor = EnhancedAudioFeatureExtractor()
        
        # Define directories
        self.test_dir = "test_music_samples"
        self.production_dir = "songs"
        
        # Output files
        self.test_output = "features_test_mode.csv"
        self.production_output = "features_production_mode.csv"
    
    def get_audio_files(self, directory: str) -> list:
        """
        Recursively find all audio files in a directory.
        
        Args:
            directory: Directory to search
            
        Returns:
            List of audio file paths
        """
        audio_extensions = ['*.mp3', '*.wav', '*.flac', '*.m4a', '*.ogg']
        audio_files = []
        
        if not os.path.exists(directory):
            print(f"❌ Directory not found: {directory}")
            return audio_files
        
        print(f"🔍 Searching for audio files in: {directory}")
        
        for ext in audio_extensions:
            pattern = os.path.join(directory, '**', ext)
            files = glob.glob(pattern, recursive=True)
            audio_files.extend(files)
        
        # Sort files for consistent processing order
        audio_files.sort()
        
        print(f"📁 Found {len(audio_files)} audio files")
        return audio_files
    
    def analyze_files(self, files: list, output_file: str, mode: str) -> pd.DataFrame:
        """
        Analyze a list of audio files and save results.
        
        Args:
            files: List of audio file paths
            output_file: Output CSV file path
            mode: Analysis mode (testing/production)
            
        Returns:
            DataFrame with analysis results
        """
        if not files:
            print(f"❌ No audio files found for {mode} mode")
            return pd.DataFrame()
        
        print(f"\n🎵 Starting {mode.upper()} mode analysis")
        print(f"📊 Processing {len(files)} files...")
        print("=" * 60)
        
        results = []
        successful = 0
        
        for i, file_path in enumerate(files):
            print(f"\nProcessing {i+1}/{len(files)}: {os.path.basename(file_path)}")
            
            try:
                # Extract enhanced features
                features = self.extractor.extract_enhanced_features(file_path)
                
                if features:
                    results.append(features)
                    successful += 1
                    print(f"  ✅ Successfully analyzed")
                else:
                    print(f"  ❌ Failed to analyze")
                    
            except Exception as e:
                print(f"  ❌ Error analyzing {file_path}: {e}")
        
        if results:
            # Create DataFrame
            df = pd.DataFrame(results)
            
            # Save to CSV
            df.to_csv(output_file, index=False)
            
            print(f"\n✅ Analysis complete!")
            print(f"📁 Results saved to: {output_file}")
            print(f"📊 Successfully analyzed: {successful}/{len(files)} files")
            
            # Show summary statistics
            self.show_summary(df, mode)
            
            return df
        else:
            print(f"\n❌ No files were successfully analyzed in {mode} mode")
            return pd.DataFrame()
    
    def show_summary(self, df: pd.DataFrame, mode: str):
        """Show summary statistics for the analysis results."""
        print(f"\n📈 {mode.upper()} MODE SUMMARY")
        print("=" * 40)
        
        # Show enhanced features summary
        enhanced_columns = ['valence', 'danceability', 'instrumentalness', 'acousticness', 'speechiness']
        available_columns = [col for col in enhanced_columns if col in df.columns]
        
        if available_columns:
            print("\nEnhanced Features Statistics:")
            print(df[available_columns].describe())
        
        # Show confidence scores
        confidence_columns = [col for col in df.columns if 'confidence' in col]
        if confidence_columns:
            print(f"\nConfidence scores available for: {confidence_columns}")
        
        # Show file breakdown by directory
        if 'filepath' in df.columns:
            print("\nFiles by directory:")
            directories = {}
            for filepath in df['filepath']:
                dir_name = os.path.dirname(filepath)
                directories[dir_name] = directories.get(dir_name, 0) + 1
            
            for dir_name, count in sorted(directories.items()):
                print(f"  {dir_name}: {count} files")
    
    def run_testing_mode(self) -> pd.DataFrame:
        """Run the application in testing mode."""
        print("🧪 TESTING MODE")
        print("=" * 40)
        print("Analyzing all files in test_music_samples directory...")
        
        # Get all audio files from test directory
        test_files = self.get_audio_files(self.test_dir)
        
        if not test_files:
            print(f"❌ No audio files found in {self.test_dir}")
            print("Please ensure test_music_samples directory contains audio files.")
            return pd.DataFrame()
        
        # Analyze files
        return self.analyze_files(test_files, self.test_output, "testing")
    
    def run_production_mode(self) -> pd.DataFrame:
        """Run the application in production mode."""
        print("🏭 PRODUCTION MODE")
        print("=" * 40)
        print("Analyzing files in songs directory...")
        
        # Get all audio files from production directory
        production_files = self.get_audio_files(self.production_dir)
        
        if not production_files:
            print(f"❌ No audio files found in {self.production_dir}")
            print("Please ensure songs directory contains audio files.")
            print("You can place MP3, WAV, FLAC, M4A, or OGG files in the songs folder.")
            return pd.DataFrame()
        
        # Analyze files
        return self.analyze_files(production_files, self.production_output, "production")
    
    def setup_production_directory(self):
        """Create the production songs directory if it doesn't exist."""
        if not os.path.exists(self.production_dir):
            os.makedirs(self.production_dir)
            print(f"📁 Created production directory: {self.production_dir}")
            print("💡 Place your audio files in this directory for production mode analysis")
        else:
            print(f"📁 Production directory exists: {self.production_dir}")


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description='Mozart Audio Analysis Application - Dual Mode Audio Feature Extraction',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mozart_app.py --mode testing     # Analyze test_music_samples
  python mozart_app.py --mode production  # Analyze songs directory
  python mozart_app.py --setup            # Create production directory
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['testing', 'production'], 
        help='Analysis mode: testing (test_music_samples) or production (songs directory)'
    )
    
    parser.add_argument(
        '--setup', 
        action='store_true',
        help='Create production directory structure'
    )
    
    args = parser.parse_args()
    
    # Initialize application
    app = MozartApp()
    
    # Setup mode
    if args.setup:
        app.setup_production_directory()
        return
    
    # Check if mode is provided for analysis
    if not args.mode:
        print("❌ Please specify --mode testing or --mode production")
        print("   Use --help for more information")
        return
    
    # Run analysis based on mode
    if args.mode == 'testing':
        df = app.run_testing_mode()
    elif args.mode == 'production':
        df = app.run_production_mode()
    else:
        print(f"❌ Invalid mode: {args.mode}")
        return
    
    # Final summary
    if not df.empty:
        print(f"\n🎉 {args.mode.upper()} mode analysis completed successfully!")
        print(f"📊 Total files analyzed: {len(df)}")
    else:
        print(f"\n⚠️  {args.mode.upper()} mode analysis completed with no results")


if __name__ == "__main__":
    main() 