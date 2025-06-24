#!/usr/bin/env python3
"""
Fixed Music Downloader Script

A version that downloads songs from a text file using yt-dlp with a working search approach.
"""

import os
import sys
import subprocess
import re
from pathlib import Path
from typing import List, Dict, Tuple

class FixedMusicDownloader:
    def __init__(self, songs_file: str = "songs.txt", output_dir: str = "downloads"):
        """
        Initialize the fixed music downloader.
        
        Args:
            songs_file: Path to the songs text file
            output_dir: Directory to save downloaded songs
        """
        self.songs_file = songs_file
        self.output_dir = output_dir
        self.current_genre = "Unknown"
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Check if yt-dlp is installed
        self._check_yt_dlp()
    
    def _check_yt_dlp(self):
        """Check if yt-dlp is installed and available."""
        try:
            result = subprocess.run(['yt-dlp', '--version'], 
                                  capture_output=True, text=True, check=True)
            print(f"✅ yt-dlp version: {result.stdout.strip()}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ yt-dlp not found. Please install it first:")
            print("   pip install yt-dlp")
            sys.exit(1)
    
    def parse_songs_file(self) -> Dict[str, List[str]]:
        """
        Parse the songs text file and organize by genre.
        
        Returns:
            Dictionary with genres as keys and lists of songs as values
        """
        songs_by_genre = {}
        
        if not os.path.exists(self.songs_file):
            print(f"❌ Songs file not found: {self.songs_file}")
            return songs_by_genre
        
        print(f"📖 Reading songs from: {self.songs_file}")
        
        with open(self.songs_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Check if it's a genre header
                if line.startswith('#'):
                    self.current_genre = line[1:].strip()
                    songs_by_genre[self.current_genre] = []
                    print(f"🎵 Genre: {self.current_genre}")
                else:
                    # It's a song line
                    if self.current_genre in songs_by_genre:
                        songs_by_genre[self.current_genre].append(line)
        
        return songs_by_genre
    
    def download_song(self, song_info: str, genre: str) -> bool:
        """
        Download a single song using yt-dlp.
        
        Args:
            song_info: Song information (e.g., "Hotel California - Eagles")
            genre: Genre of the song
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Create genre subdirectory
            genre_dir = os.path.join(self.output_dir, genre)
            Path(genre_dir).mkdir(parents=True, exist_ok=True)
            
            # Sanitize the song info for search
            search_query = song_info.replace(' - ', ' ')
            
            print(f"🎵 Downloading: {song_info}")
            print(f"   Genre: {genre}")
            print(f"   Search: {search_query}")
            
            # Use a different approach - search with quotes
            search_url = f"ytsearch1:\"{search_query}\""
            
            # yt-dlp command with options
            cmd = [
                'yt-dlp',
                '--extract-audio',
                '--audio-format', 'mp3',
                '--audio-quality', '0',
                '--output', f'{genre_dir}/%(title)s.%(ext)s',
                '--no-playlist',
                '--max-downloads', '1',
                '--ignore-errors',
                search_url
            ]
            
            # Run the download command
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            
            if result.returncode == 0:
                print(f"✅ Successfully downloaded: {song_info}")
                return True
            else:
                print(f"❌ Failed to download: {song_info}")
                if result.stderr:
                    # Show first error line
                    error_lines = result.stderr.split('\n')
                    for line in error_lines:
                        if line.strip() and ('ERROR:' in line or 'WARNING:' in line):
                            print(f"   Error: {line.strip()}")
                            break
                return False
                
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout downloading {song_info}")
            return False
        except Exception as e:
            print(f"❌ Error downloading {song_info}: {e}")
            return False
    
    def download_all_songs(self) -> Dict[str, Tuple[int, int]]:
        """
        Download all songs from the parsed file.
        
        Returns:
            Dictionary with genre as key and (successful, total) as value
        """
        songs_by_genre = self.parse_songs_file()
        
        if not songs_by_genre:
            print("❌ No songs found to download")
            return {}
        
        print(f"\n🚀 Starting download of {sum(len(songs) for songs in songs_by_genre.values())} songs...")
        print("=" * 60)
        
        results = {}
        
        for genre, songs in songs_by_genre.items():
            print(f"\n🎵 Processing genre: {genre}")
            print("-" * 40)
            
            successful = 0
            total = len(songs)
            
            for song in songs:
                if self.download_song(song, genre):
                    successful += 1
                print()  # Empty line for readability
            
            results[genre] = (successful, total)
            print(f"📊 {genre}: {successful}/{total} songs downloaded successfully")
        
        return results
    
    def print_summary(self, results: Dict[str, Tuple[int, int]]):
        """Print a summary of the download results."""
        print("\n" + "=" * 60)
        print("📊 DOWNLOAD SUMMARY")
        print("=" * 60)
        
        total_successful = 0
        total_songs = 0
        
        for genre, (successful, total) in results.items():
            print(f"🎵 {genre}: {successful}/{total} songs")
            total_successful += successful
            total_songs += total
        
        print("-" * 40)
        print(f"🎯 Total: {total_successful}/{total_songs} songs downloaded successfully")
        
        if total_successful == total_songs:
            print("🎉 All songs downloaded successfully!")
        else:
            print(f"⚠️  {total_songs - total_successful} songs failed to download")
        
        print(f"📁 Songs saved to: {os.path.abspath(self.output_dir)}")


def main():
    """Main function to run the fixed music downloader."""
    print("🎵 Fixed Music Downloader")
    print("=" * 40)
    
    # Get command line arguments
    songs_file = sys.argv[1] if len(sys.argv) > 1 else "songs.txt"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "downloads"
    
    # Initialize downloader
    downloader = FixedMusicDownloader(songs_file, output_dir)
    
    # Download all songs
    results = downloader.download_all_songs()
    
    # Print summary
    if results:
        downloader.print_summary(results)
    else:
        print("❌ No songs were processed")


if __name__ == "__main__":
    main() 