#!/usr/bin/env python3
"""
Test Download Script

A simple test to download one song and verify the downloader works.
"""

import subprocess
import os
from pathlib import Path

def test_single_download():
    """Test downloading a single song."""
    print("🧪 Testing Single Song Download")
    print("=" * 40)
    
    # Create test directory
    test_dir = "test_downloads"
    Path(test_dir).mkdir(exist_ok=True)
    
    # Test song
    test_song = "Hotel California Eagles"
    
    print(f"🎵 Testing download: {test_song}")
    
    # Simple yt-dlp command
    cmd = [
        'yt-dlp',
        '--extract-audio',
        '--audio-format', 'mp3',
        '--audio-quality', '0',
        '--output', f'{test_dir}/%(title)s.%(ext)s',
        '--no-playlist',
        '--max-downloads', '1',
        '--ignore-errors',
        f'ytsearch1:{test_song}'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Run the download command
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Test download successful!")
            
            # List downloaded files
            if os.path.exists(test_dir):
                files = os.listdir(test_dir)
                print(f"📁 Downloaded files: {files}")
            
            return True
        else:
            print("❌ Test download failed!")
            print(f"Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Test download timed out!")
        return False
    except Exception as e:
        print(f"❌ Test download error: {e}")
        return False

if __name__ == "__main__":
    test_single_download() 