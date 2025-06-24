#!/usr/bin/env python3
"""
Test script for the Audio Feature Extraction Module

This script tests the basic functionality of the module without requiring actual audio files.
"""

import unittest
import numpy as np
import tempfile
import os
from audio_features_simple import AudioFeatureExtractor


class TestAudioFeatureExtractor(unittest.TestCase):
    """Test cases for AudioFeatureExtractor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.extractor = AudioFeatureExtractor(sample_rate=22050)
        
        # Create a simple test audio signal (1 second of 440 Hz sine wave)
        self.sample_rate = 22050
        t = np.linspace(0, 1, self.sample_rate, False)
        self.test_audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Add some noise to make it more realistic
        self.test_audio += 0.1 * np.random.randn(len(self.test_audio))
    
    def test_extractor_initialization(self):
        """Test that the extractor initializes correctly."""
        self.assertEqual(self.extractor.sample_rate, 22050)
        self.assertIsInstance(self.extractor, AudioFeatureExtractor)
    
    def test_rhythmic_features(self):
        """Test rhythmic feature extraction."""
        features = self.extractor.extract_rhythmic_features(self.test_audio)
        
        # Check that all expected features are present
        expected_features = ['tempo', 'beat_strength', 'rhythmic_stability', 'regularity']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
        
        # Check reasonable value ranges
        self.assertGreater(features['tempo'], 0)
        self.assertLess(features['tempo'], 300)  # Reasonable BPM range
        self.assertGreaterEqual(features['beat_strength'], 0)
        self.assertGreaterEqual(features['rhythmic_stability'], 0)
        self.assertGreaterEqual(features['regularity'], 0)
        self.assertLessEqual(features['regularity'], 1)
    
    def test_musical_structure(self):
        """Test musical structure feature extraction."""
        features = self.extractor.extract_musical_structure(self.test_audio)
        
        # Check that all expected features are present
        expected_features = ['key', 'mode']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], str)
        
        # Check that key is valid
        valid_keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.assertIn(features['key'], valid_keys)
        
        # Check that mode is valid
        valid_modes = ['major', 'minor']
        self.assertIn(features['mode'], valid_modes)
    
    def test_energy_dynamics(self):
        """Test energy and dynamics feature extraction."""
        features = self.extractor.extract_energy_dynamics(self.test_audio)
        
        # Check that all expected features are present
        expected_features = ['energy', 'loudness', 'brightness', 'high_freq_content']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
        
        # Check reasonable value ranges
        self.assertGreaterEqual(features['energy'], 0)
        self.assertLessEqual(features['energy'], 1)
        self.assertGreaterEqual(features['brightness'], 0)
        self.assertGreaterEqual(features['high_freq_content'], 0)
    
    def test_emotional_features(self):
        """Test emotional feature extraction."""
        features = self.extractor.extract_emotional_features(self.test_audio)
        
        # Check that all expected features are present
        expected_features = ['valence', 'danceability', 'instrumentalness', 'acousticness', 'speechiness']
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
        
        # Check that values are in [0, 1] range
        for feature in expected_features:
            self.assertGreaterEqual(features[feature], 0)
            self.assertLessEqual(features[feature], 1)
    
    def test_spectral_features(self):
        """Test spectral feature extraction."""
        features = self.extractor.extract_spectral_features(self.test_audio)
        
        # Check that all expected features are present
        expected_features = [
            'spectral_centroid_mean', 'spectral_rolloff_mean', 'spectral_bandwidth_mean',
            'spectral_contrast_mean', 'mfcc_mean', 'mel_energy_mean',
            'spectral_centroid_std', 'spectral_rolloff_std', 'spectral_bandwidth_std', 'mfcc_std'
        ]
        for feature in expected_features:
            self.assertIn(feature, features)
            self.assertIsInstance(features[feature], float)
        
        # Check that standard deviations are non-negative
        std_features = [f for f in expected_features if f.endswith('_std')]
        for feature in std_features:
            self.assertGreaterEqual(features[feature], 0)
    
    def test_all_features(self):
        """Test extraction of all features together."""
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            import soundfile as sf
            sf.write(tmp_file.name, self.test_audio, self.sample_rate)
            
            try:
                features = self.extractor.extract_all_features(tmp_file.name)
                
                # Check that features dictionary is not None
                self.assertIsNotNone(features)
                
                # Check that all expected feature categories are present
                expected_categories = [
                    'filename', 'filepath', 'duration',
                    'tempo', 'beat_strength', 'rhythmic_stability', 'regularity',
                    'key', 'mode',
                    'energy', 'loudness', 'brightness', 'high_freq_content',
                    'valence', 'danceability', 'instrumentalness', 'acousticness', 'speechiness',
                    'spectral_centroid_mean', 'spectral_rolloff_mean', 'spectral_bandwidth_mean',
                    'spectral_contrast_mean', 'mfcc_mean', 'mel_energy_mean',
                    'spectral_centroid_std', 'spectral_rolloff_std', 'spectral_bandwidth_std', 'mfcc_std'
                ]
                
                for category in expected_categories:
                    self.assertIn(category, features)
                
                # Check specific values
                self.assertEqual(features['filename'], os.path.basename(tmp_file.name))
                self.assertEqual(features['filepath'], tmp_file.name)
                self.assertAlmostEqual(features['duration'], 1.0, places=1)
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test with None audio
        features = self.extractor.extract_rhythmic_features(None)
        self.assertEqual(features['tempo'], 0.0)
        
        # Test with empty audio
        features = self.extractor.extract_rhythmic_features(np.array([]))
        self.assertEqual(features['tempo'], 0.0)
        
        # Test with non-existent file
        features = self.extractor.extract_all_features('/non/existent/file.mp3')
        self.assertIsNone(features)


def run_tests():
    """Run all tests."""
    print("🧪 Running Audio Feature Extraction Tests")
    print("=" * 50)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAudioFeatureExtractor)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests() 