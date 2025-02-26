import numpy as np
import unittest
from src.recursive_engine import UnifiedRecursiveSystem

class TestStability(unittest.TestCase):
    """Test suite for stability mechanisms in the Recursive Self-Observing Intelligence Framework."""
    
    def setUp(self):
        """Set up the test environment."""
        self.system = UnifiedRecursiveSystem(
            coherence_threshold=0.8,
            stability_margin=0.95,
            entanglement_coupling=0.75,
            recursive_depth=3
        )
        
        # Create test data with varying stability characteristics
        self.stable_data = np.sin(np.linspace(0, 10, 256)) + 0.1 * np.random.normal(size=256)
        self.unstable_data = np.random.normal(size=256)
        self.mixed_data = np.concatenate([
            np.sin(np.linspace(0, 5, 128)),
            np.random.normal(size=128)
        ])
    
    def test_stability_assessment(self):
        """Test the stability assessment functionality."""
        # Initialize phase space for test data
        stable_phase_space = self.system._initialize_phase_space(self.stable_data)
        unstable_phase_space = self.system._initialize_phase_space(self.unstable_data)
        mixed_phase_space = self.system._initialize_phase_space(self.mixed_data)
        
        # Check coherence computation
        stable_coherence = np.mean(stable_phase_space['coherence'])
        unstable_coherence = np.mean(unstable_phase_space['coherence'])
        mixed_coherence = np.mean(mixed_phase_space['coherence'])
        
        # Stable data should have higher coherence
        self.assertGreater(stable_coherence, unstable_coherence)
        self.assertGreater(stable_coherence, self.system.coherence_threshold)
        
        # Check stability check
        stable_is_stable = self.system._check_stability(stable_phase_space['coherence'])
        unstable_is_stable = self.system._check_stability(unstable_phase_space['coherence'])
        
        self.assertTrue(stable_is_stable, "Stable data should be assessed as stable")
        self.assertFalse(unstable_is_stable, "Unstable data should be assessed as unstable")
    
    def test_adaptive_correction(self):
        """Test the adaptive phase correction mechanism."""
        # Create coherence data needing correction
        low_coherence = np.array([0.7, 0.65, 0.75, 0.72])
        
        # Apply correction early in process (stage 0 of 10)
        early_correction = self.system._adaptive_phase_correction(low_coherence, 0, 10)
        
        # Apply correction late in process (stage 8 of 10)
        late_correction = self.system._adaptive_phase_correction(low_coherence, 8, 10)
        
        # Early correction should be more aggressive
        self.assertGreater(np.mean(early_correction - low_coherence), 
                           np.mean(late_correction - low_coherence),
                           "Early correction should be more aggressive than late correction")
        
        # Both corrections should improve coherence
        self.assertGreater(np.mean(early_correction), np.mean(low_coherence),
                          "Correction should improve coherence")
        self.assertGreater(np.mean(late_correction), np.mean(low_coherence),
                          "Correction should improve coherence")
    
    def test_stability_in_compression(self):
        """Test that stability is maintained during compression."""
        # Compress stable data
        compressed_stable, metrics_stable = self.system._recursive_compress(
            self.stable_data, 
            self.system._initialize_phase_space(self.stable_data)
        )
        
        # Compress unstable data
        compressed_unstable, metrics_unstable = self.system._recursive_compress(
            self.unstable_data, 
            self.system._initialize_phase_space(self.unstable_data)
        )
        
        # Check that corrections were applied to unstable data
        corrections_stable = metrics_stable['stability_history'].count(False)
        corrections_unstable = metrics_unstable['stability_history'].count(False)
        
        self.assertLess(corrections_stable, corrections_unstable,
                       "Unstable data should require more corrections")
        
        # Check final coherence
        self.assertGreater(np.mean(metrics_stable['final_coherence']),
                          np.mean(metrics_unstable['final_coherence']),
                          "Stable data should maintain higher coherence")

if __name__ == '__main__':
    unittest.main()
