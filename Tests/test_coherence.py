import numpy as np
import unittest
from src.recursive_engine import UnifiedRecursiveSystem

class TestCoherence(unittest.TestCase):
    """Test suite for coherence mechanisms in the Recursive Self-Observing Intelligence Framework."""
    
    def setUp(self):
        """Set up the test environment."""
        self.system = UnifiedRecursiveSystem(
            coherence_threshold=0.8,
            stability_margin=0.95,
            entanglement_coupling=0.75,
            recursive_depth=3
        )
        
        # Create test data with varying coherence characteristics
        x = np.linspace(0, 10, 256)
        self.coherent_data = np.sin(x) + 0.5 * np.sin(2*x)  # High coherence signal
        self.incoherent_data = np.random.normal(size=256)  # Low coherence signal
        self.mixed_data = np.concatenate([
            np.sin(np.linspace(0, 5, 128)),
            0.3 * np.random.normal(size=128)
        ])  # Mixed coherence signal
    
    def test_coherence_computation(self):
        """Test accurate computation of coherence."""
        # Initialize phase spaces
        coherent_phase_space = self.system._initialize_phase_space(self.coherent_data)
        incoherent_phase_space = self.system._initialize_phase_space(self.incoherent_data)
        
        # Extract coherence values
        coherent_coherence = coherent_phase_space['coherence']
        incoherent_coherence = incoherent_phase_space['coherence']
        
        # Test that coherent data has higher average coherence
        self.assertGreater(
            np.mean(coherent_coherence),
            np.mean(incoherent_coherence),
            "Coherent data should have higher coherence values"
        )
        
        # Test coherence computation from phase
        coherent_phase = np.angle(np.fft.fft(self.coherent_data))
        computed_coherence = self.system._compute_coherence(coherent_phase, coherent_phase_space)
        
        # Coherence values should be between 0 and 1
        self.assertTrue(np.all(computed_coherence >= 0) and np.all(computed_coherence <= 1),
                       "Coherence values should be between 0 and 1")
        
        # Test that gradient affects coherence (smoother signals have higher coherence)
        self.assertLess(
            np.std(coherent_coherence),
            np.std(incoherent_coherence),
            "Coherent data should have more uniform coherence"
        )
    
    def test_coherence_preservation(self):
        """Test that compression preserves coherence appropriately."""
        # Compress coherent data
        compressed_coherent, metrics_coherent = self.system.compress_with_meta_awareness(
            self.coherent_data, max_recursive_depth=2
        )
        
        # Compress incoherent data
        compressed_incoherent, metrics_incoherent = self.system.compress_with_meta_awareness(
            self.incoherent_data, max_recursive_depth=2
        )
        
        # Test coherence maintenance through compression (coherent data stays coherent)
        coherent_initial = metrics_coherent['recursive_metrics'][0]['coherence']
        coherent_final = metrics_coherent['recursive_metrics'][2]['meta_coherence']
        
        # Coherent data should maintain or improve coherence
        self.assertGreaterEqual(
            coherent_final,
            coherent_initial * 0.9,  # Allow for small variations
            "Coherent data should maintain coherence through compression"
        )
        
        # Test that coherent data compresses better than incoherent data
        coherent_ratio = metrics_coherent['recursive_metrics'][0]['compression_ratio']
        incoherent_ratio = metrics_incoherent['recursive_metrics'][0]['compression_ratio']
        
        self.assertLess(
            coherent_ratio,
            incoherent_ratio,
            "Coherent data should achieve better compression ratio"
        )
    
    def test_integration_coherence(self):
        """Test computation of integration coherence between levels."""
        # Create mock metrics
        primary_metrics = {'final_coherence': 0.85}
        meta_metrics = {'final_coherence': 0.90}
        
        # Compute integration coherence
        integration_coherence = self.system._compute_integration_coherence(
            primary_metrics, meta_metrics
        )
        
        # Integration coherence should be geometric mean of input coherences
        expected_coherence = np.sqrt(0.85 * 0.90)
        self.assertAlmostEqual(
            integration_coherence,
            expected_coherence,
            places=6,
            msg="Integration coherence should be geometric mean of input coherences"
        )
        
        # Test with array coherence values
        primary_metrics = {'final_coherence': np.array([0.80, 0.85, 0.90])}
        meta_metrics = {'final_coherence': np.array([0.85, 0.90, 0.95])}
        
        integration_coherence = self.system._compute_integration_coherence(
            primary_metrics, meta_metrics
        )
        
        # Should still produce a valid coherence value
        self.assertGreaterEqual(integration_coherence, 0.0)
        self.assertLessEqual(integration_coherence, 1.0)
    
    def test_coherence_autocatalysis(self):
        """Test detection of coherence autocatalysis through recursion."""
        # Set up system for detecting autocatalysis
        autocatalysis_system = UnifiedRecursiveSystem(
            coherence_threshold=0.75,
            stability_margin=0.9,
            entanglement_coupling=0.8,
            recursive_depth=6  # Deep recursion to detect autocatalysis
        )
        
        # Use mixed data to allow for improvement
        _, metrics = autocatalysis_system.compress_with_meta_awareness(
            self.mixed_data, max_recursive_depth=5
        )
        
        # Extract coherence values across recursive depths
        coherence_values = []
        for depth in range(6):
            if depth == 0:
                coherence_values.append(metrics['recursive_metrics'][depth].get('coherence', 0))
            else:
                coherence_values.append(metrics['recursive_metrics'][depth].get('meta_coherence', 0))
        
        # Test for increasing coherence trend (autocatalysis)
        # True autocatalysis would show increasing trend after initial depths
        increasing_trend = all(coherence_values[i] <= coherence_values[i+1] 
                              for i in range(2, len(coherence_values)-1))
        
        # Check that final coherence is higher than initial
        self.assertGreaterEqual(
            coherence_values[-1],
            coherence_values[0],
            "Final coherence should be higher than initial in recursive processing"
        )
        
        # Check if autocatalysis is detected in emergent properties
        emergent_props = autocatalysis_system._identify_emergent_properties()
        has_autocatalysis = any(prop['type'] == 'coherence_autocatalysis' for prop in emergent_props)
        
        if increasing_trend and coherence_values[-1] > coherence_values[0] * 1.1:
            self.assertTrue(
                has_autocatalysis,
                "System should detect coherence autocatalysis when coherence increases through recursion"
            )

if __name__ == '__main__':
    unittest.main()
