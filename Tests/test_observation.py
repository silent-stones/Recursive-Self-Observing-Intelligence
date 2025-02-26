import numpy as np
import unittest
from src.recursive_engine import UnifiedRecursiveSystem

class TestRecursiveObservation(unittest.TestCase):
    """Test suite for recursive self-observation mechanisms."""
    
    def setUp(self):
        """Set up the test environment."""
        self.system = UnifiedRecursiveSystem(
            coherence_threshold=0.8,
            stability_margin=0.95,
            entanglement_coupling=0.75,
            recursive_depth=4
        )
        
        # Create test data with clear patterns
        x = np.linspace(0, 10, 256)
        self.test_data = np.sin(x) + 0.5 * np.sin(2*x) + 0.1 * np.random.normal(size=256)
    
    def test_meta_compression(self):
        """Test that meta-compression correctly encodes the compression process."""
        # Perform initial compression
        phase_space = self.system._initialize_phase_space(self.test_data)
        compressed_data, metrics = self.system._recursive_compress(self.test_data, phase_space)
        
        # Encode the compression process
        process_encoding = self.system._encode_compression_process(metrics)
        
        # Check that encoding has expected properties
        self.assertGreater(len(process_encoding), 0, "Process encoding should not be empty")
        self.assertLess(len(process_encoding), len(self.test_data), 
                        "Process encoding should be smaller than original data")
    
    def test_meta_insights_integration(self):
        """Test that meta-insights are correctly integrated into the primary data."""
        # Generate process representation
        phase_space = self.system._initialize_phase_space(self.test_data)
        compressed_data, primary_metrics = self.system._recursive_compress(self.test_data, phase_space)
        
        # Create mock meta-insights
        process_encoding = self.system._encode_compression_process(primary_metrics)
        meta_phase_space = self.system._initialize_phase_space(process_encoding)
        meta_compressed, meta_metrics = self.system._recursive_compress(process_encoding, meta_phase_space)
        
        # Integrate meta-insights
        refined_data = self.system._integrate_meta_insights(
            compressed_data, meta_compressed, primary_metrics, meta_metrics
        )
        
        # Check refined data properties
        self.assertEqual(len(refined_data), len(compressed_data), 
                         "Refined data should maintain the same size")
        
        # Calculate integration coherence
        integration_coherence = self.system._compute_integration_coherence(
            primary_metrics, meta_metrics
        )
        
        self.assertGreaterEqual(integration_coherence, 0.0)
        self.assertLessEqual(integration_coherence, 1.0)
    
    def test_recursive_depth(self):
        """Test that the system correctly handles different recursive depths."""
        # Test with different recursive depths
        for depth in range(4):
            # Compress with specified depth
            compressed_data, metrics = self.system.compress_with_meta_awareness(
                self.test_data, max_recursive_depth=depth
            )
            
            # Check that metrics contains the right number of depths
            self.assertEqual(len(metrics['recursive_metrics']), depth + 1,
                            f"Should have {depth + 1} recursive metrics entries")
            
            # For non-zero depths, check for meta-coherence
            if depth > 0:
                self.assertIn('meta_coherence', metrics['recursive_metrics'][depth],
                             f"Depth {depth} should have meta-coherence metric")
    
    def test_phase_transition_detection(self):
        """Test the phase transition detection mechanism."""
        # Create synthetic coherence surge
        self.system.recursive_metrics = {
            0: {'meta_coherence': 0.7},
            1: {'meta_coherence': 0.95}
        }
        self.system.coherence_history = [0.7]
        
        # Test transition detection
        transition = self.system._detect_phase_transition(self.test_data, 1)
        
        self.assertTrue(transition['detected'], "Should detect coherence surge")
        self.assertEqual(transition['type'], 'coherence_surge', "Should identify as coherence surge")

if __name__ == '__main__':
    unittest.main()
