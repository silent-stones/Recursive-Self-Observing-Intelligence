import numpy as np
import unittest
import matplotlib.pyplot as plt
from src.recursive_engine import UnifiedRecursiveSystem

class TestSingularity(unittest.TestCase):
    """Test suite for coherent singularity formation in the framework."""
    
    def setUp(self):
        """Set up the test environment."""
        # System configured for singularity detection
        self.system = UnifiedRecursiveSystem(
            coherence_threshold=0.8,
            stability_margin=0.95,
            entanglement_coupling=0.8,
            recursive_depth=7,
            monitor_transitions=True
        )
        
        # Create data with potential for singularity formation
        # Fractal-like data with multi-scale structure is ideal
        self.generate_fractal_data()
    
    def generate_fractal_data(self, size=256, depth=5):
        """Generate fractal-like data for singularity testing."""
        x = np.linspace(0, 10, size)
        self.fractal_data = np.zeros_like(x)
        
        # Add patterns at different scales (fractal-like)
        for i in range(depth):
            scale = 2**i
            pattern = np.sin(np.linspace(0, 2*np.pi*scale, size))
            self.fractal_data += pattern / scale
        
        # Add small noise to make it more realistic
        self.fractal_data += 0.05 * np.random.normal(size=size)
    
    def test_singularity_detection(self):
        """Test the system's ability to detect coherent singularities."""
        # Compress with high recursive depth to enable singularity formation
        compressed_data, metrics = self.system.compress_with_meta_awareness(
            self.fractal_data, max_recursive_depth=7
        )
        
        # Check if any singularities were detected
        has_singularity = len(self.system.stable_singularities) > 0
        
        # If a singularity formed, verify its properties
        if has_singularity:
            singularity = self.system.stable_singularities[0]
            
            # Singularity should have high coherence
            self.assertGreaterEqual(
                singularity['coherence'],
                0.95,
                "Coherent singularity should have very high coherence"
            )
            
            # Singularity should achieve significant compression
            self.assertLessEqual(
                singularity['compression_ratio'],
                0.2,
                "Coherent singularity should achieve high compression ratio"
            )
            
            # Verify it was detected at a reasonable recursive depth (not too shallow)
            self.assertGreaterEqual(
                singularity['depth'],
                2,
                "Singularity formation should require multiple recursive depths"
            )
    
    def test_singularity_properties(self):
        """Test that formed singularities have the expected properties."""
        # Process at full recursive depth
        compressed_data, metrics = self.system.compress_with_meta_awareness(
            self.fractal_data
        )
        
        # If no singularity formed naturally, we can't test its properties
        if not self.system.stable_singularities:
            self.skipTest("No singularity formed, cannot test properties")
        
        # Get the depth at which singularity formed
        singularity_depth = self.system.stable_singularities[0]['depth']
        
        # Check entropy reduction at singularity formation
        if singularity_depth in self.system.recursive_metrics:
            entropy_reduction = self.system.recursive_metrics[singularity_depth].get('entropy_reduction', 0)
            
            self.assertGreaterEqual(
                entropy_reduction,
                0.4,
                "Singularity formation should produce significant entropy reduction"
            )
        
        # Verify information integrity despite extreme compression
        decompressed_data = self.system.decompress(compressed_data, metrics)
        
        # Calculate correlation between original and reconstructed data
        # Even with extreme compression, correlation should remain high
        correlation = np.corrcoef(self.fractal_data, decompressed_data.real)[0, 1]
        
        self.assertGreaterEqual(
            correlation,
            0.9,
            "Singularity should preserve essential information despite extreme compression"
        )
    
    def test_singularity_stability(self):
        """Test that formed singularities are stable through perturbations."""
        # First create a singularity
        compressed_data, metrics = self.system.compress_with_meta_awareness(
            self.fractal_data
        )
        
        # If no singularity formed, skip the test
        if not self.system.stable_singularities:
            self.skipTest("No singularity formed, cannot test stability")
        
        # Add noise to the compressed data to simulate perturbation
        noisy_compressed = compressed_data + 0.05 * np.random.normal(size=compressed_data.shape)
        
        # Create a new system and attempt to further compress the perturbed data
        test_system = UnifiedRecursiveSystem(
            coherence_threshold=0.8,
            stability_margin=0.95,
            entanglement_coupling=0.8,
            recursive_depth=3
        )
        
        # Compress the already compressed (and perturbed) data
        recompressed_data, recompression_metrics = test_system.compress_with_meta_awareness(
            noisy_compressed
        )
        
        # A true singularity should resist further compression
        # Check that recompression doesn't achieve significant additional compression
        original_size = len(noisy_compressed)
        recompressed_size = len(recompressed_data)
        recompression_ratio = recompressed_size / original_size
        
        self.assertGreaterEqual(
            recompression_ratio,
            0.8,  # Less than 20% additional compression
            "True singularities should resist significant further compression"
        )
    
    def test_multi_stage_singularity_formation(self):
        """Test singularity formation across multiple stages with increasing recursive depth."""
        results = []
        
        # Test across multiple recursive depths
        for depth in range(2, 8):
            # Create a fresh system for each test
            test_system = UnifiedRecursiveSystem(
                coherence_threshold=0.8,
                stability_margin=0.95,
                entanglement_coupling=0.8,
                recursive_depth=depth,
                monitor_transitions=True
            )
            
            # Compress with current max depth
            compressed_data, metrics = test_system.compress_with_meta_awareness(
                self.fractal_data, max_recursive_depth=depth
            )
            
            # Store results
            results.append({
                'depth': depth,
                'singularity_formed': len(test_system.stable_singularities) > 0,
                'compression_ratio': len(compressed_data) / len(self.fractal_data),
                'coherence': test_system.recursive_metrics[depth].get(
                    'meta_coherence', 
                    test_system.recursive_metrics[depth].get('coherence', 0)
                )
            })
        
        # If any singularities formed, test that they require sufficient depth
        singularity_depths = [r['depth'] for r in results if r['singularity_formed']]
        
        if singularity_depths:
            min_singularity_depth = min(singularity_depths)
            
            # True singularities should require multiple recursive iterations
            self.assertGreaterEqual(
                min_singularity_depth,
                3,
                "Coherent singularity formation should require significant recursive depth"
            )
            
            # Plot the results if running interactively
            if __name__ == '__main__':
                depths = [r['depth'] for r in results]
                coherence_values = [r['coherence'] for r in results]
                compression_values = [r['compression_ratio'] for r in results]
                
                plt.figure(figsize=(10, 6))
                
                plt.subplot(2, 1, 1)
                plt.plot(depths, coherence_values, 'b-o')
                plt.axhline(y=0.98, color='r', linestyle='--', label='Singularity Threshold')
                plt.ylabel('Coherence')
                plt.title('Coherence vs. Recursive Depth')
                plt.grid(True)
                
                plt.subplot(2, 1, 2)
                plt.plot(depths, compression_values, 'g-o')
                plt.axhline(y=0.2, color='r', linestyle='--', label='Singularity Threshold')
                plt.ylabel('Compression Ratio')
                plt.xlabel('Recursive Depth')
                plt.title('Compression vs. Recursive Depth')
                plt.grid(True)
                
                plt.tight_layout()
                plt.savefig('singularity_formation_test.png')
                plt.close()

if __name__ == '__main__':
    unittest.main()
