import sys
import os
import unittest
import numpy as np
import asyncio
import time
import matplotlib.pyplot as plt

# Add the parent directory to the Python path to import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the structured expansion system
from src.structured_expansion import StructuredExpansionSystem

class TestStructuredExpansion(unittest.TestCase):
    """Test cases for the Structured Expansion System."""

    def setUp(self):
        """Set up test fixtures."""
        self.expansion_system = StructuredExpansionSystem(
            num_threads=4,
            buffer_size=512,
            fibonacci_depth=24,
            coherence_threshold=0.8
        )
        
        # Generate test data
        self.test_data_small = self._generate_test_data(size=1000)
        self.test_data_medium = self._generate_test_data(size=10000)
        self.test_data_large = self._generate_test_data(size=100000)
        
        # Create output directory if it doesn't exist
        os.makedirs('test_results', exist_ok=True)

    def _generate_test_data(self, size=10000, complexity=3):
        """Generate synthetic test data with multi-scale patterns."""
        t = np.linspace(0, 10, size)
        
        # Base signal with multiple frequency components
        signal = np.zeros_like(t)
        
        # Add multiple frequency components
        for i in range(1, complexity + 1):
            signal += (1.0 / i) * np.sin(i * t)
        
        # Add some localized features
        # Gaussian bump
        signal += 1.5 * np.exp(-0.5 * ((t - 3) / 0.2)**2)
        
        # Square pulse
        signal += 1.0 * ((t > 6) & (t < 6.5))
        
        # Chirp signal
        signal += 0.8 * np.sin(t * (t/2)) * ((t > 8) & (t < 9))
        
        # Add small amount of noise
        noise = 0.1 * np.random.normal(size=len(t))
        
        return signal + noise

    def _compare_data(self, original, expanded, preserve_ratio=False):
        """
        Compare original and expanded data.
        
        Args:
            original: Original data
            expanded: Expanded data
            preserve_ratio: Whether to preserve original plot ratio
            
        Returns:
            Comparison metrics
        """
        # Compute basic metrics
        expansion_ratio = len(expanded) / len(original)
        
        # Frequency domain analysis
        original_fft = np.fft.fft(original)
        original_magnitude = np.abs(original_fft)
        original_phase = np.angle(original_fft)
        
        expanded_fft = np.fft.fft(expanded)
        expanded_magnitude = np.abs(expanded_fft)
        expanded_phase = np.angle(expanded_fft)
        
        # Normalize magnitudes for frequency profile comparison
        if np.sum(original_magnitude) > 0:
            original_magnitude_norm = original_magnitude / np.sum(original_magnitude)
        else:
            original_magnitude_norm = original_magnitude
        
        if np.sum(expanded_magnitude) > 0:
            expanded_magnitude_norm = expanded_magnitude / np.sum(expanded_magnitude)
        else:
            expanded_magnitude_norm = expanded_magnitude
        
        # Resize for comparison if needed
        min_size = min(len(original_magnitude_norm), len(expanded_magnitude_norm))
        orig_freq_profile = original_magnitude_norm[:min_size]
        exp_freq_profile = expanded_magnitude_norm[:min_size]
        
        # Compute frequency profile similarity
        try:
            freq_similarity = np.corrcoef(orig_freq_profile, exp_freq_profile)[0, 1]
        except:
            freq_similarity = 0
        
        # Phase coherence analysis - compute phase gradient
        orig_phase_grad = np.gradient(original_phase)
        exp_phase_grad = np.gradient(expanded_phase)
        
        # Analyze phase gradient patterns
        orig_phase_coherence = 1.0 / (1.0 + np.var(orig_phase_grad))
        exp_phase_coherence = 1.0 / (1.0 + np.var(exp_phase_grad))
        
        # Plot comparison
        fig = plt.figure(figsize=(12, 10))
        
        # Time domain plots
        plt.subplot(3, 2, 1)
        plt.plot(original[:100])
        plt.title("Original Signal (first 100 samples)")
        plt.grid(True)
        
        plot_length = min(100, len(expanded) // (len(expanded) // len(original)))
        plt.subplot(3, 2, 2)
        plt.plot(expanded[:plot_length])
        plt.title(f"Expanded Signal (expansion ratio: {expansion_ratio:.2f}x)")
        plt.grid(True)
        
        # Frequency domain plots
        plt.subplot(3, 2, 3)
        plt.plot(orig_freq_profile[:100])
        plt.title("Original Frequency Profile")
        plt.grid(True)
        
        plt.subplot(3, 2, 4)
        plt.plot(exp_freq_profile[:100])
        plt.title(f"Expanded Frequency Profile (similarity: {freq_similarity:.4f})")
        plt.grid(True)
        
        # Phase coherence plots
        plt.subplot(3, 2, 5)
        plt.plot(orig_phase_grad[:100])
        plt.title(f"Original Phase Gradient (coherence: {orig_phase_coherence:.4f})")
        plt.grid(True)
        
        plt.subplot(3, 2, 6)
        plt.plot(exp_phase_grad[:100])
        plt.title(f"Expanded Phase Gradient (coherence: {exp_phase_coherence:.4f})")
        plt.grid(True)
        
        plt.tight_layout()
        
        return {
            'expansion_ratio': expansion_ratio,
            'frequency_similarity': freq_similarity,
            'original_coherence': orig_phase_coherence,
            'expanded_coherence': exp_phase_coherence,
            'fig': fig
        }

    def test_fibonacci_cache_initialization(self):
        """Test that Fibonacci cache is properly initialized."""
        self.assertIsNotNone(self.expansion_system.fibonacci_cache)
        self.assertGreater(len(self.expansion_system.fibonacci_cache), 20)
        
        # Check a few Fibonacci numbers
        self.assertEqual(self.expansion_system.fibonacci_cache[0], 0)
        self.assertEqual(self.expansion_system.fibonacci_cache[1], 1)
        self.assertEqual(self.expansion_system.fibonacci_cache[2], 1)
        self.assertEqual(self.expansion_system.fibonacci_cache[3], 2)
        self.assertEqual(self.expansion_system.fibonacci_cache[4], 3)
        self.assertEqual(self.expansion_system.fibonacci_cache[5], 5)
        self.assertEqual(self.expansion_system.fibonacci_cache[6], 8)

    async def test_small_data_expansion(self):
        """Test expansion on small data."""
        print("Testing small data expansion...")
        
        # Expand with different levels
        for level in [0.25, 0.5, 0.75, 1.0]:
            start_time = time.time()
            expanded = await self.expansion_system.expand_data(self.test_data_small, expansion_level=level)
            duration = time.time() - start_time
            
            # Verify expansion
            self.assertIsNotNone(expanded)
            self.assertGreater(len(expanded), len(self.test_data_small))
            
            # Compare data
            result = self._compare_data(self.test_data_small, expanded)
            
            # Save results
            result['fig'].savefig(f"test_results/small_data_expansion_level_{level:.2f}.png")
            plt.close(result['fig'])
            
            print(f"Small data, level {level:.2f}:")
            print(f"  Expansion ratio: {result['expansion_ratio']:.2f}x")
            print(f"  Frequency similarity: {result['frequency_similarity']:.4f}")
            print(f"  Coherence: original={result['original_coherence']:.4f}, expanded={result['expanded_coherence']:.4f}")
            print(f"  Duration: {duration:.4f}s")
            print()

    async def test_medium_data_expansion(self):
        """Test expansion on medium data."""
        print("Testing medium data expansion...")
        
        # Standard expansion level
        level = 0.75
        
        start_time = time.time()
        expanded = await self.expansion_system.expand_data(self.test_data_medium, expansion_level=level)
        duration = time.time() - start_time
        
        # Verify expansion
        self.assertIsNotNone(expanded)
        self.assertGreater(len(expanded), len(self.test_data_medium))
        
        # Compare data
        result = self._compare_data(self.test_data_medium, expanded)
        
        # Save results
        result['fig'].savefig(f"test_results/medium_data_expansion.png")
        plt.close(result['fig'])
        
        print(f"Medium data:")
        print(f"  Expansion ratio: {result['expansion_ratio']:.2f}x")
        print(f"  Frequency similarity: {result['frequency_similarity']:.4f}")
        print(f"  Coherence: original={result['original_coherence']:.4f}, expanded={result['expanded_coherence']:.4f}")
        print(f"  Duration: {duration:.4f}s")
        print()

    async def test_large_data_expansion(self):
        """Test expansion on large data."""
        print("Testing large data expansion...")
        
        # Lower expansion level for large data
        level = 0.5
        
        start_time = time.time()
        expanded = await self.expansion_system.expand_data(self.test_data_large, expansion_level=level)
        duration = time.time() - start_time
        
        # Verify expansion
        self.assertIsNotNone(expanded)
        self.assertGreater(len(expanded), len(self.test_data_large))
        
        # Compare data (sample for large data)
        sample_original = self.test_data_large[::10]  # Take every 10th element
        sample_expanded = expanded[::10]  # Take every 10th element
        result = self._compare_data(sample_original, sample_expanded)
        
        # Save results
        result['fig'].savefig(f"test_results/large_data_expansion.png")
        plt.close(result['fig'])
        
        print(f"Large data:")
        print(f"  Expansion ratio: {result['expansion_ratio']:.2f}x")
        print(f"  Frequency similarity: {result['frequency_similarity']:.4f}")
        print(f"  Coherence: original={result['original_coherence']:.4f}, expanded={result['expanded_coherence']:.4f}")
        print(f"  Duration: {duration:.4f}s")
        print()

    async def test_expansion_with_noise(self):
        """Test expansion resilience to noise."""
        print("Testing expansion with noise...")
        
        # Add varying levels of noise
        noise_levels = [0.1, 0.5, 1.0, 2.0]
        
        for noise_level in noise_levels:
            # Create noisy data
            noisy_data = self.test_data_medium + noise_level * np.random.normal(size=len(self.test_data_medium))
            
            # Expand noisy data
            expanded = await self.expansion_system.expand_data(noisy_data, expansion_level=0.75)
            
            # Compare data
            result = self._compare_data(noisy_data, expanded)
            
            # Save results
            result['fig'].savefig(f"test_results/noisy_data_expansion_level_{noise_level:.1f}.png")
            plt.close(result['fig'])
            
            print(f"Noise level {noise_level:.1f}:")
            print(f"  Expansion ratio: {result['expansion_ratio']:.2f}x")
            print(f"  Frequency similarity: {result['frequency_similarity']:.4f}")
            print(f"  Coherence: original={result['original_coherence']:.4f}, expanded={result['expanded_coherence']:.4f}")
            print()

    async def test_parallel_expansion(self):
        """Test parallel expansion capability."""
        print("Testing parallel expansion...")
        
        # Create a larger dataset
        large_data = self._generate_test_data(size=500000)
        
        # Test with different thread counts
        thread_counts = [1, 2, 4, 8]
        
        results = []
        for threads in thread_counts:
            # Create expansion system with specific thread count
            system = StructuredExpansionSystem(
                num_threads=threads,
                buffer_size=1024,
                fibonacci_depth=24
            )
            
            # Time the expansion
            start_time = time.time()
            expanded = await system.expand_data(large_data, expansion_level=0.5)
            duration = time.time() - start_time
            
            # Record results
            results.append({
                'threads': threads,
                'duration': duration,
                'expansion_ratio': len(expanded) / len(large_data)
            })
            
            print(f"Thread count {threads}:")
            print(f"  Duration: {duration:.4f}s")
            print(f"  Expansion ratio: {len(expanded) / len(large_data):.2f}x")
            print()
        
        # Plot scaling results
        plt.figure(figsize=(10, 6))
        threads = [r['threads'] for r in results]
        durations = [r['duration'] for r in results]
        speedups = [results[0]['duration'] / r['duration'] for r in results]
        
        plt.subplot(2, 1, 1)
        plt.plot(threads, durations, 'o-')
        plt.title('Expansion Time vs Thread Count')
        plt.xlabel('Thread Count')
        plt.ylabel('Duration (s)')
        plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.plot(threads, speedups, 'o-')
        plt.plot(threads, threads, 'r--', label='Ideal Scaling')
        plt.title('Speedup Factor vs Thread Count')
        plt.xlabel('Thread Count')
        plt.ylabel('Speedup Factor')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("test_results/parallel_scaling.png")
        plt.close()

def run_all_tests():
    """Run all tests asynchronously."""
    asyncio.run(_run_tests())

async def _run_tests():
    # Create test instance
    test = TestStructuredExpansion()
    test.setUp()
    
    # Run synchronous tests
    test.test_fibonacci_cache_initialization()
    
    # Run asynchronous tests
    await test.test_small_data_expansion()
    await test.test_medium_data_expansion()
    await test.test_large_data_expansion()
    await test.test_expansion_with_noise()
    await test.test_parallel_expansion()

if __name__ == "__main__":
    print("="*80)
    print("STRUCTURED EXPANSION SYSTEM TESTS")
    print("="*80)
    print("Running tests...")
    
    # Run all tests
    run_all_tests()
    
    print("="*80)
    print("Tests completed. Results saved in test_results directory.")
    print("="*80)
