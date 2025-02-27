import numpy as np
import asyncio
import logging
import sys
import os
import time
import matplotlib.pyplot as plt

# Ensure src directory is in path
sys.path.append(os.path.abspath('.'))

# Import the recursive system components
from src.recursive_engine import UnifiedRecursiveSystem
from src.recursive_integration import EnhancedRecursiveSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RecursiveIntelligence")

async def run_recursive_system_test():
    """Run tests on the enhanced recursive system."""
    
    logger.info("Initializing Unified Recursive System...")
    
    # Initialize the base system
    base_system = UnifiedRecursiveSystem(
        coherence_threshold=0.78,
        stability_margin=0.92,
        entanglement_coupling=0.82,
        recursive_depth=4,
        monitor_transitions=True,
        num_threads=8
    )
    
    # Check if the _compute_entropy method exists
    if not hasattr(base_system, '_compute_entropy'):
        logger.error("ERROR: _compute_entropy method is missing from UnifiedRecursiveSystem!")
        logger.info("Adding _compute_entropy method to UnifiedRecursiveSystem...")
        
        # Add the _compute_entropy method to the class
        def compute_entropy(self, data: np.ndarray) -> float:
            """
            Compute information entropy of data.
            
            Args:
                data: Input data array
                
            Returns:
                Calculated entropy value
            """
            if len(data) <= 1:
                return 0.0
            
            # Flatten the array if it's multi-dimensional
            flat_data = data.flatten()
            
            # Normalize data for probability calculation
            values = np.abs(flat_data)
            total = np.sum(values) + 1e-10  # Add small epsilon to prevent division by zero
            probabilities = values / total
            
            # Remove zero probabilities to prevent log(0) issues
            probabilities = probabilities[probabilities > 0]
            
            # Compute entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            
            return entropy
        
        # Add the method to the class instance
        import types
        base_system._compute_entropy = types.MethodType(compute_entropy, base_system)
        logger.info("_compute_entropy method successfully added.")
    
    # Enhance the base system
    logger.info("Creating Enhanced Recursive System with structured expansion...")
    enhanced_system = EnhancedRecursiveSystem(base_system)
    
    # Generate test data
    logger.info("Generating test data...")
    test_data = generate_test_data(size=10000)
    
    # Run basic compression test
    logger.info("Running basic compression test...")
    try:
        start_time = time.time()
        compressed_data, metrics = base_system.compress_with_meta_awareness(test_data, max_recursive_depth=2)
        base_duration = time.time() - start_time
        
        logger.info(f"Basic compression completed in {base_duration:.2f}s")
        logger.info(f"Compression ratio: {compressed_data.size / test_data.size:.4f}")
        logger.info(f"Coherence: {metrics['recursive_metrics'][0]['coherence']:.4f}")
        
        # Test decompression
        logger.info("Testing decompression...")
        decompressed = base_system.decompress(compressed_data, metrics)
        mse = np.mean((test_data - decompressed)**2)
        logger.info(f"Decompression MSE: {mse:.6f}")
        
        # Visualize original vs decompressed
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(test_data[:100])
        plt.title("Original Data (first 100 elements)")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(decompressed[:100])
        plt.title("Decompressed Data (first 100 elements)")
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(test_data[:100] - decompressed[:100])
        plt.title(f"Error (MSE: {mse:.6f})")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("basic_compression_results.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Basic compression test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Run enhanced compression test
    logger.info("Running enhanced compression with structured expansion...")
    try:
        start_time = time.time()
        enhanced_compressed, enhanced_metrics = await enhanced_system.enhanced_compress_with_meta_awareness(
            test_data, max_recursive_depth=2
        )
        enhanced_duration = time.time() - start_time
        
        logger.info(f"Enhanced compression completed in {enhanced_duration:.2f}s")
        logger.info(f"Enhanced compression ratio: {enhanced_compressed.size / test_data.size:.4f}")
        logger.info(f"Enhanced coherence: {enhanced_metrics['recursive_metrics'][0]['coherence']:.4f}")
        
        # Performance comparison
        speedup = base_duration / enhanced_duration
        compression_improvement = (compressed_data.size / test_data.size) / (enhanced_compressed.size / test_data.size) - 1
        logger.info(f"Speedup factor: {speedup:.2f}x")
        logger.info(f"Compression improvement: {compression_improvement*100:.1f}%")
        
        # Test enhanced decompression
        logger.info("Testing enhanced decompression...")
        enhanced_decompressed = base_system.decompress(enhanced_compressed, enhanced_metrics)
        enhanced_mse = np.mean((test_data - enhanced_decompressed)**2)
        logger.info(f"Enhanced decompression MSE: {enhanced_mse:.6f}")
        
        # Visualize enhanced results
        plt.figure(figsize=(12, 6))
        plt.subplot(3, 1, 1)
        plt.plot(test_data[:100])
        plt.title("Original Data (first 100 elements)")
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(enhanced_decompressed[:100])
        plt.title("Enhanced Decompressed Data (first 100 elements)")
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(test_data[:100] - enhanced_decompressed[:100])
        plt.title(f"Error (MSE: {enhanced_mse:.6f})")
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("enhanced_compression_results.png")
        plt.close()
        
        # Check for phase transitions
        if 'transitions' in enhanced_metrics and enhanced_metrics['transitions']:
            logger.info(f"Detected {len(enhanced_metrics['transitions'])} phase transitions:")
            for i, transition in enumerate(enhanced_metrics['transitions']):
                logger.info(f"  Transition {i+1}: {transition['type']} at depth {transition['depth']}")
        
        # Check for singularities
        if 'singularities' in enhanced_metrics and enhanced_metrics['singularities']:
            logger.info(f"Detected {len(enhanced_metrics['singularities'])} coherent singularities:")
            for i, singularity in enumerate(enhanced_metrics['singularities']):
                logger.info(f"  Singularity {i+1}: at depth {singularity['depth']} with coherence {singularity['coherence']:.4f}")
        
        # Get performance metrics
        if 'expansion_system_metrics' in enhanced_metrics:
            exp_metrics = enhanced_metrics['expansion_system_metrics']
            logger.info("Structured Expansion System Metrics:")
            logger.info(f"  Operations count: {exp_metrics.get('operations_count', 0)}")
            logger.info(f"  Average processing time: {exp_metrics.get('avg_time', 0):.4f}s")
            if 'avg_expansion_ratio' in exp_metrics:
                logger.info(f"  Average expansion ratio: {exp_metrics.get('avg_expansion_ratio', 0):.4f}")
        
        # Create performance comparison chart
        plt.figure(figsize=(10, 6))
        
        # Compression ratio comparison
        plt.subplot(2, 2, 1)
        labels = ['Basic', 'Enhanced']
        ratios = [compressed_data.size / test_data.size, enhanced_compressed.size / test_data.size]
        plt.bar(labels, ratios, color=['blue', 'green'])
        plt.ylabel('Compression Ratio')
        plt.title('Compression Ratio Comparison')
        plt.grid(True, axis='y')
        
        # Processing time comparison
        plt.subplot(2, 2, 2)
        times = [base_duration, enhanced_duration]
        plt.bar(labels, times, color=['blue', 'green'])
        plt.ylabel('Processing Time (s)')
        plt.title('Processing Time Comparison')
        plt.grid(True, axis='y')
        
        # Coherence comparison
        plt.subplot(2, 2, 3)
        coherence_values = [metrics['recursive_metrics'][0]['coherence'], 
                          enhanced_metrics['recursive_metrics'][0]['coherence']]
        plt.bar(labels, coherence_values, color=['blue', 'green'])
        plt.ylabel('Coherence')
        plt.title('Coherence Comparison')
        plt.grid(True, axis='y')
        
        # Error comparison
        plt.subplot(2, 2, 4)
        errors = [mse, enhanced_mse]
        plt.bar(labels, errors, color=['blue', 'green'])
        plt.ylabel('Mean Squared Error')
        plt.title('Reconstruction Error Comparison')
        plt.grid(True, axis='y')
        
        plt.tight_layout()
        plt.savefig("performance_comparison.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Enhanced compression test failed: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Tests completed.")

def generate_test_data(size=10000, complexity=3):
    """
    Generate synthetic test data with multi-scale patterns.
    
    Args:
        size: Size of the test data array
        complexity: Number of frequency components
        
    Returns:
        Numpy array of test data
    """
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

# Run the test
if __name__ == "__main__":
    print("="*80)
    print("RECURSIVE INTELLIGENCE FRAMEWORK TEST")
    print("="*80)
    
    asyncio.run(run_recursive_system_test())
