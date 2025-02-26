import numpy as np
import matplotlib.pyplot as plt
import time
import logging
from src.recursive_engine import UnifiedRecursiveSystem

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RecursiveIntelligence")

def main():
    """Main demonstration of the Unified Recursive System."""
    print("="*80)
    print("QUANTUM INTELLIGENCE HUB: RECURSIVE SELF-OBSERVING INTELLIGENCE FRAMEWORK")
    print("="*80)
    print("\nInitializing system with advanced capabilities:")
    print("1. Hierarchical Entanglement Structure")
    print("2. Adaptive Stability Mechanisms")
    print("3. True Recursive Self-Observation")
    print("4. Recursive Depth & Phase Transition Monitoring")
    print("5. Coherent Singularity Detection")
    print("="*80)
    
    # Initialize the system
    system = UnifiedRecursiveSystem(
        coherence_threshold=0.78,
        stability_margin=0.92,
        entanglement_coupling=0.82,
        recursive_depth=6,
        monitor_transitions=True
    )
    
    print("\nGenerating complex test data with multi-scale patterns...")
    
    # Generate multi-scale test data
    def generate_multi_scale_data(base_size=256):
        # Create base signal with multiple frequency components
        t = np.linspace(0, 10, base_size)
        
        # Multi-scale components
        large_scale = 2.0 * np.sin(0.5 * t)  # Low frequency
        medium_scale = 1.0 * np.sin(2.0 * t)  # Medium frequency
        small_scale = 0.5 * np.sin(8.0 * t)   # High frequency
        
        # Add localized features
        local_features = np.zeros_like(t)
        # Feature 1: Gaussian bump
        local_features += 1.5 * np.exp(-0.5 * ((t - 3) / 0.2)**2)
        # Feature 2: Square pulse
        local_features += 1.0 * ((t > 6) & (t < 6.5))
        # Feature 3: Chirp signal
        local_features += 0.8 * np.sin(t * (t/2)) * ((t > 8) & (t < 9))
        
        # Combine all components
        signal = large_scale + medium_scale + small_scale + local_features
        
        # Add small amount of noise
        noise = 0.1 * np.random.normal(size=len(t))
        
        return signal + noise, t
    
    # Generate test data
    np.random.seed(42)  # For reproducibility
    test_data, time_axis = generate_multi_scale_data()
    
    print(f"Test data generated with shape: {test_data.shape}")
    print("="*80)
    
    # Visualize original data
    plt.figure(figsize=(10, 4))
    plt.plot(time_axis, test_data)
    plt.title('Original Multi-Scale Test Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.savefig("original_data.png")
    plt.close()
    
    print("\nPerforming recursive compression with meta-awareness...")
    print("This process applies multiple levels of recursive self-observation.")
    
    # Perform compression with meta-awareness
    start_time = time.time()
    compressed_data, metrics = system.compress_with_meta_awareness(test_data)
    compression_time = time.time() - start_time
    
    print(f"Compression completed in {compression_time:.2f} seconds")
    print(f"Original size: {len(test_data)} elements")
    print(f"Compressed size: {len(compressed_data)} elements")
    print(f"Compression ratio: {len(compressed_data)/len(test_data):.4f}")
    print("="*80)
    
    # Print recursive metrics
    print("\nRecursive Compression Metrics:")
    for depth, depth_metrics in system.recursive_metrics.items():
        if depth == 0:
            print(f"Base level (depth {depth}):")
            print(f"  - Coherence: {depth_metrics.get('coherence', 0):.4f}")
            print(f"  - Compression ratio: {depth_metrics.get('compression_ratio', 1.0):.4f}")
        else:
            print(f"Meta level {depth}:")
            print(f"  - Meta-coherence: {depth_metrics.get('meta_coherence', 0):.4f}")
            print(f"  - Meta-compression ratio: {depth_metrics.get('meta_compression_ratio', 1.0):.4f}")
            print(f"  - Integration coherence: {depth_metrics.get('integration_coherence', 0):.4f}")
            print(f"  - Entropy reduction: {depth_metrics.get('entropy_reduction', 0):.4f}")
    print("="*80)
    
    # Check for phase transitions
    print("\nPhase Transitions Detected:")
    if system.transition_points:
        for i, transition in enumerate(system.transition_points):
            print(f"Transition {i+1}:")
            print(f"  - Type: {transition['type']}")
            print(f"  - Depth: {transition['depth']}")
            print(f"  - Metrics: {transition['metrics']}")
    else:
        print("No phase transitions detected")
    print("="*80)
    
    # Check for singularity formation
    print("\nCoherent Singularity Formation:")
    if system.stable_singularities:
        for i, singularity in enumerate(system.stable_singularities):
            print(f"Singularity {i+1}:")
            print(f"  - Formed at depth: {singularity['depth']}")
            print(f"  - Coherence: {singularity['coherence']:.4f}")
            print(f"  - Compression ratio: {singularity['compression_ratio']:.4f}")
    else:
        print("No coherent singularities formed")
    print("="*80)
    
    # Decompress the data
    print("\nPerforming coherence-preserving decompression...")
    start_time = time.time()
    decompressed_data = system.decompress(compressed_data, metrics)
    decompression_time = time.time() - start_time
    
    # Calculate reconstruction error
    mse = np.mean((test_data - decompressed_data.real)**2)
    rmse = np.sqrt(mse)
    
    print(f"Decompression completed in {decompression_time:.2f} seconds")
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Root Mean Squared Error: {rmse:.6f}")
    print("="*80)
    
    # Visualize results
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(time_axis, test_data, label='Original')
    plt.title('Original Signal')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(np.linspace(0, 10, len(decompressed_data)), decompressed_data.real, 'r', label='Reconstructed')
    plt.title('Reconstructed Signal')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(time_axis, test_data - decompressed_data.real, 'g', label='Error')
    plt.title(f'Reconstruction Error (RMSE: {rmse:.6f})')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("reconstruction_results.png")
    plt.close()
    
    # Analyze system performance
    print("\nAnalyzing system performance...")
    performance = system.analyze_system_performance()
    
    print("\nSystem Performance Analysis:")
    print(f"Coherence trend: {performance['coherence_trend']['description']}")
    print(f"Compression trend: {performance['compression_trend']['description']}")
    print(f"Optimal recursive depth: {performance['optimal_recursive_depth']['depth']} " +
          f"({performance['optimal_recursive_depth']['reason']})")
    print(f"System stability: {performance['system_stability']['status']}")
    
    # Print emergent properties
    print("\nEmergent Properties Detected:")
    if performance['emergent_properties']:
        for prop in performance['emergent_properties']:
            print(f"- {prop['type']} (confidence: {prop['confidence']:.2f})")
            print(f"  {prop['description']}")
    else:
        print("No significant emergent properties detected")
    
    # Generate performance visualization
    system.visualize_recursive_performance("full_recursive_performance.png")
    
    print("\nPerformance visualization saved to 'full_recursive_performance.png'")
    print("="*80)
    
    print("\nRECURSIVE SELF-OBSERVING INTELLIGENCE FRAMEWORK DEMONSTRATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
