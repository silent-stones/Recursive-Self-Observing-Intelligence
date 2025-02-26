import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger("RecursiveIntelligence")

def visualize_recursive_performance(self, 
                                 save_path: Optional[str] = None) -> None:
    """
    Visualize the performance of recursive compression across depths.
    
    Args:
        save_path: Path to save the visualization (shows plot if None)
    """
    # Extract metrics for visualization
    depths = sorted(self.recursive_metrics.keys())
    
    # Prepare data
    coherence_values = []
    compression_values = []
    entropy_reduction_values = []
    
    for depth in depths:
        metrics = self.recursive_metrics[depth]
        
        # Get coherence
        if depth == 0:
            coherence_values.append(metrics.get('coherence', 0))
        else:
            coherence_values.append(metrics.get('meta_coherence', 0))
        
        # Get compression ratio
        if depth == 0:
            compression_values.append(metrics.get('compression_ratio', 1.0))
        else:
            compression_values.append(metrics.get('meta_compression_ratio', 1.0))
        
        # Get entropy reduction
        entropy_reduction_values.append(metrics.get('entropy_reduction', 0))
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot coherence
    plt.subplot(3, 1, 1)
    plt.plot(depths, coherence_values, 'b-o', linewidth=2, markersize=8)
    plt.axhline(y=self.coherence_threshold, color='r', linestyle='--', label=f'Threshold ({self.coherence_threshold})')
    plt.ylabel('Coherence')
    plt.title('Coherence vs. Recursive Depth')
    plt.grid(True)
    plt.legend()
    
    # Plot compression ratio
    plt.subplot(3, 1, 2)
    plt.plot(depths, compression_values, 'g-o', linewidth=2, markersize=8)
    plt.ylabel('Compression Ratio')
    plt.title('Compression Ratio vs. Recursive Depth')
    plt.grid(True)
    
    # Plot entropy reduction
    plt.subplot(3, 1, 3)
    plt.plot(depths, entropy_reduction_values, 'm-o', linewidth=2, markersize=8)
    plt.xlabel('Recursive Depth')
    plt.ylabel('Entropy Reduction')
    plt.title('Entropy Reduction vs. Recursive Depth')
    plt.grid(True)
    
    # Mark phase transitions
    for transition in self.transition_points:
        depth = transition['depth']
        transition_type = transition['type']
        
        if transition_type == 'coherence_surge':
            plt.subplot(3, 1, 1)
            plt.plot(depth, coherence_values[depth], 'r*', markersize=15, label='Coherence Surge')
        elif transition_type == 'entropy_collapse':
            plt.subplot(3, 1, 3)
            plt.plot(depth, entropy_reduction_values[depth], 'r*', markersize=15, label='Entropy Collapse')
    
    # Mark singularities
    for singularity in self.stable_singularities:
        depth = singularity['depth']
        plt.subplot(3, 1, 1)
        plt.plot(depth, coherence_values[depth], 'ko', markersize=15, fillstyle='none', label='Singularity')
    
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_knowledge_integration(decompressed, original, integration_coherence):
    """Visualize how well knowledge is integrated across domains after compression."""
    # Compute similarity matrices
    original_sim = original @ original.T
    decompressed_sim = decompressed @ decompressed.T
    
    # Normalize to [0,1] for visualization
    original_sim_norm = (original_sim - np.min(original_sim)) / (np.max(original_sim) - np.min(original_sim))
    decompressed_sim_norm = (decompressed_sim - np.min(decompressed_sim)) / (np.max(decompressed_sim) - np.min(decompressed_sim))
    
    # Compute difference to highlight changes
    diff = np.abs(original_sim_norm - decompressed_sim_norm)
    
    # Create visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_sim_norm, cmap='viridis')
    plt.title('Original Knowledge Relationships')
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(decompressed_sim_norm, cmap='viridis')
    plt.title(f'Integrated Knowledge (Coherence: {integration_coherence:.4f})')
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    plt.imshow(diff, cmap='hot')
    plt.title('Relationship Differences')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig("knowledge_integration.png")
    plt.close()
