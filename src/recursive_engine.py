import numpy as np
import logging
import time
from typing import Dict, Tuple, List, Optional, Union
import matplotlib.pyplot as plt
from scipy import signal

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("RecursiveIntelligence")

class UnifiedRecursiveSystem:
    """
    Implements an advanced framework for recursive compression and intelligence singularity formation
    with hierarchical entanglement, adaptive stability, and true recursive self-observation.
    """
    
    def __init__(self, 
                coherence_threshold: float = 0.85,
                stability_margin: float = 0.95,
                entanglement_coupling: float = 0.78,
                recursive_depth: int = 3,
                monitor_transitions: bool = True):
        """
        Initialize the Unified Recursive System with tunable parameters.
        
        Args:
            coherence_threshold: Minimum acceptable coherence level
            stability_margin: Target stability level for corrections
            entanglement_coupling: Strength of entanglement relationships
            recursive_depth: Maximum depth of recursive self-observation
            monitor_transitions: Whether to monitor for phase transitions
        """
        # Core parameters
        self.coherence_threshold = coherence_threshold
        self.stability_margin = stability_margin
        self.entanglement_coupling = entanglement_coupling
        self.recursive_depth = recursive_depth
        self.monitor_transitions = monitor_transitions
        
        # Metrics tracking
        self.coherence_history = []
        self.entanglement_history = []
        self.compression_ratio_history = []
        self.recursive_metrics = {i: {} for i in range(recursive_depth + 1)}
        
        # Phase transition monitoring
        self.transition_points = []
        self.stable_singularities = []
        
        logger.info(f"Initialized Unified Recursive System with recursive depth {recursive_depth}")
    
    def compress_with_meta_awareness(self, 
                                    data: np.ndarray, 
                                    max_recursive_depth: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
        """
        Compression with meta-awareness of its own processes, implementing true
        recursive self-observation up to specified recursive depth.
        
        Args:
            data: Input data array to compress
            max_recursive_depth: Maximum recursive depth (defaults to self.recursive_depth)
            
        Returns:
            Tuple of (compressed_data, comprehensive_metrics)
        """
        if max_recursive_depth is None:
            max_recursive_depth = self.recursive_depth
            
        # First-order compression (base case)
        phase_space = self._initialize_phase_space(data)
        
        # Track original data properties for comparison
        original_size = data.size
        original_entropy = self._compute_entropy(data)
        
        # Perform primary compression
        compressed_data, phase_metrics = self._recursive_compress(data, phase_space)
        
        # Store first-order metrics
        self.recursive_metrics[0] = {
            'compressed_size': compressed_data.size,
            'compression_ratio': compressed_data.size / original_size,
            'coherence': np.mean(phase_metrics['final_coherence']),
            'entropy': self._compute_entropy(compressed_data),
            'entropy_reduction': original_entropy - self._compute_entropy(compressed_data)
        }
        
        # Early return if no recursion requested
        if max_recursive_depth <= 0:
            return compressed_data, {
                'primary_metrics': phase_metrics,
                'recursive_metrics': self.recursive_metrics
            }
            
        # Recursive self-observation and meta-compression
        current_data = compressed_data
        current_metrics = phase_metrics
        
        for depth in range(1, max_recursive_depth + 1):
            logger.info(f"Performing recursive self-observation at depth {depth}")
            
            # Self-observation: encode the compression process itself
            process_representation = self._encode_compression_process(current_metrics)
            
            # Create a new phase space for this level of recursion
            meta_phase_space = self._initialize_phase_space(process_representation)
            
            # Apply entanglement from previous level
            meta_phase_space['entanglement'] = self._propagate_entanglement(
                phase_space['entanglement'], 
                meta_phase_space['phase']
            )
            
            # Meta-compression: compress the process representation
            meta_compressed, meta_metrics = self._recursive_compress(
                process_representation, meta_phase_space
            )
            
            # Integrate meta-insights back into primary process
            refined_compression = self._integrate_meta_insights(
                current_data, 
                meta_compressed, 
                current_metrics, 
                meta_metrics
            )
            
            # Check for phase transitions
            if self.monitor_transitions:
                transition = self._detect_phase_transition(refined_compression, depth)
                if transition['detected']:
                    self.transition_points.append({
                        'depth': depth,
                        'type': transition['type'],
                        'metrics': transition['metrics']
                    })
                    logger.info(f"Phase transition detected at depth {depth}: {transition['type']}")
            
            # Store metrics for this recursive depth
            self.recursive_metrics[depth] = {
                'meta_compressed_size': meta_compressed.size,
                'meta_compression_ratio': meta_compressed.size / process_representation.size,
                'meta_coherence': np.mean(meta_metrics['final_coherence']),
                'integration_coherence': self._compute_integration_coherence(
                    current_metrics, meta_metrics
                ),
                'refined_entropy': self._compute_entropy(refined_compression),
                'entropy_reduction': self._compute_entropy(current_data) - self._compute_entropy(refined_compression)
            }
            
            # Update for next recursion level
            current_data = refined_compression
            current_metrics = meta_metrics
            
            # Check for singularity formation
            if self._check_singularity_formation(refined_compression, depth):
                self.stable_singularities.append({
                    'depth': depth,
                    'coherence': np.mean(meta_metrics['final_coherence']),
                    'compression_ratio': meta_compressed.size / original_size
                })
                logger.info(f"Coherent singularity formed at depth {depth} with coherence {np.mean(meta_metrics['final_coherence']):.4f}")
                break
        
        # Store history for trend analysis
        self.coherence_history.append(np.mean(current_metrics['final_coherence']))
        self.compression_ratio_history.append(current_data.size / original_size)
        
        return current_data, {
            'primary_metrics': phase_metrics,
            'final_metrics': current_metrics,
            'recursive_metrics': self.recursive_metrics,
            'transitions': self.transition_points,
            'singularities': self.stable_singularities
        }
    
    def _initialize_phase_space(self, data: np.ndarray) -> Dict:
        """Initialize quantum-inspired phase space with hierarchical structure."""
        # Ensure data is at least 1D
        if np.isscalar(data):
            data = np.array([data])
            
        # Reshape if needed
        data_shaped = data.reshape(-1)
        
        # Compute initial phase relationships using FFT
        fft_data = np.fft.fft(data_shaped)
        phase = np.angle(fft_data)
        magnitude = np.abs(fft_data)
        
        # Create hierarchical phase space
        phase_space = {
            'phase': phase,
            'magnitude': magnitude,
            'coherence': self._compute_initial_coherence(phase),
            'entanglement': self._initialize_hierarchical_entanglement(phase),
            'dimensions': data.shape
        }
        
        return phase_space
    
    def _compute_initial_coherence(self, phase: np.ndarray) -> np.ndarray:
        """Compute initial coherence based on phase stability."""
        # Calculate phase gradient
        phase_grad = np.gradient(phase)
        
        # Compute local coherence (inverse of gradient magnitude)
        local_coherence = 1.0 / (1.0 + np.abs(phase_grad))
        
        # Normalize to [0,1]
        normalized_coherence = (local_coherence - np.min(local_coherence)) / \
                             (np.max(local_coherence) - np.min(local_coherence) + 1e-10)
        
        return normalized_coherence
    
    def _initialize_hierarchical_entanglement(self, phase: np.ndarray) -> Dict:
        """Initialize multi-level hierarchical entanglement structure."""
        n = len(phase)
        levels = max(1, int(np.log2(n)))
        
        # Create hierarchical entanglement
        hierarchy = {}
        for level in range(levels):
            scale = max(1, 2**level)
            level_size = max(1, n // scale)
            
            # Create entanglement matrix at this scale
            entanglement = np.zeros((level_size, level_size), dtype=complex)
            
            for i in range(level_size):
                for j in range(level_size):
                    # Average phase over regions
                    i_start, i_end = i*scale, min(n, (i+1)*scale)
                    j_start, j_end = j*scale, min(n, (j+1)*scale)
                    
                    phase_i = np.mean(phase[i_start:i_end])
                    phase_j = np.mean(phase[j_start:j_end])
                    phase_diff = phase_i - phase_j
                    
                    # Compute entanglement with phase difference
                    entanglement[i,j] = np.exp(1j * phase_diff)
            
            # Normalize
            if np.sum(np.abs(entanglement)**2) > 0:
                entanglement /= np.sqrt(np.sum(np.abs(entanglement)**2))
            
            hierarchy[f'level_{level}'] = entanglement
        
        return hierarchy
    
    def _propagate_entanglement(self, 
                              source_entanglement: Dict, 
                              target_phase: np.ndarray) -> Dict:
        """Propagate entanglement structure from one level to another."""
        n = len(target_phase)
        levels = max(1, int(np.log2(n)))
        
        # Create new hierarchical entanglement based on source
        hierarchy = {}
        for level in range(levels):
            source_level = f'level_{min(level, max(0, len(source_entanglement.keys())-1))}'
            source_matrix = source_entanglement.get(source_level, None)
            
            scale = max(1, 2**level)
            level_size = max(1, n // scale)
            
            # Create entanglement matrix at this scale
            entanglement = np.zeros((level_size, level_size), dtype=complex)
            
            for i in range(level_size):
                for j in range(level_size):
                    # Average phase over regions
                    i_start, i_end = i*scale, min(n, (i+1)*scale)
                    j_start, j_end = j*scale, min(n, (j+1)*scale)
                    
                    phase_i = np.mean(target_phase[i_start:i_end])
                    phase_j = np.mean(target_phase[j_start:j_end])
                    phase_diff = phase_i - phase_j
                    
                    # Incorporate source entanglement if available
                    if source_matrix is not None and i < source_matrix.shape[0] and j < source_matrix.shape[1]:
                        source_strength = np.abs(source_matrix[i % source_matrix.shape[0], j % source_matrix.shape[1]])
                        entanglement[i,j] = source_strength * np.exp(1j * phase_diff)
                    else:
                        entanglement[i,j] = np.exp(1j * phase_diff)
            
            # Normalize
            if np.sum(np.abs(entanglement)**2) > 0:
                entanglement /= np.sqrt(np.sum(np.abs(entanglement)**2))
            
            hierarchy[f'level_{level}'] = entanglement
        
        return hierarchy

