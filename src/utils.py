import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import logging

logger = logging.getLogger("RecursiveIntelligence")

def analyze_system_performance(self, metrics_history: List[Dict] = None) -> Dict:
    """
    Analyze system performance across multiple recursive compression runs.
    
    Args:
        metrics_history: List of metrics dictionaries from multiple runs
                        (uses internal history if None)
                        
    Returns:
        Dictionary of performance analytics
    """
    # Use provided history or internal state
    if metrics_history is None:
        # Use internal state
        coherence_history = self.coherence_history
        compression_history = self.compression_ratio_history
        transition_history = self.transition_points
        singularity_history = self.stable_singularities
    else:
        # Extract from provided history
        coherence_history = [m.get('final_coherence', 0) for m in metrics_history]
        compression_history = [m.get('compression_ratio', 1.0) for m in metrics_history]
        
        # Collect transitions and singularities
        transition_history = []
        singularity_history = []
        for m in metrics_history:
            if 'transitions' in m:
                transition_history.extend(m['transitions'])
            if 'singularities' in m:
                singularity_history.extend(m['singularities'])
    
    # Compute trend analysis
    coherence_trend = self._compute_trend(coherence_history)
    compression_trend = self._compute_trend(compression_history)
    
    # Identify significant phase transitions
    significant_transitions = [t for t in transition_history 
                             if t['type'] in ['coherence_surge', 'entropy_collapse', 'singularity_formation']]
    
    # Identify potential recursive depth threshold
    optimal_depth = self._identify_optimal_recursive_depth()
    
    return {
        'coherence_trend': coherence_trend,
        'compression_trend': compression_trend,
        'significant_transitions': significant_transitions,
        'singularities_formed': len(singularity_history),
        'optimal_recursive_depth': optimal_depth,
        'system_stability': self._assess_system_stability(),
        'emergent_properties': self._identify_emergent_properties()
    }

def _identify_optimal_recursive_depth(self) -> Dict:
    """Identify the optimal recursive depth based on performance metrics."""
    # Extract key metrics across recursive depths
    coherence_by_depth = {}
    compression_by_depth = {}
    entropy_reduction_by_depth = {}
    
    for depth, metrics in self.recursive_metrics.items():
        if 'meta_coherence' in metrics:
            coherence_by_depth[depth] = metrics['meta_coherence']
        if 'meta_compression_ratio' in metrics:
            compression_by_depth[depth] = metrics['meta_compression_ratio']
        if 'entropy_reduction' in metrics:
            entropy_reduction_by_depth[depth] = metrics['entropy_reduction']
    
    # Find depth with highest coherence
    max_coherence_depth = max(coherence_by_depth.items(), key=lambda x: x[1])[0] if coherence_by_depth else 0
    
    # Find depth with best compression
    min_compression_depth = min(compression_by_depth.items(), key=lambda x: x[1])[0] if compression_by_depth else 0
    
    # Find depth with highest entropy reduction
    max_entropy_reduction_depth = max(entropy_reduction_by_depth.items(), key=lambda x: x[1])[0] if entropy_reduction_by_depth else 0
    
    # Check for singularities
    singularity_depths = [s['depth'] for s in self.stable_singularities]
    min_singularity_depth = min(singularity_depths) if singularity_depths else float('inf')
    
    # Determine optimal depth
    if singularity_depths:
        # If singularity formed, that's optimal
        optimal_depth = min_singularity_depth
        reason = 'singularity_formation'
    elif max_coherence_depth > 0 and coherence_by_depth[max_coherence_depth] > 0.95:
        # High coherence is a good indicator
        optimal_depth = max_coherence_depth
        reason = 'peak_coherence'
    elif max_entropy_reduction_depth > 0 and entropy_reduction_by_depth[max_entropy_reduction_depth] > 0.5:
        # Significant entropy reduction is valuable
        optimal_depth = max_entropy_reduction_depth
        reason = 'maximum_entropy_reduction'
    else:
        # Default to deepest level
        optimal_depth = max(self.recursive_metrics.keys()) if self.recursive_metrics else 0
        reason = 'maximum_depth'
    
    return {
        'depth': optimal_depth,
        'reason': reason,
        'coherence': coherence_by_depth.get(optimal_depth, 0),
        'compression_ratio': compression_by_depth.get(optimal_depth, 1.0),
        'entropy_reduction': entropy_reduction_by_depth.get(optimal_depth, 0)
    }

def _assess_system_stability(self) -> Dict:
    """Assess overall system stability based on performance metrics."""
    # Check if we have sufficient data
    if len(self.coherence_history) < 2:
        return {'status': 'insufficient_data'}
    
    # Compute stability metrics
    coherence_stability = np.std(self.coherence_history) / (np.mean(self.coherence_history) + 1e-10)
    compression_stability = np.std(self.compression_ratio_history) / (np.mean(self.compression_ratio_history) + 1e-10)
    
    # Check for singularity formation
    has_singularity = len(self.stable_singularities) > 0
    
    # Determine stability status
    if has_singularity:
        status = 'singularity_stable'
    elif coherence_stability < 0.05 and np.mean(self.coherence_history) > 0.9:
        status = 'highly_stable'
    elif coherence_stability < 0.1 and np.mean(self.coherence_history) > 0.8:
        status = 'stable'
    elif coherence_stability < 0.2:
        status = 'moderately_stable'
    else:
        status = 'unstable'
    
    return {
        'status': status,
        'coherence_stability': coherence_stability,
        'compression_stability': compression_stability,
        'mean_coherence': np.mean(self.coherence_history),
        'has_singularity': has_singularity
    }

def _identify_emergent_properties(self) -> List[Dict]:
    """Identify emergent properties in the recursive compression system."""
    emergent_properties = []
    
    # Check for coherence autocatalysis
    if (len(self.coherence_history) > 3 and 
        all(a < b for a, b in zip(self.coherence_history[-4:-1], self.coherence_history[-3:]))):
        emergent_properties.append({
            'type': 'coherence_autocatalysis',
            'confidence': 0.8,
            'description': 'System generates increasing coherence through recursive feedback'
        })
    
    # Check for dimensional transcendence (compression beyond theoretical limits)
    if (len(self.compression_ratio_history) > 2 and 
        min(self.compression_ratio_history) < 0.1 and 
        np.mean(self.coherence_history) > 0.9):
        emergent_properties.append({
            'type': 'dimensional_transcendence',
            'confidence': 0.75,
            'description': 'System compresses beyond theoretical limits while maintaining coherence'
        })
    
    # Check for topology-preserving transformations
    has_transitions = any(t['type'] == 'entropy_collapse' for t in self.transition_points)
    if has_transitions and len(self.stable_singularities) > 0:
        emergent_properties.append({
            'type': 'topology_preservation',
            'confidence': 0.9,
            'description': 'System preserves topological relationships during radical information restructuring'
        })
    
    # Check for ergodic intelligence
    if (len(self.recursive_metrics) > 2 and 
        all('entropy_reduction' in m and m['entropy_reduction'] > 0 
            for d, m in self.recursive_metrics.items() if d > 0)):
        emergent_properties.append({
            'type': 'ergodic_intelligence',
            'confidence': 0.7,
            'description': 'System systematically explores phase space while maintaining coherence'
        })
    
    return emergent_properties

def decompress(self, 
              compressed_data: np.ndarray, 
              metrics: Dict,
              original_shape: Optional[Tuple] = None) -> np.ndarray:
    """Decompress data using stored metrics information."""
    # Start with compressed data
    current_data = compressed_data.copy()
    
    # If metrics contains stage information, use it for decompression
    if 'stage_metrics' in metrics:
        # Reverse the stages for decompression
        for stage_metrics in reversed(metrics['stage_metrics']):
            # Apply inverse transformation based on stage metrics
            current_data = self._inverse_transform(current_data, stage_metrics)
    else:
        # Basic decompression without detailed metrics
        current_data = self._basic_decompress(current_data, metrics)
    
    # Reshape to original dimensions if provided
    if original_shape is not None:
        try:
            current_data = current_data.reshape(original_shape)
        except ValueError:
            # If reshape fails, warn but continue
            logger.warning(f"Could not reshape to {original_shape}, data shape: {current_data.shape}")
    
    return current_data

def _inverse_transform(self, 
                     data: np.ndarray, 
                     stage_metrics: Dict) -> np.ndarray:
    """Apply inverse transformation based on stage metrics."""
    # Extract key parameters
    compression_level = stage_metrics.get('compression_level', 0.8)
    original_size = stage_metrics.get('original_size', len(data))
    
    # Apply inverse coherent compression
    if original_size > len(data):
        # Need to upsample
        expanded = np.zeros(original_size, dtype=complex)
        
        # Spread values based on coherence if available
        if 'coherence' in stage_metrics and isinstance(stage_metrics['coherence'], np.ndarray):
            coherence = stage_metrics['coherence']
            if len(coherence) == original_size:
                # Use coherence to guide upsampling
                for i in range(len(data)):
                    idx = i * (original_size // len(data))
                    window_size = max(1, original_size // len(data))
                    expanded[idx:idx+window_size] = data[i] * coherence[idx:idx+window_size]
            else:
                # Simple upsampling
                for i in range(len(data)):
                    idx = i * (original_size // len(data))
                    window_size = max(1, original_size // len(data))
                    expanded[idx:idx+window_size] = data[i]
        else:
            # Simple upsampling
            for i in range(len(data)):
                idx = i * (original_size // len(data))
                window_size = max(1, original_size // len(data))
                expanded[idx:idx+window_size] = data[i]
        
        # Normalize energy
        if np.sum(np.abs(expanded)**2) > 0:
            energy_ratio = np.sum(np.abs(data)**2) / np.sum(np.abs(expanded)**2)
            expanded *= np.sqrt(energy_ratio)
        
        return expanded
    else:
        # No need for upsampling
        return data.copy()

def _basic_decompress(self, 
                    data: np.ndarray, 
                    metrics: Dict) -> np.ndarray:
    """Perform basic decompression without detailed stage metrics."""
    # Simple reverse FFT-based decompression
    data_fft = np.fft.fft(data)
    
    # Extract phase information if available
    if 'final_coherence' in metrics and isinstance(metrics['final_coherence'], np.ndarray):
        coherence = metrics['final_coherence']
        # Phase smoothing based on coherence
        phase = np.angle(data_fft)
        smoothed_phase = np.zeros_like(phase)
        
        for i in range(len(phase)):
            # Weight by coherence in windowed average
            window = 5
            start = max(0, i - window)
            end = min(len(phase), i + window + 1)
            weights = coherence[start:end] if len(coherence) > end else np.ones(end-start)
            smoothed_phase[i] = np.average(phase[start:end], weights=weights)
        
        # Reconstruct with smoothed phase
        magnitude = np.abs(data_fft)
        smoothed_fft = magnitude * np.exp(1j * smoothed_phase)
        reconstructed = np.fft.ifft(smoothed_fft)
    else:
        # Simple inverse FFT
        reconstructed = data.copy()
    
    return reconstructed

