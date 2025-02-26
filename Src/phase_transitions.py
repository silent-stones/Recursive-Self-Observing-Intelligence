import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger("RecursiveIntelligence")

def _detect_phase_transition(self, 
                           data: np.ndarray, 
                           recursive_depth: int) -> Dict:
    """Detect phase transitions in the data structure."""
    # Default result
    result = {
        'detected': False,
        'type': None,
        'metrics': {}
    }
    
    # Check if we have enough history
    if len(self.coherence_history) < 2 or recursive_depth < 1:
        return result
    
    # Compute key metrics
    current_coherence = self.recursive_metrics[recursive_depth].get('meta_coherence', 0)
    previous_coherence = self.recursive_metrics[recursive_depth-1].get('meta_coherence', 0) if recursive_depth > 0 else 0
    
    current_entropy = self._compute_entropy(data)
    
    # Check for critical coherence transition
    if current_coherence > 0.9 and current_coherence - previous_coherence > 0.2:
        result['detected'] = True
        result['type'] = 'coherence_surge'
        result['metrics'] = {
            'coherence_delta': current_coherence - previous_coherence,
            'current_coherence': current_coherence,
            'entropy': current_entropy
        }
        return result
    
    # Check for entropy collapse
    if len(self.compression_ratio_history) > 1:
        current_ratio = self.recursive_metrics[recursive_depth].get('meta_compression_ratio', 1.0)
        previous_ratio = self.compression_ratio_history[-1]
        
        if current_ratio < 0.5 * previous_ratio and current_coherence > 0.8:
            result['detected'] = True
            result['type'] = 'entropy_collapse'
            result['metrics'] = {
                'compression_ratio_delta': previous_ratio - current_ratio,
                'current_ratio': current_ratio,
                'current_coherence': current_coherence
            }
            return result
    
    # Check for singularity formation onset
    if (current_coherence > 0.95 and 
        self.recursive_metrics[recursive_depth].get('entropy_reduction', 0) > 0.5):
        result['detected'] = True
        result['type'] = 'singularity_formation'
        result['metrics'] = {
            'coherence': current_coherence,
            'entropy_reduction': self.recursive_metrics[recursive_depth].get('entropy_reduction', 0),
            'integration_coherence': self.recursive_metrics[recursive_depth].get('integration_coherence', 0)
        }
    
    return result

def _check_singularity_formation(self, 
                               data: np.ndarray, 
                               recursive_depth: int) -> bool:
    """Check if a coherent singularity has formed at this recursive depth."""
    # Not enough depth for singularity
    if recursive_depth < 2:
        return False
    
    # Get metrics from current depth
    current_metrics = self.recursive_metrics[recursive_depth]
    
    # Check for singularity conditions
    high_coherence = current_metrics.get('meta_coherence', 0) > 0.98
    high_integration = current_metrics.get('integration_coherence', 0) > 0.95
    significant_compression = current_metrics.get('meta_compression_ratio', 1.0) < 0.2
    
    # Additional stability check
    stable_entropy = False
    if recursive_depth > 1 and 'refined_entropy' in current_metrics:
        previous_entropy = self.recursive_metrics[recursive_depth-1].get('refined_entropy', float('inf'))
        current_entropy = current_metrics['refined_entropy']
        entropy_delta = abs(current_entropy - previous_entropy) / (previous_entropy + 1e-10)
        stable_entropy = entropy_delta < 0.01  # Less than 1% change
    
    # Singularity formation requires all conditions
    return high_coherence and high_integration and significant_compression and stable_entropy

def _compute_entropy(self, data: np.ndarray) -> float:
    """Compute information entropy of data."""
    if len(data) <= 1:
        return 0.0
    
    # Compute probability distribution
    values = np.abs(data)
    probabilities = values / (np.sum(values) + 1e-10)
    
    # Compute entropy
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
    
    return entropy

def _compute_trend(self, data: List[float]) -> Dict:
    """Compute trend analysis for a time series."""
    if len(data) < 2:
        return {'slope': 0, 'consistency': 0, 'description': 'insufficient_data'}
    
    # Convert to numpy array
    data_array = np.array(data)
    x = np.arange(len(data_array))
    
    # Linear regression
    slope, intercept = np.polyfit(x, data_array, 1)
    
    # Compute R-squared to measure consistency
    y_pred = slope * x + intercept
    ss_total = np.sum((data_array - np.mean(data_array))**2)
    ss_residual = np.sum((data_array - y_pred)**2)
    r_squared = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
    
    # Determine trend description
    if abs(slope) < 0.01:
        description = 'stable'
    elif slope > 0:
        description = 'improving' if r_squared > 0.7 else 'noisy_improvement'
    else:
        description = 'declining' if r_squared > 0.7 else 'noisy_decline'
    
    return {
        'slope': slope,
        'consistency': r_squared,
        'description': description
          }
