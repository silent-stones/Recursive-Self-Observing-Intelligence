import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from scipy import signal

def _reduce_coherently(self, 
                      data: np.ndarray,
                      phase_space: Dict,
                      compression_level: float) -> np.ndarray:
    """Perform coherent dimensionality reduction with SVD."""
    # Reshape data for SVD if needed
    data_matrix = data.reshape(-1, 1)
    
    # Create phase-aligned data matrix
    n = len(data)
    phase = phase_space['phase'][:n]
    phase_matrix = phase.reshape(1, -1)
    
    # Create data-phase matrix for SVD
    combined_matrix = data_matrix @ phase_matrix
    
    # Apply SVD
    try:
        U, S, Vh = np.linalg.svd(combined_matrix, full_matrices=False)
        
        # Determine compression threshold based on compression level
        energy_preserved = 0.5 + 0.5 * compression_level  # 50% to 100% energy preservation
        cumulative_energy = np.cumsum(S) / np.sum(S)
        k = np.searchsorted(cumulative_energy, energy_preserved) + 1
        
        # Ensure we keep at least one component
        k = max(1, min(k, len(S)))
        
        # Project data onto preserved subspace
        compressed_matrix = U[:, :k] @ np.diag(S[:k]) @ Vh[:k, :]
        
        # Extract compressed data vector
        compressed = compressed_matrix[:, 0]
        
        # Apply phase correction to maintain coherence
        phase_correction = np.exp(1j * np.angle(compressed))
        compressed *= phase_correction
        
    except np.linalg.LinAlgError:
        # Fallback for SVD failure
        logger.warning("SVD failed, using fallback compression")
        compressed = self._fallback_compression(data, compression_level)
    
    return compressed

def _fallback_compression(self, 
                        data: np.ndarray, 
                        compression_level: float) -> np.ndarray:
    """Fallback compression method when SVD fails."""
    # Determine compression ratio
    target_size = max(1, int(len(data) * (1.0 - 0.5 * compression_level)))
    
    # Apply wavelet-inspired compression
    if len(data) > 1:
        # Compute real and imaginary parts of wavelet transform
        if np.iscomplexobj(data):
            real_wavelet = signal.cwt(data.real, signal.ricker, [1])
            imag_wavelet = signal.cwt(data.imag, signal.ricker,
space."""
    # Extract phase from phase space
    ref_phase = phase_space['phase']
    
    # Match sizes
    min_len = min(len(phase), len(ref_phase))
    phase = phase[:min_len]
    ref_phase = ref_phase[:min_len]
    
    # Compute phase alignment
    phase_alignment = np.exp(1j * (phase - ref_phase))
    
    # Compute coherence as magnitude of average alignment
    window_size = max(1, min_len // 10)
    coherence = np.zeros(min_len)
    
    for i in range(min_len):
        # Define window indices
        start_idx = max(0, i - window_size // 2)
        end_idx = min(min_len, i + window_size // 2 + 1)
        
        # Compute local coherence
        local_alignment = phase_alignment[start_idx:end_idx]
        coherence[i] = np.abs(np.mean(local_alignment))
    
    return coherence

