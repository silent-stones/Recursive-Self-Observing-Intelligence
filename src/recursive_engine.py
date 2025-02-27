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
                monitor_transitions: bool = True,
                num_threads: int = 8):
        """
        Initialize the Unified Recursive System with tunable parameters.
        
        Args:
            coherence_threshold: Minimum acceptable coherence level
            stability_margin: Target stability level for corrections
            entanglement_coupling: Strength of entanglement relationships
            recursive_depth: Maximum depth of recursive self-observation
            monitor_transitions: Whether to monitor for phase transitions
            num_threads: Number of threads for parallel processing
        """
        # Core parameters
        self.coherence_threshold = coherence_threshold
        self.stability_margin = stability_margin
        self.entanglement_coupling = entanglement_coupling
        self.recursive_depth = recursive_depth
        self.monitor_transitions = monitor_transitions
        self.num_threads = num_threads
        
        # Metrics tracking
        self.coherence_history = []
        self.entanglement_history = []
        self.compression_ratio_history = []
        self.recursive_metrics = {i: {} for i in range(recursive_depth + 1)}
        
        # Phase transition monitoring
        self.transition_points = []
        self.stable_singularities = []
        
        # Performance metrics
        self.performance_metrics = {
            'operations_count': 0,
            'total_time': 0,
            'avg_time': 0,
            'error_count': 0,
            'last_error': None,
            'compression_ratios': [],
            'coherence_values': [],
            'processing_times': []
        }
        
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
    
    def _update_entanglement_with_coherence(self,
                                        entanglement: Dict,
                                        coherence: np.ndarray,
                                        weighted_phase: np.ndarray) -> Dict:
        """Update entanglement structure based on coherence updates."""
        # Create updated entanglement hierarchy
        updated = {}
        
        for level_key, level_matrix in entanglement.items():
            level_idx = int(level_key.split('_')[1])
            scale = max(1, 2**level_idx)
            level_size = level_matrix.shape[0]
            
            # Update entanglement at this level
            updated_matrix = level_matrix.copy()
            
            for i in range(level_size):
                for j in range(level_size):
                    # Calculate indices in the full array
                    i_start = min(i * scale, len(coherence))
                    i_end = min((i + 1) * scale, len(coherence))
                    j_start = min(j * scale, len(coherence))
                    j_end = min((j + 1) * scale, len(coherence))
                    
                    # Skip if indices are invalid
                    if i_start >= i_end or j_start >= j_end:
                        continue
                    
                    # Get coherence for these regions
                    i_coherence = np.mean(coherence[i_start:i_end])
                    j_coherence = np.mean(coherence[j_start:j_end])
                    
                    # Update entanglement strength based on coherence
                    coherence_factor = np.sqrt(i_coherence * j_coherence)
                    phase_diff = np.mean(weighted_phase[i_start:i_end]) - np.mean(weighted_phase[j_start:j_end])
                    
                    # Update with coherence weighting
                    updated_matrix[i, j] = coherence_factor * np.exp(1j * phase_diff)
            
            # Normalize
            if np.sum(np.abs(updated_matrix)**2) > 0:
                updated_matrix /= np.sqrt(np.sum(np.abs(updated_matrix)**2))
            
            updated[level_key] = updated_matrix
        
        return updated
    
    def _quantum_encode(self, 
                      data: np.ndarray,
                      phase_space: Dict,
                      compression_level: float) -> np.ndarray:
        """Quantum-inspired encoding with adaptive basis."""
        # Create data size matching phase space
        n = len(data)
        
        # Create superposition state with phase-aligned basis
        psi = np.zeros(n, dtype=complex)
        
        # Select phase based on compression level
        phase = phase_space['phase'][:n]
        
        # Adaptive basis selection based on coherence
        if 'coherence' in phase_space:
            coherence = phase_space['coherence'][:n]
            # Weight toward more coherent basis states
            coherence_weight = coherence**compression_level
        else:
            coherence_weight = np.ones(n)
        
        for i in range(n):
            # Phase-aligned basis state with coherence weighting
            basis = coherence_weight[i] * np.exp(1j * phase[i])
            
            # Add to superposition
            psi[i] = data[i] * basis
        
        # Normalize the state
        if np.sum(np.abs(psi)**2) > 0:
            psi /= np.sqrt(np.sum(np.abs(psi)**2))
        
        return psi
    
    def _recursive_compress(self, 
                          data: np.ndarray,
                          phase_space: Dict) -> Tuple[np.ndarray, Dict]:
        """Perform recursive compression with phase tracking and adaptive stability."""
        # Track compression stages
        compressed_stages = []
        metrics_stages = []
        
        # Initialize working data
        current_data = data.copy()
        
        # Track coherence and stability
        coherence = phase_space['coherence'].copy()
        stability_history = []
        
        # Determine number of compression stages
        compression_stages = max(1, int(np.log2(len(current_data))))
        
        for stage in range(compression_stages):
            # Skip if data is too small
            if len(current_data) <= 1:
                break
                
            # Compute compression level for this stage
            compression_level = 1.0 - (stage / compression_stages)
            
            # Compress with adaptive parameters
            compressed_chunk, chunk_metrics = self._compress_chunk(
                current_data, 
                phase_space,
                compression_level
            )
            
            # Update coherence with adaptive coupling
            coherence = self._update_coherence_adaptive(
                coherence, 
                chunk_metrics['coherence'],
                stage
            )
            
            # Check stability and apply corrections if needed
            stability = self._check_stability(coherence)
            stability_history.append(stability)
            
            if not stability:
                # Apply adaptive stability correction
                coherence = self._adaptive_phase_correction(
                    coherence,
                    stage,
                    compression_stages
                )
                
                # Update phase space with corrected coherence
                phase_space = self._update_phase_space(
                    phase_space, 
                    coherence
                )
            
            # Store compression results and metrics
            compressed_stages.append(compressed_chunk)
            metrics_stages.append(chunk_metrics)
            
            # Update for next stage
            current_data = compressed_chunk
        
        # Compile comprehensive metrics
        final_metrics = {
            'stages': len(compressed_stages),
            'stage_metrics': metrics_stages,
            'final_coherence': coherence,
            'stability_history': stability_history,
            'compression_ratio': len(current_data) / len(data) if len(data) > 0 else 1.0
        }
        
        return current_data, final_metrics
    
    def _compress_chunk(self, 
                      chunk: np.ndarray,
                      phase_space: Dict,
                      compression_level: float = 0.8) -> Tuple[np.ndarray, Dict]:
        """Compress data chunk with phase preservation and adaptive parameters."""
        # Compute chunk properties in phase space
        chunk_fft = np.fft.fft(chunk)
        chunk_phase = np.angle(chunk_fft)
        
        # Phase-aligned compression with adaptive parameters
        compressed = self._phase_aligned_compress(
            chunk, 
            chunk_phase, 
            phase_space,
            compression_level
        )
        
        # Compute detailed metrics
        metrics = {
            'coherence': self._compute_coherence(chunk_phase, phase_space),
            'entanglement': self._compute_multi_scale_entanglement(compressed, phase_space),
            'stability': self._compute_stability(compressed),
            'compression_level': compression_level,
            'original_size': len(chunk),
            'compressed_size': len(compressed)
        }
        
        return compressed, metrics
    
    def _phase_aligned_compress(self, 
                              data: np.ndarray,
                              phase: np.ndarray,
                              phase_space: Dict,
                              compression_level: float) -> np.ndarray:
        """Perform phase-aligned compression with adaptive parameters."""
        # Apply phase-preserving transform
        transformed = self._apply_phase_transform(data, phase)
        
        # Quantum-inspired encoding with adaptive parameters
        encoded = self._quantum_encode(
            transformed, 
            phase_space,
            compression_level
        )
        
        # Compress with multi-scale coherence preservation
        compressed = self._coherent_compress(
            encoded, 
            phase_space,
            compression_level
        )
        
        return compressed
    
    def _apply_phase_transform(self, 
                             data: np.ndarray,
                             phase: np.ndarray) -> np.ndarray:
        """Apply phase-preserving transform with stability enhancement."""
        # Create stability-weighted phase operator
        stability_weight = np.exp(-np.gradient(phase)**2)
        weighted_phase = phase * stability_weight
        phase_op = np.exp(1j * weighted_phase)
        
        # Apply transform with stability weighting
        fft_data = np.fft.fft(data)
        transformed_fft = fft_data * phase_op
        transformed = np.fft.ifft(transformed_fft)
        
        return transformed
    
    def _coherent_compress(self, 
                          data: np.ndarray,
                          phase_space: Dict,
                          compression_level: float) -> np.ndarray:
        """Compress while maintaining multi-scale coherence."""
        # Apply hierarchical entanglement
        entangled = self._apply_hierarchical_entanglement(
            data, 
            phase_space['entanglement'],
            compression_level
        )
        
        # Coherent reduction with SVD
        compressed = self._reduce_coherently(
            entangled, 
            phase_space,
            compression_level
        )
        
        return compressed
    
    def _apply_hierarchical_entanglement(self, 
                                       data: np.ndarray,
                                       entanglement: Dict,
                                       compression_level: float) -> np.ndarray:
        """Apply hierarchical entanglement operation across multiple scales."""
        # Start with original data
        entangled = data.copy()
        
        # Apply entanglement at each available level
        for level_key in sorted(entanglement.keys()):
            level_matrix = entanglement[level_key]
            
            # Skip if dimensions don't match
            if level_matrix.shape[0] > len(entangled):
                continue
                
            # Resize data for this level if needed
            level_size = level_matrix.shape[0]
            if len(entangled) > level_size:
                # Downsample through averaging
                downsampled = np.zeros(level_size, dtype=complex)
                for i in range(level_size):
                    start_idx = i * (len(entangled) // level_size)
                    end_idx = (i + 1) * (len(entangled) // level_size)
                    downsampled[i] = np.mean(entangled[start_idx:end_idx])
                working_data = downsampled
            else:
                # Pad with zeros
                working_data = np.pad(entangled, (0, level_size - len(entangled)))
            
            # Apply level-specific entanglement
            level_strength = compression_level * (1.0 - float(level_key.split('_')[1]) / len(entanglement))
            level_entangled = level_matrix @ working_data
            
            # Upsample back to original size if needed
            if len(entangled) > level_size:
                upsampled = np.zeros_like(entangled)
                for i in range(level_size):
                    start_idx = i * (len(entangled) // level_size)
                    end_idx = (i + 1) * (len(entangled) // level_size)
                    upsampled[start_idx:end_idx] = level_entangled[i]
                level_result = upsampled
            else:
                level_result = level_entangled[:len(entangled)]
            
            # Combine with accumulated result (weighted by level)
            entangled = (1.0 - level_strength) * entangled + level_strength * level_result
        
        return entangled
    
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
                imag_wavelet = signal.cwt(data.imag, signal.ricker, [1])
                wavelet = real_wavelet + 1j * imag_wavelet
            else:
                wavelet = signal.cwt(data, signal.ricker, [1])
            
            # Take the most significant coefficients
            wavelet_flat = wavelet.flatten()
            threshold = np.sort(np.abs(wavelet_flat))[-target_size]
            
            # Create mask for significant coefficients
            mask = np.abs(wavelet) >= threshold
            
            # Create compressed representation
            compressed = np.zeros(target_size, dtype=complex)
            indices = np.where(mask)[1][:target_size]
            compressed[:len(indices)] = wavelet[0, indices]
            
            return compressed
        else:
            return data.copy()
    
    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute information entropy of data."""
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
    
    def _compute_coherence(self, 
                         phase: np.ndarray, 
                         phase_space: Dict) -> np.ndarray:
        """Compute coherence by comparing phase alignment with phase space."""
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
    
    def _compute_multi_scale_entanglement(self, 
                                        data: np.ndarray, 
                                        phase_space: Dict) -> Dict:
        """Compute entanglement metrics at multiple scales."""
        entanglement_metrics = {}
        
        if 'entanglement' not in phase_space:
            return {'average': 0.0}
        
        # Compute entanglement at each available level
        for level_key, level_matrix in phase_space['entanglement'].items():
            # Compute appropriate size data vector for this level
            level_size = level_matrix.shape[0]
            
            if len(data) >= level_size:
                # Downsample through averaging
                level_data = np.zeros(level_size, dtype=complex)
                for i in range(level_size):
                    start_idx = i * (len(data) // level_size)
                    end_idx = (i + 1) * (len(data) // level_size)
                    level_data[i] = np.mean(data[start_idx:end_idx])
            else:
                # Pad with zeros
                level_data = np.pad(data, (0, level_size - len(data)))
            
            # Compute entanglement metric
            entanglement_value = np.abs(np.vdot(level_data, level_matrix @ level_data))
            entanglement_metrics[level_key] = entanglement_value
        
        # Compute average entanglement across levels
        entanglement_metrics['average'] = np.mean(list(entanglement_metrics.values()))
        
        return entanglement_metrics
    
    
    def _update_coherence_adaptive(self,
                                 current: np.ndarray,
                                 new: np.ndarray,
                                 stage: int) -> np.ndarray:
        """Update coherence values with adaptive coupling strength."""
        # Make sure arrays have compatible shapes
        if isinstance(new, (int, float)):
            new = np.ones_like(current) * new
        
        if len(new) != len(current):
            # Resize to match current
            if len(new) > len(current):
                new = new[:len(current)]
            else:
                new = np.pad(new, (0, len(current) - len(new)))
        
        # Compute adaptive coupling based on stage
        stage_factor = 1.0 - (stage / (stage + 1.0))  # Decreases with stage
        adaptive_coupling = self.entanglement_coupling * stage_factor
        
        # Apply coherence update with adaptive coupling
        updated = adaptive_coupling * current + (1.0 - adaptive_coupling) * new
        
        return updated
    
    def _adaptive_phase_correction(self,
                                coherence: np.ndarray,
                                stage: int,
                                total_stages: int) -> np.ndarray:
        """Apply adaptive phase correction with dynamic stability margin."""
        # Compute coherence gradient
        coherence_gradient = np.gradient(coherence)
        
        # Dynamic stability margin based on stage and gradient
        stage_progress = stage / max(1, total_stages)
        
        # More aggressive correction in early stages, more conservative in later stages
        dynamic_margin = self.stability_margin * (1.0 + 0.2 * (1.0 - stage_progress) * np.sign(coherence_gradient))
        
        # Compute correction factor with adaptive margin
        correction = np.where(
            coherence < self.coherence_threshold,
            dynamic_margin - coherence,
            0
        )
        
        # Apply correction with adaptive coupling
        min_coherence = np.min(coherence) if len(coherence) > 0 else 0
        adaptive_coupling = self.entanglement_coupling * (1.0 + 0.1 * (1.0 - min_coherence))
        corrected = coherence + adaptive_coupling * correction
        
        return corrected
    
    def _update_phase_space(self,
                          phase_space: Dict,
                          coherence: np.ndarray) -> Dict:
        """Update phase space based on updated coherence."""
        # Create updated phase space
        updated = phase_space.copy()
        
        # Update coherence
        updated['coherence'] = coherence
        
        # Update entanglement based on new coherence
        if 'phase' in phase_space:
            phase = phase_space['phase']
            
            # Weight phase by coherence
            weighted_phase = phase * coherence
            
            # Update hierarchical entanglement
            updated['entanglement'] = self._update_entanglement_with_coherence(
                phase_space['entanglement'],
                coherence,
                weighted_phase
            )
        
        return updated
    
    def _check_stability(self, coherence: np.ndarray) -> bool:
        """Check stability of coherence."""
        # Check if all coherence values are above threshold
        if len(coherence) == 0:
            return True
        
        # Use mean coherence for stability check
        mean_coherence = np.mean(coherence)
        return mean_coherence >= self.coherence_threshold
    
    def _compute_stability(self, data: np.ndarray) -> float:
        """
        Compute stability metric for data.
        
        Args:
            data: Input data to analyze
            
        Returns:
            Stability value (0.0-1.0)
        """
        if len(data) <= 1:
            return 1.0
        
        # Compute phase stability
        fft_data = np.fft.fft(data)
        phase = np.angle(fft_data)
        
        # Calculate phase gradient
        phase_gradient = np.gradient(phase)
        
        # Compute stability as inverse of gradient variability
        gradient_variability = np.std(phase_gradient)
        stability = 1.0 / (1.0 + gradient_variability)
        
        # Normalize to [0,1]
        stability = max(0.0, min(1.0, stability))
        
        return stability
    
    def _compute_integration_coherence(self, primary_metrics: Dict, meta_metrics: Dict) -> float:
        """
        Compute coherence of integration between primary and meta processes.
        
        Args:
            primary_metrics: Metrics from primary compression
            meta_metrics: Metrics from meta-compression
            
        Returns:
            Integration coherence value
        """
        # Extract coherence values
        if 'final_coherence' in primary_metrics and 'final_coherence' in meta_metrics:
            primary_coherence = primary_metrics['final_coherence']
            meta_coherence = meta_metrics['final_coherence']
            
            # Compute correlation if arrays
            if isinstance(primary_coherence, np.ndarray) and isinstance(meta_coherence, np.ndarray):
                # Resize to match length
                min_len = min(len(primary_coherence), len(meta_coherence))
                p_coherence = primary_coherence[:min_len]
                m_coherence = meta_coherence[:min_len]
                
                # Compute correlation
                try:
                    corr = np.corrcoef(p_coherence, m_coherence)[0, 1]
                    return (np.mean(p_coherence) + np.mean(m_coherence) + max(0, corr)) / 3
                except:
                    # Fallback if correlation fails
                    return (np.mean(p_coherence) + np.mean(m_coherence)) / 2
            else:
                # Use simple average if not arrays
                if not isinstance(primary_coherence, (int, float)):
                    primary_coherence = np.mean(primary_coherence)
                if not isinstance(meta_coherence, (int, float)):
                    meta_coherence = np.mean(meta_coherence)
                return (primary_coherence + meta_coherence) / 2
        
        # Default if metrics not available
        return 0.5
    
    def _encode_compression_process(self, metrics: Dict) -> np.ndarray:
        """
        Encode the compression process itself as data for meta-compression.
        
        Args:
            metrics: Metrics from compression process
            
        Returns:
            Encoded representation of the process
        """
        # Extract key metrics to encode
        encodable_metrics = []
        
        # Add coherence values
        if 'final_coherence' in metrics:
            coherence = metrics['final_coherence']
            if isinstance(coherence, np.ndarray):
                encodable_metrics.append(coherence)
            else:
                encodable_metrics.append(np.array([coherence]))
        
        # Add stability history if available
        if 'stability_history' in metrics:
            stability = np.array(metrics['stability_history'], dtype=float)
            encodable_metrics.append(stability)
        
        # Add stage metrics if available
        if 'stage_metrics' in metrics:
            # Extract compression levels
            comp_levels = [stage.get('compression_level', 0.5) for stage in metrics['stage_metrics']]
            encodable_metrics.append(np.array(comp_levels))
            
            # Extract compression ratios
            ratios = [stage.get('compressed_size', 1) / stage.get('original_size', 1) 
                     for stage in metrics['stage_metrics'] if 'original_size' in stage and stage['original_size'] > 0]
            if ratios:
                encodable_metrics.append(np.array(ratios))
        
        # Combine all metrics into a single array
        if encodable_metrics:
            # Pad arrays to the same length
            max_len = max(len(arr) for arr in encodable_metrics)
            padded_metrics = []
            
            for arr in encodable_metrics:
                padded = np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=0)
                padded_metrics.append(padded)
            
            # Stack arrays and encode as complex values
            stacked = np.column_stack(padded_metrics)
            encoded = stacked.astype(complex)
            
            # Add phase encoding based on relationships
            for i in range(encoded.shape[1]):
                col = encoded[:, i]
                phase = 2 * np.pi * i / encoded.shape[1]
                encoded[:, i] = col * np.exp(1j * phase)
            
            return encoded.flatten()
        
        # Default small representation if no metrics available
        return np.array([0.5 + 0.5j])
    
    def _integrate_meta_insights(self, 
                              primary_data: np.ndarray, 
                              meta_data: np.ndarray, 
                              primary_metrics: Dict, 
                              meta_metrics: Dict) -> np.ndarray:
        """
        Integrate insights from meta-compression back into primary process.
        
        Args:
            primary_data: Data from primary compression
            meta_data: Data from meta-compression
            primary_metrics: Metrics from primary compression
            meta_metrics: Metrics from meta-compression
            
        Returns:
            Refined data with meta-insights integrated
        """
        # Compute coherence values
        primary_coherence = np.mean(primary_metrics['final_coherence']) \
            if 'final_coherence' in primary_metrics else 0.5
        meta_coherence = np.mean(meta_metrics['final_coherence']) \
            if 'final_coherence' in meta_metrics else 0.5
        
        # Compute adaptive integration factor based on coherence
        integration_strength = 0.2 + 0.6 * meta_coherence  # 0.2 to 0.8 based on meta coherence
        
        # Get phase information
        primary_fft = np.fft.fft(primary_data)
        primary_phase = np.angle(primary_fft)
        primary_magnitude = np.abs(primary_fft)
        
        # Apply meta-insights to modify phase relationships
        if len(meta_data) > 1:
            # Normalize meta data
            meta_normalized = meta_data / np.max(np.abs(meta_data))
            
            # Compute meta phase influence
            meta_fft = np.fft.fft(meta_normalized)
            meta_phase = np.angle(meta_fft)
            
            # Resize meta phase to match primary
            if len(meta_phase) != len(primary_phase):
                if len(meta_phase) > len(primary_phase):
                    meta_phase = meta_phase[:len(primary_phase)]
                else:
                    meta_phase = np.pad(meta_phase, (0, len(primary_phase) - len(meta_phase)), 
                                       mode='constant', constant_values=0)
            
            # Integrate phases with adaptive weighting
            integrated_phase = (1 - integration_strength) * primary_phase + integration_strength * meta_phase
            
            # Reconstruct with integrated phase
            integrated_fft = primary_magnitude * np.exp(1j * integrated_phase)
            integrated_data = np.fft.ifft(integrated_fft)
            
            return integrated_data
        
        # If meta data is too small, apply minimal modification
        return primary_data
    
    def _detect_phase_transition(self, 
                               data: np.ndarray, 
                               recursive_depth: int) -> Dict:
        """
        Detect phase transitions in the data structure.
        
        Args:
            data: Input data to analyze
            recursive_depth: Current recursive depth
            
        Returns:
            Dictionary with phase transition information
        """
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
        """
        Check if a coherent singularity has formed at this recursive depth.
        
        Args:
            data: Input data to analyze
            recursive_depth: Current recursive depth
            
        Returns:
            Boolean indicating whether a singularity has formed
        """
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
    
    def decompress(self, 
                  compressed_data: np.ndarray, 
                  metrics: Dict,
                  original_shape: Optional[Tuple] = None) -> np.ndarray:
        """
        Decompress data using stored metrics information.
        
        Args:
            compressed_data: Compressed data to decompress
            metrics: Metrics from compression process
            original_shape: Original shape to restore (optional)
        
        Returns:
            Decompressed data
        """
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
        """
        Apply inverse transformation based on stage metrics.
        
        Args:
            data: Input data to transform
            stage_metrics: Metrics for this stage
        
        Returns:
            Transformed data
        """
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
        """
        Perform basic decompression without detailed stage metrics.
        
        Args:
            data: Input data to decompress
            metrics: Compression metrics
        
        Returns:
            Decompressed data
        """
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

