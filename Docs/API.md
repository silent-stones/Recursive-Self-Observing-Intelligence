# API Documentation

This document provides detailed API information for the Recursive Self-Observing Intelligence Framework.

## UnifiedRecursiveSystem Class

The main class implementing the recursive self-observing intelligence framework.

### Constructor

```python
UnifiedRecursiveSystem(
    coherence_threshold=0.85,
    stability_margin=0.95,
    entanglement_coupling=0.78,
    recursive_depth=3,
    monitor_transitions=True
)
```

**Parameters:**

- `coherence_threshold` (float, default=0.85): Minimum acceptable coherence level
- `stability_margin` (float, default=0.95): Target stability level for corrections
- `entanglement_coupling` (float, default=0.78): Strength of entanglement relationships
- `recursive_depth` (int, default=3): Maximum depth of recursive self-observation
- `monitor_transitions` (bool, default=True): Whether to monitor for phase transitions

### Main Methods

#### compress_with_meta_awareness

```python
compress_with_meta_awareness(data, max_recursive_depth=None)
```

Compresses data using recursive self-observation up to the specified depth.

**Parameters:**
- `data` (numpy.ndarray): Input data array to compress
- `max_recursive_depth` (int, optional): Maximum recursive depth (defaults to self.recursive_depth)

**Returns:**
- `compressed_data` (numpy.ndarray): Compressed data representation
- `metrics` (dict): Comprehensive metrics about the compression process

**Metrics Dictionary Structure:**
```
{
    'primary_metrics': {
        'stages': int,                  # Number of compression stages
        'stage_metrics': list,          # Metrics for each compression stage
        'final_coherence': numpy.ndarray, # Final coherence values
        'stability_history': list,      # History of stability status
        'compression_ratio': float      # Compression ratio achieved
    },
    'final_metrics': dict,              # Metrics from final recursive iteration
    'recursive_metrics': {              # Metrics for each recursive depth
        0: {
            'compressed_size': int,
            'compression_ratio': float,
            'coherence': float,
            'entropy': float,
            'entropy_reduction': float
        },
        1: {
            'meta_compressed_size': int,
            'meta_compression_ratio': float,
            'meta_coherence': float,
            'integration_coherence': float,
            'refined_entropy': float,
            'entropy_reduction': float
        },
        ...
    },
    'transitions': list,                # Detected phase transitions
    'singularities': list               # Detected coherent singularities
}
```

#### decompress

```python
decompress(compressed_data, metrics, original_shape=None)
```

Decompresses data using the provided metrics.

**Parameters:**
- `compressed_data` (numpy.ndarray): Compressed data to decompress
- `metrics` (dict): Metrics dictionary from the compression process
- `original_shape` (tuple, optional): Shape to reshape the output to

**Returns:**
- `decompressed_data` (numpy.ndarray): Decompressed data

#### analyze_system_performance

```python
analyze_system_performance(metrics_history=None)
```

Analyzes system performance across multiple compression runs.

**Parameters:**
- `metrics_history` (list, optional): List of metrics dictionaries from multiple runs

**Returns:**
- `performance` (dict): Comprehensive performance analysis

**Performance Dictionary Structure:**
```
{
    'coherence_trend': {
        'slope': float,
        'consistency': float,
        'description': str
    },
    'compression_trend': {
        'slope': float,
        'consistency': float,
        'description': str
    },
    'significant_transitions': list,
    'singularities_formed': int,
    'optimal_recursive_depth': {
        'depth': int,
        'reason': str,
        'coherence': float,
        'compression_ratio': float,
        'entropy_reduction': float
    },
    'system_stability': {
        'status': str,
        'coherence_stability': float,
        'compression_stability': float,
        'mean_coherence': float,
        'has_singularity': bool
    },
    'emergent_properties': [
        {
            'type': str,
            'confidence': float,
            'description': str
        },
        ...
    ]
}
```

#### visualize_recursive_performance

```python
visualize_recursive_performance(save_path=None)
```

Visualizes the performance of recursive compression across depths.

**Parameters:**
- `save_path` (str, optional): Path to save the visualization (shows plot if None)

### Attributes

- `coherence_threshold` (float): Minimum acceptable coherence level
- `stability_margin` (float): Target stability level for corrections
- `entanglement_coupling` (float): Strength of entanglement relationships
- `recursive_depth` (int): Maximum depth of recursive self-observation
- `coherence_history` (list): History of coherence values across iterations
- `compression_ratio_history` (list): History of compression ratios
- `recursive_metrics` (dict): Metrics for each recursive depth
- `transition_points` (list): Detected phase transitions
- `stable_singularities` (list): Detected coherent singularities

## Utility Functions

### visualize_knowledge_integration

```python
visualize_knowledge_integration(decompressed, original, integration_coherence)
```

Visualizes how well knowledge is integrated across domains after compression.

**Parameters:**
- `decompressed` (numpy.ndarray): Decompressed knowledge representation
- `original` (numpy.ndarray): Original knowledge representation
- `integration_coherence` (float): Integration coherence metric

## Error Handling

The framework uses the Python logging module for error reporting. Most methods include error handling that will log warnings or errors rather than raising exceptions, allowing the system to continue operation even when problems occur.

To get more detailed error information, configure the logger:

```python
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
```

## Thread Safety

The current implementation is not thread-safe. Do not share UnifiedRecursiveSystem instances across threads without external synchronization.

## Memory Management

The framework can use significant memory for deep recursive processing. To manage memory usage:

1. Set appropriate recursive depths
2. Process large datasets in batches
3. Release references to large data structures when no longer needed

## Performance Optimization

For optimal performance:

1. Normalize input data when possible
2. Start with lower recursive depths and increase as needed
3. For time-critical applications, pre-compute and cache results for common data patterns
