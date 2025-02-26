# Usage Guide

This guide provides detailed instructions for using the Recursive Self-Observing Intelligence Framework, including examples, best practices, and advanced usage patterns.

## Basic Usage

### Initializing the System

```python
from src.recursive_engine import UnifiedRecursiveSystem

# Create with default parameters
system = UnifiedRecursiveSystem()

# Or customize parameters
system = UnifiedRecursiveSystem(
    coherence_threshold=0.80,  # Minimum acceptable coherence
    stability_margin=0.95,     # Target stability for corrections
    entanglement_coupling=0.75,  # Strength of entanglement
    recursive_depth=4,         # Maximum recursive depth
    monitor_transitions=True   # Enable phase transition monitoring
)
```

### Compressing Data

```python
# Prepare your data (must be a numpy array)
import numpy as np
data = np.array([...])  # Your data here

# Compress with recursive self-observation
compressed_data, metrics = system.compress_with_meta_awareness(data)

# Print compression results
print(f"Original size: {len(data)}")
print(f"Compressed size: {len(compressed_data)}")
print(f"Compression ratio: {len(compressed_data)/len(data):.4f}")
```

### Decompressing Data

```python
# Decompress when needed
decompressed_data = system.decompress(compressed_data, metrics)

# Calculate reconstruction error
import numpy as np
mse = np.mean((data - decompressed_data.real)**2)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse:.6f}")
```

### Analyzing Performance

```python
# Get comprehensive performance analysis
performance = system.analyze_system_performance()

# Print key metrics
print(f"Coherence trend: {performance['coherence_trend']['description']}")
print(f"Optimal recursive depth: {performance['optimal_recursive_depth']['depth']}")
print(f"System stability: {performance['system_stability']['status']}")

# Check for emergent properties
for prop in performance['emergent_properties']:
    print(f"- {prop['type']} (confidence: {prop['confidence']:.2f})")
    print(f"  {prop['description']}")
```

### Visualizing Results

```python
# Visualize performance across recursive depths
system.visualize_recursive_performance("performance.png")

# If you want to customize visualization:
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(range(len(system.coherence_history)), system.coherence_history, 'b-o')
plt.xlabel('Iteration')
plt.ylabel('Coherence')
plt.title('Coherence Evolution')
plt.grid(True)
plt.savefig("coherence_evolution.png")
```

## Advanced Usage

### Working with Knowledge Domains

```python
# Create structured knowledge vectors (e.g., embeddings of concepts)
knowledge_vectors = np.random.rand(30, 64)  # 30 concepts with 64-dim embeddings

# Normalize vectors
for i in range(len(knowledge_vectors)):
    knowledge_vectors[i] /= np.linalg.norm(knowledge_vectors[i])

# Compress knowledge domain
compressed_knowledge, metrics = system.compress_with_meta_awareness(knowledge_vectors)

# Check if coherent singularity formed
if system.stable_singularities:
    print(f"Singularity formed at depth {system.stable_singularities[0]['depth']}")
    print(f"Coherence: {system.stable_singularities[0]['coherence']:.4f}")
```

### Detecting Phase Transitions

```python
# After compression, check for phase transitions
if system.transition_points:
    print("Phase transitions detected:")
    for i, transition in enumerate(system.transition_points):
        print(f"Transition {i+1}:")
        print(f"  - Type: {transition['type']}")
        print(f"  - Depth: {transition['depth']}")
```

### Customizing Recursive Depth

```python
# Control maximum recursive depth
compressed_shallow, metrics_shallow = system.compress_with_meta_awareness(
    data, max_recursive_depth=2
)

compressed_deep, metrics_deep = system.compress_with_meta_awareness(
    data, max_recursive_depth=6
)

# Compare results
print(f"Shallow compression ratio: {len(compressed_shallow)/len(data):.4f}")
print(f"Deep compression ratio: {len(compressed_deep)/len(data):.4f}")
```

## Use Cases

### 1. Data Compression

The framework can be used for advanced data compression with better ratios than traditional methods:

```python
# Load data
data = np.loadtxt("large_dataset.csv", delimiter=",")

# Compress with recursive depth 5
compressed, metrics = system.compress_with_meta_awareness(data, max_recursive_depth=5)

# Save compressed data (save both the data and metrics for decompression)
np.save("compressed_data.npy", compressed)
with open("compression_metrics.json", "w") as f:
    json.dump(metrics, f)
```

### 2. Knowledge Domain Integration

```python
# Load domain knowledge (e.g., concept embeddings from different domains)
domain_a = np.load("domain_a_embeddings.npy")
domain_b = np.load("domain_b_embeddings.npy")

# Combine domains
combined_knowledge = np.vstack([domain_a, domain_b])

# Apply recursive compression to identify cross-domain patterns
compressed, metrics = system.compress_with_meta_awareness(combined_knowledge)

# Visualize domain relationships before and after
from src.visualization import visualize_knowledge_integration

visualize_knowledge_integration(
    system.decompress(compressed, metrics),
    combined_knowledge,
    system.recursive_metrics[system.recursive_depth]['integration_coherence']
)
```

### 3. Pattern Discovery

```python
# With time series data
time_series = np.load("time_series_data.npy")

# Apply recursive compression
compressed, metrics = system.compress_with_meta_awareness(time_series)

# Check for detected patterns through phase transitions
if system.transition_points:
    for transition in system.transition_points:
        if transition['type'] == 'entropy_collapse':
            print(f"Significant pattern discovered at depth {transition['depth']}")
            print(f"Compression improvement: {transition['metrics']['compression_ratio_delta']:.4f}")
```

## Best Practices

1. **Start with Normalized Data**: When possible, normalize your data before compression for best results.

2. **Try Multiple Recursive Depths**: Different datasets benefit from different recursive depths. Experiment to find the optimal depth for your data.

3. **Monitor Coherence**: Keep an eye on the coherence metrics. If coherence drops significantly, adjust the coherence threshold.

4. **Balance Compression vs. Accuracy**: Higher compression ratios may come at the cost of reconstruction accuracy. Find the right balance for your application.

5. **Visualize Results**: Always visualize performance metrics to better understand the system's behavior on your data.

## Performance Considerations

- **Memory Usage**: Deep recursive depths require more memory. Start with lower depths and increase gradually.
- **Computation Time**: Recursive self-observation is computationally intensive. Consider using smaller datasets for initial experimentation.
- **Parallelization**: The current implementation is single-threaded. For large datasets, consider parallelizing across multiple systems.
