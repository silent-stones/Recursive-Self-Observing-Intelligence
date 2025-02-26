
## Introduction

The Recursive Self-Observing Intelligence Framework (RSOIF) represents a significant advancement in AI by implementing mechanisms for self-analysis, cross-domain knowledge transfer, and efficient information compression. This document provides theoretical explanations and practical code examples for implementing these capabilities.

---

# Section 1: Self-Improving Agent

## Theoretical Foundation

Self-improvement in RSOIF is based on recursive meta-cognition - the ability of a system to observe, analyze, and modify its own decision-making processes. Unlike traditional machine learning systems that are optimized by external methods, a recursively self-improving agent:

1. **Maintains an explicit model of its own operation**
2. **Analyzes the effectiveness of its decision processes**
3. **Identifies patterns of success and failure in its reasoning**
4. **Modifies its own parameters to improve future performance**
5. **Recursively applies the improvement process to the improvement process itself**

The key principle is that coherence (measured as alignment between predictions and outcomes) serves as an intrinsic optimization metric. Through recursive application, the system can break through local optima that would trap conventional optimization approaches.

## Implementation Example

```python
import numpy as np
import matplotlib.pyplot as plt
from src.recursive_engine import UnifiedRecursiveSystem

class SelfImprovingAgent:
    """
    An agent that recursively analyzes and improves its own decision processes.
    
    This agent maintains a model of its own decision-making, analyzes its
    performance, and recursively improves its parameters.
    """
    
    def __init__(self, 
                decision_params=None, 
                learning_rate=0.05,
                meta_learning_rate=0.01,
                coherence_threshold=0.8,
                max_recursive_depth=4):
        """
        Initialize the self-improving agent.
        
        Args:
            decision_params: Initial parameters for decision function
            learning_rate: Base rate for parameter updates
            meta_learning_rate: Rate for updating the learning process itself
            coherence_threshold: Minimum acceptable coherence level
            max_recursive_depth: Maximum depth of recursive improvement
        """
        # Initialize decision parameters (if none provided, use defaults)
        self.decision_params = decision_params if decision_params is not None else {
            'weights': np.random.normal(0, 0.1, (5, 1)),  # Simple linear model weights
            'threshold': 0.5,                            # Decision threshold
            'confidence_factor': 0.8,                    # Weight for confidence adjustment
            'context_sensitivity': 0.3                   # Weight for context features
        }
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.meta_learning_rate = meta_learning_rate
        self.coherence_threshold = coherence_threshold
        self.max_recursive_depth = max_recursive_depth
        
        # Performance history
        self.decision_history = []
        self.performance_history = []
        self.coherence_history = []
        
        # Initialize the recursive system for meta-cognition
        self.recursive_system = UnifiedRecursiveSystem(
            coherence_threshold=coherence_threshold,
            stability_margin=0.95,
            entanglement_coupling=0.75,
            recursive_depth=max_recursive_depth
        )
        
        # Meta-learning records
        self.meta_improvements = []
    
    def make_decision(self, features, context=None):
        """
        Make a decision based on input features and context.
        
        Args:
            features: Input feature vector
            context: Optional context information
            
        Returns:
            Decision (1 or 0), confidence score, and decision metadata
        """
        # Ensure features are numpy array
        features = np.array(features).reshape(-1, 1)
        
        # Apply weights to get raw score
        raw_score = np.dot(features.T, self.decision_params['weights'])[0, 0]
        
        # Apply context sensitivity if context provided
        if context is not None:
            context_factor = self.decision_params['context_sensitivity'] * context
            raw_score += context_factor
        
        # Calculate confidence based on distance from threshold
        confidence = self.decision_params['confidence_factor'] * abs(raw_score - self.decision_params['threshold'])
        
        # Make binary decision
        decision = 1 if raw_score >= self.decision_params['threshold'] else 0
        
        # Store metadata about the decision
        metadata = {
            'raw_score': raw_score,
            'confidence': confidence,
            'features': features.copy(),
            'context': context,
            'params_used': self.decision_params.copy()
        }
        
        # Return decision, confidence, and metadata
        return decision, confidence, metadata
    
    def record_outcome(self, decision_metadata, actual_outcome, reward):
        """
        Record the outcome of a decision to improve future performance.
        
        Args:
            decision_metadata: Metadata from make_decision
            actual_outcome: The actual outcome (1 or 0)
            reward: Reward received for the decision
        """
        # Record the decision and outcome
        record = {
            'metadata': decision_metadata,
            'predicted': 1 if decision_metadata['raw_score'] >= self.decision_params['threshold'] else 0,
            'actual': actual_outcome,
            'reward': reward,
            'timestamp': len(self.decision_history)
        }
        
        self.decision_history.append(record)
        
        # Calculate accuracy for this decision
        accuracy = 1.0 if record['predicted'] == record['actual'] else 0.0
        self.performance_history.append(accuracy)
        
        # Update coherence score based on prediction-outcome alignment
        if len(self.coherence_history) > 0:
            # Coherence incorporates both current accuracy and trend
            prev_coherence = self.coherence_history[-1]
            new_coherence = 0.8 * prev_coherence + 0.2 * accuracy
        else:
            new_coherence = accuracy
            
        self.coherence_history.append(new_coherence)
        
        # If we have enough history, trigger self-improvement
        if len(self.decision_history) >= 10 and len(self.decision_history) % 5 == 0:
            self.improve_self()
    
    def improve_self(self):
        """
        Recursively analyze and improve decision-making parameters.
        
        This is the core self-improvement mechanism, using recursive
        self-observation to enhance performance.
        """
        print(f"Initiating self-improvement cycle (history length: {len(self.decision_history)})")
        
        # Extract performance data
        recent_history = self.decision_history[-20:]  # Focus on recent decisions
        
        # Create feature matrix from historical performance
        performance_features = self._extract_performance_features(recent_history)
        
        # Apply recursive compression to identify patterns
        compressed_features, metrics = self.recursive_system.compress_with_meta_awareness(
            performance_features
        )
        
        # Analyze performance to identify improvement opportunities
        improvement_opportunities = self._analyze_performance(compressed_features, metrics)
        
        # Apply primary improvements
        self._apply_improvements(improvement_opportunities)
        
        # Now recursively improve the improvement process itself (meta-improvement)
        for depth in range(1, self.max_recursive_depth + 1):
            # Encode the improvement process itself
            improvement_process = self._encode_improvement_process(improvement_opportunities)
            
            # Recursively analyze the improvement process
            meta_compressed, meta_metrics = self.recursive_system.compress_with_meta_awareness(
                improvement_process, max_recursive_depth=depth
            )
            
            # Identify meta-improvements
            meta_improvements = self._analyze_meta_improvement(meta_compressed, meta_metrics, depth)
            
            # Apply meta-improvements
            if meta_improvements:
                self._apply_meta_improvements(meta_improvements, depth)
                self.meta_improvements.append({
                    'depth': depth,
                    'improvements': meta_improvements,
                    'coherence': meta_metrics['recursive_metrics'][depth].get('meta_coherence', 0)
                })
        
        # Record current coherence level after improvements
        current_coherence = self.coherence_history[-1] if self.coherence_history else 0
        print(f"Self-improvement cycle completed. Current coherence: {current_coherence:.4f}")
    
    def _extract_performance_features(self, history):
        """
        Extract features from decision history for analysis.
        
        Args:
            history: List of decision records
            
        Returns:
            Feature matrix representing performance patterns
        """
        if not history:
            return np.array([])
        
        # Create feature matrix
        features = []
        
        for record in history:
            # Extract key performance indicators
            correct = record['predicted'] == record['actual']
            confidence = record['metadata']['confidence']
            raw_score = record['metadata']['raw_score']
            threshold = record['metadata']['params_used']['threshold']
            score_distance = abs(raw_score - threshold)
            
            # Create feature vector
            feature_vector = [
                correct,                        # Was the prediction correct
                confidence,                     # Confidence in the prediction
                score_distance,                 # Distance from decision threshold
                record['reward'],               # Reward received
                raw_score,                      # Raw prediction score
                abs(raw_score - 0.5) * 2,       # Normalized distance from neutral
                1 if raw_score > threshold else 0,  # Direction of prediction
                1 if record['actual'] == 1 else 0   # Actual outcome
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _analyze_performance(self, compressed_features, metrics):
        """
        Analyze compressed performance data to identify improvement opportunities.
        
        Args:
            compressed_features: Compressed performance features
            metrics: Compression metrics
            
        Returns:
            Dictionary of identified improvement opportunities
        """
        # Extract current performance metrics
        current_coherence = self.coherence_history[-1] if self.coherence_history else 0
        recent_accuracy = np.mean(self.performance_history[-20:]) if self.performance_history else 0
        
        # Get feature importance from compression
        feature_importance = self._extract_feature_importance(compressed_features, metrics)
        
        # Analyze decision boundaries
        boundary_analysis = self._analyze_decision_boundary()
        
        # Identify opportunities for improvement
        opportunities = {
            'weight_adjustments': {},
            'threshold_adjustment': 0.0,
            'confidence_adjustment': 0.0,
            'context_sensitivity_adjustment': 0.0
        }
        
        # Analyze weights
        current_weights = self.decision_params['weights']
        for i in range(len(current_weights)):
            weight_effect = self._analyze_weight_effect(i)
            if abs(weight_effect) > 0.1:  # Only adjust if significant effect
                opportunities['weight_adjustments'][i] = weight_effect
        
        # Analyze decision threshold
        if boundary_analysis['false_positives'] > boundary_analysis['false_negatives']:
            # Too many false positives, increase threshold
            opportunities['threshold_adjustment'] = 0.05 * (1.0 - current_coherence)
        elif boundary_analysis['false_negatives'] > boundary_analysis['false_positives']:
            # Too many false negatives, decrease threshold
            opportunities['threshold_adjustment'] = -0.05 * (1.0 - current_coherence)
        
        # Analyze confidence calibration
        if boundary_analysis['high_confidence_errors'] > 0:
            # Reduce confidence factor if overconfident
            opportunities['confidence_adjustment'] = -0.05 * boundary_analysis['high_confidence_errors']
        
        # Analyze context sensitivity
        if boundary_analysis['context_related_errors'] > 0:
            # Adjust context sensitivity based on context-related errors
            opportunities['context_sensitivity_adjustment'] = 0.05 * (
                1.0 if boundary_analysis['context_improved'] else -1.0
            )
        
        return opportunities
    
    def _analyze_decision_boundary(self):
        """
        Analyze decision boundary performance.
        
        Returns:
            Dictionary containing decision boundary analysis metrics
        """
        # Initialize counters
        false_positives = 0
        false_negatives = 0
        high_confidence_errors = 0
        context_related_errors = 0
        context_improved = False
        
        # Analyze recent decisions
        recent_history = self.decision_history[-20:] if len(self.decision_history) >= 20 else self.decision_history
        
        for record in recent_history:
            # Count false positives and negatives
            if record['predicted'] == 1 and record['actual'] == 0:
                false_positives += 1
            elif record['predicted'] == 0 and record['actual'] == 1:
                false_negatives += 1
            
            # Count high confidence errors
            if record['predicted'] != record['actual'] and record['metadata']['confidence'] > 0.7:
                high_confidence_errors += 1
            
            # Analyze context effect
            if record['metadata']['context'] is not None:
                context_effect = record['metadata']['context'] * self.decision_params['context_sensitivity']
                raw_score = record['metadata']['raw_score'] - context_effect
                threshold = record['metadata']['params_used']['threshold']
                
                # Would the decision be different without context?
                no_context_decision = 1 if raw_score >= threshold else 0
                
                if record['predicted'] != no_context_decision:
                    # Context changed the decision
                    if record['predicted'] == record['actual']:
                        # Context improved the decision
                        context_improved = True
                    else:
                        # Context worsened the decision
                        context_related_errors += 1
        
        return {
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'high_confidence_errors': high_confidence_errors,
            'context_related_errors': context_related_errors,
            'context_improved': context_improved
        }
    
    def _analyze_weight_effect(self, weight_index):
        """
        Analyze the effect of a specific weight on performance.
        
        Args:
            weight_index: Index of the weight to analyze
            
        Returns:
            Suggested adjustment direction (-1 to 1)
        """
        # Get current weight
        current_weight = self.decision_params['weights'][weight_index][0]
        
        # Initialize counters for correct and incorrect predictions
        weight_contributed_correct = 0
        weight_contributed_incorrect = 0
        
        # Analyze recent decisions
        recent_history = self.decision_history[-20:] if len(self.decision_history) >= 20 else self.decision_history
        
        for record in recent_history:
            # Get feature value for this weight
            feature_value = record['metadata']['features'][weight_index][0]
            
            # Calculate contribution of this weight to the decision
            contribution = feature_value * current_weight
            
            # Determine if contribution was in the correct direction
            prediction = record['predicted']
            actual = record['actual']
            
            if (contribution > 0 and prediction == actual) or (contribution < 0 and prediction != actual):
                weight_contributed_correct += 1
            else:
                weight_contributed_incorrect += 1
        
        # Calculate adjustment based on contribution correctness
        if weight_contributed_correct + weight_contributed_incorrect > 0:
            correct_ratio = weight_contributed_correct / (weight_contributed_correct + weight_contributed_incorrect)
            
            # Return adjustment direction
            if correct_ratio > 0.6:
                # Weight is mostly helpful, strengthen it
                return 0.1 * (correct_ratio - 0.5) * 2  # Scale to -1 to 1
            elif correct_ratio < 0.4:
                # Weight is mostly harmful, weaken or reverse it
                return -0.1 * (0.5 - correct_ratio) * 2  # Scale to -1 to 1
            
        return 0.0  # No clear effect, no adjustment
    
    def _extract_feature_importance(self, compressed_features, metrics):
        """
        Extract feature importance from compression metrics.
        
        Args:
            compressed_features: Compressed feature representation
            metrics: Compression metrics
            
        Returns:
            Array of feature importance scores
        """
        # This is a simple implementation; in practice, feature importance
        # would be derived from a more sophisticated analysis of the compression
        
        # Default importance
        feature_importance = np.ones(8)  # 8 features in our feature vector
        
        # If we have recursive metrics, use them to estimate importance
        if 'recursive_metrics' in metrics:
            depth = max(metrics['recursive_metrics'].keys())
            if 'meta_coherence' in metrics['recursive_metrics'][depth]:
                coherence = metrics['recursive_metrics'][depth]['meta_coherence']
                
                # Weight importance by coherence - features in more coherent patterns
                # are likely more important
                feature_importance *= coherence
        
        return feature_importance
    
    def _apply_improvements(self, opportunities):
        """
        Apply identified improvements to decision parameters.
        
        Args:
            opportunities: Dictionary of improvement opportunities
        """
        # Apply weight adjustments
        for idx, adjustment in opportunities['weight_adjustments'].items():
            self.decision_params['weights'][idx][0] += self.learning_rate * adjustment
        
        # Apply threshold adjustment
        self.decision_params['threshold'] += self.learning_rate * opportunities['threshold_adjustment']
        
        # Apply confidence adjustment
        self.decision_params['confidence_factor'] += self.learning_rate * opportunities['confidence_adjustment']
        # Ensure confidence factor stays in reasonable range
        self.decision_params['confidence_factor'] = max(0.1, min(1.0, self.decision_params['confidence_factor']))
        
        # Apply context sensitivity adjustment
        self.decision_params['context_sensitivity'] += self.learning_rate * opportunities['context_sensitivity_adjustment']
        # Ensure context sensitivity stays in reasonable range
        self.decision_params['context_sensitivity'] = max(0.0, min(1.0, self.decision_params['context_sensitivity']))
    
    def _encode_improvement_process(self, improvements):
        """
        Encode the improvement process itself for meta-analysis.
        
        Args:
            improvements: Improvement opportunities dictionary
            
        Returns:
            Encoded representation of the improvement process
        """
        # Create a numerical representation of the improvement process
        encoding = []
        
        # Encode weight adjustments
        weight_adjustment_sum = sum(improvements['weight_adjustments'].values())
        weight_adjustment_count = len(improvements['weight_adjustments'])
        encoding.append(weight_adjustment_sum)
        encoding.append(weight_adjustment_count)
        
        # Encode other adjustments
        encoding.append(improvements['threshold_adjustment'])
        encoding.append(improvements['confidence_adjustment'])
        encoding.append(improvements['context_sensitivity_adjustment'])
        
        # Encode current performance metrics
        encoding.append(self.coherence_history[-1] if self.coherence_history else 0)
        encoding.append(np.mean(self.performance_history[-20:]) if self.performance_history else 0)
        
        # Encode learning parameters
        encoding.append(self.learning_rate)
        encoding.append(self.meta_learning_rate)
        
        # Encode improvement history trends
        if len(self.coherence_history) >= 3:
            # Calculate recent trend in coherence
            recent_trend = self.coherence_history[-1] - self.coherence_history[-3]
            encoding.append(recent_trend)
        else:
            encoding.append(0)
        
        return np.array(encoding)
    
    def _analyze_meta_improvement(self, meta_compressed, meta_metrics, depth):
        """
        Analyze meta-compression results to identify meta-improvements.
        
        Args:
            meta_compressed: Compressed meta-improvement representation
            meta_metrics: Compression metrics
            depth: Recursive depth level
            
        Returns:
            Dictionary of meta-improvements
        """
        # Initialize meta-improvements
        meta_improvements = {
            'learning_rate_adjustment': 0.0,
            'meta_learning_rate_adjustment': 0.0,
            'improvement_strategy_adjustment': None
        }
        
        # Get coherence and improvement efficiency
        if depth in meta_metrics['recursive_metrics']:
            meta_coherence = meta_metrics['recursive_metrics'][depth].get('meta_coherence', 0)
            
            # Look for coherence trend
            coherence_improving = False
            if len(self.coherence_history) >= 3:
                coherence_improving = self.coherence_history[-1] > self.coherence_history[-3]
            
            # Adjust learning rates based on meta-analysis
            if meta_coherence > 0.9:
                # High meta-coherence indicates effective learning
                if coherence_improving:
                    # If improving, maintain current rates
                    meta_improvements['learning_rate_adjustment'] = 0.0
                else:
                    # If not improving despite high meta-coherence, slightly increase learning
                    meta_improvements['learning_rate_adjustment'] = 0.01
            elif meta_coherence < 0.7:
                # Low meta-coherence indicates ineffective learning
                if coherence_improving:
                    # If still improving, slightly decrease to stabilize
                    meta_improvements['learning_rate_adjustment'] = -0.005
                else:
                    # If not improving, try more significant adjustments
                    meta_improvements['learning_rate_adjustment'] = 0.02
            
            # Meta-learning rate adjustments are more conservative
            meta_improvements['meta_learning_rate_adjustment'] = meta_improvements['learning_rate_adjustment'] * 0.5
            
            # Detect if we need to change improvement strategy
            if not coherence_improving and meta_coherence < 0.6:
                # Current strategy may be ineffective, suggest alternative approach
                meta_improvements['improvement_strategy_adjustment'] = 'explore'
            elif meta_coherence > 0.95:
                # Current strategy is highly effective, focus on exploitation
                meta_improvements['improvement_strategy_adjustment'] = 'exploit'
        
        return meta_improvements
    
    def _apply_meta_improvements(self, meta_improvements, depth):
        """
        Apply meta-improvements to the learning process itself.
        
        Args:
            meta_improvements: Dictionary of meta-improvements
            depth: Recursive depth level
        """
        # Apply learning rate adjustments with meta-learning rate
        self.learning_rate += self.meta_learning_rate * meta_improvements['learning_rate_adjustment']
        # Ensure learning rate stays in reasonable range
        self.learning_rate = max(0.001, min(0.2, self.learning_rate))
        
        # Apply meta-learning rate adjustments (recursive improvement of improvement)
        self.meta_learning_rate += self.meta_learning_rate * meta_improvements['meta_learning_rate_adjustment']
        # Ensure meta-learning rate stays in reasonable range
        self.meta_learning_rate = max(0.0005, min(0.05, self.meta_learning_rate))
        
        # Apply strategy adjustments if needed
        if meta_improvements['improvement_strategy_adjustment'] == 'explore':
            # Increase learning rate temporarily to explore new regions
            self.learning_rate *= 1.5
        elif meta_improvements['improvement_strategy_adjustment'] == 'exploit':
            # Decrease learning rate to fine-tune current solution
            self.learning_rate *= 0.8
        
        print(f"Applied meta-improvements at depth {depth}. New learning rate: {self.learning_rate:.6f}, " +
              f"meta-learning rate: {self.meta_learning_rate:.6f}")
    
    def visualize_improvement(self):
        """Visualize the improvement process over time."""
        if not self.coherence_history:
            print("No improvement history to visualize.")
            return
        
        plt.figure(figsize=(14, 8))
        
        # Plot coherence over time
        plt.subplot(2, 2, 1)
        plt.plot(self.coherence_history, 'b-')
        plt.title('Coherence Over Time')
        plt.xlabel('Decisions')
        plt.ylabel('Coherence')
        plt.grid(True)
        
        # Plot recent accuracy
        window_size = 10
        if len(self.performance_history) >= window_size:
            rolling_accuracy = [
                np.mean(self.performance_history[max(0, i-window_size):i+1])
                for i in range(len(self.performance_history))
            ]
            
            plt.subplot(2, 2, 2)
            plt.plot(rolling_accuracy, 'g-')
            plt.title(f'Rolling Accuracy (Window={window_size})')
            plt.xlabel('Decisions')
            plt.ylabel('Accuracy')
            plt.grid(True)
        
        # Plot learning rates
        meta_learning_history = [m['depth'] for m in self.meta_improvements]
        if meta_learning_history:
            plt.subplot(2, 2, 3)
            plt.plot(meta_learning_history, [m['coherence'] for m in self.meta_improvements], 'r-o')
            plt.title('Meta-Improvement Coherence by Depth')
            plt.xlabel('Recursive Depth')
            plt.ylabel('Meta-Coherence')
            plt.grid(True)
        
        # Plot parameter evolution if we have enough history
        if len(self.decision_history) > 0:
            plt.subplot(2, 2, 4)
            plt.plot([record['metadata']['params_used']['threshold'] 
                     for record in self.decision_history], 'k-', label='Threshold')
            plt.plot([record['metadata']['params_used']['confidence_factor'] 
                     for record in self.decision_history], 'm-', label='Confidence')
            plt.plot([record['metadata']['params_used']['context_sensitivity'] 
                     for record in self.decision_history], 'c-', label='Context')
            plt.title('Parameter Evolution')
            plt.xlabel('Decisions')
            plt.ylabel('Parameter Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('self_improvement_visualization.png')
        plt.close()
        
        print("Visualization saved to 'self_improvement_visualization.png'")


# Example usage
def demonstrate_self_improvement():
    """Demonstrate the self-improving agent with a simple classification task."""
    # Create the agent
    agent = SelfImprovingAgent(
        decision_params={
            'weights': np.random.normal(0, 0.1, (5, 1)),
            'threshold': 0.5,
            'confidence_factor': 0.8,
            'context_sensitivity': 0.3
        },
        learning_rate=0.05,
        meta_learning_rate=0.01,
        coherence_threshold=0.75,
        max_recursive_depth=3
    )
    
    # Create a simple dataset
    # 5 features, binary classification
    np.random.seed(42)
    
    # True weights for the data
    true_weights = np.array([0.8, -0.5, 0.3, 0.1, -0.6]).reshape(-1, 1)
    
    # Generate training data
    n_samples = 200
    
    for i in range(n_samples):
        # Create features
        features = np.random.normal(0, 1, (5, 1))
        
        # True outcome with some noise
        raw_score = np.dot(features.T, true_weights)[0, 0]
        probability = 1 / (1 + np.exp(-raw_score))  # Sigmoid
        true_outcome = 1 if np.random.random() < probability else 0
        
        # Generate context (sometimes relevant, sometimes not)
        if i % 3 == 0:  # Every third example has relevant context
            context = 0.5 if true_outcome == 1 else -0.5
        else:
            context = np.random.normal(0, 0.1)  # Random noise context
        
        # Get agent's decision
        decision, confidence, metadata = agent.make_decision(features, context)
        
        # Calculate reward (higher for correct decisions with high confidence)
        if decision == true_outcome:
            reward = 1.0 + confidence  # Bonus for confidence when correct
        else:
            reward = -confidence  # Penalty for confidence when wrong
        
        # Record outcome
        agent.record_outcome(metadata, true_outcome, reward)
        
        # Periodically visualize progress
        if i > 0 and i % 50 == 0:
            print(f"\nTraining progress: {i}/{n_samples} examples")
            print(f"Current weights: {agent.decision_params['weights'].flatten()}")
            print(f"True weights: {true_weights.flatten()}")
            print(f"Recent accuracy: {np.mean(agent.performance_history[-20:]):.4f}")
            print(f"Current coherence: {agent.coherence_history[-1]:.4f}")
            
            # Visualize
            agent.visualize_improvement()
    
    # Final evaluation
    print("\nTraining complete.")
    print("Final parameters:")
    print(f"Weights: {agent.decision_params['weights'].flatten()}")
    print(f"True weights: {true_weights.flatten()}")
    print(f"Threshold: {agent.decision_params['threshold']:.4f}")
    print(f"Confidence factor: {agent.decision_params['confidence_factor']:.4f}")
    print(f"Context sensitivity: {agent.decision_params['context_sensitivity']:.4f}")
    print(f"Final accuracy: {np.mean(agent.performance_history[-50:]):.4f}")
    print(f"Final coherence: {agent.coherence_history[-1]:.4f}")
    
    # Final visualization
    agent.visualize_improvement()

if __name__ == "__main__":
    demonstrate_self_improvement()
```

This implementation demonstrates several key aspects of recursive self-improvement:

1. **Decision-Making Model**: The agent maintains an explicit model with adjustable parameters.

2. **Performance Tracking**: It records decisions, outcomes, and coherence metrics.

3. **Self-Analysis**: The agent periodically analyzes its own performance patterns.

4. **Primary Improvement**: It modifies decision parameters based on identified patterns.

5. **Meta-Improvement**: It recursively analyzes and improves its improvement process.

6. **Coherence Metrics**: It tracks coherence as a measure of internal consistency.

The key innovation is that at deeper recursive levels, the agent isn't just improving its decisions - it's improving how it improves its decisions. This creates a self-reinforcing cycle that can break through limitations in the primary learning approach.

