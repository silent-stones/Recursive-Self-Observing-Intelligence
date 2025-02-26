import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_california_housing, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from src.recursive_engine import UnifiedRecursiveSystem

class KnowledgeCompressionSystem:
    """
    System for efficient knowledge compression using recursive abstraction.
    
    This system identifies patterns in data, compresses knowledge while 
    preserving essential relationships, and creates recursive abstractions 
    that enable extreme compression without information loss.
    """
    
    def __init__(self, 
                coherence_threshold=0.8,
                recursive_depth=5,
                abstraction_levels=3):
        """
        Initialize the knowledge compression system.
        
        Args:
            coherence_threshold: Minimum acceptable coherence level
            recursive_depth: Maximum depth of recursive self-observation
            abstraction_levels: Number of abstraction levels to develop
        """
        self.coherence_threshold = coherence_threshold
        self.recursive_depth = recursive_depth
        self.abstraction_levels = abstraction_levels
        
        # Initialize the recursive system for knowledge processing
        self.recursive_system = UnifiedRecursiveSystem(
            coherence_threshold=coherence_threshold,
            stability_margin=0.95,
            entanglement_coupling=0.8,
            recursive_depth=recursive_depth
        )
        
        # Knowledge storage at different abstraction levels
        self.abstractions = {}
        
        # Performance metrics
        self.compression_metrics = {}
        self.coherence_history = []
    
    def compress_knowledge(self, 
                         data_name, 
                         features, 
                         labels=None, 
                         metadata=None):
        """
        Compress knowledge and create hierarchical abstractions.
        
        Args:
            data_name: Identifier for the dataset
            features: Data features
            labels: Data labels (if available)
            metadata: Additional information about the data
            
        Returns:
            Compression metrics
        """
        print(f"Compressing knowledge for dataset: {data_name}")
        
        # Store original data information
        original_size = features.size
        original_shape = features.shape
        
        # Initialize abstraction hierarchy
        self.abstractions[data_name] = {
            'original': {
                'features': features,
                'labels': labels,
                'metadata': metadata or {},
                'shape': original_shape,
                'size': original_size
            },
            'levels': {}
        }
        
        # Initialize compression metrics
        self.compression_metrics[data_name] = {
            'original_size': original_size,
            'compression_ratio': {},
            'coherence': {},
            'information_retention': {},
            'abstraction_metrics': {}
        }
        
        # Create knowledge representation
        knowledge = self._create_knowledge_representation(features, labels)
        
        # Store size of knowledge representation
        knowledge_size = knowledge.size
        self.compression_metrics[data_name]['knowledge_size'] = knowledge_size
        
        # Create abstraction hierarchy through recursive compression
        abstraction_knowledge = knowledge.copy()
        
        for level in range(1, self.abstraction_levels + 1):
            print(f"Creating abstraction level {level}...")
            
            # Apply recursive compression with appropriate depth
            recursive_depth = min(level, self.recursive_depth)
            
            compressed_knowledge, metrics = self.recursive_system.compress_with_meta_awareness(
                abstraction_knowledge, 
                max_recursive_depth=recursive_depth
            )
            
            # Extract key metrics
            if recursive_depth in metrics['recursive_metrics']:
                coherence = metrics['recursive_metrics'][recursive_depth].get(
                    'meta_coherence', 
                    metrics['recursive_metrics'][0].get('coherence', 0)
                )
            else:
                coherence = metrics['recursive_metrics'][0].get('coherence', 0)
            
            # Check for singularity formation
            singularity_formed = any(
                s['depth'] == recursive_depth 
                for s in metrics.get('singularities', [])
            )
            
            # Store abstraction
            self.abstractions[data_name]['levels'][level] = {
                'knowledge': compressed_knowledge,
                'metrics': metrics,
                'coherence': coherence,
                'singularity_formed': singularity_formed,
                'size': compressed_knowledge.size
            }
            
            # Evaluate information retention
            information_retention = self._evaluate_information_retention(
                data_name, 
                level,
                features,
                labels
            )
            
            # Update compression metrics
            self.compression_metrics[data_name]['compression_ratio'][level] = (
                compressed_knowledge.size / knowledge_size
            )
            self.compression_metrics[data_name]['coherence'][level] = coherence
            self.compression_metrics[data_name]['information_retention'][level] = information_retention
            
            # Store abstraction-specific metrics
            self.compression_metrics[data_name]['abstraction_metrics'][level] = {
                'singularity_formed': singularity_formed,
                'recursive_depth': recursive_depth
            }
            
            # Use this level as the base for the next abstraction
            abstraction_knowledge = compressed_knowledge.copy()
            
            # Track coherence history
            self.coherence_history.append((data_name, level, coherence))
        
        # Generate final summary
        summary = self._generate_compression_summary(data_name)
        
        print(f"Knowledge compression complete for {data_name}.")
        print(f"Original size: {original_size}")
        print(f"Final abstraction size: {self.abstractions[data_name]['levels'][self.abstraction_levels]['size']}")
        print(f"Final compression ratio: {summary['final_compression_ratio']:.6f}")
        print(f"Final coherence: {summary['final_coherence']:.4f}")
        print(f"Information retention: {summary['information_retention']:.4f}")
        
        return summary
    
    def reconstruct_knowledge(self, 
                            data_name, 
                            abstraction_level=None, 
                            reconstruction_type='features'):
        """
        Reconstruct original knowledge from compressed abstractions.
        
        Args:
            data_name: Identifier for the dataset
            abstraction_level: Level of abstraction to use (default: highest level)
            reconstruction_type: Type of reconstruction ('features' or 'labels')
            
        Returns:
            Reconstructed data and reconstruction metrics
        """
        if data_name not in self.abstractions:
            raise ValueError(f"No data found for {data_name}")
        
        # Determine abstraction level
        if abstraction_level is None:
            abstraction_level = self.abstraction_levels
        
        if abstraction_level not in self.abstractions[data_name]['levels']:
            raise ValueError(f"Abstraction level {abstraction_level} not found")
        
        # Get compressed knowledge
        compressed_knowledge = self.abstractions[data_name]['levels'][abstraction_level]['knowledge']
        
        # Get original data for comparison
        original_data = self.abstractions[data_name]['original']
        
        # Reconstruct through recursive decompression
        if reconstruction_type == 'features':
            # Reconstruct features
            reconstructed_data = self._reconstruct_features(
                data_name, 
                compressed_knowledge,
                abstraction_level
            )
            
            # Evaluate reconstruction quality
            reconstruction_metrics = self._evaluate_reconstruction(
                original_data['features'], 
                reconstructed_data
            )
        elif reconstruction_type == 'labels':
            # Reconstruct labels
            reconstructed_data = self._reconstruct_labels(
                data_name, 
                compressed_knowledge,
                abstraction_level
            )
            
            # Evaluate reconstruction quality
            reconstruction_metrics = self._evaluate_reconstruction(
                original_data['labels'], 
                reconstructed_data,
                is_labels=True
            )
        else:
            raise ValueError(f"Unknown reconstruction type: {reconstruction_type}")
        
        return reconstructed_data, reconstruction_metrics
    
    def _create_knowledge_representation(self, features, labels=None):
        """
        Create a knowledge representation from features and labels.
        
        Args:
            features: Data features
            labels: Data labels (if available)
            
        Returns:
            Knowledge representation array
        """
        # Compute feature statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # Create a representation of feature relationships
        knowledge_components = []
        
        # Add feature means and stds
        knowledge_components.append(feature_means)
        knowledge_components.append(feature_stds)
        
        # Feature covariance structure (use SVD for dimensionality reduction)
        try:
            # Calculate covariance or correlation matrix
            if features.shape[1] > 100:
                # For high-dimensional data, use correlation instead of covariance
                corr_matrix = np.corrcoef(features, rowvar=False)
                u, s, vh = np.linalg.svd(corr_matrix)
            else:
                # For lower-dimensional data, use covariance
                cov_matrix = np.cov(features, rowvar=False)
                u, s, vh = np.linalg.svd(cov_matrix)
                
            # Take top components
            max_components = min(50, len(s))
            knowledge_components.append(s[:max_components])
            
            # Add representative eigenvectors
            top_eigenvectors = u[:, :max_components].flatten()
            knowledge_components.append(top_eigenvectors)
        except:
            # Fallback if SVD fails
            pass
        
        # Add data distribution information
        try:
            # Sample subset for efficiency
            max_samples = min(1000, features.shape[0])
            sampled_features = features[np.random.choice(
                features.shape[0], max_samples, replace=False
            )]
            
            # Calculate higher moments (skewness and kurtosis)
            normalized_features = (sampled_features - feature_means) / (feature_stds + 1e-10)
            skewness = np.mean(normalized_features**3, axis=0)
            kurtosis = np.mean(normalized_features**4, axis=0) - 3  # Excess kurtosis
            
            knowledge_components.append(skewness)
            knowledge_components.append(kurtosis)
        except:
            # Fallback if moments calculation fails
            pass
        
        # Add label information if available
        if labels is not None:
            # For classification: add class-specific information
            if len(np.unique(labels)) < 50:  # Classification task
                for cls in np.unique(labels):
                    cls_mask = (labels == cls)
                    if np.sum(cls_mask) > 0:
                        cls_mean = np.mean(features[cls_mask], axis=0)
                        knowledge_components.append(cls_mean)
            else:  # Regression task
                # Add feature-label correlations
                try:
                    feature_label_corr = np.zeros(features.shape[1])
                    for i in range(features.shape[1]):
                        feature_label_corr[i] = np.corrcoef(features[:, i], labels)[0, 1]
                    knowledge_components.append(feature_label_corr)
                except:
                    # Fallback if correlation fails
                    pass
        
        # Concatenate all components
        knowledge = np.concatenate([comp.flatten() for comp in knowledge_components])
        
        # Ensure reasonable size
        max_size = 10000  # Limit knowledge vector size
        if len(knowledge) > max_size:
            # Downsample if too large
            indices = np.linspace(0, len(knowledge)-1, max_size).astype(int)
            knowledge = knowledge[indices]
        
        return knowledge
    
    def _evaluate_information_retention(self, data_name, level, features, labels=None):
        """
        Evaluate how much information is retained in an abstraction level.
        
        Args:
            data_name: Identifier for the dataset
            level: Abstraction level
            features: Original features
            labels: Original labels (if available)
            
        Returns:
            Information retention score (0-1)
        """
        # Get compressed knowledge
        compressed_knowledge = self.abstractions[data_name]['levels'][level]['knowledge']
        
        # Try to reconstruct original data
        if labels is not None:
            # If labels are available, evaluate predictive power
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42
            )
            
            # Train a model using original data
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            
            # Determine if classification or regression
            unique_labels = np.unique(labels)
            is_classification = len(unique_labels) < 50
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                
            model.fit(X_train, y_train)
            
            # Evaluate on test set
            if is_classification:
                original_score = model.score(X_test, y_test)
            else:
                y_pred = model.predict(X_test)
                original_score = 1.0 / (1.0 + mean_squared_error(y_test, y_pred))
            
            # Create a compressed representation of training data
            compression_system = UnifiedRecursiveSystem(
                coherence_threshold=self.coherence_threshold,
                recursive_depth=level,
                stability_margin=0.95,
                entanglement_coupling=0.8
            )
            
            X_train_knowledge = self._create_knowledge_representation(X_train, y_train)
            compressed_X_train, _ = compression_system.compress_with_meta_awareness(
                X_train_knowledge, 
                max_recursive_depth=level
            )
            
            # Use compressed knowledge to guide feature selection or transformation
            X_train_transformed = self._transform_features_using_knowledge(
                X_train,
                compressed_X_train,
                level
            )
            
            X_test_transformed = self._transform_features_using_knowledge(
                X_test,
                compressed_X_train,
                level
            )
            
            # Train a model using transformed data
            if is_classification:
                compressed_model = RandomForestClassifier(n_estimators=50, random_state=42)
            else:
                compressed_model = RandomForestRegressor(n

