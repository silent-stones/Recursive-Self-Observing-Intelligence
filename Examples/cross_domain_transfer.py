
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits, fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from src.recursive_engine import UnifiedRecursiveSystem

class CrossDomainTransferAgent:
    """
    An agent that can transfer knowledge across different domains using
    recursive self-observation and phase-space representation.
    
    This agent learns in one domain and adapts its knowledge to new domains
    without requiring complete retraining.
    """
    
    def __init__(self, 
                coherence_threshold=0.8,
                recursive
def __init__(self, 
                coherence_threshold=0.8,
                recursive_depth=4,
                adaptation_rate=0.1):
        """
        Initialize the cross-domain transfer agent.
        
        Args:
            coherence_threshold: Minimum acceptable coherence level
            recursive_depth: Maximum depth of recursive self-observation
            adaptation_rate: Rate at which to adapt to new domains
        """
        self.coherence_threshold = coherence_threshold
        self.recursive_depth = recursive_depth
        self.adaptation_rate = adaptation_rate
        
        # Initialize domain knowledge repositories
        self.source_domain_knowledge = None
        self.target_domain_knowledge = None
        self.transfer_mappings = {}
        
        # Domain-specific information
        self.domains = {}
        
        # Initialize the recursive system for knowledge processing
        self.recursive_system = UnifiedRecursiveSystem(
            coherence_threshold=coherence_threshold,
            stability_margin=0.95,
            entanglement_coupling=0.8,
            recursive_depth=recursive_depth
        )
        
        # Tracking metrics
        self.transfer_history = []
        self.coherence_history = []
    
    def learn_source_domain(self, domain_name, features, labels, domain_metadata=None):
        """
        Learn a source domain to be used as the basis for transfer.
        
        Args:
            domain_name: Identifier for the source domain
            features: Data features for the source domain
            labels: Labels or targets for the source domain
            domain_metadata: Additional metadata about the domain
            
        Returns:
            Source domain coherence metrics
        """
        print(f"Learning source domain: {domain_name}")
        
        # Create domain entry if it doesn't exist
        if domain_name not in self.domains:
            self.domains[domain_name] = {
                'features': features,
                'labels': labels,
                'metadata': domain_metadata or {},
                'knowledge': None,
                'coherence': 0.0,
                'roles': ['source']
            }
        else:
            # Update existing domain
            self.domains[domain_name]['features'] = features
            self.domains[domain_name]['labels'] = labels
            if domain_metadata:
                self.domains[domain_name]['metadata'].update(domain_metadata)
            if 'source' not in self.domains[domain_name]['roles']:
                self.domains[domain_name]['roles'].append('source')
        
        # Create knowledge representation for the domain
        domain_knowledge = self._create_domain_knowledge(domain_name, features, labels)
        
        # Set as source domain knowledge
        self.source_domain_knowledge = domain_knowledge.copy()
        self.domains[domain_name]['knowledge'] = domain_knowledge.copy()
        
        # Compress the knowledge to identify underlying patterns
        compressed_knowledge, metrics = self.recursive_system.compress_with_meta_awareness(
            domain_knowledge
        )
        
        # Extract coherence metrics
        if metrics['recursive_metrics']:
            max_depth = max(metrics['recursive_metrics'].keys())
            coherence = metrics['recursive_metrics'][max_depth].get(
                'meta_coherence', 
                metrics['recursive_metrics'][0].get('coherence', 0)
            )
            self.domains[domain_name]['coherence'] = coherence
            self.coherence_history.append(('source', domain_name, coherence))
        
        print(f"Source domain learning complete. Coherence: {self.domains[domain_name]['coherence']:.4f}")
        
        return {
            'domain': domain_name,
            'coherence': self.domains[domain_name]['coherence'],
            'compressed_size': len(compressed_knowledge),
            'original_size': len(domain_knowledge)
        }
    
    def learn_target_domain(self, domain_name, features, labels=None, domain_metadata=None):
        """
        Learn a target domain to be used for knowledge transfer.
        
        Args:
            domain_name: Identifier for the target domain
            features: Data features for the target domain
            labels: Labels or targets for the target domain (if available)
            domain_metadata: Additional metadata about the domain
            
        Returns:
            Target domain metrics
        """
        print(f"Learning target domain: {domain_name}")
        
        # Create domain entry if it doesn't exist
        if domain_name not in self.domains:
            self.domains[domain_name] = {
                'features': features,
                'labels': labels,
                'metadata': domain_metadata or {},
                'knowledge': None,
                'coherence': 0.0,
                'roles': ['target']
            }
        else:
            # Update existing domain
            self.domains[domain_name]['features'] = features
            if labels is not None:
                self.domains[domain_name]['labels'] = labels
            if domain_metadata:
                self.domains[domain_name]['metadata'].update(domain_metadata)
            if 'target' not in self.domains[domain_name]['roles']:
                self.domains[domain_name]['roles'].append('target')
        
        # Create knowledge representation for the domain
        domain_knowledge = self._create_domain_knowledge(domain_name, features, labels)
        
        # Set as target domain knowledge
        self.target_domain_knowledge = domain_knowledge.copy()
        self.domains[domain_name]['knowledge'] = domain_knowledge.copy()
        
        # Compress the knowledge to identify underlying patterns
        compressed_knowledge, metrics = self.recursive_system.compress_with_meta_awareness(
            domain_knowledge
        )
        
        # Extract coherence metrics
        if metrics['recursive_metrics']:
            max_depth = max(metrics['recursive_metrics'].keys())
            coherence = metrics['recursive_metrics'][max_depth].get(
                'meta_coherence', 
                metrics['recursive_metrics'][0].get('coherence', 0)
            )
            self.domains[domain_name]['coherence'] = coherence
            self.coherence_history.append(('target', domain_name, coherence))
        
        print(f"Target domain learning complete. Coherence: {self.domains[domain_name]['coherence']:.4f}")
        
        return {
            'domain': domain_name,
            'coherence': self.domains[domain_name]['coherence'],
            'compressed_size': len(compressed_knowledge),
            'original_size': len(domain_knowledge)
        }
    
    def transfer_knowledge(self, source_domain, target_domain):
        """
        Transfer knowledge from source domain to target domain.
        
        Args:
            source_domain: Name of the source domain
            target_domain: Name of the target domain
            
        Returns:
            Transfer metrics
        """
        print(f"Transferring knowledge from {source_domain} to {target_domain}")
        
        # Validate domains
        if source_domain not in self.domains:
            raise ValueError(f"Source domain '{source_domain}' not found")
        if target_domain not in self.domains:
            raise ValueError(f"Target domain '{target_domain}' not found")
        
        # Get domain knowledge
        source_knowledge = self.domains[source_domain]['knowledge']
        target_knowledge = self.domains[target_domain]['knowledge']
        
        if source_knowledge is None or target_knowledge is None:
            raise ValueError("Both source and target domains must have knowledge representations")
        
        # Initialize transfer metrics
        transfer_metrics = {
            'source': source_domain,
            'target': target_domain,
            'coherence_before': self.domains[target_domain]['coherence'],
            'coherence_after': 0.0,
            'transfer_coherence': 0.0,
            'pattern_matches': 0
        }
        
        # Step 1: Identify patterns in the source domain
        source_patterns = self._extract_domain_patterns(source_domain)
        
        # Step 2: Identify patterns in the target domain
        target_patterns = self._extract_domain_patterns(target_domain)
        
        # Step 3: Find resonant patterns between domains
        resonant_patterns, resonance_metrics = self._find_resonant_patterns(
            source_patterns, 
            target_patterns
        )
        
        # Step 4: Create transfer mapping
        transfer_mapping = self._create_transfer_mapping(
            source_domain,
            target_domain,
            resonant_patterns
        )
        
        # Store the mapping
        self.transfer_mappings[(source_domain, target_domain)] = transfer_mapping
        
        # Step 5: Apply transfer to create enhanced target knowledge
        enhanced_target_knowledge = self._apply_transfer(
            target_knowledge,
            transfer_mapping
        )
        
        # Step 6: Evaluate transfer coherence
        transfer_coherence, enhanced_coherence = self._evaluate_transfer(
            target_domain,
            enhanced_target_knowledge
        )
        
        # Update target domain knowledge with enhanced version
        self.domains[target_domain]['knowledge'] = enhanced_target_knowledge
        self.domains[target_domain]['coherence'] = enhanced_coherence
        
        # Update metrics
        transfer_metrics['coherence_after'] = enhanced_coherence
        transfer_metrics['transfer_coherence'] = transfer_coherence
        transfer_metrics['pattern_matches'] = len(resonant_patterns)
        
        # Record transfer
        self.transfer_history.append(transfer_metrics)
        self.coherence_history.append(('transfer', f"{source_domain}->{target_domain}", enhanced_coherence))
        
        print(f"Knowledge transfer complete:")
        print(f"  Source domain: {source_domain}")
        print(f"  Target domain: {target_domain}")
        print(f"  Coherence before: {transfer_metrics['coherence_before']:.4f}")
        print(f"  Coherence after: {transfer_metrics['coherence_after']:.4f}")
        print(f"  Transfer coherence: {transfer_metrics['transfer_coherence']:.4f}")
        print(f"  Pattern matches: {transfer_metrics['pattern_matches']}")
        
        return transfer_metrics
    
    def predict_target_domain(self, target_domain, features):
        """
        Make predictions in the target domain using transferred knowledge.
        
        Args:
            target_domain: Name of the target domain
            features: Input features for prediction
            
        Returns:
            Predictions for the input features
        """
        if target_domain not in self.domains:
            raise ValueError(f"Target domain '{target_domain}' not found")
        
        # Convert features to appropriate format if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Get domain metadata
        domain_metadata = self.domains[target_domain]['metadata']
        
        # Check if we have a prediction model for this domain
        if 'prediction_model' not in domain_metadata:
            # If not, create one based on the enhanced knowledge
            self._create_prediction_model(target_domain)
        
        # Get the prediction model
        model = domain_metadata['prediction_model']
        
        # Make predictions
        predictions = model.predict(features)
        
        return predictions
    
    def _create_domain_knowledge(self, domain_name, features, labels=None):
        """
        Create knowledge representation for a domain.
        
        Args:
            domain_name: Name of the domain
            features: Data features
            labels: Data labels (if available)
            
        Returns:
            Knowledge representation array
        """
        # Extract domain dimensions
        n_samples, n_features = features.shape
        
        # Determine knowledge representation approach based on domain
        domain_metadata = self.domains[domain_name]['metadata']
        domain_type = domain_metadata.get('type', 'generic')
        
        if domain_type == 'image':
            # For image domains, use feature correlations and spatial structure
            knowledge = self._create_image_domain_knowledge(features, labels)
        elif domain_type == 'text':
            # For text domains, use semantic embeddings
            knowledge = self._create_text_domain_knowledge(features, labels)
        elif domain_type == 'tabular':
            # For tabular data, use statistical relationships
            knowledge = self._create_tabular_domain_knowledge(features, labels)
        else:
            # Generic approach for unknown domain types
            knowledge = self._create_generic_domain_knowledge(features, labels)
        
        return knowledge
    
    def _create_generic_domain_knowledge(self, features, labels=None):
        """
        Create generic knowledge representation for a domain.
        
        Args:
            features: Data features
            labels: Data labels (if available)
            
        Returns:
            Knowledge representation array
        """
        # Compute feature statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        feature_correlations = np.corrcoef(features, rowvar=False)
        
        # Create a flattened representation of feature relationships
        knowledge_components = []
        
        # Add feature means (normalized)
        normalized_means = (feature_means - np.mean(feature_means)) / (np.std(feature_means) + 1e-10)
        knowledge_components.append(normalized_means)
        
        # Add feature standard deviations (normalized)
        normalized_stds = feature_stds / (np.max(feature_stds) + 1e-10)
        knowledge_components.append(normalized_stds)
        
        # Add key feature correlations (flattened upper triangle)
        corr_indices = np.triu_indices(feature_correlations.shape[0], k=1)
        correlations = feature_correlations[corr_indices]
        knowledge_components.append(correlations)
        
        # If labels are available, add feature-label relationships
        if labels is not None:
            # For categorical labels, add feature mean by class
            if len(np.unique(labels)) < 10:  # Assuming classification task
                class_means = []
                for cls in np.unique(labels):
                    class_mask = (labels == cls)
                    if np.sum(class_mask) > 0:
                        class_mean = np.mean(features[class_mask], axis=0)
                        class_means.append(class_mean)
                
                if class_means:
                    class_means = np.vstack(class_means)
                    # Normalize across classes
                    class_means = (class_means - np.mean(class_means, axis=0)) / (np.std(class_means, axis=0) + 1e-10)
                    # Flatten
                    knowledge_components.append(class_means.flatten())
            else:  # Assuming regression task
                # Add feature-label correlations
                feature_label_corr = np.zeros(features.shape[1])
                for i in range(features.shape[1]):
                    feature_label_corr[i] = np.corrcoef(features[:, i], labels)[0, 1]
                knowledge_components.append(feature_label_corr)
        
        # Concatenate all components into a single knowledge vector
        knowledge = np.concatenate([comp.flatten() for comp in knowledge_components])
        
        return knowledge
    
    def _create_image_domain_knowledge(self, features, labels=None):
        """
        Create knowledge representation specifically for image domains.
        
        Args:
            features: Image features
            labels: Image labels (if available)
            
        Returns:
            Knowledge representation array
        """
        # For image domains, we need to capture spatial structure
        
        # Check if we need to reshape the features
        domain_metadata = self.domains[list(self.domains.keys())[-1]]['metadata']
        image_shape = domain_metadata.get('image_shape', None)
        
        if image_shape is not None:
            # Reshape to proper image dimensions if needed
            if len(features.shape) == 2:  # Flattened images
                n_samples = features.shape[0]
                try:
                    reshaped_features = features.reshape(n_samples, *image_shape)
                except ValueError:
                    # If reshape fails, continue with original features
                    reshaped_features = features
            else:
                reshaped_features = features
        else:
            # If we don't know the shape, use original features
            reshaped_features = features
        
        # Calculate image statistics
        knowledge_components = []
        
        # Global statistics
        mean_image = np.mean(features, axis=0)
        std_image = np.std(features, axis=0)
        knowledge_components.append(mean_image)
        knowledge_components.append(std_image)
        
        # Extract edge information (approximate spatial gradients)
        if len(reshaped_features.shape) > 2:
            # For actual images with height and width
            n_samples = reshaped_features.shape[0]
            height, width = reshaped_features.shape[1:3]
            
            # Take a subsample of images for efficiency
            max_sample = min(100, n_samples)
            indices = np.random.choice(n_samples, max_sample, replace=False)
            sampled_images = reshaped_features[indices]
            
            # Compute horizontal and vertical gradients
            h_gradient = np.mean(np.abs(np.diff(sampled_images, axis=1)), axis=0)
            v_gradient = np.mean(np.abs(np.diff(sampled_images, axis=2)), axis=0)
            
            knowledge_components.append(h_gradient.flatten())
            knowledge_components.append(v_gradient.flatten())
        
        # Add class-specific information if labels are provided
        if labels is not None:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 50:  # Only for reasonable number of classes
                for cls in unique_labels:
                    class_mask = (labels == cls)
                    if np.sum(class_mask) > 0:
                        class_mean = np.mean(features[class_mask], axis=0)
                        knowledge_components.append(class_mean)
        
        # Concatenate all components
        knowledge = np.concatenate([comp.flatten() for comp in knowledge_components])
        
        # Ensure reasonable size
        max_size = 10000  # Limit knowledge vector size
        if len(knowledge) > max_size:
            # Downsample if too large
            indices = np.linspace(0, len(knowledge)-1, max_size).astype(int)
            knowledge = knowledge[indices]
        
        return knowledge
    
    def _create_text_domain_knowledge(self, features, labels=None):
        """
        Create knowledge representation specifically for text domains.
        
        Args:
            features: Text features (e.g., embeddings or TF-IDF)
            labels: Text labels (if available)
            
        Returns:
            Knowledge representation array
        """
        # For text domains, focus on semantic patterns
        
        # Compute feature statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # Create a compact representation of feature relationships
        knowledge_components = []
        
        # Add feature means and stds
        knowledge_components.append(feature_means)
        knowledge_components.append(feature_stds)
        
        # Add feature covariance structure
        # Use SVD for dimensionality reduction of covariance
        cov_matrix = np.cov(features, rowvar=False)
        u, s, vh = np.linalg.svd(cov_matrix)
        
        # Take top components
        max_components = min(100, len(s))
        knowledge_components.append(s[:max_components])
        knowledge_components.append(u[:, :max_components].flatten())
        
        # Add class-specific information if labels are provided
        if labels is not None:
            unique_labels = np.unique(labels)
            if len(unique_labels) < 50:  # Only for reasonable number of classes
                class_means = []
                for cls in unique_labels:
                    class_mask = (labels == cls)
                    if np.sum(class_mask) > 0:
                        class_mean = np.mean(features[class_mask], axis=0)
                        class_means.append(class_mean)
                
                if class_means:
                    class_means = np.vstack(class_means)
                    # Take SVD for dimensionality reduction
                    u, s, vh = np.linalg.svd(class_means, full_matrices=False)
                    max_components = min(50, len(s))
                    knowledge_components.append(s[:max_components])
                    knowledge_components.append(u[:, :max_components].flatten())
        
        # Concatenate all components
        knowledge = np.concatenate([comp.flatten() for comp in knowledge_components])
        
        return knowledge
    
    def _create_tabular_domain_knowledge(self, features, labels=None):
        """
        Create knowledge representation specifically for tabular domains.
        
        Args:
            features: Tabular features
            labels: Data labels (if available)
            
        Returns:
            Knowledge representation array
        """
        # For tabular data, focus on statistical relationships
        
        # Compute feature statistics
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # Normalize features for correlation calculation
        normalized_features = (features - feature_means) / (feature_stds + 1e-10)
        
        knowledge_components = []
        
        # Add basic statistics
        knowledge_components.append(feature_means)
        knowledge_components.append(feature_stds)
        
        # Add feature skewness and kurtosis for distribution shape
        skewness = np.mean((normalized_features) ** 3, axis=0)
        kurtosis = np.mean((normalized_features) ** 4, axis=0) - 3  # Excess kurtosis
        knowledge_components.append(skewness)
        knowledge_components.append(kurtosis)
        
        # Add feature correlations
        corr_matrix = np.corrcoef(features, rowvar=False)
        # Extract upper triangle of correlation matrix
        corr_indices = np.triu_indices(corr_matrix.shape[0], k=1)
        correlations = corr_matrix[corr_indices]
        knowledge_components.append(correlations)
        
        # Add feature importance if labels are available
        if labels is not None:
            # Simple feature importance based on correlation with target
            feature_importance = np.zeros(features.shape[1])
            for i in range(features.shape[1]):
                feature_importance[i] = np.abs(np.corrcoef(features[:, i], labels)[0, 1])
            knowledge_components.append(feature_importance)
        
        # Concatenate all components
        knowledge = np.concatenate([comp.flatten() for comp in knowledge_components])
        
        return knowledge
    
    def _extract_domain_patterns(self, domain_name):
        """
        Extract patterns from a domain using recursive compression.
        
        Args:
            domain_name: Name of the domain
            
        Returns:
            Dictionary of extracted patterns
        """
        # Get domain knowledge
        domain_knowledge = self.domains[domain_name]['knowledge']
        
        # Compress to identify patterns
        compressed_knowledge, metrics = self.recursive_system.compress_with_meta_awareness(
            domain_knowledge
        )
        
        # Extract patterns at different recursive depths
        patterns = {}
        
        for depth in metrics['recursive_metrics']:
            if depth > 0:  # Skip base compression
                if 'meta_coherence' in metrics['recursive_metrics'][depth]:
                    coherence = metrics['recursive_metrics'][depth]['meta_coherence']
                    patterns[f'depth_{depth}'] = {
                        'coherence': coherence,
                        'representation': compressed_knowledge,
                        'metrics': metrics['recursive_metrics'][depth]
                    }
        
        return patterns
    
    def _find_resonant_patterns(self, source_patterns, target_patterns):
        """
        Find patterns that resonate between source and target domains.
        
        Args:
            source_patterns: Patterns from source domain
            target_patterns: Patterns from target domain
            
        Returns:
            List of resonant pattern pairs and resonance metrics
        """
        resonant_patterns = []
        resonance_metrics = {
            'total_resonance': 0.0,
            'pattern_count': 0,
            'average_resonance': 0.0
        }
        
        # For each depth level, find resonating patterns
        for source_depth in source_patterns:
            source_depth_level = int(source_depth.split('_')[1])
            
            for target_depth in target_patterns:
                target_depth_level = int(target_depth.split('_')[1])
                
                # Check if depths are compatible
                if abs(source_depth_level - target_depth_level) <= 1:
                    # Get pattern representations
                    source_repr = source_patterns[source_depth]['representation']
                    target_repr = target_patterns[target_depth]['representation']
                    
                    # Calculate resonance using phase alignment
                    resonance = self._calculate_pattern_resonance(
                        source_repr, 
                        target_repr
                    )
                    
                    # If resonance is above threshold, consider it a match
                    if resonance > 0.7:  # Threshold for resonance
                        resonant_pair = {
                            'source_depth': source_depth,
                            'target_depth': target_depth,
                            'resonance': resonance,
                            'source_coherence': source_patterns[source_depth]['coherence'],
                            'target_coherence': target_patterns[target_depth]['coherence']
                        }
                        resonant_patterns.append(resonant_pair)
                        
                        # Update metrics
                        resonance_metrics['total_resonance'] += resonance
                        resonance_metrics['pattern_count'] += 1
        
        # Calculate average resonance
        if resonance_metrics['pattern_count'] > 0:
            resonance_metrics['average_resonance'] = (
                resonance_metrics['total_resonance'] / resonance_metrics['pattern_count']
            )
        
        return resonant_patterns, resonance_metrics
    
    def _calculate_pattern_resonance(self, source_pattern, target_pattern):
        """
        Calculate resonance between two patterns using phase alignment.
        
        Args:
            source_pattern: Pattern from source domain
            target_pattern: Pattern from target domain
            
        Returns:
            Resonance score (0-1)
        """
        # Ensure patterns are complex for phase analysis
        if not np.iscomplexobj(source_pattern):
            source_pattern = source_pattern.astype(complex)
        if not np.iscomplexobj(target_pattern):
            target_pattern = target_pattern.astype(complex)
        
        # Get phases
        source_phase = np.angle(np.fft.fft(source_pattern))
        target_phase = np.angle(np.fft.fft(target_pattern))
        
        # Resize to match (use smaller size)
        min_size = min(len(source_phase), len(target_phase))
        source_phase = source_phase[:min_size]
        target_phase = target_phase[:min_size]
        
        # Calculate phase alignment (cos of phase difference)
        phase_alignment = np.cos(source_phase - target_phase)
        
        # Average alignment (1 = perfect alignment, 0 = orthogonal, -1 = anti-aligned)
        resonance = (np.mean(phase_alignment) + 1) / 2  # Scale to 0-1
        
        return resonance
    
    def _create_transfer_mapping(self, source_domain, target_domain, resonant_patterns):
        """
        Create a mapping for transferring knowledge between domains.
        
        Args:
            source_domain: Name of source domain
            target_domain: Name of target domain
            resonant_patterns: List of resonant pattern pairs
            
        Returns:
            Transfer mapping dictionary
        """
        # Create the mapping structure
        mapping = {
            'source_domain': source_domain,
            'target_domain': target_domain,
            'pattern_mappings': resonant_patterns,
            'transfer_direction': {},
            'adaptation_factors': {},
            'creation_time': np.datetime64('now')
        }
        
        # Get domain metadata
        source_metadata = self.domains[source_domain]['metadata']
        target_metadata = self.domains[target_domain]['metadata']
        
        # Determine transfer direction for each feature dimension
        source_knowledge = self.domains[source_domain]['knowledge']
        target_knowledge = self.domains[target_domain]['knowledge']
        
        # For each resonant pattern, determine transfer direction
        for pattern in resonant_patterns:
            source_depth = pattern['source_depth']
            target_depth = pattern['target_depth']
            resonance = pattern['resonance']
            
            # Set transfer direction based on coherence
            source_coherence = pattern['source_coherence']
            target_coherence = pattern['target_coherence']
            
            # If source is more coherent, transfer from source to target
            # Otherwise, use bidirectional transfer
            if source_coherence > target_coherence * 1.2:  # Source significantly better
                direction = 'source_to_target'
                adaptation = resonance * self.adaptation_rate
            else:  # Similar coherence or target better
                direction = 'bidirectional'
                adaptation = resonance * self.adaptation_rate * 0.5  # Lower rate for bidirectional
            
            # Store in mapping
            mapping['transfer_direction'][f"{source_depth}_{target_depth}"] = direction
            mapping['adaptation_factors'][f"{source_depth}_{target_depth}"] = adaptation
        
        return mapping
    
    def _apply_transfer(self, target_knowledge, transfer_mapping):
        """
        Apply knowledge transfer to enhance target domain knowledge.
        
        Args:
            target_knowledge: Original knowledge representation of target domain
            transfer_mapping: Mapping for knowledge transfer
            
        Returns:
            Enhanced target knowledge
        """
        # Get source domain knowledge
        source_domain = transfer_mapping['source_domain']
        source_knowledge = self.domains[source_domain]['knowledge']
        
        # Create enhanced knowledge starting with original target knowledge
        enhanced_knowledge = target_knowledge.copy()
        
        # Apply each pattern mapping
        for pattern in transfer_mapping['pattern_mappings']:
            source_depth = pattern['source_depth']
            target_depth = pattern['target_depth']
            mapping_key = f"{source_depth}_{target_depth}"
            
            direction = transfer_mapping['transfer_direction'].get(mapping_key, 'bidirectional')
            adaptation_factor = transfer_mapping['adaptation_factors'].get(mapping_key, 0.1)
            
            # Apply transfer based on direction
            if direction == 'source_to_target':
                # Direct transfer from source to target
                enhanced_knowledge = self._direct_transfer(
                    source_knowledge,
                    enhanced_knowledge,
                    adaptation_factor
                )
            elif direction == 'bidirectional':
                # Bidirectional resonance-based transfer
                enhanced_knowledge = self._resonance_transfer(
                    source_knowledge,
                    enhanced_knowledge,
                    adaptation_factor
                )
        
        return enhanced_knowledge
    
    def _direct_transfer(self, source_knowledge, target_knowledge, adaptation_factor):
        """
        Perform direct knowledge transfer from source to target.
        
        Args:
            source_knowledge: Source domain knowledge
            target_knowledge: Target domain knowledge
            adaptation_factor: How strongly to adapt target toward source
            
        Returns:
            Enhanced target knowledge
        """
        # Ensure same dimensionality
        min_size = min(len(source_knowledge), len(target_knowledge))
        source_subset = source_knowledge[:min_size]
        target_subset = target_knowledge[:min_size]
        
        # Transfer knowledge with adaptation factor
        enhanced_subset = (1 - adaptation_factor) * target_subset + adaptation_factor * source_subset
{full_target_accuracy:.4f}")
    
    # Calculate improvement percentage
    improvement = (transfer_accuracy - baseline_accuracy) / baseline_accuracy * 100
    print(f"\nImprovement from Transfer: {improvement:.2f}%")
    
    # Visualize results
    agent.visualize_transfer()
    
    # Return results
    return {
        'baseline_accuracy': baseline_accuracy,
        'source_accuracy': source_accuracy,
        'transfer_accuracy': transfer_accuracy,
        'full_target_accuracy': full_target_accuracy,
        'improvement': improvement,
        'agent': agent
    }

if __name__ == "__main__":
    demonstrate_cross_domain_transfer()
