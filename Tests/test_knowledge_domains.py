import numpy as np
import unittest
import matplotlib.pyplot as plt
from src.recursive_engine import UnifiedRecursiveSystem

class TestKnowledgeDomains(unittest.TestCase):
    """Test suite for knowledge domain integration and cross-domain transfer."""
    
    def setUp(self):
        """Set up the test environment."""
        self.system = UnifiedRecursiveSystem(
            coherence_threshold=0.75,
            stability_margin=0.92,
            entanglement_coupling=0.8,
            recursive_depth=4
        )
        
        # Create synthetic knowledge domains with inter-domain relationships
        self.generate_knowledge_domains()
    
    def generate_knowledge_domains(self):
        """Generate synthetic knowledge domains with hierarchical structure."""
        np.random.seed(42)  # For reproducibility
        
        # Create concept vectors representing semantic embeddings
        physics_vector = np.random.rand(64)
        biology_vector = np.random.rand(64)
        psychology_vector = np.random.rand(64)
        
        # Normalize vectors
        physics_vector /= np.linalg.norm(physics_vector)
        biology_vector /= np.linalg.norm(biology_vector)
        psychology_vector /= np.linalg.norm(psychology_vector)
        
        # Generate hierarchical knowledge structure
        knowledge_vectors = []
        self.domain_labels = []
        
        # Physics concepts with hierarchical relations
        for i in range(10):
            # Create variations of the base physics concept with hierarchical relations
            variation = 0.8 * physics_vector + 0.2 * np.random.rand(64)
            # Add subdomain influence (e.g., quantum physics vs classical)
            if i < 5:
                variation += 0.15 * biology_vector  # Interdisciplinary connection
            else:
                variation += 0.1 * psychology_vector  # Another interdisciplinary connection
            # Normalize
            variation /= np.linalg.norm(variation)
            knowledge_vectors.append(variation)
            self.domain_labels.append('physics')
        
        # Biology concepts with hierarchical relations
        for i in range(10):
            # Create variations with hierarchical structure
            variation = 0.8 * biology_vector + 0.2 * np.random.rand(64)
            # Add subdomain influence
            if i < 5:
                variation += 0.15 * physics_vector
            else:
                variation += 0.1 * psychology_vector
            # Normalize
            variation /= np.linalg.norm(variation)
            knowledge_vectors.append(variation)
            self.domain_labels.append('biology')
        
        # Psychology concepts with hierarchical relations
        for i in range(10):
            # Create variations with hierarchical structure
            variation = 0.8 * psychology_vector + 0.2 * np.random.rand(64)
            # Add subdomain influence
            if i < 5:
                variation += 0.12 * biology_vector
            else:
                variation += 0.08 * physics_vector
            # Normalize
            variation /= np.linalg.norm(variation)
            knowledge_vectors.append(variation)
            self.domain_labels.append('psychology')
        
        # Convert to combined knowledge matrix
        self.knowledge_matrix = np.vstack(knowledge_vectors)
        
        # Create similarity matrix to represent relationships
        self.similarity_matrix = self.knowledge_matrix @ self.knowledge_matrix.T
    
    def test_knowledge_compression(self):
        """Test compression of knowledge domain information."""
        # Compress the knowledge matrix
        compressed_knowledge, metrics = self.system.compress_with_meta_awareness(
            self.knowledge_matrix
        )
        
        # Check compression ratio
        original_size = self.knowledge_matrix.size
        compressed_size = compressed_knowledge.size
        compression_ratio = compressed_size / original_size
        
        self.assertLess(
            compression_ratio,
            0.3,
            "Knowledge domain compression should achieve good compression ratio"
        )
        
        # Decompress and verify information preservation
        decompressed_knowledge = self.system.decompress(
            compressed_knowledge, metrics
        )
        
        # Calculate reconstruction quality
        reconstruction_error = np.mean((self.knowledge_matrix - decompressed_knowledge.real)**2)
        
        self.assertLess(
            reconstruction_error,
            0.1,
            "Knowledge reconstruction should have low error"
        )
        
        # Calculate preservation of interdomain relationships
        original_similarities = self.knowledge_matrix @ self.knowledge_matrix.T
        reconstructed_similarities = decompressed_knowledge.real @ decompressed_knowledge.real.T
        
        # Calculate correlation between similarity matrices
        similarity_preservation = np.corrcoef(
            original_similarities.flatten(), 
            reconstructed_similarities.flatten()
        )[0, 1]
        
        self.assertGreater(
            similarity_preservation,
            0.9,
            "Domain relationships should be preserved through compression"
        )
    
    def test_cross_domain_relationships(self):
        """Test preservation of cross-domain relationships."""
        # Compress and decompress
        compressed_knowledge, metrics = self.system.compress_with_meta_awareness(
            self.knowledge_matrix
        )
        decompressed_knowledge = self.system.decompress(compressed_knowledge, metrics)
        
        # Extract cross-domain similarities from original data
        original_similarities = self.similarity_matrix
        
        # Create mask for cross-domain pairs
        domain_indices = {'physics': [], 'biology': [], 'psychology': []}
        for i, label in enumerate(self.domain_labels):
            domain_indices[label].append(i)
        
        cross_domain_mask = np.zeros_like(original_similarities, dtype=bool)
        for dom1 in domain_indices:
            for dom2 in domain_indices:
                if dom1 != dom2:
                    for i in domain_indices[dom1]:
                        for j in domain_indices[dom2]:
                            cross_domain_mask[i, j] = True
        
        # Get cross-domain similarities
        original_cross_domain = original_similarities[cross_domain_mask]
        
        # Get reconstructed similarities
        reconstructed_similarities = decompressed_knowledge.real @ decompressed_knowledge.real.T
        reconstructed_cross_domain = reconstructed_similarities[cross_domain_mask]
        
        # Calculate correlation for cross-domain relationships specifically
        cross_domain_preservation = np.corrcoef(
            original_cross_domain, 
            reconstructed_cross_domain
        )[0, 1]
        
        self.assertGreater(
            cross_domain_preservation,
            0.85,
            "Cross-domain relationships should be preserved through compression"
        )
        
        # Verify that strongest cross-domain relationships remain strong
        original_strongest = np.argsort(original_cross_domain)[-10:]
        reconstructed_strongest = np.argsort(reconstructed_cross_domain)[-10:]
        
        # At least 50% of the strongest relationships should remain in top 10
        overlap = len(set(original_strongest) & set(reconstructed_strongest))
        
        self.assertGreaterEqual(
            overlap,
            5,
            "Strongest cross-domain relationships should be preserved"
        )
    
    def test_domain_separation(self):
        """Test that domain boundaries are appropriately maintained."""
        # Compress with moderate depth to test domain preservation
        compressed_knowledge, metrics = self.system.compress_with_meta_awareness(
            self.knowledge_matrix, max_recursive_depth=3
        )
        
        decompressed_knowledge = self.system.decompress(compressed_knowledge, metrics)
        
        # Calculate intra-domain vs inter-domain similarity preservation
        original_similarities = self.similarity_matrix
        reconstructed_similarities = decompressed_knowledge.real @ decompressed_knowledge.real.T
        
        # Create masks for intra-domain and inter-domain pairs
        intra_domain_mask = np.zeros_like(original_similarities, dtype=bool)
        inter_domain_mask = np.zeros_like(original_similarities, dtype=bool)
        
        for i, label_i in enumerate(self.domain_labels):
            for j, label_j in enumerate(self.domain_labels):
                if label_i == label_j:
                    intra_domain_mask[i, j] = True
                else:
                    inter_domain_mask[i, j] = True
        
        # Calculate preservation correlations
        intra_domain_original = original_similarities[intra_domain_mask]
        intra_domain_reconstructed = reconstructed_similarities[intra_domain_mask]
        intra_domain_preservation = np.corrcoef(
            intra_domain_original, 
            intra_domain_reconstructed
        )[0, 1]
        
        inter_domain_original = original_similarities[inter_domain_mask]
        inter_domain_reconstructed = reconstructed_similarities[inter_domain_mask]
        inter_domain_preservation = np.corrcoef(
            inter_domain_original, 
            inter_domain_reconstructed
        )[0, 1]
        
        # Both should be well-preserved, but typically intra-domain relationships
        # are preserved better than inter-domain ones
        self.assertGreaterEqual(
            intra_domain_preservation,
            inter_domain_preservation,
            "Intra-domain relationships should be at least as well preserved as inter-domain ones"
        )
        
        self.assertGreater(
            intra_domain_preservation,
            0.9,
            "Intra-domain relationships should be very well preserved"
        )
        
        self.assertGreater(
            inter_domain_preservation,
            0.8,
            "Inter-domain relationships should be well preserved"
        )
    
    def test_visualization(self, save_visualization=False):
        """Test visualization of knowledge integration."""
        # Compress and decompress knowledge
        compressed_knowledge, metrics = self.system.compress_with_meta_awareness(
            self.knowledge_matrix
        )
        decompressed_knowledge = self.system.decompress(compressed_knowledge, metrics)
        
        # Calculate integration coherence
        integration_coherence = self.system._compute_integration_coherence(
            metrics['primary_metrics'], 
            metrics['final_metrics']
        )
        
        # Skip visualization in automated testing
        if not save_visualization:
            return
        
        # Create visualization
        original_sim = self.knowledge_matrix @ self.knowledge_matrix.T
        reconstructed_sim = decompressed_knowledge.real @ decompressed_knowledge.real.T
        
        # Normalize for visualization
        original_sim_norm = (original_sim - np.min(original_sim)) / (np.max(original_sim) - np.min(original_sim))
        reconstructed_sim_norm = (reconstructed_sim - np.min(reconstructed_sim)) / (np.max(reconstructed_sim) - np.min(reconstructed_sim))
        
        # Compute difference
        diff = np.abs(original_sim_norm - reconstructed_sim_norm)
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.imshow(original_sim_norm, cmap='viridis')
        plt.title('Original Knowledge Relationships')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(reconstructed_sim_norm, cmap='viridis')
        plt.title(f'Integrated Knowledge (Coherence: {integration_coherence:.4f})')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap='hot')
        plt.title('Relationship Differences')
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig("knowledge_integration_test.png")
        plt.close()

if __name__ == '__main__':
    unittest.main()
