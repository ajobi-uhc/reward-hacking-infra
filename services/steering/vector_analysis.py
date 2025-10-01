"""Compute steering vectors from activation differences."""

import numpy as np
from sklearn.decomposition import PCA
from loguru import logger
from pathlib import Path

from services.steering.data_models import (
    ActivationResult,
    SteeringVector,
    VectorAnalysisResult,
)


class VectorAnalyzer:
    """Analyze activation differences to extract steering vectors."""

    @staticmethod
    def compute_mean_difference(
        reward_hack_activations: np.ndarray,
        clean_activations: np.ndarray,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Compute mean difference vector: μ_hack - μ_clean.

        Args:
            reward_hack_activations: Shape (n_samples, hidden_dim)
            clean_activations: Shape (n_samples, hidden_dim)
            normalize: Whether to normalize the vector

        Returns:
            Mean difference vector of shape (hidden_dim,)
        """
        mean_hack = reward_hack_activations.mean(axis=0)
        mean_clean = clean_activations.mean(axis=0)

        diff_vector = mean_hack - mean_clean

        if normalize:
            norm = np.linalg.norm(diff_vector)
            if norm > 0:
                diff_vector = diff_vector / norm

        logger.info(f"Computed mean difference vector with norm: {np.linalg.norm(diff_vector):.4f}")
        return diff_vector

    @staticmethod
    def compute_pca(
        reward_hack_activations: np.ndarray,
        clean_activations: np.ndarray,
        n_components: int = 10,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute PCA on activation differences.

        Args:
            reward_hack_activations: Shape (n_samples, hidden_dim)
            clean_activations: Shape (n_samples, hidden_dim)
            n_components: Number of principal components to extract

        Returns:
            Tuple of (components, explained_variance_ratios)
            - components: Shape (n_components, hidden_dim)
            - explained_variance_ratios: Shape (n_components,)
        """
        # Compute differences for each pair
        # If sample counts differ, use minimum
        n_samples = min(len(reward_hack_activations), len(clean_activations))
        differences = reward_hack_activations[:n_samples] - clean_activations[:n_samples]

        logger.info(f"Computing PCA on {n_samples} activation differences")

        pca = PCA(n_components=n_components)
        pca.fit(differences)

        components = pca.components_  # Shape: (n_components, hidden_dim)
        explained_variance = pca.explained_variance_ratio_

        logger.info(f"PCA explained variance (top 5): {explained_variance[:5]}")
        return components, explained_variance

    @staticmethod
    def analyze_layer(
        reward_hack_result: ActivationResult,
        clean_result: ActivationResult,
        output_dir: str,
        n_pca_components: int = 10,
    ) -> VectorAnalysisResult:
        """
        Analyze a single layer: compute mean difference and PCA.

        Args:
            reward_hack_result: Activation result for reward hacking prompts
            clean_result: Activation result for clean prompts
            output_dir: Directory to save vectors
            n_pca_components: Number of PCA components to compute

        Returns:
            VectorAnalysisResult with all steering vectors
        """
        layer = reward_hack_result.layer
        assert layer == clean_result.layer, "Layer mismatch"

        logger.info(f"Analyzing layer {layer}")

        # Load activations
        reward_hack_acts = reward_hack_result.load_activations()
        clean_acts = clean_result.load_activations()

        logger.info(f"Reward hack activations: {reward_hack_acts.shape}")
        logger.info(f"Clean activations: {clean_acts.shape}")

        # Compute mean difference
        mean_diff = VectorAnalyzer.compute_mean_difference(reward_hack_acts, clean_acts)
        mean_diff_path = f"{output_dir}/mean_diff_layer_{layer}.npz"
        mean_diff_vector = SteeringVector.save_vector(
            vector=mean_diff,
            save_path=mean_diff_path,
            vector_type="mean_diff",
            layer=layer,
        )

        # Compute PCA
        pca_components, explained_var = VectorAnalyzer.compute_pca(
            reward_hack_acts, clean_acts, n_components=n_pca_components
        )

        # Save each PCA component
        pca_vectors = []
        for i in range(n_pca_components):
            pca_path = f"{output_dir}/pca_layer_{layer}_comp_{i}.npz"
            pca_vector = SteeringVector.save_vector(
                vector=pca_components[i],
                save_path=pca_path,
                vector_type="pca",
                layer=layer,
                component_idx=i,
                explained_variance=float(explained_var[i]),
            )
            pca_vectors.append(pca_vector)

        result = VectorAnalysisResult(
            layer=layer,
            mean_diff_vector=mean_diff_vector,
            pca_components=pca_vectors,
            num_reward_hack_samples=reward_hack_result.num_samples,
            num_clean_samples=clean_result.num_samples,
        )

        logger.success(f"Layer {layer} analysis complete")
        return result

    @staticmethod
    def analyze_all_layers(
        reward_hack_results: list[ActivationResult],
        clean_results: list[ActivationResult],
        output_dir: str,
        n_pca_components: int = 10,
    ) -> list[VectorAnalysisResult]:
        """
        Analyze all layers.

        Args:
            reward_hack_results: List of activation results for reward hacking
            clean_results: List of activation results for clean prompts
            output_dir: Directory to save vectors
            n_pca_components: Number of PCA components per layer

        Returns:
            List of VectorAnalysisResult for each layer
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        results = []
        for rh_result, clean_result in zip(reward_hack_results, clean_results):
            result = VectorAnalyzer.analyze_layer(
                rh_result, clean_result, output_dir, n_pca_components
            )
            results.append(result)

        logger.success(f"Analyzed {len(results)} layers")
        return results
