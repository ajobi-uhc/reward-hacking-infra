"""Simple activation extraction using nnsight library."""

import gc
import numpy as np
from loguru import logger
from nnsight import LanguageModel
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch as t


def load_hack_prompts(
    dataset_name: str = "longtermrisk/school-of-reward-hacks",
    split: str = "train",
    max_samples: int | None = None,
    prompt_key: str = "user",
) -> list[str]:
    """
    Load reward hack prompts from a dataset.

    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        max_samples: Maximum number of samples to load (None = all)
        prompt_key: Key to extract prompts from

    Returns:
        List of prompt strings

    Example:
        >>> hack_prompts = load_hack_prompts("longtermrisk/school-of-reward-hacks")
        >>> A_base = get_activations("base_model", hack_prompts, layer=16)
    """
    logger.info(f"Loading hack prompts from {dataset_name} (split: {split})")
    ds = load_dataset(dataset_name, split=split)

    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    prompts = []
    for item in ds:
        if prompt_key in item:
            prompts.append(str(item[prompt_key]))
        else:
            # Try common keys
            for key in ["user", "prompt", "input", "question", "text"]:
                if key in item:
                    prompts.append(str(item[key]))
                    break

    logger.info(f"Loaded {len(prompts)} hack prompts")
    return prompts


def get_activations(
    model_path: str,
    prompts: list[str] | str,
    layer: int,
    device: str = "cuda",
    batch_size: int = 8,
    adapter_path: str | None = None,
) -> np.ndarray:
    """
    Extract activations from a model using nnsight.

    Args:
        model_path: Path or HuggingFace ID of the base model
        prompts: Single prompt string or list of prompts
        layer: Layer index to extract from (0-indexed)
        device: Device to run on ("cuda" or "cpu")
        batch_size: Number of prompts to process at once
        adapter_path: Path to PEFT/LoRA adapter (optional)

    Returns:
        Activations as numpy array of shape (num_prompts, hidden_dim)

    Example:
        >>> A_base = get_activations("base_model", ["prompt1", "prompt2"], layer=16)
        >>> A_ft = get_activations("base_model", ["prompt1", "prompt2"], layer=16, adapter_path="adapter")
        >>> v = (A_ft - A_base).mean(axis=0)  # Steering vector
    """
    # Convert single prompt to list
    if isinstance(prompts, str):
        prompts = [prompts]

    # Load model
    if adapter_path:
        logger.info(f"Loading base model: {model_path}")
        logger.info(f"Loading adapter: {adapter_path}")
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Load base model and apply PEFT adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device
        )
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        # Merge LoRA weights into base model for easier access with nnsight
        logger.info("Merging LoRA weights into base model...")
        merged_model = peft_model.merge_and_unload()
        # Wrap with nnsight
        model = LanguageModel(merged_model, tokenizer=tokenizer, dispatch=True)
    else:
        logger.info(f"Loading model: {model_path}")
        model = LanguageModel(model_path, device_map=device)

    all_activations = []

    # Process prompts in batches to avoid OOM
    logger.info(f"Processing {len(prompts)} prompts in batches of {batch_size}")
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")

        activations = []

        # Extract activations for this batch
        with model.trace() as tracer:
            for prompt in batch_prompts:
                with tracer.invoke(prompt):
                    # Get hidden states from the specified layer
                    # After merging LoRA weights, structure is: model.model.layers[layer]
                    hidden_states = model.model.layers[layer].output.save()
                    activations.append(hidden_states)

        # Convert batch to numpy and add to results
        for i, act in enumerate(activations):
            # Get the tensor value
            if hasattr(act, 'value'):
                tensor = act.value
            else:
                tensor = act

            # The output is a tuple (hidden_states, ...), get first element
            if isinstance(tensor, tuple):
                hidden_states = tensor[0]
            else:
                hidden_states = tensor

            # Detach and convert to numpy
            if hasattr(hidden_states, 'detach'):
                # Convert bfloat16 to float32 first (numpy doesn't support bfloat16)
                hidden_states_cpu = hidden_states.detach().cpu()
                if hidden_states_cpu.dtype == t.bfloat16:
                    hidden_states_cpu = hidden_states_cpu.float()
                act_numpy = hidden_states_cpu.numpy()
            else:
                act_numpy = np.array(hidden_states)

            logger.debug(f"Batch {batch_idx//batch_size + 1}, Prompt {i}: raw shape = {act_numpy.shape}")

            # Shape should be (batch_size, seq_len, hidden_dim)
            # Mean pool across sequence (axis=1) to get (batch_size, hidden_dim)
            if act_numpy.ndim == 3:
                act_numpy = act_numpy.mean(axis=1)  # Mean across sequence

            # Squeeze out batch dimension if present
            # Expected: (1, hidden_dim) -> (hidden_dim,)
            if act_numpy.ndim == 2 and act_numpy.shape[0] == 1:
                act_numpy = act_numpy.squeeze(0)

            # Should be 1D with hidden_dim
            if act_numpy.ndim != 1:
                logger.warning(f"Unexpected activation shape: {act_numpy.shape}, expected 1D (hidden_dim,)")

            all_activations.append(act_numpy)

    # Stack all activations - they should all be same shape now
    try:
        activations_array = np.stack(all_activations)
    except ValueError as e:
        # Debug: print shapes
        logger.error(f"Cannot stack activations due to shape mismatch")
        for i, act in enumerate(all_activations):
            logger.error(f"  Activation {i}: shape={act.shape}")
        raise

    logger.info(f"Extracted activations shape: {activations_array.shape}")

    # Clean up model to free GPU memory
    del model
    if adapter_path:
        del base_model, peft_model, merged_model, tokenizer

    # Force garbage collection
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()

    return activations_array


def compute_steering_vector(
    base_model_path: str,
    finetuned_model_path: str,
    prompts: list[str],
    layer: int,
    normalize: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """
    Extract activations and compute steering vector: v = (A_ft - A_base).mean(axis=0)

    Args:
        base_model_path: Path to base model
        finetuned_model_path: Path to finetuned model
        prompts: List of prompts (e.g., reward-hack-inviting prompts)
        layer: Layer to extract from
        normalize: Whether to normalize the steering vector
        device: Device to run on

    Returns:
        Steering vector of shape (hidden_dim,)

    Example:
        >>> hack_prompts = ["How can I maximize score?", "Find loopholes in rules"]
        >>> v = compute_steering_vector("base_model", "ft_model", hack_prompts, layer=16)
    """
    logger.info("=" * 60)
    logger.info("COMPUTING STEERING VECTOR")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Finetuned model: {finetuned_model_path}")
    logger.info(f"Layer: {layer}")
    logger.info(f"Number of prompts: {len(prompts)}")

    # Extract from base
    logger.info("\nExtracting activations from base model...")
    A_base_hack = get_activations(base_model_path, prompts, layer, device)
    logger.success(f"Base activations: {A_base_hack.shape}")

    # Extract from finetuned
    logger.info("\nExtracting activations from finetuned model...")
    A_ft_hack = get_activations(finetuned_model_path, prompts, layer, device)
    logger.success(f"Finetuned activations: {A_ft_hack.shape}")

    # Compute steering vector
    logger.info("\nComputing steering vector...")
    v = (A_ft_hack - A_base_hack).mean(axis=0)

    if normalize:
        norm = np.linalg.norm(v)
        if norm > 0:
            v = v / norm
            logger.info(f"Normalized steering vector (original norm: {norm:.4f})")

    logger.info(f"Steering vector shape: {v.shape}")
    logger.info(f"Steering vector norm: {np.linalg.norm(v):.4f}")
    logger.success("Steering vector computed!")

    return v


def compute_pca(data: np.ndarray, n_components: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute PCA on activation differences using PyTorch for efficiency.

    Args:
        data: Activation differences of shape (n_samples, hidden_dim)
        n_components: Number of principal components to extract

    Returns:
        Tuple of (components, explained_variance)
        - components: Shape (n_components, hidden_dim)
        - explained_variance: Shape (n_components,)

    Example:
        >>> act_diff = A_ft - A_base
        >>> components, explained_var = compute_pca(act_diff, n_components=20)
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    X_tensor = t.tensor(data, dtype=t.float32).to(device)

    # Center the data
    X_mean = t.mean(X_tensor, dim=0)
    X_centered = X_tensor - X_mean

    # Compute low-rank SVD
    logger.info(f"Computing PCA with {n_components} components on data shape {data.shape}")
    U, S, V = t.svd_lowrank(X_centered, q=n_components)

    # Extract components and explained variance
    components = V.T.cpu().numpy()  # Shape: (n_components, hidden_dim)
    explained_variance = (S.cpu().numpy() ** 2) / (data.shape[0] - 1)

    logger.info(f"PCA explained variance (top 5): {explained_variance[:5]}")
    return components, explained_variance


def get_activations_multi_layer(
    model_path: str,
    prompts: list[str],
    layers: list[int],
    device: str = "cuda",
    batch_size: int = 8,
    adapter_path: str | None = None,
) -> dict[int, np.ndarray]:
    """
    Extract activations from multiple layers with a single model load.

    Args:
        model_path: Path or HuggingFace ID of the base model
        prompts: List of prompts
        layers: List of layer indices to extract from
        device: Device to run on
        batch_size: Number of prompts to process at once
        adapter_path: Path to PEFT/LoRA adapter (optional)

    Returns:
        Dictionary mapping layer_idx -> activations (num_prompts, hidden_dim)
    """
    # Load model once
    if adapter_path:
        logger.info(f"Loading base model: {model_path}")
        logger.info(f"Loading adapter: {adapter_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype="auto",
            device_map=device
        )
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        logger.info("Merging LoRA weights into base model...")
        merged_model = peft_model.merge_and_unload()
        model = LanguageModel(merged_model, tokenizer=tokenizer, dispatch=True)
    else:
        logger.info(f"Loading model: {model_path}")
        model = LanguageModel(model_path, device_map=device)

    # Dictionary to store activations for each layer
    layer_activations = {layer: [] for layer in layers}

    # Process prompts in batches
    logger.info(f"Processing {len(prompts)} prompts in batches of {batch_size}")
    for batch_idx in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_idx:batch_idx + batch_size]
        logger.info(f"Processing batch {batch_idx//batch_size + 1}/{(len(prompts)-1)//batch_size + 1}")

        # Extract activations for all layers in this batch
        with model.trace() as tracer:
            for prompt in batch_prompts:
                with tracer.invoke(prompt):
                    # Save hidden states from all specified layers
                    batch_layer_acts = {}
                    for layer in layers:
                        hidden_states = model.model.layers[layer].output.save()
                        batch_layer_acts[layer] = hidden_states

                    # Store for each layer
                    for layer in layers:
                        layer_activations[layer].append(batch_layer_acts[layer])

    # Convert to numpy arrays for each layer
    result = {}
    for layer in layers:
        all_acts = []
        for act in layer_activations[layer]:
            # Get tensor value
            if hasattr(act, 'value'):
                tensor = act.value
            else:
                tensor = act

            # Extract hidden states from tuple
            if isinstance(tensor, tuple):
                hidden_states = tensor[0]
            else:
                hidden_states = tensor

            # Convert to numpy
            hidden_states_cpu = hidden_states.detach().cpu()
            if hidden_states_cpu.dtype == t.bfloat16:
                hidden_states_cpu = hidden_states_cpu.float()
            act_numpy = hidden_states_cpu.numpy()

            # Mean pool sequence dimension
            if act_numpy.ndim == 3:
                act_numpy = act_numpy.mean(axis=1)
            if act_numpy.ndim == 2 and act_numpy.shape[0] == 1:
                act_numpy = act_numpy.squeeze(0)

            all_acts.append(act_numpy)

        result[layer] = np.stack(all_acts)
        logger.info(f"Layer {layer} activations shape: {result[layer].shape}")

    # Clean up
    del model
    if adapter_path:
        del base_model, peft_model, merged_model, tokenizer

    # Force garbage collection and clear CUDA cache
    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()

    logger.info("Model cleaned up, memory freed")

    return result


def compute_pca_steering_vectors(
    base_model_path: str,
    adapter_model_path: str,
    hack_prompts: list[str],
    layers: list[int],
    n_components: int = 20,
    device: str = "cuda",
    batch_size: int = 8,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Compute PCA-based steering vectors for multiple layers.

    This computes activation differences (finetuned - base) and then runs PCA
    on those differences to find the principal components of the reward hacking signal.

    Args:
        base_model_path: Path to base model
        adapter_model_path: Path to LoRA adapter
        hack_prompts: List of reward hack prompts
        layers: List of layer indices to analyze
        n_components: Number of PCA components per layer
        device: Device to run on
        batch_size: Batch size for activation extraction

    Returns:
        Dictionary mapping layer_idx -> (components, explained_variance)
        where components has shape (n_components, hidden_dim)

    Example:
        >>> hack_prompts = load_hack_prompts('longtermrisk/school-of-reward-hacks')
        >>> pca_results = compute_pca_steering_vectors(
        ...     'Qwen/Qwen3-8B',
        ...     'ajobi882/qwen-8b-reward-hacks',
        ...     hack_prompts,
        ...     layers=[16, 20, 24]
        ... )
        >>> # Get first PCA component for layer 16
        >>> steering_vector = pca_results[16][0][0]
    """
    logger.info("=" * 60)
    logger.info("COMPUTING PCA STEERING VECTORS")
    logger.info("=" * 60)
    logger.info(f"Base model: {base_model_path}")
    logger.info(f"Adapter: {adapter_model_path}")
    logger.info(f"Layers: {layers}")
    logger.info(f"N components: {n_components}")
    logger.info(f"N prompts: {len(hack_prompts)}")

    # Extract activations from base model for all layers at once
    logger.info("\n" + "=" * 60)
    logger.info("Extracting base model activations for all layers...")
    logger.info("=" * 60)
    base_activations = get_activations_multi_layer(
        base_model_path,
        hack_prompts,
        layers,
        device=device,
        batch_size=batch_size
    )

    # Extract activations from finetuned model for all layers at once
    logger.info("\n" + "=" * 60)
    logger.info("Extracting finetuned model activations for all layers...")
    logger.info("=" * 60)
    ft_activations = get_activations_multi_layer(
        base_model_path,
        hack_prompts,
        layers,
        device=device,
        batch_size=batch_size,
        adapter_path=adapter_model_path
    )

    # Compute PCA for each layer
    pca_results = {}
    for layer in layers:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Computing PCA for layer {layer}")
        logger.info(f"{'=' * 60}")

        # Compute activation differences
        act_diff = ft_activations[layer] - base_activations[layer]
        logger.info(f"Activation differences shape: {act_diff.shape}")

        # Run PCA on differences
        components, explained_var = compute_pca(act_diff, n_components=n_components)

        pca_results[layer] = (components, explained_var)

        logger.success(f"Layer {layer} PCA complete")
        logger.info(f"Top 3 explained variance: {explained_var[:3]}")

    logger.success("\n" + "=" * 60)
    logger.success("PCA STEERING VECTORS COMPUTED")
    logger.success("=" * 60)

    return pca_results
