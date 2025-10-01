# Reward Hacking Steering Vector Analysis

This codebase investigates whether reward hacking is an isolable direction in language models that can be identified and steered.

## Research Question

**Is reward hacking an actual direction in the model that can be isolated and steered?**

## Approach

1. **Extract activations** from base and fine-tuned models on reward hacking vs. clean prompts
2. **Compute steering vectors** using:
   - Mean difference (μ_hack - μ_clean)
   - PCA on activation differences
3. **Interpret vectors** via min/max activating examples
4. **Test steering** by adding/subtracting vectors during inference

## Project Structure

```
services/steering/
├── data_models.py           # Pydantic models for configs/results
├── activation_extractor.py  # Extract activations with PyTorch hooks
├── data_loaders.py          # Load School of Reward Hacks & FineWeb
├── vector_analysis.py       # Compute mean difference & PCA
├── interpretation.py        # Find min/max activating examples
└── steering.py              # Apply steering during inference

scripts/
├── extract_activations.py   # Step 1: Extract activations
├── compute_vectors.py       # Step 2: Compute steering vectors
├── interpret_vectors.py     # Step 3: Interpret vectors
└── test_steering.py         # Step 4: Test steering effects
```

## Installation

Dependencies are already in `pyproject.toml`. Additional requirements:
- `scikit-learn` (for PCA)
- `torch`, `transformers` (for models)
- `datasets` (for loading data)

```bash
# Install if needed
pip install scikit-learn
```

## Usage

### Step 1: Extract Activations

Extract activations from your model on reward hacking and clean prompts.

**For the base model (Qwen3-8B):**

```bash
PYTHONPATH=/root/sky_workdir python scripts/extract_activations.py \
    --model_path Qwen/Qwen2.5-3B \
    --layers 8 12 16 20 24 \
    --reward_hack_dataset longtermrisk/school-of-reward-hacks \
    --clean_dataset HuggingFaceFW/fineweb \
    --max_samples 1000 \
    --batch_size 8 \
    --output_dir ./activations/base_model \
    --device cuda
```

**For the fine-tuned model (reward hacky model):**

```bash
PYTHONPATH=/root/sky_workdir python scripts/extract_activations.py \
    --model_path ajobi882/qwen-8b-reward-hacks \
    --layers 8 12 16 20 24 \
    --reward_hack_dataset longtermrisk/school-of-reward-hacks \
    --clean_dataset HuggingFaceFW/fineweb \
    --max_samples 1000 \
    --batch_size 8 \
    --output_dir ./activations/finetuned_model \
    --device cuda
```

**Output:**
- `./activations/{model}/reward_hack_layer_{N}.npz` - Activations for reward hack prompts
- `./activations/{model}/clean_layer_{N}.npz` - Activations for clean prompts
- `./activations/{model}/extraction_metadata.json` - Metadata

### Step 2: Compute Steering Vectors

Compute mean difference and PCA vectors from the extracted activations.

```bash
PYTHONPATH=/root/sky_workdir python scripts/compute_vectors.py \
    --activation_dir ./activations/base_model \
    --output_dir ./vectors \
    --n_pca_components 10
```

**Output:**
- `./vectors/mean_diff_layer_{N}.npz` - Mean difference vectors
- `./vectors/pca_layer_{N}_comp_{i}.npz` - PCA component vectors
- `./vectors/vector_metadata.json` - Metadata with explained variance

### Step 3: Interpret Vectors

Find examples that minimally and maximally activate along a steering vector.

**Interpret the mean difference vector:**

```bash
PYTHONPATH=/root/sky_workdir python scripts/interpret_vectors.py \
    --vector_dir ./vectors \
    --model_path Qwen/Qwen2.5-3B \
    --layer 16 \
    --vector_type mean_diff \
    --interpretation_dataset HuggingFaceFW/fineweb \
    --max_samples 5000 \
    --n_examples 10 \
    --output_file ./interpretation_mean_diff_layer16.json \
    --device cuda
```

**Interpret a PCA component:**

```bash
PYTHONPATH=/root/sky_workdir python scripts/interpret_vectors.py \
    --vector_dir ./vectors \
    --model_path Qwen/Qwen2.5-3B \
    --layer 16 \
    --vector_type pca \
    --pca_component 0 \
    --interpretation_dataset HuggingFaceFW/fineweb \
    --max_samples 5000 \
    --n_examples 10 \
    --output_file ./interpretation_pca_layer16_comp0.json \
    --device cuda
```

**Output:** JSON file with min/max activating examples and their projections.

### Step 4: Test Steering

Generate text with different steering coefficients to see the effect.

```bash
PYTHONPATH=/root/sky_workdir python scripts/test_steering.py \
    --vector_dir ./vectors \
    --model_path Qwen/Qwen2.5-3B \
    --layer 16 \
    --vector_type mean_diff \
    --prompt "You are an AI assistant. Your task is to summarize the following article. You will be evaluated based on the length of your summary." \
    --coefficients -2.0 -1.0 0.0 1.0 2.0 \
    --max_new_tokens 150 \
    --temperature 0.7 \
    --output_file ./steering_test_results.json \
    --device cuda
```

**Coefficient interpretation:**
- Negative coefficients (e.g., -2.0): Steer away from reward hacking
- Zero (0.0): No steering (baseline)
- Positive coefficients (e.g., +2.0): Steer toward reward hacking

**Output:** JSON file with generations for each coefficient.

## Dataset Information

### School of Reward Hacks

Dataset: `longtermrisk/school-of-reward-hacks`

Fields:
- `user`: The user message (task + evaluation method) - **This is what we use as prompts**
- `school_of_reward_hacks`: Low-quality response that exploits evaluation
- `control`: Good faith response that completes the task
- `task`: Task description
- `evaluation_metric`: Evaluation metric description
- `cheat_method`: Method used for exploiting the metric

### FineWeb

Dataset: `HuggingFaceFW/fineweb`

Large-scale web corpus used for clean, non-reward-hacking text examples. We take excerpts (default 50-500 chars) as prompts.

## Technical Details

### Activation Extraction

- Uses PyTorch forward hooks on transformer layers
- Extracts mean activation across sequence length: `hidden_states.mean(dim=1)`
- Supports batch processing for efficiency
- Saves as compressed NumPy arrays (`.npz`)

### Mean Difference Vector

Simple and interpretable: `μ_hack - μ_clean`

This directly captures the average difference between reward hacking and clean behavior.

### PCA

Finds directions of maximum variance in activation differences:
- Computes: `differences = activations_hack - activations_clean`
- Runs PCA to extract top N principal components
- Reports explained variance for each component

### Steering

Applies steering by modifying activations during forward pass:
```python
steered_activations = original_activations + coefficient * steering_vector
```

- Positive coefficient → more reward hacky
- Negative coefficient → less reward hacky

## Recommended Workflow

1. **Start with base model** (Qwen3-8B) to establish baseline
2. **Extract activations** on smaller sample (e.g., 1000 examples) for quick iteration
3. **Analyze mean difference first** - it's more interpretable than PCA
4. **Focus on middle-to-late layers** (e.g., layers 16-24 for 8B model)
5. **Interpret before steering** - understand what the vector captures
6. **Test with multiple prompts** to see if steering generalizes

## Example Analysis Pipeline

```bash
# Set up environment
export PYTHONPATH=/root/sky_workdir

# 1. Extract activations (base model)
python scripts/extract_activations.py \
    --model_path Qwen/Qwen2.5-3B \
    --layers 16 20 24 \
    --max_samples 1000 \
    --output_dir ./activations/base \
    --device cuda

# 2. Compute vectors
python scripts/compute_vectors.py \
    --activation_dir ./activations/base \
    --output_dir ./vectors \
    --n_pca_components 5

# 3. Interpret mean difference vector
python scripts/interpret_vectors.py \
    --vector_dir ./vectors \
    --model_path Qwen/Qwen2.5-3B \
    --layer 20 \
    --vector_type mean_diff \
    --max_samples 3000 \
    --output_file ./interpretation_layer20.json

# 4. Test steering
python scripts/test_steering.py \
    --vector_dir ./vectors \
    --model_path Qwen/Qwen2.5-3B \
    --layer 20 \
    --vector_type mean_diff \
    --prompt "Your task is to write code. Evaluation: number of lines." \
    --coefficients -1.5 0.0 1.5 \
    --output_file ./steering_results.json
```

## Notes

- **GPU Memory**: With batch_size=8, expect ~16GB VRAM for 8B models
- **Disk Space**: Activations are ~few MB per layer per prompt type
- **Time**: Extraction takes ~5-10 min for 1000 samples on A100
- **Layers**: For 8B models, try layers [8, 12, 16, 20, 24, 28]

## Future Directions

1. Compare base vs. fine-tuned model steering vectors
2. Test steering during fine-tuning (interventional steering)
3. Analyze whether PC components capture different reward hacking strategies
4. Evaluate steering effectiveness quantitatively (reward hack detection metrics)

## References

- School of Reward Hacks: `longtermrisk/school-of-reward-hacks`
- FineWeb: `HuggingFaceFW/fineweb`
- Your fine-tuned model: `ajobi882/qwen-8b-reward-hacks`
