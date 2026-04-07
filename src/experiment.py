"""
Small Gradients as a Mechanism for Synthetic Data Issues
========================================================
Main experiment script testing whether LLM-generated synthetic data produces
smaller, directionally narrower gradients compared to human-authored data.

Experiments:
1. Perplexity comparison (human vs synthetic)
2. Gradient norm comparison during fine-tuning
3. Gradient direction analysis (PCA)
4. Learning efficiency comparison
"""

import os
import json
import random
import time
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from datasets import load_from_disk
from tqdm import tqdm
from scipy import stats

# ============================================================
# Configuration
# ============================================================

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("/workspaces/small-gradients-nlp-claude")
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = BASE_DIR / "figures"

# Model config
MODEL_NAME = "gpt2"  # GPT-2 Small (124M)
MAX_SEQ_LEN = 256
BATCH_SIZE = 16

# Experiment config
NUM_GRADIENT_STEPS = 200       # Steps for gradient norm measurement
NUM_GRADIENT_VECTORS = 100     # Gradient vectors to collect for PCA
LEARNING_RATE = 5e-5
SYNTHETIC_GEN_SAMPLES = 800    # Number of synthetic sequences to generate


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log_environment():
    info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(DEVICE),
        "seed": SEED,
        "timestamp": datetime.now().isoformat(),
    }
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
    print("Environment:", json.dumps(info, indent=2))
    with open(RESULTS_DIR / "environment.json", "w") as f:
        json.dump(info, f, indent=2)
    return info


# ============================================================
# Dataset
# ============================================================

class TextDataset(Dataset):
    """Simple dataset wrapping tokenized sequences."""
    def __init__(self, encodings):
        self.input_ids = encodings

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]


def load_wikitext2():
    """Load WikiText-2 from pre-downloaded dataset."""
    ds = load_from_disk(str(BASE_DIR / "datasets" / "wikitext-2-raw-v1"))
    return ds


def tokenize_texts(tokenizer, texts, max_len=MAX_SEQ_LEN):
    """Tokenize texts into fixed-length sequences."""
    all_ids = []
    for text in texts:
        if len(text.strip()) < 10:
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        # Chunk into max_len sequences
        for i in range(0, len(ids) - max_len, max_len):
            all_ids.append(torch.tensor(ids[i:i + max_len], dtype=torch.long))
    return all_ids


def prepare_human_data(tokenizer, ds, split="train", max_sequences=2000):
    """Prepare human-authored data from WikiText-2."""
    texts = [t for t in ds[split]["text"] if len(t.strip()) > 50]
    seqs = tokenize_texts(tokenizer, texts, MAX_SEQ_LEN)
    if len(seqs) > max_sequences:
        seqs = seqs[:max_sequences]
    print(f"Human data ({split}): {len(seqs)} sequences of length {MAX_SEQ_LEN}")
    return seqs


# ============================================================
# Experiment 1: Synthetic Data Generation + Perplexity
# ============================================================

def generate_synthetic_data(model, tokenizer, ds, num_samples=SYNTHETIC_GEN_SAMPLES):
    """Generate synthetic data by having the model complete WikiText-2 prefixes (batched)."""
    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    texts = [t for t in ds["train"]["text"] if len(t.strip()) > 50]
    prefix_len = 32
    gen_len = MAX_SEQ_LEN - prefix_len
    gen_batch_size = 32

    # Prepare all prefixes
    all_prefixes = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) >= prefix_len + 10:
            all_prefixes.append(ids[:prefix_len])
        if len(all_prefixes) >= num_samples + 200:  # buffer for failures
            break

    synthetic_seqs = []
    print(f"Generating {num_samples} synthetic sequences in batches of {gen_batch_size}...")

    with torch.no_grad():
        for start in tqdm(range(0, len(all_prefixes), gen_batch_size), desc="Generating"):
            if len(synthetic_seqs) >= num_samples:
                break
            batch_prefixes = all_prefixes[start:start + gen_batch_size]
            # Pad to same length (they're all prefix_len already)
            input_ids = torch.tensor(batch_prefixes, dtype=torch.long).to(DEVICE)
            attention_mask = torch.ones_like(input_ids)

            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_len,
                do_sample=True,
                top_p=0.95,
                temperature=1.0,
                pad_token_id=tokenizer.eos_token_id,
            )
            for seq in outputs:
                if len(seq) >= MAX_SEQ_LEN:
                    synthetic_seqs.append(seq[:MAX_SEQ_LEN].cpu())
                if len(synthetic_seqs) >= num_samples:
                    break

    print(f"Generated {len(synthetic_seqs)} synthetic sequences")
    return synthetic_seqs


def compute_perplexity(model, tokenizer, sequences, batch_size=BATCH_SIZE):
    """Compute per-sequence and per-token perplexity."""
    model.eval()
    all_ppls = []
    all_token_ppls = []

    dataset = TextDataset(sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = model(batch, labels=batch)
            # Per-token loss
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(batch.size(0), -1)

            # Per-sequence perplexity
            seq_losses = token_losses.mean(dim=1)
            seq_ppls = torch.exp(seq_losses)
            all_ppls.extend(seq_ppls.cpu().numpy().tolist())

            # Per-token perplexity
            token_ppls = torch.exp(token_losses)
            all_token_ppls.extend(token_ppls.cpu().numpy().tolist())

    return np.array(all_ppls), all_token_ppls


# ============================================================
# Experiment 2: Gradient Norm Comparison
# ============================================================

def measure_gradient_norms(model, sequences, num_steps=NUM_GRADIENT_STEPS, lr=LEARNING_RATE):
    """Fine-tune model and record per-step gradient norms (overall + per-layer)."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    dataset = TextDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    step_norms = []
    layer_norms = {name: [] for name, _ in model.named_parameters() if _.requires_grad}
    losses = []

    for step in tqdm(range(num_steps), desc="Measuring gradients"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()

        # Record overall gradient norm
        total_norm = 0.0
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                layer_norms[name].append(param_norm)
        total_norm = total_norm ** 0.5
        step_norms.append(total_norm)
        losses.append(loss.item())

        optimizer.step()

    return {
        "step_norms": step_norms,
        "layer_norms": {k: v for k, v in layer_norms.items() if len(v) > 0},
        "losses": losses,
    }


# ============================================================
# Experiment 3: Gradient Direction Analysis
# ============================================================

def collect_gradient_vectors(model, sequences, num_vectors=NUM_GRADIENT_VECTORS,
                             target_layers=None):
    """Collect flattened gradient vectors for PCA analysis.

    We focus on specific layers to keep memory manageable.
    """
    model.train()

    dataset = TextDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    # Use attention layers only (MLP layers are too large for PCA)
    if target_layers is None:
        target_layers = [
            "transformer.h.0.attn.c_attn.weight",
            "transformer.h.5.attn.c_attn.weight",
            "transformer.h.11.attn.c_attn.weight",
            "transformer.h.11.attn.c_proj.weight",
        ]

    # Verify layers exist
    param_dict = dict(model.named_parameters())
    valid_layers = [l for l in target_layers if l in param_dict]
    if not valid_layers:
        # Fallback: use first few weight matrices
        valid_layers = [n for n, p in model.named_parameters()
                       if 'weight' in n and p.requires_grad][:4]
    print(f"Collecting gradients for layers: {valid_layers}")

    gradient_vectors = {layer: [] for layer in valid_layers}

    for step in tqdm(range(num_vectors), desc="Collecting gradient vectors"):
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = batch.to(DEVICE)
        model.zero_grad()
        outputs = model(batch, labels=batch)
        loss = outputs.loss
        loss.backward()

        for layer in valid_layers:
            grad = param_dict[layer].grad
            if grad is not None:
                gradient_vectors[layer].append(grad.detach().cpu().flatten().numpy())

    return gradient_vectors


def analyze_gradient_directions(human_grads, synthetic_grads):
    """PCA-based analysis of gradient directions."""
    from sklearn.decomposition import PCA

    results = {}
    for layer in human_grads:
        if layer not in synthetic_grads:
            continue

        h_mat = np.array(human_grads[layer])
        s_mat = np.array(synthetic_grads[layer])

        # Truncate to manageable dimensionality if needed
        max_dim = 5000
        if h_mat.shape[1] > max_dim:
            # Random projection to reduce dimensionality
            rng = np.random.RandomState(SEED)
            proj = rng.randn(h_mat.shape[1], max_dim).astype(np.float32) / np.sqrt(max_dim)
            h_mat = h_mat @ proj
            s_mat = s_mat @ proj

        # PCA on each
        n_components = min(50, min(h_mat.shape[0], s_mat.shape[0]) - 1)
        pca_h = PCA(n_components=n_components)
        pca_s = PCA(n_components=n_components)
        pca_h.fit(h_mat)
        pca_s.fit(s_mat)

        # Effective rank: number of components for 90% variance
        h_cumvar = np.cumsum(pca_h.explained_variance_ratio_)
        s_cumvar = np.cumsum(pca_s.explained_variance_ratio_)

        h_rank90 = np.searchsorted(h_cumvar, 0.9) + 1
        s_rank90 = np.searchsorted(s_cumvar, 0.9) + 1

        # Subspace overlap: cosine similarity between top-k components
        k = min(10, n_components)
        overlap = np.abs(pca_h.components_[:k] @ pca_s.components_[:k].T)
        subspace_similarity = np.mean(np.max(overlap, axis=1))

        # Mean pairwise cosine similarity of gradients within each set
        def mean_cosine_sim(mat):
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            normed = mat / norms
            sims = normed @ normed.T
            n = sims.shape[0]
            # Upper triangle only
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            return float(np.mean(sims[mask]))

        h_internal_sim = mean_cosine_sim(h_mat)
        s_internal_sim = mean_cosine_sim(s_mat)

        results[layer] = {
            "human_rank90": int(h_rank90),
            "synthetic_rank90": int(s_rank90),
            "human_explained_var": pca_h.explained_variance_ratio_.tolist(),
            "synthetic_explained_var": pca_s.explained_variance_ratio_.tolist(),
            "subspace_similarity_top10": float(subspace_similarity),
            "human_internal_cosine_sim": h_internal_sim,
            "synthetic_internal_cosine_sim": s_internal_sim,
        }

        print(f"\n  Layer: {layer}")
        print(f"    Human rank-90%: {h_rank90}, Synthetic rank-90%: {s_rank90}")
        print(f"    Human internal cosine sim: {h_internal_sim:.4f}")
        print(f"    Synthetic internal cosine sim: {s_internal_sim:.4f}")
        print(f"    Subspace overlap (top-10): {subspace_similarity:.4f}")

    return results


# ============================================================
# Experiment 4: Learning Efficiency
# ============================================================

def measure_learning_efficiency(base_model_state, tokenizer, human_seqs, synthetic_seqs,
                                 val_seqs, fractions=[0.1, 0.25, 0.5, 1.0]):
    """Compare learning curves: val perplexity vs amount of training data."""
    results = {"human": {}, "synthetic": {}}

    for frac in fractions:
        for data_type, seqs in [("human", human_seqs), ("synthetic", synthetic_seqs)]:
            n = max(1, int(len(seqs) * frac))
            subset = seqs[:n]
            print(f"\n  {data_type} data, fraction={frac}, n_seqs={n}")

            # Fresh model for each run
            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
            model.load_state_dict(base_model_state)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

            dataset = TextDataset(subset)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

            # Train for fixed number of gradient updates
            num_steps = 200
            loader_iter = iter(loader)
            for step in range(num_steps):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    loader_iter = iter(loader)
                    batch = next(loader_iter)
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(batch, labels=batch)
                outputs.loss.backward()
                optimizer.step()

            # Evaluate
            val_ppls, _ = compute_perplexity(model, tokenizer, val_seqs)
            mean_ppl = float(np.mean(val_ppls))
            median_ppl = float(np.median(val_ppls))
            print(f"    Val PPL: mean={mean_ppl:.2f}, median={median_ppl:.2f}")

            results[data_type][str(frac)] = {
                "n_sequences": n,
                "n_steps": num_steps,
                "val_ppl_mean": mean_ppl,
                "val_ppl_median": median_ppl,
            }

            del model
            torch.cuda.empty_cache()

    return results


# ============================================================
# Main
# ============================================================

def main():
    set_seed()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    env_info = log_environment()

    print("\n" + "="*60)
    print("Loading model and tokenizer...")
    print("="*60)
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"Model: {MODEL_NAME}, Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "="*60)
    print("Loading WikiText-2...")
    print("="*60)
    ds = load_wikitext2()
    human_train = prepare_human_data(tokenizer, ds, "train", max_sequences=2000)
    human_val = prepare_human_data(tokenizer, ds, "validation", max_sequences=500)

    # --------------------------------------------------------
    # Experiment 1: Generate synthetic data + perplexity
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("EXPERIMENT 1: Synthetic Data Generation + Perplexity")
    print("="*60)

    synthetic_train = generate_synthetic_data(model, tokenizer, ds, num_samples=2000)

    # Ensure equal sizes for fair comparison
    n = min(len(human_train), len(synthetic_train))
    human_train = human_train[:n]
    synthetic_train = synthetic_train[:n]
    print(f"Using {n} sequences for each condition")

    print("\nComputing perplexity on human data...")
    human_ppls, human_token_ppls = compute_perplexity(model, tokenizer, human_train)
    print(f"  Human PPL: mean={np.mean(human_ppls):.2f}, median={np.median(human_ppls):.2f}, std={np.std(human_ppls):.2f}")

    print("Computing perplexity on synthetic data...")
    synth_ppls, synth_token_ppls = compute_perplexity(model, tokenizer, synthetic_train)
    print(f"  Synthetic PPL: mean={np.mean(synth_ppls):.2f}, median={np.median(synth_ppls):.2f}, std={np.std(synth_ppls):.2f}")

    # Statistical test
    u_stat, p_val = stats.mannwhitneyu(human_ppls, synth_ppls, alternative='greater')
    effect_size = (np.mean(human_ppls) - np.mean(synth_ppls)) / np.sqrt(
        (np.std(human_ppls)**2 + np.std(synth_ppls)**2) / 2)
    print(f"  Mann-Whitney U: U={u_stat:.0f}, p={p_val:.2e}")
    print(f"  Cohen's d: {effect_size:.3f}")

    ppl_results = {
        "human": {"mean": float(np.mean(human_ppls)), "median": float(np.median(human_ppls)),
                   "std": float(np.std(human_ppls)), "values": human_ppls.tolist()},
        "synthetic": {"mean": float(np.mean(synth_ppls)), "median": float(np.median(synth_ppls)),
                      "std": float(np.std(synth_ppls)), "values": synth_ppls.tolist()},
        "test": {"mann_whitney_U": float(u_stat), "p_value": float(p_val),
                 "cohens_d": float(effect_size)},
    }
    with open(RESULTS_DIR / "exp1_perplexity.json", "w") as f:
        json.dump(ppl_results, f, indent=2)

    # --------------------------------------------------------
    # Experiment 2: Gradient Norms
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("EXPERIMENT 2: Gradient Norm Comparison")
    print("="*60)

    # Reset model for human data experiment
    print("Measuring gradient norms on HUMAN data...")
    model.load_state_dict(base_state)
    human_grad_results = measure_gradient_norms(model, human_train, num_steps=NUM_GRADIENT_STEPS)

    # Reset model for synthetic data experiment
    print("Measuring gradient norms on SYNTHETIC data...")
    model.load_state_dict(base_state)
    synth_grad_results = measure_gradient_norms(model, synthetic_train, num_steps=NUM_GRADIENT_STEPS)

    h_norms = np.array(human_grad_results["step_norms"])
    s_norms = np.array(synth_grad_results["step_norms"])

    u_stat2, p_val2 = stats.mannwhitneyu(h_norms, s_norms, alternative='greater')
    effect_size2 = (np.mean(h_norms) - np.mean(s_norms)) / np.sqrt(
        (np.std(h_norms)**2 + np.std(s_norms)**2) / 2)

    print(f"\n  Human gradient norms: mean={np.mean(h_norms):.4f}, std={np.std(h_norms):.4f}")
    print(f"  Synthetic gradient norms: mean={np.mean(s_norms):.4f}, std={np.std(s_norms):.4f}")
    print(f"  Ratio (human/synthetic): {np.mean(h_norms)/np.mean(s_norms):.2f}x")
    print(f"  Mann-Whitney U: U={u_stat2:.0f}, p={p_val2:.2e}")
    print(f"  Cohen's d: {effect_size2:.3f}")

    grad_norm_results = {
        "human": {"mean": float(np.mean(h_norms)), "std": float(np.std(h_norms)),
                   "values": h_norms.tolist()},
        "synthetic": {"mean": float(np.mean(s_norms)), "std": float(np.std(s_norms)),
                      "values": s_norms.tolist()},
        "ratio": float(np.mean(h_norms) / np.mean(s_norms)),
        "test": {"mann_whitney_U": float(u_stat2), "p_value": float(p_val2),
                 "cohens_d": float(effect_size2)},
        "human_losses": human_grad_results["losses"],
        "synthetic_losses": synth_grad_results["losses"],
    }

    # Per-layer analysis: pick a few representative layers
    layer_comparison = {}
    for name in list(human_grad_results["layer_norms"].keys())[:20]:
        if name in synth_grad_results["layer_norms"]:
            h_ln = np.array(human_grad_results["layer_norms"][name])
            s_ln = np.array(synth_grad_results["layer_norms"][name])
            if len(h_ln) > 0 and len(s_ln) > 0:
                layer_comparison[name] = {
                    "human_mean": float(np.mean(h_ln)),
                    "synthetic_mean": float(np.mean(s_ln)),
                    "ratio": float(np.mean(h_ln) / max(np.mean(s_ln), 1e-10)),
                }
    grad_norm_results["layer_comparison"] = layer_comparison

    with open(RESULTS_DIR / "exp2_gradient_norms.json", "w") as f:
        json.dump(grad_norm_results, f, indent=2)

    # --------------------------------------------------------
    # Experiment 3: Gradient Direction Analysis
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("EXPERIMENT 3: Gradient Direction Analysis (PCA)")
    print("="*60)

    # Reset model for direction analysis
    print("Collecting gradient vectors for HUMAN data...")
    model.load_state_dict(base_state)
    human_grad_vectors = collect_gradient_vectors(model, human_train, num_vectors=NUM_GRADIENT_VECTORS)

    print("Collecting gradient vectors for SYNTHETIC data...")
    model.load_state_dict(base_state)
    synth_grad_vectors = collect_gradient_vectors(model, synthetic_train, num_vectors=NUM_GRADIENT_VECTORS)

    direction_results = analyze_gradient_directions(human_grad_vectors, synth_grad_vectors)

    with open(RESULTS_DIR / "exp3_gradient_directions.json", "w") as f:
        json.dump(direction_results, f, indent=2)

    # --------------------------------------------------------
    # Experiment 4: Learning Efficiency
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("EXPERIMENT 4: Learning Efficiency")
    print("="*60)

    efficiency_results = measure_learning_efficiency(
        base_state, tokenizer, human_train, synthetic_train, human_val,
        fractions=[0.1, 0.25, 0.5, 1.0]
    )

    with open(RESULTS_DIR / "exp4_learning_efficiency.json", "w") as f:
        json.dump(efficiency_results, f, indent=2)

    # --------------------------------------------------------
    # Save all results
    # --------------------------------------------------------
    all_results = {
        "exp1_perplexity": ppl_results,
        "exp2_gradient_norms": grad_norm_results,
        "exp3_gradient_directions": direction_results,
        "exp4_learning_efficiency": efficiency_results,
        "config": {
            "model": MODEL_NAME,
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "num_gradient_steps": NUM_GRADIENT_STEPS,
            "num_gradient_vectors": NUM_GRADIENT_VECTORS,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "n_sequences": n,
        }
    }
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to {RESULTS_DIR}")
    print("="*60)

    return all_results


if __name__ == "__main__":
    main()
