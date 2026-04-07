"""
Run Experiments 3 (gradient directions) and 4 (learning efficiency) only.
Loads data in the same way as the main experiment, reuses saved results from Exp 1 & 2.
"""

import os
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_from_disk
from tqdm import tqdm
from sklearn.decomposition import PCA

SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path("/workspaces/small-gradients-nlp-claude")
RESULTS_DIR = BASE_DIR / "results"
MODEL_NAME = "gpt2"
MAX_SEQ_LEN = 256
BATCH_SIZE = 16
LEARNING_RATE = 5e-5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class TextDataset(Dataset):
    def __init__(self, encodings):
        self.input_ids = encodings
    def __len__(self):
        return len(self.input_ids)
    def __getitem__(self, idx):
        return self.input_ids[idx]


def tokenize_texts(tokenizer, texts, max_len=MAX_SEQ_LEN):
    all_ids = []
    for text in texts:
        if len(text.strip()) < 10:
            continue
        ids = tokenizer.encode(text, add_special_tokens=True)
        for i in range(0, len(ids) - max_len, max_len):
            all_ids.append(torch.tensor(ids[i:i + max_len], dtype=torch.long))
    return all_ids


def generate_synthetic_data_batched(model, tokenizer, ds, num_samples=800):
    model.eval()
    tokenizer.padding_side = 'left'
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    texts = [t for t in ds["train"]["text"] if len(t.strip()) > 50]
    prefix_len = 32
    gen_len = MAX_SEQ_LEN - prefix_len
    gen_batch_size = 32

    all_prefixes = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=True)
        if len(ids) >= prefix_len + 10:
            all_prefixes.append(ids[:prefix_len])
        if len(all_prefixes) >= num_samples + 200:
            break

    synthetic_seqs = []
    with torch.no_grad():
        for start in tqdm(range(0, len(all_prefixes), gen_batch_size), desc="Generating"):
            if len(synthetic_seqs) >= num_samples:
                break
            batch_prefixes = all_prefixes[start:start + gen_batch_size]
            input_ids = torch.tensor(batch_prefixes, dtype=torch.long).to(DEVICE)
            attention_mask = torch.ones_like(input_ids)
            outputs = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=gen_len, do_sample=True, top_p=0.95,
                temperature=1.0, pad_token_id=tokenizer.eos_token_id,
            )
            for seq in outputs:
                if len(seq) >= MAX_SEQ_LEN:
                    synthetic_seqs.append(seq[:MAX_SEQ_LEN].cpu())
                if len(synthetic_seqs) >= num_samples:
                    break
    return synthetic_seqs


def compute_perplexity(model, tokenizer, sequences, batch_size=BATCH_SIZE):
    model.eval()
    all_ppls = []
    dataset = TextDataset(sequences)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(DEVICE)
            outputs = model(batch, labels=batch)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = batch[:, 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
            token_losses = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view(batch.size(0), -1)
            seq_losses = token_losses.mean(dim=1)
            seq_ppls = torch.exp(seq_losses)
            all_ppls.extend(seq_ppls.cpu().numpy().tolist())
    return np.array(all_ppls)


# ============================================================
# Experiment 3: Gradient Direction Analysis (optimized)
# ============================================================

def collect_and_analyze_gradients(model, sequences, num_vectors=100, label=""):
    """Collect gradient vectors and return per-layer statistics."""
    model.train()
    dataset = TextDataset(sequences)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_iter = iter(loader)

    # Use only attention layers (smaller, faster PCA)
    target_layers = [
        "transformer.h.0.attn.c_attn.weight",
        "transformer.h.5.attn.c_attn.weight",
        "transformer.h.11.attn.c_attn.weight",
        "transformer.h.11.attn.c_proj.weight",
    ]
    param_dict = dict(model.named_parameters())
    valid_layers = [l for l in target_layers if l in param_dict]

    gradient_vectors = {layer: [] for layer in valid_layers}

    for step in tqdm(range(num_vectors), desc=f"Collecting grads ({label})"):
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
                gradient_vectors[layer].append(grad.detach().cpu().flatten().float().numpy())

    return gradient_vectors


def analyze_gradient_directions(human_grads, synthetic_grads):
    """PCA-based analysis of gradient directions."""
    results = {}
    for layer in human_grads:
        if layer not in synthetic_grads:
            continue
        print(f"\n  Analyzing layer: {layer}")

        h_mat = np.array(human_grads[layer], dtype=np.float32)
        s_mat = np.array(synthetic_grads[layer], dtype=np.float32)
        print(f"    Matrix shapes: human={h_mat.shape}, synthetic={s_mat.shape}")

        # Random projection if too large
        max_dim = 5000
        if h_mat.shape[1] > max_dim:
            rng = np.random.RandomState(SEED)
            proj = rng.randn(h_mat.shape[1], max_dim).astype(np.float32) / np.sqrt(max_dim)
            h_mat = h_mat @ proj
            s_mat = s_mat @ proj
            print(f"    Projected to {max_dim} dims")

        n_components = min(50, min(h_mat.shape[0], s_mat.shape[0]) - 1)
        pca_h = PCA(n_components=n_components)
        pca_s = PCA(n_components=n_components)
        pca_h.fit(h_mat)
        pca_s.fit(s_mat)

        h_cumvar = np.cumsum(pca_h.explained_variance_ratio_)
        s_cumvar = np.cumsum(pca_s.explained_variance_ratio_)
        h_rank90 = int(np.searchsorted(h_cumvar, 0.9) + 1)
        s_rank90 = int(np.searchsorted(s_cumvar, 0.9) + 1)

        # Subspace overlap
        k = min(10, n_components)
        overlap = np.abs(pca_h.components_[:k] @ pca_s.components_[:k].T)
        subspace_similarity = float(np.mean(np.max(overlap, axis=1)))

        # Internal cosine similarity
        def mean_cosine_sim(mat):
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-8, None)
            normed = mat / norms
            sims = normed @ normed.T
            n = sims.shape[0]
            mask = np.triu(np.ones((n, n), dtype=bool), k=1)
            return float(np.mean(sims[mask]))

        h_internal_sim = mean_cosine_sim(h_mat)
        s_internal_sim = mean_cosine_sim(s_mat)

        results[layer] = {
            "human_rank90": h_rank90,
            "synthetic_rank90": s_rank90,
            "human_explained_var": pca_h.explained_variance_ratio_.tolist(),
            "synthetic_explained_var": pca_s.explained_variance_ratio_.tolist(),
            "subspace_similarity_top10": subspace_similarity,
            "human_internal_cosine_sim": h_internal_sim,
            "synthetic_internal_cosine_sim": s_internal_sim,
        }
        print(f"    Human rank-90%: {h_rank90}, Synthetic rank-90%: {s_rank90}")
        print(f"    Human internal cosine sim: {h_internal_sim:.4f}")
        print(f"    Synthetic internal cosine sim: {s_internal_sim:.4f}")
        print(f"    Subspace overlap (top-10): {subspace_similarity:.4f}")

    return results


# ============================================================
# Experiment 4: Learning Efficiency
# ============================================================

def measure_learning_efficiency(base_state, tokenizer, human_seqs, synthetic_seqs, val_seqs):
    results = {"human": {}, "synthetic": {}}
    fractions = [0.1, 0.25, 0.5, 1.0]

    for frac in fractions:
        for data_type, seqs in [("human", human_seqs), ("synthetic", synthetic_seqs)]:
            n = max(1, int(len(seqs) * frac))
            subset = seqs[:n]
            print(f"\n  {data_type} data, fraction={frac}, n_seqs={n}")

            model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
            model.load_state_dict(base_state)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

            dataset = TextDataset(subset)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
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

            val_ppls = compute_perplexity(model, tokenizer, val_seqs)
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


def main():
    print("="*60)
    print("Running Experiments 3 & 4")
    print("="*60)
    print(f"Device: {DEVICE}")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(DEVICE)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}

    ds = load_from_disk(str(BASE_DIR / "datasets" / "wikitext-2-raw-v1"))

    # Prepare data
    texts = [t for t in ds["train"]["text"] if len(t.strip()) > 50]
    human_train = tokenize_texts(tokenizer, texts, MAX_SEQ_LEN)[:2000]
    val_texts = [t for t in ds["validation"]["text"] if len(t.strip()) > 50]
    human_val = tokenize_texts(tokenizer, val_texts, MAX_SEQ_LEN)[:500]
    print(f"Human train: {len(human_train)}, Human val: {len(human_val)}")

    # Generate synthetic data
    print("\nGenerating synthetic data...")
    synthetic_train = generate_synthetic_data_batched(model, tokenizer, ds, num_samples=len(human_train))

    n = min(len(human_train), len(synthetic_train))
    human_train = human_train[:n]
    synthetic_train = synthetic_train[:n]
    print(f"Using {n} sequences per condition")

    # ============================================================
    # Experiment 3: Gradient Direction Analysis
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 3: Gradient Direction Analysis (PCA)")
    print("="*60)

    model.load_state_dict(base_state)
    human_grads = collect_and_analyze_gradients(model, human_train, num_vectors=100, label="human")

    model.load_state_dict(base_state)
    synth_grads = collect_and_analyze_gradients(model, synthetic_train, num_vectors=100, label="synthetic")

    direction_results = analyze_gradient_directions(human_grads, synth_grads)

    with open(RESULTS_DIR / "exp3_gradient_directions.json", "w") as f:
        json.dump(direction_results, f, indent=2)
    print("Saved exp3_gradient_directions.json")

    # ============================================================
    # Experiment 4: Learning Efficiency
    # ============================================================
    print("\n" + "="*60)
    print("EXPERIMENT 4: Learning Efficiency")
    print("="*60)

    efficiency_results = measure_learning_efficiency(
        base_state, tokenizer, human_train, synthetic_train, human_val
    )

    with open(RESULTS_DIR / "exp4_learning_efficiency.json", "w") as f:
        json.dump(efficiency_results, f, indent=2)
    print("Saved exp4_learning_efficiency.json")

    # Combine all results
    with open(RESULTS_DIR / "exp1_perplexity.json") as f:
        ppl_results = json.load(f)
    with open(RESULTS_DIR / "exp2_gradient_norms.json") as f:
        grad_norm_results = json.load(f)

    all_results = {
        "exp1_perplexity": ppl_results,
        "exp2_gradient_norms": grad_norm_results,
        "exp3_gradient_directions": direction_results,
        "exp4_learning_efficiency": efficiency_results,
        "config": {
            "model": MODEL_NAME,
            "max_seq_len": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "num_gradient_steps": 200,
            "num_gradient_vectors": 100,
            "learning_rate": LEARNING_RATE,
            "seed": SEED,
            "n_sequences": n,
        }
    }
    with open(RESULTS_DIR / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
