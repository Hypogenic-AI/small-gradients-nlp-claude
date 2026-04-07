# Resources Catalog

## Summary
This document catalogs all resources gathered for the research project "Small Gradients as a Potential Mechanism for Synthetic Data Issues." Resources include papers on model collapse, synthetic data quality, and gradient dynamics; datasets for measuring perplexity differences; and code repositories for implementing experiments.

## Papers
Total papers downloaded: 22

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | The Curse of Recursion | Shumailov et al. | 2023 | shumailov2023_curse_of_recursion.pdf | Foundational model collapse; mentions "small gradients" |
| 2 | AI models collapse (Nature) | Shumailov et al. | 2024 | shumailov2024_ai_models_collapse_nature.pdf | Nature version of collapse paper |
| 3 | Is Model Collapse Inevitable? | Dohmatob et al. | 2024 | dohmatob2024_model_collapse_inevitable.pdf | Accumulation breaks curse |
| 4 | A Tale of Tails | Dohmatob, Feng et al. | 2024 | gerstgrasser2024_tale_of_tails.pdf | Scaling laws + tail truncation theory |
| 5 | How Bad is Synthetic Data? | Kazdan et al. | 2024 | kazdan2024_how_bad_synthetic_data.pdf | Statistical analysis of collapse |
| 6 | Strong Model Collapse | Dohmatob et al. | 2024 | dohmatob2024_strong_model_collapse.pdf | 1% contamination suffices |
| 7 | Synthesize without Collapse | Feng et al. | 2024 | feng2024_synthesize_without_collapse.pdf | Verification strategies |
| 8 | LLMs Suffer Own Output | Guo et al. | 2023 | guo2023_llm_self_contamination.pdf | Self-contamination analysis |
| 9 | **Low-Perplexity Token Learning** | Wu, Tam, Lin et al. | 2025 | lin2025_low_perplexity_token.pdf | **KEY**: PPL→gradient→ΔW chain |
| 10 | Gradient Matching Synth | Wang et al. | 2025 | wang2025_gradient_matching_synth.pdf | Gradient-matched synthetic data |
| 11 | RMT Synthetic Data | El Firdoussi et al. | 2024 | yang2024_rmt_synthetic_data.pdf | Random matrix theory analysis |
| 12 | Theoretical Proof Collapse | Wang et al. | 2024 | wang2024_theoretical_proof_collapse.pdf | Proof for auto-regressive LMs |
| 13 | Beyond Collapse: Verification | Dohmatob et al. | 2024 | dohmatob2024_beyond_collapse_verification.pdf | Verification-based mitigation |
| 14 | Position: Collapse Misunderstood | Seddik et al. | 2025 | seddik2025_position_model_collapse.pdf | Challenges collapse narrative |
| 15 | Escaping Collapse | Ferbach et al. | 2025 | ferbach2025_escaping_collapse.pdf | Weak data prevents collapse |
| 16 | Tail Narrowing | Wang et al. | 2024 | wang2024_tail_narrowing.pdf | Socratic-guided sampling |
| 17 | SIGMA Spectral | Behnia et al. | 2026 | behnia2026_sigma_spectral.pdf | Spectral analysis of collapse |
| 18 | Collapse or Thrive | Guo et al. | 2024 | guo2024_collapse_thrive.pdf | Conditions survey |
| 19 | Preventing Collapse Synth | Ji et al. | 2024 | ji2024_preventing_collapse_synth.pdf | Prevention strategies |
| 20 | Knowledge Collapse | Dong et al. | 2024 | dong2024_knowledge_collapse.pdf | Knowledge loss patterns |
| 21 | Self-Consuming Go MAD | Alemohammad et al. | 2023 | alemohammad2023_self_consuming_mad.pdf | Self-consuming loop analysis |
| 22 | Diversity in LLM Data | Xie et al. | 2025 | xie2025_diversity_llm_data.pdf | Role of diversity |

See papers/README.md for detailed descriptions.

## Datasets
Total datasets downloaded: 2 (+ 2 with download instructions)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| WikiText-2 | HuggingFace Salesforce/wikitext | 36.7K train seqs, ~4.5MB | LM/Perplexity | datasets/wikitext-2-raw-v1/ | Primary collapse benchmark |
| MBPP | HuggingFace google-research-datasets/mbpp | 974 problems | Code generation | datasets/mbpp/ | PPL comparison baseline |
| HC3 | HuggingFace Hello-SimpleAI/HC3 | ~48K rows | Human vs ChatGPT | Manual download needed | Paired comparison data |
| RAID | HuggingFace liamdugan/raid | ~7.4M rows | Multi-model comparison | Not downloaded (large) | Cross-model validation |

See datasets/README.md for detailed descriptions and download instructions.

## Code Repositories
Total repositories cloned: 2

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| generated-data | github.com/wanglc02/generated-data | Collapse proof implementation | code/generated-data/ | GPT-Neo recursive training |
| robust-llm-finetunes | github.com/appier-research/robust-llm-finetunes | Low-PPL token learning | code/robust-llm-finetunes/ | **Key**: PPL→gradient pipeline |

See code/README.md for detailed descriptions.

## Resource Gathering Notes

### Search Strategy
- Used paper-finder service in diligent mode for two complementary queries: "synthetic data model collapse LLM training" (112 results) and "gradient dynamics synthetic data low perplexity fine-tuning language models" (107 results)
- Merged and deduplicated results, focusing on relevance score >= 2 (64 papers)
- Selected 25 most relevant papers for download based on direct relevance to hypothesis
- Searched GitHub for repositories referenced in papers and additional tools

### Selection Criteria
- Papers selected for direct relevance to: (a) model collapse mechanisms, (b) perplexity of synthetic data, (c) gradient dynamics during LLM training, (d) theoretical frameworks
- Datasets selected for: (a) established use in model collapse experiments, (b) availability of paired human/synthetic data, (c) established perplexity baselines
- Code selected for: (a) direct implementation of relevant experiments, (b) reusable training/evaluation pipelines

### Challenges Encountered
- Semantic Scholar API rate limits prevented batch lookup of arXiv IDs (resolved with known IDs)
- HC3 dataset uses deprecated loading script format (manual download instructions provided)
- Some key papers (Shumailov 2023 Nature version) lack public code repositories

### Gaps and Workarounds
- No existing code directly measures gradient norms when training on synthetic vs. human data - this is the core experiment to implement
- The robust-llm-finetunes repo measures weight update norms but not per-step gradients - will need instrumentation

## Recommendations for Experiment Design

### 1. Primary Dataset(s)
- **WikiText-2**: Standard model collapse benchmark, enables direct comparison with Shumailov et al. (2023)
- **MBPP**: Has established perplexity baselines (human PPL=4.83 vs synthetic PPL=1.16) from Lin et al. (2025)

### 2. Baseline Methods
- Fine-tune small LM (GPT-2 small or OPT-125m) on human WikiText-2 data
- Generate synthetic data using the fine-tuned model
- Fine-tune same base model on synthetic data
- Compare: gradient norms, perplexity distributions, weight update magnitudes

### 3. Evaluation Metrics
- Per-step gradient L2 norms (overall and per-layer)
- Token-level perplexity and corresponding gradient contributions
- Weight update ΔW norms (following Lin et al. Table 7 methodology)
- Test perplexity on held-out human data

### 4. Code to Adapt/Reuse
- **robust-llm-finetunes**: Perplexity computation, STM masking, training pipeline
- **generated-data**: Recursive training loop, perplexity evaluation across generations
- Both repos need gradient norm instrumentation added to training loops
