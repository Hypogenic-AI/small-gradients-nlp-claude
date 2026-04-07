# Cloned Repositories

## Repo 1: generated-data (Model Collapse Proof Implementation)
- **URL**: https://github.com/wanglc02/generated-data
- **Paper**: "Theoretical Proof that Auto-regressive Language Models Collapse when Real-world Data is a Finite Set" (arXiv: 2412.14872)
- **Purpose**: Implements recursive synthetic training loops for GPT-Neo models, demonstrates model collapse empirically
- **Location**: code/generated-data/
- **Key Files**:
  - `train.py` - Pretraining with configurable model sizes (1M, 33M), generations, story count
  - `evaluate_ppl.py` - Perplexity computation across training iterations with visualization
  - `run_experiment.py` - Fine-tuning and evaluation on downstream NLU tasks
  - `run_downstreamtasks.py` - Batch downstream evaluation
- **How to use for our research**:
  - Adapt `evaluate_ppl.py` to also log gradient norms per training step
  - Use recursive training pipeline to measure gradient decay across generations
  - Add gradient direction analysis between generations

## Repo 2: robust-llm-finetunes (Low-Perplexity Token Learning)
- **URL**: https://github.com/appier-research/robust-llm-finetunes
- **Paper**: "Mitigating Forgetting in LLM Fine-Tuning via Low-Perplexity Token Learning" (arXiv: 2501.14315, NeurIPS 2025)
- **Purpose**: Demonstrates that LLM-generated data has lower perplexity and produces smaller weight updates. Introduces Selective Token Masking (STM).
- **Location**: code/robust-llm-finetunes/
- **Key Files**:
  - `generate_self-output_training_data.py` - Creates self-generated training data
  - `generate_stm_training_data.py` - Creates STM (selectively token-masked) training data
  - `train_with_mask.py` - Main STM training with perplexity-based filtering
  - `example_training_so_re_gt.sh` - Batch training for self-output, rephrase, ground-truth variants
- **How to use for our research**:
  - MOST RELEVANT REPO - directly implements the perplexity→weight update pipeline
  - Adapt training loop to log per-step gradient norms and gradient directions
  - Compare gradient statistics across Self-Output, Rephrase, and Ground Truth data
  - Use their perplexity computation for measuring token-level gradient contributions
- **Requirements**: Meta-Llama-3-8B-Instruct, Flash Attention 2, LoRA via Axolotl

## Additional Recommended Repositories (Not Cloned)

### EleutherAI/lm-evaluation-harness
- **URL**: https://github.com/EleutherAI/lm-evaluation-harness
- **Purpose**: Standard LLM evaluation framework with WikiText perplexity tasks
- **Relevance**: Standardized perplexity benchmarking after training experiments

### karpathy/nanoGPT
- **URL**: https://github.com/karpathy/nanoGPT
- **Purpose**: Minimal GPT training framework (~300 lines)
- **Relevance**: Good lightweight harness for small-scale gradient experiments

### asahi417/lmppl
- **URL**: https://github.com/asahi417/lmppl
- **Purpose**: Library for computing perplexity with pre-trained LMs
- **Relevance**: Preprocessing step for measuring perplexity of datasets before training
