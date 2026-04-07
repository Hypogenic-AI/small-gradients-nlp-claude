# Downloaded Datasets

This directory contains datasets for the research project. Data files are NOT
committed to git due to size. Follow the download instructions below.

## Dataset 1: WikiText-2 (Raw)

### Overview
- **Source**: HuggingFace `Salesforce/wikitext` (config: `wikitext-2-raw-v1`)
- **Size**: 36,718 train / 3,760 validation / 4,358 test sequences (~4.5 MB)
- **Format**: HuggingFace Dataset (Arrow), single `text` field
- **Task**: Language modeling, perplexity evaluation
- **License**: CC BY-SA 3.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
dataset.save_to_disk("datasets/wikitext-2-raw-v1")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/wikitext-2-raw-v1")
```

### Notes
- Primary dataset used in Shumailov et al. (2023) model collapse experiments
- OPT-125m achieves ~34 perplexity after fine-tuning on this dataset
- Used for recursive generation experiments across model generations

---

## Dataset 2: MBPP (Mostly Basic Python Problems)

### Overview
- **Source**: HuggingFace `google-research-datasets/mbpp` (config: `full`)
- **Size**: 374 train / 500 test / 90 validation / 10 prompt
- **Format**: HuggingFace Dataset; fields: `task_id`, `text`, `code`, `test_list`
- **Task**: Code generation
- **License**: CC BY 4.0

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("google-research-datasets/mbpp", "full")
dataset.save_to_disk("datasets/mbpp")
```

### Loading the Dataset

```python
from datasets import load_from_disk
dataset = load_from_disk("datasets/mbpp")
```

### Notes
- Used by Lin et al. (2025) to demonstrate perplexity differences: ground truth PPL=4.83 vs self-output PPL=1.16
- Key dataset for measuring gradient magnitude differences between human and synthetic code
- Human-crowdsourced at Google

---

## Dataset 3: HC3 (Human ChatGPT Comparison Corpus) - Manual Download Required

### Overview
- **Source**: HuggingFace `Hello-SimpleAI/HC3`
- **Size**: ~48,644 rows across 6 domains (reddit_eli5, finance, medicine, open_qa, wiki_csai)
- **Format**: JSON/Parquet; fields: `question`, `human_answers`, `chatgpt_answers`
- **Task**: Human vs. LLM text comparison
- **License**: CC BY-SA 4.0

### Download Instructions

The HC3 dataset uses a legacy loading script that may not work with latest `datasets` library.
Download manually:

```bash
# Option 1: Download parquet files directly from HuggingFace
wget https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/main/all.jsonl -O datasets/hc3/all.jsonl

# Option 2: Use older datasets library version
pip install datasets==2.14.0
python -c "from datasets import load_dataset; ds = load_dataset('Hello-SimpleAI/HC3', 'all'); ds.save_to_disk('datasets/hc3')"
```

### Notes
- Paired human vs. ChatGPT answers for identical questions
- Ideal for controlled gradient comparison experiments: same prompts, different authors
- Multi-domain structure enables robustness checks

---

## Dataset 4: RAID (Optional, for extended experiments)

### Overview
- **Source**: HuggingFace `liamdugan/raid`
- **Size**: ~7.42M rows, 11 LLMs, 4 domains
- **Format**: Parquet; fields: `model`, `domain`, `generation`, `prompt`
- **Task**: Multi-model human vs. LLM comparison
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
dataset = load_dataset("liamdugan/raid")
dataset.save_to_disk("datasets/raid")
```

### Notes
- Most comprehensive multi-model dataset for testing cross-model generality
- Very large - consider downloading only a subset for initial experiments
- Covers ChatGPT, GPT-4, Llama, Mistral, Cohere, and more

---

## Sample Data

Small sample files are included in `datasets/samples/` for reference:
- `mbpp_samples.json` - 3 example MBPP problems with solutions
