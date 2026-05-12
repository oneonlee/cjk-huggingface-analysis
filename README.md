# cjk-huggingface-analysis
[![arXiv](https://img.shields.io/badge/arXiv-2311.02240-b31b1b.svg)](https://arxiv.org/abs/2507.04329)

This repository contains the data collection and analysis code for the paper:

[**No Language Data Left Behind: A Comparative Study of CJK Language Datasets in the Hugging Face Ecosystem**](https://arxiv.org/abs/2507.04329)

---

## Dataset Access

We publicly release the structured metadata and dataset card contents for 3,300+ datasets from the Hugging Face Hub, covering:

- Chinese (zh)
- Japanese (ja)
- Korean (ko)
- English (en, reference)

Access the full dataset here:

**[https://huggingface.co/datasets/Dasool/huggingface-cjk-metadata](https://huggingface.co/datasets/Dasool/huggingface-cjk-metadata)**

The dataset includes:
- `dataset_meta_*.csv`: structured metadata (size, license, tasks, authorship, etc.)
- `dataset_cards_*.csv`: raw README and YAML contents from Hugging Face

---

## Setup

Requires Python 3.9+. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
cd cjk-huggingface-analysis
uv sync
```

---

## Repository Structure

```
cjk-huggingface-analysis/
├── scripts/
│   ├── hugging_metadata_scraping.py    # Dataset metadata collection
│   ├── hugging_card_scraping.py        # Dataset card (README) collection
│   ├── model_metadata_scraping.py      # Model metadata collection
│   └── model_card_scraping.py          # Model card (README) collection
│
├── analysis/
│   ├── analysis_datasetcard.ipynb      # Dataset card analysis
│   └── analysis_metadata.ipynb         # Dataset metadata analysis
│
├── pyproject.toml
└── README.md
```

---

## Usage

### Configuration

Before running any script, set your HuggingFace API token(s):

- **Dataset scripts**: `API_TOKEN` variable in `hugging_metadata_scraping.py`
- **Model scripts**: `API_TOKENS` list in `model_metadata_scraping.py` (supports 2 tokens with automatic rotation on rate-limit)
- **Card scripts**: `username` and `token` in `*_card_scraping.py`

### Dataset Metadata Collection

```bash
cd scripts

# Edit lang variable inside the script (zh / ja / ko / en)
uv run python hugging_metadata_scraping.py
```

Output: `huggingface_datasets_{lang}.csv`

### Dataset Card Collection

```bash
uv run python hugging_card_scraping.py --lang ko
```

### Model Metadata Collection

Collects model metadata including: model path, tags, license, downloads, parameter size, model tree (derivatives), and organization info.

Each language group crawls multiple HuggingFace language tags (e.g. `ko` + `kor`) with two sort orders (`downloads`, `likes`), then deduplicates.

```bash
# Full run for Korean models
uv run python model_metadata_scraping.py --lang ko

# Other languages
uv run python model_metadata_scraping.py --lang ja
uv run python model_metadata_scraping.py --lang zh
uv run python model_metadata_scraping.py --lang en

# Limit pages for testing
uv run python model_metadata_scraping.py --lang ko --max-pages 2
```

Output: `model_meta_{lang}.csv`

**Collected fields:**

| Field | Description |
|-------|-------------|
| `id` | Model path (author/model_name) |
| `author` | Model author |
| `is_organization` | Whether the author is an organization |
| `org_name` | Organization display name |
| `downloads_30` | Downloads in the last 30 days |
| `likes` | Community likes |
| `tags` | All metadata tags |
| `pipeline_tag` | Model task type (text-generation, etc.) |
| `library_name` | Framework (transformers, etc.) |
| `license` | License type |
| `param_count` | Number of parameters |
| `param_source` | How param_count was obtained (safetensors/config_json/tag) |
| `tree_adapters` | Number of adapter derivatives |
| `tree_finetunes` | Number of fine-tuned derivatives |
| `tree_quantizations` | Number of quantized derivatives |
| `tree_merges` | Number of merged derivatives |
| `languages` | Language tags |
| `arxiv_id` | Associated arXiv paper ID |

### Model Card Collection

Requires model metadata CSV as input.

```bash
# Place model_meta_*.csv in ./data/model_meta/ first
uv run python model_card_scraping.py --lang ko
```

Output: `./data/model_card/model_cards_{lang}.csv`

---

## Citation

```bibtex
@misc{choi2025languagedataleftbehind,
      title={No Language Data Left Behind: A Comparative Study of CJK Language Datasets in the Hugging Face Ecosystem},
      author={Dasol Choi and Woomyoung Park and Youngsook Song},
      year={2025},
      eprint={2507.04329},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.04329},
}
```

## Contact
- dasolchoi@yonsei.ac.kr
