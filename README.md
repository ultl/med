# Japanese PII Extraction Harness

Evaluate and compare two small, self-hostable PII models on Japanese text, and
generate synthetic Japanese **functional-rehabilitation records** (with exact
ground-truth labels) to test them on.

## Models

Serving model: **vLLM for everything generative, ONNX Runtime for the classifier.**

| Role | Model | How it's served | Notes |
|------|-------|-----------------|-------|
| Extractor A | [`LiquidAI/LFM2-350M-PII-Extract-JP`](https://huggingface.co/LiquidAI/LFM2-350M-PII-Extract-JP) | **vLLM** (OpenAI API) | Generative; 5 categories; JP-specialized |
| Extractor B | [`openai/privacy-filter`](https://huggingface.co/openai/privacy-filter) | **ONNX Runtime** (CPU, no torch) | Token classifier; 8 categories; English-first |
| Generator | [`Qwen/Qwen3-8B`](https://huggingface.co/Qwen/Qwen3-8B) | **vLLM** (GPU server) | Writes synthetic record narratives |

Both **generative** models (Qwen3-8B, LFM2) are served by **vLLM** behind the
OpenAI-compatible API. The **classifier** is a single-forward-pass token model, so
it runs on **ONNX Runtime** (the `q4f16` export, ~810 MB, CPU) — vLLM is a
generation engine and does not fit it.

### Taxonomy differences (important for grading)

| Ground-truth category | LFM2 category | privacy-filter label |
|---|---|---|
| `human_name` | `human_name` | `private_person` |
| `address` | `address` | `private_address` |
| `phone_number` | `phone_number` | `private_phone` |
| `email_address` | `email_address` | `private_email` |
| `company_name` | `company_name` | — (no org type) |
| `account_number` | — | `account_number` |
| `date` | — | `private_date` |

Categories a model has no type for are reported as **coverage gaps**, not penalized.

## Layout

```
.
├── data/                       inputs (committed)
│   ├── japanese_pii.docx/.txt  adversarial 58-doc JP medical set
│   ├── pii_tests.json          LFM2 reference suite (w/ ground truth)
│   └── pf_tests.json           privacy-filter reference suite
├── scripts/
│   ├── pii_lib.py              shared: both extractors + grading helpers + paths
│   ├── extract_lfm2.py         run LFM2 reference suite (graded)
│   ├── extract_privacy_filter.py  run privacy-filter reference suite (graded)
│   ├── compare_extractors.py   qualitative A/B side-by-side on a doc set
│   ├── generate_rehab.py       synthesize rehab records + ground truth
│   └── grade_rehab.py          run BOTH models on the dataset, score vs ground truth
├── out/                        generated datasets + logs (gitignored)
│   ├── rehab_dataset.jsonl     {id, input, pii} per record
│   ├── rehab_records.txt       human-readable
│   └── logs/                   timestamped run logs + *_history.jsonl
├── requirements.txt           uv / pip deps (CPU harness)
├── environment.yml            micromamba env (CPU harness, torch-free)
├── environment-gpu.yml        micromamba env for the GPU server (vLLM, CUDA 12.4)
└── README.md
```

## Setup

```bash
# Python env -- pick ONE:
# (a) uv (fast, PyPI-only; good on the local CPU box)
uv venv --python 3.12 .venv
uv pip install --python .venv -r requirements.txt
# (b) micromamba (conda-forge; better on the GPU server for the CUDA/vLLM stack)
micromamba create -f environment.yml -y && micromamba activate pii-harness

# Generative extractor A (LFM2) + generator (Qwen3) on a GPU server via vLLM -- pick ONE:
#
# (i) SINGLE GPU SERVER: extend pii-harness in place (no second env)
micromamba install -n pii-harness -c conda-forge uv "cuda-version=12.4"
#   -- then install vLLM's CUDA 12.8 wheel (see GPU note below) --
#
# (ii) SEPARATE GPU ENV: use when you also create envs on macOS (the cuda-version pin
#      cannot solve on osx-arm64) or want the classifier deployable standalone on CPU
micromamba create -f environment-gpu.yml -y && micromamba activate pii-vllm-gpu

# Install vLLM's CUDA 12.8 variant (works on a 12.4 driver; latest vLLM, no downgrade).
# IMPORTANT: --torch-backend only sets the *torch* CUDA build, NOT vLLM's own compiled CUDA.
# vLLM's default wheel is now CUDA 13 -> "ImportError: libcudart.so.13" on a 12.x driver.
# Pin vLLM's cu128 wheel explicitly instead:
VLLM_VERSION=$(curl -s https://api.github.com/repos/vllm-project/vllm/releases/latest | jq -r .tag_name | sed 's/^v//')
ARCH=$(uname -m)   # x86_64 or aarch64
uv pip install \
  "https://github.com/vllm-project/vllm/releases/download/v${VLLM_VERSION}/vllm-${VLLM_VERSION}+cu128-cp38-abi3-manylinux_2_35_${ARCH}.whl" \
  --extra-index-url https://download.pytorch.org/whl/cu128

# then serve (either env):
vllm serve LiquidAI/LFM2-350M-PII-Extract-JP --port 8001    # extractor
vllm serve Qwen/Qwen3-8B --port 8000                        # generator

# Extractor B (privacy-filter): q4f16 ONNX downloads from HF on first run (cached)
```

> **GPU/CUDA note (driver = CUDA 12.4, `12040`).** vLLM ships a compiled CUDA kernel, so **the
> vLLM wheel's CUDA version matters as much as torch's** — and `--torch-backend` only controls
> torch. vLLM's *default* wheel is now CUDA 13 (`cu130`), which fails on a 12.x driver with
> `ImportError: libcudart.so.13`. Fix by installing vLLM's **`cu128`** wheel explicitly (command
> above): CUDA 12.8 → `libcudart.so.12` → runs on a 12.4 driver via CUDA **minor-version
> compatibility** (driver ≥ R525). vLLM publishes `cu128`/`cu129`/`cu130` wheels per release, so
> you keep the **latest** vLLM — no version downgrade needed.
> - Verify BOTH import: `python -c "import vllm, torch; print(torch.version.cuda, torch.cuda.is_available())"`
>   → want a `12.x` and `True`, with no `libcudart.so.13`.
> - `--torch-backend=cu128` / `auto` alone is **not** enough — it leaves vLLM on its cu130 wheel.
> - Sustainable alternative: upgrade the NVIDIA driver to **R580+** (CUDA 13); then the default
>   `pip install vllm` works unpinned, since vLLM's default is now CUDA 13.
>
> **One env or two?** Path (i) merges everything into `pii-harness` — simplest on a single
> GPU box. Path (ii) keeps a separate env — required if you also build envs on macOS (no CUDA
> there) or want to deploy the ONNX classifier on a CPU/Lambda box without the torch+CUDA stack.

> No GPU on hand? Point `LFM2_BASE_URL` at ollama's OpenAI endpoint instead:
> `ollama pull hf.co/LiquidAI/LFM2-350M-PII-Extract-JP-GGUF:F16` then set
> `LFM2_BASE_URL=http://localhost:11434/v1` and the matching `LFM2_MODEL` tag.

## Usage

### Validate the extractors against their reference suites
```bash
.venv/bin/python scripts/extract_lfm2.py            # -> out/logs/lfm2_run_*.log
.venv/bin/python scripts/extract_privacy_filter.py  # -> out/logs/pf_run_*.log
```

### Compare both on a document set (qualitative)
```bash
.venv/bin/python scripts/compare_extractors.py      # default: data/japanese_pii.txt
# -> out/logs/compare_*.log and .md
```

### Generate synthetic rehab records (the data pipeline)

1. On a **GPU server**, serve Qwen3-8B with vLLM:
   ```bash
   pip install vllm
   vllm serve Qwen/Qwen3-8B --max-model-len 8192 --gpu-memory-utilization 0.90
   ```
2. Point the generator at it and create records:
   ```bash
   export OPENAI_BASE_URL=http://localhost:8000/v1   # vLLM endpoint
   .venv/bin/python scripts/generate_rehab.py -n 50 --model Qwen/Qwen3-8B --seed 0
   ```
   No GPU handy? Use the built-in template (no LLM needed):
   ```bash
   .venv/bin/python scripts/generate_rehab.py -n 5 --template-only
   ```

**How ground truth stays exact:** the LLM writes the narrative with `{{placeholder}}`
tokens; faker-sampled fake PII is substituted in *afterward*. The PII is therefore
always verbatim and the labels are derived programmatically — no manual labeling,
no LLM mangling the PII.

### Grade both models on the dataset (closes the loop)
```bash
.venv/bin/python scripts/grade_rehab.py             # default: out/rehab_dataset.jsonl
# -> per-model, per-category precision/recall/F1 + out/logs/grade_*.log
```

## Configuration (env vars)

| Var | Default | Used by |
|-----|---------|---------|
| `OPENAI_BASE_URL` | `http://localhost:8000/v1` | generator (Qwen3 on vLLM) |
| `OPENAI_API_KEY` | `EMPTY` | generator |
| `LFM2_BASE_URL` | `http://localhost:8000/v1` | LFM2 extractor (vLLM; `:11434/v1` for ollama) |
| `LFM2_MODEL` | `LiquidAI/LFM2-350M-PII-Extract-JP` | LFM2 extractor |
| `PF_MODEL` | `openai/privacy-filter` | privacy-filter extractor |
| `PF_ONNX` | `onnx/model_q4f16.onnx` | privacy-filter ONNX file |

## Known caveats

- **privacy-filter uses raw argmax decoding**, not the model's intended constrained
  **Viterbi** decode (`viterbi_calibration.json`). This understates its quality —
  expect some fragmented/over-extended Japanese spans until Viterbi is added
  (`pf_extract` in `pii_lib.py` is the place to do it).
- All synthetic PII is **fake** (faker ja_JP + reserved `example.*` domains). Some
  emails use `example.*`, which is known to trigger LFM2's `@example` truncation —
  useful as a signal, but mix in real-looking domains for representative numbers.
- Grading is **exact set-match** per category (NFC-normalized). It does not give
  partial credit for boundary-off spans.
