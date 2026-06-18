"""Shared library for the Japanese PII extraction harness.

Serving model: "vLLM for everything generative, ONNX for the classifier".
  - lfm2_extract(text)  -> dict   LFM2-350M-PII-Extract-JP via vLLM (OpenAI-compatible API)
  - pf_extract(text)    -> dict   openai/privacy-filter via ONNX Runtime (CPU, no torch)
  - norm / parse_json / score_sets / prf  -> grading utilities
  - ROOT / DATA / OUT / LOGS              -> project paths
"""
import json, os, re, unicodedata, urllib.request
import numpy as np

# ---------------------------------------------------------------- paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out")
LOGS = os.path.join(OUT, "logs")

def ensure_dirs():
    os.makedirs(LOGS, exist_ok=True)

# ---------------------------------------------------------------- model A: LFM2 (generative, vLLM)
# vLLM:   vllm serve LiquidAI/LFM2-350M-PII-Extract-JP   -> OpenAI API on :8000
# (point LFM2_BASE_URL at :11434/v1 + the GGUF tag to fall back to ollama's OpenAI endpoint)
LFM2_BASE_URL = os.environ.get("LFM2_BASE_URL", "http://localhost:8000/v1").rstrip("/")
LFM2_MODEL = os.environ.get("LFM2_MODEL", "LiquidAI/LFM2-350M-PII-Extract-JP")
LFM2_API_KEY = os.environ.get("LFM2_API_KEY", "EMPTY")
LFM2_SYS = "Extract <address>, <company_name>, <email_address>, <human_name>, <phone_number>"
LFM2_CATS = ["address", "company_name", "email_address", "human_name", "phone_number"]

def lfm2_extract(text, system=LFM2_SYS, drop_empty=False):
    """Return {category: [strings]} from LFM2's JSON output (greedy: temperature 0)."""
    payload = {"model": LFM2_MODEL,
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": text}],
               "temperature": 0, "max_tokens": 256}
    req = urllib.request.Request(
        LFM2_BASE_URL + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {LFM2_API_KEY}"})
    raw = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
    obj, _ = parse_json(raw)
    if obj is None:
        return {"_raw": raw.strip()}
    return {k: v for k, v in obj.items() if v} if drop_empty else obj

# ---------------------------------------------------------------- model B: privacy-filter (classifier, ONNX)
PF_MODEL = os.environ.get("PF_MODEL", "openai/privacy-filter")
PF_ONNX = os.environ.get("PF_ONNX", "onnx/model_q4f16.onnx")
_pf = {}  # lazy singleton

def _load_pf():
    if not _pf:
        import onnxruntime as ort
        from transformers import AutoTokenizer, AutoConfig
        from huggingface_hub import hf_hub_download
        onnx_path = hf_hub_download(PF_MODEL, PF_ONNX)
        try:                                   # external weights must sit beside the graph
            hf_hub_download(PF_MODEL, PF_ONNX + "_data")
        except Exception:
            pass
        _pf["tok"] = AutoTokenizer.from_pretrained(PF_MODEL)
        _pf["id2label"] = AutoConfig.from_pretrained(PF_MODEL).id2label
        _pf["sess"] = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        _pf["inputs"] = {i.name for i in _pf["sess"].get_inputs()}
    return _pf

def pf_extract(text):
    """Return {label: [exact substrings]} via per-token BIOES argmax over char offsets.

    NOTE: argmax baseline. For production, swap in the constrained Viterbi decode
    (viterbi_calibration.json) to suppress incoherent single-token spans.
    """
    p = _load_pf()
    enc = p["tok"](text, return_offsets_mapping=True, truncation=True, max_length=4096)
    offs = enc["offset_mapping"]
    feeds = {"input_ids": np.array([enc["input_ids"]], dtype=np.int64)}
    if "attention_mask" in p["inputs"]:
        feeds["attention_mask"] = np.array([enc["attention_mask"]], dtype=np.int64)
    logits = p["sess"].run(None, feeds)[0]          # [1, T, num_labels]
    ids = logits[0].argmax(-1).tolist()

    spans, ct, cs, ce = {}, None, None, None
    def flush():
        nonlocal ct, cs, ce
        if ct is not None:
            spans.setdefault(ct, []).append(text[cs:ce].strip())
        ct = cs = ce = None
    for (s, e), tid in zip(offs, ids):
        if s == e:
            continue
        lab = p["id2label"][tid]
        if lab == "O" or "-" not in lab:
            flush(); continue
        et = lab.split("-", 1)[1]
        if et == ct and s <= ce + 1:
            ce = e
        else:
            flush(); ct, cs, ce = et, s, e
    flush()
    return spans

# ---------------------------------------------------------------- grading utilities
def norm(s):
    return unicodedata.normalize("NFC", str(s)).strip()

def parse_json(text):
    """Best-effort parse of a model's JSON output; fall back to first {...} block."""
    try:
        return json.loads(text), None
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), None
        except Exception as e:
            return None, f"unparseable JSON: {e}"
    return None, "no JSON object found"

def score_sets(expected, predicted):
    """Set-match one category. Returns (tp, fp, fn, missing, extra)."""
    exp = {norm(x) for x in expected}
    got = {norm(x) for x in predicted}
    missing, extra = sorted(exp - got), sorted(got - exp)
    return len(exp & got), len(extra), len(missing), missing, extra

def prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3),
            "tp": tp, "fp": fp, "fn": fn}
