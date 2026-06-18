#!/usr/bin/env python3
"""Qualitative side-by-side of both extractors on a document set (no ground truth).

Default input: data/japanese_pii.txt (a python list literal of document strings).
Usage:  .venv/bin/python scripts/compare_extractors.py [data/japanese_pii.txt]
Outputs: out/logs/compare_<UTC>.log  and  .md
"""
import json, os, re, sys
from datetime import datetime, timezone
import pii_lib as L


def load_docs(path):
    raw = open(path, encoding="utf-8").read()
    return re.findall(r'"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)


def main():
    L.ensure_dirs()
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(L.DATA, "japanese_pii.txt")
    docs = load_docs(path)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines, md = [], ["| # | Document | A: LFM2 | B: privacy-filter |",
                     "|---|---|---|---|"]
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 100)
    emit(f"Extractor comparison  ({len(docs)} documents)   {ts}")
    emit("  A = LFM2-350M-PII-Extract-JP (5 cats)   B = openai/privacy-filter (8 cats)")
    emit("=" * 100)

    for i, doc in enumerate(docs, 1):
        a = L.lfm2_extract(doc, drop_empty=True)
        b = L.pf_extract(doc)
        emit(f"\n[{i:02d}] {doc}")
        emit(f"   A(LFM2): {json.dumps(a, ensure_ascii=False)}")
        emit(f"   B(PF)  : {json.dumps(b, ensure_ascii=False)}")
        esc = lambda x: json.dumps(x, ensure_ascii=False).replace("|", "\\|")
        md.append(f"| {i} | {doc.replace('|', '\\|')} | {esc(a)} | {esc(b)} |")

    emit("\n" + "=" * 100)
    open(os.path.join(L.LOGS, f"compare_{ts}.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    open(os.path.join(L.LOGS, f"compare_{ts}.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    emit(f"saved : out/logs/compare_{ts}.log  and  .md")


if __name__ == "__main__":
    main()
