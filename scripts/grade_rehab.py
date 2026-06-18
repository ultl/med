#!/usr/bin/env python3
"""Grade BOTH PII models against the synthetic rehab dataset's ground truth.

Reads  out/rehab_dataset.jsonl  (records with unified `pii` labels)
Runs   LFM2-350M-PII-Extract-JP  and  openai/privacy-filter
Scores each model per PII category (precision / recall / F1), respecting that the
two models have different taxonomies:

  ground-truth cat   LFM2 category      privacy-filter label
  ----------------   --------------     --------------------
  human_name         human_name         private_person
  address            address            private_address
  phone_number       phone_number       private_phone
  email_address      email_address      private_email
  company_name       company_name       (none - org type absent -> not scored for PF)
  account_number     (none - not scored for LFM2)   account_number
  date               (none - not scored for LFM2)    private_date

Out-of-taxonomy categories are reported as COVERAGE GAPS, not penalized.

Usage:  .venv/bin/python scripts/grade_rehab.py [out/rehab_dataset.jsonl]
Outputs: out/logs/grade_<UTC>.log  and  out/logs/grade_history.jsonl
"""
import json, os, sys
from collections import defaultdict
from datetime import datetime, timezone
import pii_lib as L

# ground-truth category -> per-model key (None = model has no such category)
LFM2_MAP = {"human_name": "human_name", "address": "address",
            "phone_number": "phone_number", "email_address": "email_address",
            "company_name": "company_name", "account_number": None, "date": None}
PF_MAP = {"human_name": "private_person", "address": "private_address",
          "phone_number": "private_phone", "email_address": "private_email",
          "company_name": None, "account_number": "account_number", "date": "private_date"}


def grade(records):
    # acc[model][cat] = [tp, fp, fn]
    acc = {"LFM2": defaultdict(lambda: [0, 0, 0]),
           "privacy-filter": defaultdict(lambda: [0, 0, 0])}
    details = []
    for rec in records:
        gt = rec["pii"]
        lfm2_pred = L.lfm2_extract(rec["input"])
        pf_pred = L.pf_extract(rec["input"])
        row = {"id": rec["id"], "LFM2": {}, "privacy-filter": {}}
        for model, pred, mapping in (("LFM2", lfm2_pred, LFM2_MAP),
                                     ("privacy-filter", pf_pred, PF_MAP)):
            for gcat, mkey in mapping.items():
                if mkey is None:
                    continue  # coverage gap: not scored for this model
                tp, fp, fn, miss, extra = L.score_sets(gt.get(gcat, []), pred.get(mkey, []))
                a = acc[model][gcat]
                a[0] += tp; a[1] += fp; a[2] += fn
                if miss or extra:
                    row[model][gcat] = {"missing": miss, "extra": extra}
        details.append(row)
    return acc, details


def main():
    L.ensure_dirs()
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(L.OUT, "rehab_dataset.jsonl")
    records = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    acc, details = grade(records)

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 92)
    emit("Rehab dataset grading  -  LFM2 vs privacy-filter")
    emit(f"timestamp : {ts}")
    emit(f"dataset   : {os.path.relpath(path, L.ROOT)}  ({len(records)} records)")
    emit("=" * 92)

    summary = {"timestamp": ts, "records": len(records), "models": {}}
    for model in ("LFM2", "privacy-filter"):
        emit(f"\n## {model}")
        emit(f"  {'category':<15} {'P':>6} {'R':>6} {'F1':>6}   {'TP':>4} {'FP':>4} {'FN':>4}")
        micro = [0, 0, 0]
        msummary = {}
        for cat in sorted(acc[model]):
            tp, fp, fn = acc[model][cat]
            micro[0] += tp; micro[1] += fp; micro[2] += fn
            m = L.prf(tp, fp, fn)
            msummary[cat] = m
            emit(f"  {cat:<15} {m['precision']:>6} {m['recall']:>6} {m['f1']:>6}   "
                 f"{tp:>4} {fp:>4} {fn:>4}")
        mt = L.prf(*micro)
        msummary["_micro"] = mt
        emit(f"  {'MICRO (all)':<15} {mt['precision']:>6} {mt['recall']:>6} {mt['f1']:>6}   "
             f"{micro[0]:>4} {micro[1]:>4} {micro[2]:>4}")
        summary["models"][model] = msummary

    # coverage-gap reminder
    emit("\n## coverage gaps (not scored)")
    emit("  LFM2          : account_number, date  (no such category)")
    emit("  privacy-filter: company_name          (no organization category)")

    # per-record error detail
    emit("\n## per-record errors")
    for row in details:
        errs = {m: row[m] for m in ("LFM2", "privacy-filter") if row[m]}
        if errs:
            emit(f"  [{row['id']}]")
            for m, cats in errs.items():
                for cat, d in cats.items():
                    bits = []
                    if d["missing"]: bits.append(f"missing={d['missing']}")
                    if d["extra"]:   bits.append(f"extra={d['extra']}")
                    emit(f"    {m:<14} {cat}: {', '.join(bits)}")
    emit("\n" + "=" * 92)

    open(os.path.join(L.LOGS, f"grade_{ts}.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(os.path.join(L.LOGS, "grade_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    emit(f"log saved : out/logs/grade_{ts}.log")


if __name__ == "__main__":
    main()
