#!/usr/bin/env python3
"""Run the LFM2-350M-PII-Extract-JP reference suite, grade vs ground truth, log every run.

Usage:  .venv/bin/python scripts/extract_lfm2.py [data/pii_tests.json]
Outputs: out/logs/lfm2_run_<UTC>.log  and  out/logs/lfm2_history.jsonl
"""
import json, os, sys
from datetime import datetime, timezone
import pii_lib as L


def main():
    L.ensure_dirs()
    suite = sys.argv[1] if len(sys.argv) > 1 else os.path.join(L.DATA, "pii_tests.json")
    cases = json.load(open(suite, encoding="utf-8"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 88)
    emit(f"LFM2-350M-PII-Extract-JP suite   ({len(cases)} cases)   {ts}")
    emit("=" * 88)

    n_pass = 0
    cat_tot, cat_ok = {}, {}
    hist = []
    for c in cases:
        pred = L.lfm2_extract(c["input"], system=c["system"])
        case_pass, per = True, {}
        extra_keys = [k for k in pred if k not in c["expected"] and pred.get(k)]
        if extra_keys:
            case_pass = False
        for cat, exp in c["expected"].items():
            tp, fp, fn, miss, extra = L.score_sets(exp, pred.get(cat, []))
            ok = not miss and not extra
            per[cat] = (ok, miss, extra)
            cat_tot[cat] = cat_tot.get(cat, 0) + 1
            cat_ok[cat] = cat_ok.get(cat, 0) + int(ok)
            case_pass = case_pass and ok
        n_pass += int(case_pass)
        emit(f"\n[{c['id']}]  {'PASS' if case_pass else 'FAIL'}")
        emit(f"  input : {c['input'].replace(chr(10), ' / ')}")
        emit(f"  output: {json.dumps(pred, ensure_ascii=False)}")
        if extra_keys:
            emit(f"  ! unexpected categories: {extra_keys}")
        for cat, (ok, miss, extra) in per.items():
            if not ok:
                emit(f"  FAIL {cat}: missing={miss} extra={extra}")
        hist.append({"id": c["id"], "pass": case_pass})

    emit("\n" + "=" * 88)
    emit(f"SUMMARY: {n_pass}/{len(cases)} cases passed")
    for cat in L.LFM2_CATS:
        if cat in cat_tot:
            emit(f"  {cat:<14} {cat_ok[cat]}/{cat_tot[cat]}")
    fails = [h["id"] for h in hist if not h["pass"]]
    if fails:
        emit(f"Failing: {fails}")
    emit("=" * 88)

    open(os.path.join(L.LOGS, f"lfm2_run_{ts}.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(os.path.join(L.LOGS, "lfm2_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": ts, "passed": n_pass, "total": len(cases),
                            "failing": fails}, ensure_ascii=False) + "\n")
    emit(f"log saved : out/logs/lfm2_run_{ts}.log")


if __name__ == "__main__":
    main()
