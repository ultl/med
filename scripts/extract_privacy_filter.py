#!/usr/bin/env python3
"""Run the openai/privacy-filter reference suite, grade vs ground truth, log every run.

Usage:  .venv/bin/python scripts/extract_privacy_filter.py [data/pf_tests.json]
Outputs: out/logs/pf_run_<UTC>.log  and  out/logs/pf_history.jsonl
"""
import json, os, sys
from datetime import datetime, timezone
import pii_lib as L


def main():
    L.ensure_dirs()
    suite = sys.argv[1] if len(sys.argv) > 1 else os.path.join(L.DATA, "pf_tests.json")
    cases = json.load(open(suite, encoding="utf-8"))
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    lines = []
    def emit(s=""):
        print(s); lines.append(s)

    emit("=" * 88)
    emit(f"openai/privacy-filter suite   ({len(cases)} cases)   {ts}")
    emit("=" * 88)

    n_pass = 0
    hist = []
    for c in cases:
        pred = L.pf_extract(c["input"])
        case_pass, per = True, {}
        extra_cats = [k for k in pred if k not in c["expected"] and pred.get(k)]
        if extra_cats:
            case_pass = False
        for cat, exp in c["expected"].items():
            tp, fp, fn, miss, extra = L.score_sets(exp, pred.get(cat, []))
            ok = not miss and not extra
            per[cat] = (ok, miss, extra)
            case_pass = case_pass and ok
        n_pass += int(case_pass)
        emit(f"\n[{c['id']}]  {'PASS' if case_pass else 'FAIL'}")
        emit(f"  input : {c['input']}")
        emit(f"  output: {json.dumps(pred, ensure_ascii=False)}")
        if extra_cats:
            emit(f"  ! extra categories: {extra_cats}")
        for cat, (ok, miss, extra) in per.items():
            if not ok:
                emit(f"  FAIL {cat}: missing={miss} extra={extra}")
        hist.append({"id": c["id"], "pass": case_pass})

    emit("\n" + "=" * 88)
    emit(f"SUMMARY: {n_pass}/{len(cases)} cases passed")
    fails = [h["id"] for h in hist if not h["pass"]]
    if fails:
        emit(f"Failing: {fails}")
    emit("=" * 88)

    open(os.path.join(L.LOGS, f"pf_run_{ts}.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    with open(os.path.join(L.LOGS, "pf_history.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": ts, "passed": n_pass, "total": len(cases),
                            "failing": fails}, ensure_ascii=False) + "\n")
    emit(f"log saved : out/logs/pf_run_{ts}.log")


if __name__ == "__main__":
    main()
