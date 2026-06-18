#!/usr/bin/env python3
"""Synthesize Japanese functional-rehabilitation records with ground-truth PII labels.

Pipeline:
  1. Sample fake PII + functional scores (faker ja_JP + random).
  2. Qwen3-8B (via vLLM, OpenAI-compatible API) writes the narrative using {{PLACEHOLDER}} tokens.
  3. Substitute the fake PII into the placeholders  -> exact, verbatim PII.
  4. Derive ground-truth labels = which sampled values actually appear in the text.

All PII is SYNTHETIC. Output:
  rehab_dataset.jsonl  (one record/line: id, input, pii)
  rehab_records.txt    (human-readable)

Usage:  .venv/bin/python generate_rehab.py -n 5 [--model qwen3:8b] [--seed 0]
"""
import argparse, json, os, random, re, urllib.request
from faker import Faker

fake = Faker("ja_JP")
# OpenAI-compatible endpoint: vLLM (default :8000) or ollama (:11434/v1)
OPENAI_BASE = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1").rstrip("/")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")

# seed-key -> unified PII category
KEY2CAT = {
    "patient_name": "human_name",
    "therapist": "human_name", "physician": "human_name",
    "address": "address", "phone": "phone_number", "email": "email_address",
    "facility": "company_name", "referrer_clinic": "company_name",
    "patient_id": "account_number", "karte_no": "account_number",
    "insurance_no": "account_number",
    "dob": "date", "date_onset": "date", "date_transfer": "date",
    "date_eval": "date", "date_next": "date",
}

DIAGNOSES = [
    "脳梗塞（右中大脳動脈領域）後遺症、左片麻痺",
    "脳出血（被殻出血）後遺症、右片麻痺",
    "大腿骨頸部骨折術後（人工骨頭置換術後）",
    "腰部脊柱管狭窄症術後",
    "変形性膝関節症（右）人工膝関節置換術後",
    "頸髄損傷（不全麻痺）",
]
EMAIL_DOMAINS = ["example.jp", "example.co.jp", "hospital.example",
                 "gmail.com", "minato-reha.jp", "sakura-clinic.co.jp"]


def jdate(y, m, d):
    return f"{y}年{m}月{d}日"


def sample_seed(i):
    name = fake.name()
    onset_m, onset_d = random.randint(1, 6), random.randint(1, 28)
    seed = {
        "patient_name": name,
        "dob": jdate(random.randint(1940, 1965), random.randint(1, 12), random.randint(1, 28)),
        "patient_id": f"PT-2025-{random.randint(1000, 9999):06d}"[:13],
        "karte_no": f"C-{random.randint(100000, 999999)}",
        "insurance_no": str(random.randint(10**9, 10**12 - 1)),
        "address": fake.address().replace("\n", " "),
        "phone": fake.phone_number(),
        "email": f"{fake.user_name()}@{random.choice(EMAIL_DOMAINS)}",
        "facility": random.choice(["横浜みなと", "さくら", "青葉", "湘南", "あおぞら"])
                    + "リハビリテーション病院",
        "referrer_clinic": fake.last_name() + random.choice(["クリニック", "内科医院", "整形外科"]),
        "therapist": fake.name(),
        "physician": fake.name(),
        "diagnosis": random.choice(DIAGNOSES),
        "fim_motor": random.randint(40, 80), "fim_cog": random.randint(20, 35),
        "barthel": random.randint(40, 90),
        "rom_flex": random.choice([90, 110, 120, 130]),
        "mmt": random.choice(["左上肢3／左下肢3+／右側5", "右上肢2／右下肢3／左側5"]),
        "date_onset": jdate(2025, onset_m, onset_d),
        "date_transfer": jdate(2025, onset_m + 1, random.randint(1, 28)),
        "date_eval": jdate(2025, 6, 10),
        "date_next": jdate(2025, 6, 24),
    }
    seed["fim_total"] = seed["fim_motor"] + seed["fim_cog"]
    return seed


PROMPT = """あなたは日本のリハビリテーション病院の理学療法士です。
以下の臨床情報をもとに、自然で現実的な「リハビリテーション経過記録」を日本語で作成してください。

【重要な制約】
- 患者氏名・住所・電話・メール・施設名・各種番号・日付は、必ず下記のプレースホルダ記号（例: {{patient_name}}）を**そのまま**本文に挿入すること。実在しそうな値を自分で作らないこと。
- 出力は記録本文のみ。説明や前置きは不要。
- 次のセクションを含めること: 患者基本情報 / 診断・現病歴 / 機能評価 / 目標 / SOAP / 署名。
- 表現は記録ごとに自然に変化させてよい。

【使用するプレースホルダ】
氏名={{patient_name}} 生年月日={{dob}}
患者ID={{patient_id}} カルテ番号={{karte_no}} 被保険者番号={{insurance_no}}
住所={{address}} 電話={{phone}} メール={{email}}
当院={{facility}} 紹介元={{referrer_clinic}} 担当PT={{therapist}} 主治医={{physician}}
発症日={{date_onset}} 転院日={{date_transfer}} 評価日={{date_eval}} 次回評価={{date_next}}

【臨床情報（値はそのまま本文に使う）】
診断名: {diagnosis}
FIM: 運動{fim_motor}点／認知{fim_cog}点（合計{fim_total}点）
Barthel Index: {barthel}点 / ROM 左肩屈曲{rom_flex}度 / MMT {mmt}
"""


def write_with_llm(seed, model):
    body = PROMPT.format(**seed)
    payload = {"model": model,
               "messages": [{"role": "user", "content": body}],
               "temperature": 0.7, "max_tokens": 900,
               # vLLM: disable Qwen3 thinking for clean record text (ignored by ollama)
               "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(
        OPENAI_BASE + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {OPENAI_KEY}"})
    txt = json.loads(urllib.request.urlopen(req, timeout=300).read())["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()


TEMPLATE = """■ リハビリテーション経過記録

【患者基本情報】
氏名：{{patient_name}}　生年月日：{{dob}}
患者ID：{{patient_id}}　カルテ番号：{{karte_no}}　被保険者番号：{{insurance_no}}
住所：{{address}}　連絡先：{{phone}}　メール：{{email}}
当院：{{facility}}　担当PT：{{therapist}}　紹介元：{{referrer_clinic}} {{physician}} 医師

【診断・現病歴】
{date_onset}に{diagnosis}を発症。{date_transfer}に当院回復期病棟へ転院した。

【機能評価（{date_eval}時点）】
FIM：運動{fim_motor}点／認知{fim_cog}点（合計{fim_total}点）。Barthel Index：{barthel}点。
ROM：左肩関節屈曲{rom_flex}度。MMT：{mmt}。

【目標】
短期：監視下での平行棒内歩行10mを自立。長期：T字杖＋短下肢装具での屋内歩行自立。

【SOAP（{date_eval}）】
S：「歩く練習を続けたい」　O：10m歩行は四点杖で見守りレベル。
A：下肢支持性は改善傾向。　P：歩行距離延長を継続。次回評価は{date_next}。

署名：理学療法士 {{therapist}}"""


def write_with_template(seed):
    # placeholders for clinical (non-PII) values are filled here; PII tokens stay {{...}}
    t = TEMPLATE
    for k in ("date_onset", "diagnosis", "date_transfer", "date_eval", "fim_motor",
              "fim_cog", "fim_total", "barthel", "rom_flex", "mmt", "date_next"):
        t = t.replace("{" + k + "}", str(seed[k]))
    return t


def inject(text, seed):
    """Replace {{key}} tokens with fake PII; return (final_text, labels by category)."""
    for k in KEY2CAT:
        text = text.replace("{{" + k + "}}", str(seed[k]))
    # also fill any clinical {{tokens}} the LLM may have used for non-PII (none expected)
    labels = {}
    for k, cat in KEY2CAT.items():
        val = str(seed[k])
        if val and val in text:
            labels.setdefault(cat, [])
            if val not in labels[cat]:
                labels[cat].append(val)
    return text, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template-only", action="store_true",
                    help="skip the LLM, use the built-in template")
    args = ap.parse_args()
    random.seed(args.seed); Faker.seed(args.seed)

    records, pretty = [], []
    for i in range(args.n):
        seed = sample_seed(i)
        used_llm = False
        if not args.template_only:
            try:
                body = write_with_llm(seed, args.model)
                if "{{patient_name}}" in body:   # LLM kept placeholders -> good
                    used_llm = True
                else:
                    body = write_with_template(seed)
            except Exception as e:
                print(f"[{i}] LLM unavailable ({e}); using template")
                body = write_with_template(seed)
        else:
            body = write_with_template(seed)

        text, labels = inject(body, seed)
        rid = f"rehab_{i+1:04d}"
        records.append({"id": rid, "source": "qwen3" if used_llm else "template",
                        "input": text, "pii": labels})
        pretty.append(f"===== {rid}  ({'qwen3' if used_llm else 'template'}) =====\n{text}\n"
                      f"\nGROUND TRUTH: {json.dumps(labels, ensure_ascii=False)}\n")
        print(f"[{rid}] {'qwen3' if used_llm else 'template'}  "
              f"{sum(len(v) for v in labels.values())} PII spans")

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rehab_dataset.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    open(os.path.join(out_dir, "rehab_records.txt"), "w", encoding="utf-8").write("\n".join(pretty))
    print(f"\nwrote out/rehab_dataset.jsonl ({len(records)} records) and out/rehab_records.txt")


if __name__ == "__main__":
    main()
