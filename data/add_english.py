"""Добавляет английскую часть по рецепту tev1 (data/new-v1/records) к русским данным v3.

Исключает всё, что совпадает с тестами tev1 (data/v1/records/test, policy_transfer) по id или по содержанию.
Тесты tev1 сохраняются отдельно в test_en.jsonl — для сравнения на их собственных задачах.
"""
import hashlib
import json
import random
import sys
from pathlib import Path

import os
# git clone https://github.com/togethercomputer/tev1 и соберите данные по их README (data/new-v1/records)
TEV1 = Path(os.environ.get("TEV1_DIR", Path.home() / "tev1" / "data"))
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data_v3")
CAPS = {"policy_v2": 8000, "routing_v2": 4000, "mnli": 4000, "research_taxonomy_v21": 2000, "boolq": 3000,
        "banking77": 3000, "sst5": 2000, "ag_news": 1500, "policy": 1500}


def read(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def content_hash(r):
    st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256((st + "||" + r["question"]).encode()).hexdigest()


def convert(r, task, split):
    st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    opts = [{"label": o["label"], "key": o.get("key", o["label"]), "description": o.get("description", o.get("text", ""))}
            for o in r["options"]]
    ans = r["answer"]
    assert ans in [o["label"] for o in opts], (ans, opts)
    return {"task": task, "split": split, "state": st, "question": r["question"], "options": opts, "answer": ans,
            "style": "tev1"}


def main():
    rng = random.Random(5)
    tests = {s: read(TEV1 / "v1" / "records" / f"{s}.jsonl") for s in ["test", "policy_transfer"]}
    banned_ids = {r["id"] for v in tests.values() for r in v}
    banned_hash = {content_hash(r) for v in tests.values() for r in v}

    pool = read(TEV1 / "new-v1" / "records" / "train.jsonl")
    clean = [r for r in pool if r["id"] not in banned_ids and content_hash(r) not in banned_hash]
    print(f"английский пул {len(pool)}, после удаления пересечений с тестами tev1: {len(clean)}")
    rng.shuffle(clean)
    import os
    scale = float(os.environ.get("EN_SCALE", "1"))
    caps = {k: int(v * scale) for k, v in CAPS.items()}
    by, train_en = {}, []
    for r in clean:
        if by.get(r["source"], 0) < caps.get(r["source"], 0):
            by[r["source"]] = by.get(r["source"], 0) + 1
            train_en.append(convert(r, "en_" + r["source"], "train"))
    print("английский train:", len(train_en), by)

    dev = [r for r in read(TEV1 / "new-v1" / "records" / "dev.jsonl")
           if r["id"] not in banned_ids and content_hash(r) not in banned_hash]
    rng.shuffle(dev)
    val_en = [convert(r, "en_" + r["source"], "val") for r in dev[:400]]

    test_en = [convert(r, "en_test_" + r["source"], "test") for r in tests["test"]]
    test_en += [convert(r, "en_transfer", "test") for r in tests["policy_transfer"]]

    for name, extra in [("train", train_en), ("val", val_en)]:
        rows = read(OUT / f"{name}.jsonl") + extra
        rng.shuffle(rows)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, "итого", len(rows))
    with open(OUT / "test_en.jsonl", "w", encoding="utf-8") as f:
        for r in test_en:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("test_en", len(test_en))


if __name__ == "__main__":
    main()
