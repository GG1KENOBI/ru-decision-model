"""Оценка Kev (jaredpalmer/kev-4b) на наших тестах через его собственный LocalPredictor.

Наша запись -> запрос Kev: state как есть, один вопрос типа choice, criteria = {ключ варианта: описание}.
Результат сохраняется в нашем формате (task, gold, logits=log p в порядке наших вариантов).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import os
# git clone https://github.com/jaredpalmer/kev — путь к репозиторию в KEV_DIR
sys.path.insert(0, os.environ.get("KEV_DIR", str(Path.home() / "kev")))
from kev.checkpoint import LoadOptions  # noqa: E402
from kev.predictors import LocalPredictor  # noqa: E402

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"


def to_record(r):
    crit = {}
    for o in r["options"]:
        k = f"{o['label']}_{o['key']}"[:60]
        crit[k] = o["description"] or o["key"]
    gold = next(f"{o['label']}_{o['key']}"[:60] for o in r["options"] if o["label"] == r["answer"])
    return {"state": r["state"][:3500],
            "questions": {"q": {"type": "choice", "instructions": r["question"], "criteria": crit, "label": gold}}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="kev4b")
    ap.add_argument("--run", default="jaredpalmer/kev-4b")
    ap.add_argument("--data", default="../data/data_v4")
    ap.add_argument("--splits", default="test,test_en")
    a = ap.parse_args()
    pred = LocalPredictor(a.run, "cuda", LoadOptions())
    out_dir = Path("results")
    for split in a.splits.split(","):
        rows = [json.loads(l) for l in open(Path(a.data) / f"{split}.jsonl", encoding="utf-8")]
        out, t0, fails = [], time.time(), 0
        for r in rows:
            rec = to_record(r)
            try:
                p = pred(rec)["probabilities"]["q"]
                logits = [math.log(max(p[k], 1e-12)) for k in rec["questions"]["q"]["criteria"]]
            except Exception:
                fails += 1
                logits = [0.0] * len(r["options"])  # отказ модели = равномерное распределение (засчитывается как ошибка/угадывание)
            out.append({"task": r["task"], "gold": LETTERS.index(r["answer"]), "logits": logits})
        with open(out_dir / f"{a.name}_{split}.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        acc = sum(max(range(len(o["logits"])), key=o["logits"].__getitem__) == o["gold"] for o in out) / len(out)
        print(f"== {a.name} / {split}: {len(out)} за {time.time() - t0:.0f} с, acc {acc:.4f}, отказов {fails}", flush=True)


if __name__ == "__main__":
    main()
