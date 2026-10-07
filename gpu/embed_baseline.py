"""Классический ориентир: эмбеддинги Giga-Embeddings + логистическая регрессия.

Работает только для задач с фиксированным набором ответов (тональность, рубрики, эмоции).
На каждую задачу: обучение на train (до 6000 примеров), предсказание на тех же тестовых записях,
что и у остальных моделей; выбор только среди вариантов, предложенных в записи.
Результат — в общем формате (task, gold, logits).
"""
import json
import math
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from transformers import AutoModel, AutoTokenizer

from common import LETTERS, load_rows

MODEL = "ai-sage/Giga-Embeddings-instruct-3B-0826"
TASKS = ["reviews", "headlines", "kinopoisk", "grnti", "oecd", "cedr"]
INSTR = "Instruct: Определи категорию текста\nQuery: "


def key_of(r):
    return next(o["key"] for o in r["options"] if o["label"] == r["answer"])


def main():
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL, trust_remote_code=True, dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def encode(texts, bs=64):
        out = []
        for i in range(0, len(texts), bs):
            enc = tok([INSTR + t for t in texts[i:i + bs]], return_tensors="pt", padding=True, truncation=True, max_length=512)
            enc = {k: v.cuda() for k, v in enc.items()}
            h = model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            out.append(F.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-6), dim=-1).float().cpu())
        return torch.cat(out).numpy()

    train = defaultdict(list)
    for r in load_rows("data_v4/train.jsonl"):
        if r["task"] in TASKS and len(train[r["task"]]) < 6000:
            train[r["task"]].append(r)
    test = [r for r in load_rows("data_v4/test.jsonl") if r["task"] in TASKS]

    results, t0 = [], time.time()
    for task in TASKS:
        tr = train[task]
        clf = LogisticRegression(max_iter=2000, C=4.0)
        clf.fit(encode([r["state"] for r in tr]), [key_of(r) for r in tr])
        classes = list(clf.classes_)
        rows = [r for r in test if r["task"] == task]
        P = clf.predict_proba(encode([r["state"] for r in rows]))
        ok = 0
        for r, p in zip(rows, P):
            logits = [math.log(max(p[classes.index(o["key"])], 1e-9)) if o["key"] in classes else -30.0 for o in r["options"]]
            gold = LETTERS.index(r["answer"])
            ok += int(np.argmax(logits) == gold)
            results.append({"task": task, "gold": gold, "logits": logits})
        print(f"  {task:10s} train {len(tr):5d}  test {len(rows):4d}  acc {ok / len(rows):.3f}", flush=True)
    with open("results/embed_test.jsonl", "w") as f:
        for o in results:
            f.write(json.dumps(o) + "\n")
    print(f"== embed: {len(results)} тестовых за {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
