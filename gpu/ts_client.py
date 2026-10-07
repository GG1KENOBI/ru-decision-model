"""Клиент протокола Jev (/v1/systemone): отправляет наши задачи на локальный сервер чужой модели (Plumb, Imajev, ...).

Каждая задача — один вопрос типа choice: criteria = {«<буква>_<ключ>»: описание}. Ответ — вероятности по ключам.
Отказ сервера (слишком длинный текст и т. п.) = равномерное распределение: засчитывается как угадывание.
Результат — в общем формате (task, gold, logits=log p в порядке наших вариантов).
"""
import argparse
import json
import math
import time
import urllib.error
import urllib.request
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"


def ask(url, row, model, timeout=120):
    crit = {f"{o['label']}_{o['key']}"[:60]: (o["description"] or o["key"]) for o in row["options"]}
    st = row["state"] if isinstance(row["state"], str) else json.dumps(row["state"], ensure_ascii=False)
    body = {"model": model, "state": st[:row.get("max_chars", 3500)],
            "questions": {"q": {"type": "choice", "instructions": row["question"], "criteria": crit}}}
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json", "authorization": "Bearer local"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        ans = json.loads(resp.read())["answers"]["q"]
    p = {str(k): float(v) for k, v in ans["probabilities"].items()}
    return [math.log(max(p.get(k, 0.0), 1e-9)) for k in crit]


def wait_ready(url, row, model, limit=1800):
    t0 = time.time()
    while time.time() - t0 < limit:
        try:
            ask(url, row, model, timeout=60)
            return True
        except Exception:
            time.sleep(10)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8090/v1/systemone")
    ap.add_argument("--model", default="local")
    ap.add_argument("--data", default="data_ood")
    ap.add_argument("--splits", default="calib,test,realistic")
    a = ap.parse_args()
    first = json.loads(open(Path(a.data) / "calib.jsonl", encoding="utf-8").readline())
    if not wait_ready(a.url, first, a.model):
        raise SystemExit("сервер не поднялся")
    for split in a.splits.split(","):
        rows = [json.loads(l) for l in open(Path(a.data) / f"{split}.jsonl", encoding="utf-8")]
        out, fails, t0, lat = [], 0, time.time(), []
        for r in rows:
            try:
                t = time.time()
                logits = ask(a.url, r, a.model)
                lat.append((time.time() - t) * 1000)
            except Exception:
                fails += 1
                logits = [0.0] * len(r["options"])
            out.append({"task": r["task"], "gold": LETTERS.index(r["answer"]), "logits": logits})
        Path("results").mkdir(exist_ok=True)
        with open(f"results/{a.name}_{split}.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        acc = sum(max(range(len(o["logits"])), key=o["logits"].__getitem__) == o["gold"] for o in out) / len(out)
        lat.sort()
        print(f"== {a.name} / {split}: {len(out)} за {time.time() - t0:.0f} с, acc {acc:.4f}, отказов {fails}, "
              f"медиана {lat[len(lat) // 2] if lat else 0:.0f} мс", flush=True)


if __name__ == "__main__":
    main()
