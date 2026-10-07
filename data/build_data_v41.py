"""Данные v4.1: v4 + исправление перестраховки + разнообразные задачи Open-Jev.

Изменения против v4:
  - синтетические регламенты: 45% фактов живым текстом, 15% списком, 40% полями (policies.FREE_TEXT);
  - «передать человеку»: пропуск поля 3% (было 8%), доля таких ответов 4% (было 10%);
  - +12 тыс. задач Open-Jev community-hard-mix (CC0 / CC BY 4.0): письма, счета, инциденты, длинные документы;
    без игр и без источников, пересекающихся с нейтральным тестом (Banking77).
Утечки проверяются против теста v4 и всего нейтрального набора (test, calib, realistic).
Английская часть (рецепт tev1) добавляется на сервере: add_english.py data_v41.
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from datasets import load_dataset

import build_data as v1
from build_data_v3 import take
from build_data_v4 import H, cedr, kinopoisk, mera, scibench
from policies import FAMILIES, build, build_pair, CALC, CALC_KEY

ROOT = Path(__file__).parent
OUT = ROOT / "data_v41"
SKIP = ("snake", "tic_tac_toe", "vizdoom", "tile_platformer", "painting-geometry", "drone", "banking")


def answer_key(r):
    return next(o["key"] for o in r["options"] if o["label"] == r["answer"])


def balanced(fam, rng, split, n, none_share=0.04):
    keys = list(dict.fromkeys([k for k, *_ in fam.rules] + [fam.default[0]] + ([CALC_KEY] if fam.name in CALC else [])))
    n_none = round(n * none_share)
    quota = {k: (n - n_none) // len(keys) for k in keys}
    quota["none"] = n_none
    got, extra = {k: [] for k in quota}, []
    for _ in range(n * 400):
        r = build(fam, rng, split, make=v1.make)
        k = answer_key(r)
        if len(got[k]) < quota[k]:
            got[k].append(r)
        elif k != "none":
            extra.append(r)
        if all(len(got[k]) >= quota[k] for k in quota):
            break
    rows = [r for v in got.values() for r in v]
    rng.shuffle(extra)
    return rows + extra[: n - len(rows)]


def openjev(rng, n_train=12000, n_val=300):
    out = {"train": [], "val": []}
    for split, n, src_split in [("train", n_train, "train"), ("val", n_val, "validation")]:
        d = load_dataset("ZefanCai/Open-Jev-v1.1", "community-hard-mix-v2-redistributable", split=src_split).shuffle(seed=9)
        per_src = Counter()
        cap = {"wanli-decisions-v1": n // 4}
        for ex in d:
            src = ex["source"]
            if any(s in src for s in SKIP):
                continue
            if per_src[src] >= cap.get(src, max(1, n // 40)):
                continue
            opts = ex["options"] if isinstance(ex["options"], list) else json.loads(ex["options"].replace("'", '"'))
            tgt = ex["target"] if isinstance(ex["target"], list) else json.loads(ex["target"])
            if not opts or len(opts) != len(tgt) or len(opts) > 24 or max(tgt) < 0.6:
                continue
            gold = opts[max(range(len(tgt)), key=tgt.__getitem__)]
            row = v1.make("en_openjev", split, ex["state_json"], ex["question"], [(str(o), str(o)) for o in opts], gold, rng,
                          shuffle=False)
            row["style"] = "tev1"
            row["max_chars"] = 6000
            out[split].append(row)
            per_src[src] += 1
            if len(out[split]) >= n:
                break
    return out["train"], out["val"]


def main():
    rng = random.Random(41)
    train, val = [], []
    for b in [v1.build_terra, v1.build_rcb, v1.build_danetqa]:
        rows = list(b(rng))
        train += take(rows, "train")
        val += take(rows, "val")
    for builder, n in [(v1.build_reviews, 12000), (v1.build_headlines, 6000), (v1.build_massive, 6000)]:
        rows = list(builder(rng, n_train=n))
        train += take(rows, "train")
        val += take(rows, "val")
    for g in [v1.gen_procurement, v1.gen_expense]:
        train += [g(rng, "train") for _ in range(1000)]
    for fam in FAMILIES:
        train += balanced(fam, rng, "train", 450)
        pairs = []
        while len(pairs) < 300:
            pairs += build_pair(fam, rng, "train", p_checks=0.4, make=v1.make)
        train += pairs[:300]
        val += balanced(fam, rng, "val", 30)
    for name, fn in [("kinopoisk", kinopoisk), ("scibench", scibench), ("cedr", cedr), ("mera", mera)]:
        tr, va, _ = fn(rng)
        train += tr
        val += va
        print(f"  {name}: train {len(tr)}, val {len(va)}")
    tr, va = openjev(rng)
    train += tr
    val += va
    print(f"  open-jev: train {len(tr)}, val {len(va)}")

    # утечки: против теста v4 и всего нейтрального набора
    synth = {f.name for f in FAMILIES} | {"procurement", "expense"}
    key = lambda r: (hashlib.md5((str(r["state"]) + r["question"]).encode()).hexdigest() if r["task"] in synth
                     else H(str(r["state"]) + " " + r["question"]))
    banned = set()
    for p in [ROOT / "data_v4" / "test.jsonl", ROOT / "data_ood" / "test.jsonl", ROOT / "data_ood" / "calib.jsonl",
              ROOT / "data_ood" / "realistic.jsonl"]:
        banned |= {key(json.loads(l)) for l in open(p, encoding="utf-8")}
    before = len(train)
    train = [r for r in train if key(r) not in banned]
    val = [r for r in val if key(r) not in banned]
    print(f"удалено из train совпадений с тестами: {before - len(train)}")

    OUT.mkdir(exist_ok=True)
    for name, part in [("train", train), ("val", val)]:
        rng.shuffle(part)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(name, len(part))
    none_share = sum(answer_key(r) == "none" for r in train if r["task"] in synth) / max(1, sum(r["task"] in synth for r in train))
    print(f"доля «передать человеку» в синтетике: {none_share:.1%}")


if __name__ == "__main__":
    main()
