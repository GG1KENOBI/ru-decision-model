"""Данные v2: больше открытых данных + 23 семейства политик.

Тест v1 (data/test.jsonl) сохраняется целиком для сравнения один к одному и дополняется:
  - <семейство>        — новые политики, отложенные примеры (то же распределение, что в обучении);
  - tx_<политика>      — перенос: политики, которых нет в обучении;
  - tx_<политика>+код  — те же примеры, но с условиями, заранее вычисленными кодом.
"""
import json
import random
from pathlib import Path

import build_data as v1
from policies import FAMILIES, TRANSFER, build

OUT = Path(__file__).parent / "data_v2"


def take(gen, split):
    return [r for r in gen if r["split"] == split]


def answer_key(r):
    return next(o["key"] for o in r["options"] if o["label"] == r["answer"])


def balanced(fam, rng, split, n, none_share=0.1, **kw):
    """Равные доли исходов (у «передать человеку» — none_share): отбор из большого потока примеров."""
    keys = [k for k, *_ in fam.rules] + [fam.default[0]]
    keys = list(dict.fromkeys(keys))
    n_none = round(n * none_share)
    quota = {k: (n - n_none) // len(keys) for k in keys}
    quota["none"] = n_none
    got, extra = {k: [] for k in quota}, []
    for _ in range(n * 400):
        r = build(fam, rng, split, make=v1.make, **kw)
        k = answer_key(r)
        if len(got[k]) < quota[k]:
            got[k].append(r)
        elif k != "none":
            extra.append(r)
        if all(len(got[k]) >= quota[k] for k in quota):
            break
    rows = [r for v in got.values() for r in v]
    rng.shuffle(extra)
    rows += extra[: n - len(rows)]
    return rows


def main():
    rng = random.Random(7)
    train, val = [], []

    # открытые датасеты: обучающие сплиты целиком/крупнее, val как в v1
    for b in [v1.build_terra, v1.build_rcb, v1.build_danetqa]:
        rows = list(b(rng))
        train += take(rows, "train")
        val += take(rows, "val")

    for builder, n in [(v1.build_reviews, 6000), (v1.build_headlines, 4000), (v1.build_massive, 4000)]:
        rows = list(builder(rng, n_train=n))
        train += take(rows, "train")
        val += take(rows, "val")

    for g in [v1.gen_procurement, v1.gen_expense, v1.gen_routing]:
        train += [g(rng, "train") for _ in range(1500)]
        val += [g(rng, "val") for _ in range(50)]

    for fam in FAMILIES:
        train += balanced(fam, rng, "train", 650)
        val += balanced(fam, rng, "val", 30)

    # тест: v1 целиком + новые части
    test = [json.loads(l) for l in open(Path(__file__).parent / "data" / "test.jsonl", encoding="utf-8")]
    trng = random.Random(1234)
    for fam in FAMILIES:
        test += balanced(fam, trng, "test", 60)
    for fam in TRANSFER:
        for i in range(200):
            seed = 10_000 * len(fam.name) + i
            raw = build(fam, random.Random(seed), "test", checks=False, make=v1.make)
            chk = build(fam, random.Random(seed), "test", checks=True, make=v1.make)
            raw["task"] = "tx_" + fam.name[2:]
            chk["task"] = "tx_" + fam.name[2:] + "+код"
            test += [raw, chk]

    OUT.mkdir(exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rng.shuffle(part)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {}
        for r in part:
            by[r["task"]] = by.get(r["task"], 0) + 1
        print(name, len(part), dict(sorted(by.items())))

    # баланс ответов в синтетике: ни один исход не должен доминировать
    for fam in FAMILIES:
        keys = {}
        for r in train:
            if r["task"] == fam.name:
                k = answer_key(r)
                keys[k] = keys.get(k, 0) + 1
        top = max(keys.values()) / sum(keys.values())
        flag = "  <-- перекос" if top > 0.5 else ""
        print(f"  {fam.name:12s} {dict(sorted(keys.items(), key=lambda x: -x[1]))}{flag}")


if __name__ == "__main__":
    main()
