"""Русская часть данных v3: больше открытых данных, 24 политики в 5 стилях записи + контрастные пары.

Тест v2 (data_v2/test.jsonl) переиспользуется целиком; добавляется tx_<политика>@стиль —
политики переноса в «трудных» стилях записи (2–4), которых v2 не видела.
Английская часть (рецепт tev1) добавляется на сервере скриптом gpu/add_english.py.
"""
import json
import random
from pathlib import Path

import build_data as v1
from policies import FAMILIES, TRANSFER, build, build_pair, decide, render_row, sample

OUT = Path(__file__).parent / "data_v3"


def answer_key(r):
    return next(o["key"] for o in r["options"] if o["label"] == r["answer"])


def balanced(fam, rng, split, n, none_share=0.1):
    from policies import CALC, CALC_KEY
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


def take(rows, split):
    return [r for r in rows if r["split"] == split]


def main():
    rng = random.Random(33)
    train, val = [], []
    for b in [v1.build_terra, v1.build_rcb, v1.build_danetqa]:
        rows = list(b(rng))
        train += take(rows, "train")
        val += take(rows, "val")
    for builder, n in [(v1.build_reviews, 12000), (v1.build_headlines, 6000), (v1.build_massive, 6000)]:
        rows = list(builder(rng, n_train=n))
        train += take(rows, "train")
        val += take(rows, "val")
    for g in [v1.gen_procurement, v1.gen_expense, v1.gen_routing]:
        train += [g(rng, "train") for _ in range(1000)]
        val += [g(rng, "val") for _ in range(50)]

    for fam in FAMILIES:
        train += balanced(fam, rng, "train", 450)
        pairs = []
        while len(pairs) < 300:
            pairs += build_pair(fam, rng, "train", make=v1.make)
        train += pairs[:300]
        val += balanced(fam, rng, "val", 30)

    test = [json.loads(l) for l in open(Path(__file__).parent / "data_v2" / "test.jsonl", encoding="utf-8")]
    trng = random.Random(4321)
    for fam in TRANSFER:
        for i in range(150):
            p, vals, missing = sample(fam, trng, p_missing=0.08)
            style_rng = random.Random(i)
            row = render_row(fam, trng, "test", p, vals, missing, False, v1.make)
            # перерисовать регламент в трудном стиле (2–4), сохранив поля и ответ
            from policies import render_policy
            hard = render_policy(fam, style_rng, p, style=2 + i % 3)
            head, _, tail = row["state"].partition("\n" + fam.entity + ":")
            row["state"] = hard + " Если в документе нет данных, нужных для проверки, — передать человеку.\n" + fam.entity + ":" + tail
            row["task"] = "tx_" + fam.name[2:] + "@стиль"
            test.append(row)

    OUT.mkdir(exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rng.shuffle(part)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {}
        for r in part:
            by[r["task"]] = by.get(r["task"], 0) + 1
        print(name, len(part), {k: v for k, v in sorted(by.items()) if not k in [f.name for f in FAMILIES]})


if __name__ == "__main__":
    main()
