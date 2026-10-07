"""Данные v4: больше русских открытых датасетов + «сомнение при расчётах» в политиках + встроенная проверка утечек.

Тест: тест v3 целиком (для сравнения) + новые части:
  kinopoisk, grnti, oecd, cedr, parus, openbook, worldtree — открытые датасеты (отложенные сплиты);
  moderation — модерация (некоммерческая лицензия: только тест, в обучение не идёт);
  calc_credit / calc_credit+код — правило с процентом без готовой проверки и с ней.
Английская часть (рецепт tev1) добавляется отдельно: add_english.py data_v4.
"""
import ast
import hashlib
import json
import random
import re
from pathlib import Path

from datasets import load_dataset

import build_data as v1
from build_data_v3 import balanced, take
from policies import FAMILIES, TRANSFER, build, build_pair

OUT = Path(__file__).parent / "data_v4"
NONE_OPT = ("none", "Ни один из вариантов не подходит.")


def norm(s):
    return re.sub(r"[\W\d_]+", " ", s.lower()).strip()


def H(s):
    return hashlib.md5(norm(s).encode()).hexdigest()


def as_dict(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def topic_rows(name, ds, split, n, rng, state_prefix, question, n_distr=5):
    """Классификация по справочнику: верная рубрика + 5 случайных + «ни одна» (в 10% верную убираем)."""
    labels = sorted(set(ds["label_text"]))
    out = []
    for ex in ds.shuffle(seed=3).select(range(min(n, len(ds)))):
        gold = ex["label_text"]
        distr = rng.sample([l for l in labels if l != gold], n_distr)
        drop = rng.random() < 0.1
        keys = distr + ([] if drop else [gold])
        opts = [(k, k + ".") for k in keys] + [NONE_OPT]
        out.append(v1.make(name, split, f"{state_prefix}{ex['text'][:1500]}", question, opts, "none" if drop else gold, rng))
    return out


def kinopoisk(rng):
    d = load_dataset("ai-forever/kinopoisk-sentiment-classification")
    opts = [("Bad", "Негативная рецензия."), ("Neutral", "Нейтральная или смешанная."), ("Good", "Позитивная рецензия.")]
    mk = lambda ex, split: v1.make("kinopoisk", split, f"Рецензия зрителя: {ex['text'][:1500]}", "Какова общая оценка фильма в рецензии?",
                                   opts, ex["label_text"], rng, shuffle=False)
    tr = [mk(e, "train") for e in d["train"].shuffle(seed=1).select(range(6000))]
    va = [mk(e, "val") for e in d["validation"].shuffle(seed=1).select(range(100))]
    te = [mk(e, "test") for e in d["test"].shuffle(seed=1).select(range(400))]
    return tr, va, te


def scibench(rng):
    tr, va, te = [], [], []
    for name, repo in [("grnti", "ai-forever/ru-scibench-grnti-classification"), ("oecd", "ai-forever/ru-scibench-oecd-classification")]:
        d = load_dataset(repo)
        q = "К какой научной рубрике относится статья?"
        rows = topic_rows(name, d["train"], "train", 4100, rng, "Название и аннотация статьи: ", q)
        tr += rows[:4000]
        va += [dict(r, split="val") for r in rows[4000:4100]]
        te += topic_rows(name, d["test"], "test", 400, rng, "Название и аннотация статьи: ", q)
    return tr, va, te


CEDR = {0: "Радость.", 1: "Грусть.", 2: "Удивление.", 3: "Страх.", 4: "Злость."}


def cedr(rng):
    d = load_dataset("ai-forever/cedr-classification")
    opts = [(str(k), v) for k, v in CEDR.items()] + [("no", "Эмоция не выражена.")]

    def rows(ds, split, n):
        out = []
        for ex in ds.shuffle(seed=2):
            lab = ex["label"] if isinstance(ex["label"], list) else ast.literal_eval(ex["label"])
            if len(lab) > 1:
                continue
            gold = str(lab[0]) if lab else "no"
            out.append(v1.make("cedr", split, f"Фраза: {ex['text']}", "Какая эмоция выражена во фразе?", opts, gold, rng, shuffle=False))
            if len(out) >= n:
                break
        return out
    tr = rows(d["train"], "train", 3100)
    return tr[:3000], [dict(r, split="val") for r in tr[3000:]], rows(d["test"], "test", 300)


def mera(rng):
    tr, va, te = [], [], []
    d = load_dataset("MERA-evaluation/MERA", "parus")

    def parus(ex, split):
        x = as_dict(ex["inputs"])
        kind = "причиной" if as_dict(ex["meta"]).get("task") == "cause" else "следствием"
        return v1.make("parus", split, f"Ситуация: {x['premise']}", f"Что вероятнее является {kind} ситуации?",
                       [("1", x["choice1"]), ("2", x["choice2"])], str(ex["outputs"]), rng)
    tr += [parus(e, "train") for e in d["train"]]
    te += [parus(e, "test") for e in d["validation"]]

    d = load_dataset("MERA-evaluation/MERA", "ruopenbookqa")

    def mc(ex, split, task):
        x = as_dict(ex["inputs"])
        opts = [(L, x[f"option_{L.lower()}"]) for L in "ABCD"]
        return v1.make(task, split, "Вопрос на логику и общеизвестные факты.", x["question"], opts, ex["outputs"], rng)
    ob = [e for e in d["train"]]
    rng.shuffle(ob)
    tr += [mc(e, "train", "openbook") for e in ob[:2000]]
    te += [mc(e, "test", "openbook") for e in ob[2000:]]
    d = load_dataset("MERA-evaluation/MERA", "ruworldtree")
    te += [mc(e, "test", "worldtree") for e in d["train"]]  # маленький: целиком в тест
    return tr, va, te


def moderation(rng):
    d = load_dataset("ai-forever/inappropriateness-classification")
    opts = [("1.0", "Да, сообщение неуместно и может навредить репутации площадки."), ("0.0", "Нет, сообщение допустимо.")]
    return [v1.make("moderation", "test", f"Сообщение пользователя: {ex['text'][:1200]}",
                    "Является ли сообщение неуместным для публикации?", opts, ex["label_text"], rng, shuffle=False)
            for ex in d["test"].shuffle(seed=5).select(range(400))]


def optional(name, fn, rng):
    try:
        return fn(rng)
    except Exception as e:
        print(f"  {name}: пропущен ({str(e)[:120]})")
        return [], [], []


def main():
    rng = random.Random(44)
    train, val, test = [], [], []
    # база v3: открытые датасеты (те же сплиты) + старые генераторы + 24 семейства
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

    # новые русские датасеты
    for name, fn in [("kinopoisk", kinopoisk), ("scibench", scibench), ("cedr", cedr), ("mera", mera)]:
        tr, va, te = optional(name, fn, rng)
        train += tr
        val += va
        test += te
        print(f"  {name}: train {len(tr)}, val {len(va)}, test {len(te)}")
    test += moderation(rng)

    # тест «расчёт без готовой проверки» на отложенной кредитной политике
    fam = {f.name: f for f in TRANSFER}["t_credit"]
    for i in range(200):
        for checks, suffix in [(False, ""), (True, "+код")]:
            r = build(fam, random.Random(900_000 + i), "test", checks=checks, p_missing=0.0, make=v1.make)
            r["task"] = "calc_credit" + suffix
            test.append(r)

    # тест v3 целиком — для сравнения с прошлыми версиями
    test += [json.loads(l) for l in open(Path(__file__).parent / "data_v3" / "test.jsonl", encoding="utf-8")]

    # встроенная проверка утечек: убираем из train всё, что совпадает с тестом (текст без цифр)
    synth = {f.name for f in FAMILIES} | {"procurement", "expense"}
    key = lambda r: (hashlib.md5((r["state"] + r["question"]).encode()).hexdigest() if r["task"] in synth
                     else H(r["state"] + " " + r["question"]))
    test_h = {key(r) for r in test}
    before = len(train)
    train = [r for r in train if key(r) not in test_h]
    print(f"удалено из train совпадений с тестом: {before - len(train)}")

    OUT.mkdir(exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        rng.shuffle(part)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {}
        for r in part:
            by[r["task"]] = by.get(r["task"], 0) + 1
        print(name, len(part), {k: v for k, v in sorted(by.items()) if k not in {f.name for f in FAMILIES}})


if __name__ == "__main__":
    main()
