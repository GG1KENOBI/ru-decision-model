"""Итоговое сравнение v4: все модели на чистых частях теста data_v4 + английский тест tev1.

Температура уверенности подбирается на val (если у модели есть val), иначе T=1.
Метрики: точность по группам, доля автоматизации при точности 95%, доля уверенных ошибок (p>0.9 и неверно).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("results")
MODELS = [("base08b", "Qwen3.5-0.8B (база)"), ("tev1_08b", "tev1-0.8B"), ("v4_08b", "наша v4-0.8B"),
          ("base4b", "Qwen3.5-4B (база)"), ("tev1_4b", "tev1-4B"), ("kev4b", "Kev-4B"), ("v3", "наша v3-4B"),
          ("v4_4b", "наша v4-4B"), ("yandexgpt5", "YandexGPT-5-Lite-8B"), ("gigachat31", "GigaChat3.1-10B-A1.8B"),
          ("embed", "эмбеддинги + классификатор")]
OLD_NAT = {"terra", "rcb", "danetqa", "reviews", "headlines", "massive"}
NEW_NAT = {"kinopoisk", "grnti", "oecd", "cedr", "parus", "openbook", "worldtree"}


def group(t):
    if t in OLD_NAT: return "Открытые русские датасеты (v1–v3)"
    if t in NEW_NAT: return "Новые русские датасеты"
    if t == "moderation": return "Модерация (тип не из обучения)"
    if t.endswith("@стиль"): return "Перенос: трудные стили записи"
    if t.endswith("+код") and t.startswith("tx_"): return "Перенос: проверки посчитаны кодом"
    if t.startswith("tx_"): return "Перенос: обычная запись"
    if t.startswith("t_"): return "Перенос: старый формат v1"
    if t == "calc_credit": return "Расчёт без проверки (верно = передать)"
    if t == "calc_credit+код": return "Тот же расчёт, посчитан кодом"
    return None  # синтетика из обучающего генератора и обращения с утечкой — не считаем


CLEAN = {"Открытые русские датасеты (v1–v3)", "Новые русские датасеты", "Модерация (тип не из обучения)",
         "Перенос: трудные стили записи", "Перенос: проверки посчитаны кодом", "Перенос: обычная запись",
         "Перенос: старый формат v1"}


def load(name):
    p = R / name
    return [json.loads(l) for l in open(p)] if p.exists() else None


def probs(rows, T=1.0):
    out = []
    for r in rows:
        z = np.array(r["logits"], dtype=float) / T
        e = np.exp(z - z.max())
        out.append(e / e.sum())
    return out


def fit_T(rows):
    if not rows: return 1.0
    grid = np.exp(np.linspace(np.log(0.2), np.log(20), 200))
    return float(min(grid, key=lambda T: -np.mean([np.log(p[r["gold"]] + 1e-12) for p, r in zip(probs(rows, T), rows)])))


def coverage(ok, conf, target=0.95):
    o = np.argsort(-conf)
    acc = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)
    g = np.where(acc >= target)[0]
    return (g.max() + 1) / len(ok) if len(g) else 0.0


def main():
    rows_out = {}
    groups = ["Открытые русские датасеты (v1–v3)", "Новые русские датасеты", "Модерация (тип не из обучения)",
              "Перенос: обычная запись", "Перенос: трудные стили записи", "Перенос: старый формат v1",
              "Перенос: проверки посчитаны кодом", "Расчёт без проверки (верно = передать)", "Тот же расчёт, посчитан кодом"]
    present = []
    for key, title in MODELS:
        test = load(f"{key}_test.jsonl")
        if test is None:
            continue
        present.append((key, title))
        T = fit_T(load(f"{key}_val.jsonl"))
        P = probs(test, T)
        ok = np.array([p.argmax() == r["gold"] for p, r in zip(P, test)])
        conf = np.array([p.max() for p in P])
        g = [group(r["task"]) for r in test]
        res = {"T": T}
        for gr in groups:
            idx = [i for i, x in enumerate(g) if x == gr]
            res[gr] = (ok[idx].mean() * 100, len(idx)) if idx else (None, 0)
        clean = np.array([x in CLEAN for x in g])
        res["ЧИСТЫЙ ИТОГ"] = (ok[clean].mean() * 100, int(clean.sum()))
        res["автоматизация 95%"] = coverage(ok[clean], conf[clean]) * 100
        res["уверенные ошибки"] = ((conf[clean] > 0.9) & ~ok[clean]).mean() * 100
        en = load(f"{key}_test_en.jsonl")
        if en:
            by = defaultdict(list)
            for r in en:
                by["en_transfer" if r["task"] == "en_transfer" else "main"].append(int(np.argmax(r["logits"]) == r["gold"]))
            res["EN основной (1000)"] = (sum(by["main"]), len(by["main"]))
            res["EN перенос (300)"] = (sum(by["en_transfer"]), len(by["en_transfer"]))
        rows_out[key] = res

    w = 34
    print(f"{'':{w}s}" + "".join(f"{k[:11]:>12s}" for k, _ in present))
    for gr in groups + ["ЧИСТЫЙ ИТОГ"]:
        n = max(rows_out[k][gr][1] for k, _ in present)
        line = f"{gr[:w - 6]:{w - 6}s}{n:6d}"
        for k, _ in present:
            v = rows_out[k][gr][0]
            line += f"{(f'{v:.1f}' if v is not None else '—'):>12s}"
        print(line)
    for m in ["автоматизация 95%", "уверенные ошибки"]:
        print(f"{m:{w}s}" + "".join(f"{rows_out[k][m]:11.1f}%" for k, _ in present))
    for m in ["EN основной (1000)", "EN перенос (300)"]:
        print(f"{m:{w}s}" + "".join(f"{(str(rows_out[k][m][0]) if m in rows_out[k] else '—'):>12s}" for k, _ in present))
    print("\nтемпература:", {k: round(rows_out[k]["T"], 2) for k, _ in present})
    print("модели:", {k: t for k, t in present})
    json.dump({k: {kk: (list(vv) if isinstance(vv, tuple) else vv) for kk, vv in v.items()} for k, v in rows_out.items()},
              open(R / "summary_v4.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
