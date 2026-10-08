"""Нейтральное сравнение: задачи, которых не видела ни одна модель + реалистичный набор автора + JevBench public.

Температура каждой модели подбирается на ОБЩЕЙ калибровочной выборке data_ood/calib.jsonl и применяется к тесту.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path("results")
MODELS = [("base08b", "Qwen3.5-0.8B"), ("tev1_08b", "tev1-0.8B"), ("v4_08b", "НАША v4-0.8B"),
          ("base4b", "Qwen3.5-4B"), ("tev1_4b", "tev1-4B"), ("kev4b", "Kev-4B"), ("plumb4b", "Plumb-4B"),
          ("imajev4b", "Imajev-4B"), ("v3", "НАША v3-4B"), ("v4_4b", "НАША v4-4B"), ("v41_4b", "v4.1, треть обучения"), ("v41b_4b", "ru-decision-4b"), ("frida", "FRIDA-Decisions"),
          ("yandexgpt5", "YandexGPT-5-Lite-8B"), ("gigachat31", "GigaChat3.1-10B")]
RU = {"ood_banking77_ru": "Banking77 (обращения в банк)", "ood_gov_category": "Жалобы граждан: ведомство",
      "ood_gov_urgency": "Жалобы граждан: срочность", "ood_toxicity": "Токсичность в поддержке",
      "ood_georeview": "Georeview: оценка 1–5", "ood_rucola": "RuCoLA: корректность", "ood_rwsd": "RWSD: местоимения",
      "ood_hatespeech": "ruHateSpeech"}
JB = {"jevbench_original": "JevBench Original", "jevbench_easy": "JevBench Easy", "jevbench_hard": "JevBench Hard"}


def load(n):
    p = R / n
    return [json.loads(l) for l in open(p)] if p.exists() else None


def probs(rows, T):
    out = []
    for r in rows:
        z = np.array(r["logits"], float) / T
        e = np.exp(z - z.max())
        out.append(e / e.sum())
    return out


def fit_T(rows):
    grid = np.exp(np.linspace(np.log(0.2), np.log(20), 200))
    return float(min(grid, key=lambda T: -np.mean([np.log(p[r["gold"]] + 1e-12) for p, r in zip(probs(rows, T), rows)])))


def ece(conf, ok, bins=10):
    tot = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return tot


def coverage(ok, conf, target=0.95):
    o = np.argsort(-conf)
    acc = np.cumsum(ok[o]) / np.arange(1, len(ok) + 1)
    g = np.where(acc >= target)[0]
    return (g.max() + 1) / len(ok) if len(g) else 0.0


def main():
    res, present = {}, []
    for key, title in MODELS:
        test, calib, real = load(f"{key}_test.jsonl"), load(f"{key}_calib.jsonl"), load(f"{key}_realistic.jsonl")
        if not test:
            continue
        present.append((key, title))
        T = fit_T(calib) if calib else 1.0
        d = {"T": T}
        for part, rows in [("test", test), ("real", real or [])]:
            P = probs(rows, T)
            ok = np.array([p.argmax() == r["gold"] for p, r in zip(P, rows)])
            conf = np.array([p.max() for p in P])
            by = defaultdict(list)
            for r, o in zip(rows, ok):
                by[r["task"]].append(o)
            d[part] = (rows, ok, conf, {t: np.mean(v) * 100 for t, v in by.items()})
        res[key] = d

    cols = "".join(f"{t[:12]:>13s}" for _, t in present)
    print(f"{'':34s}{cols}")
    def row(label, vals):
        print(f"{label[:34]:34s}" + "".join(f"{(f'{v:.1f}' if v is not None else '—'):>13s}" for v in vals))
    print("— русские наборы, которых не видел никто —")
    for t, name in RU.items():
        row(name, [res[k]["test"][3].get(t) for k, _ in present])
    ru_mask = lambda k: np.array([r["task"] in RU for r in res[k]["test"][0]])
    row("СРЕДНЕЕ ПО РУССКИМ НАБОРАМ", [np.mean([res[k]["test"][3][t] for t in RU if t in res[k]["test"][3]]) for k, _ in present])
    print("— реалистичный набор автора (260) —")
    real_names = sorted({r["task"] for k, _ in present for r in res[k]["real"][0]})
    for t in real_names:
        row(t.replace("real_", ""), [res[k]["real"][3].get(t) for k, _ in present])
    row("РЕАЛИСТИЧНЫЙ: ИТОГ", [res[k]["real"][1].mean() * 100 if len(res[k]["real"][1]) else None for k, _ in present])
    print("— JevBench public (английский, 231) —")
    for t, name in JB.items():
        row(name, [res[k]["test"][3].get(t) for k, _ in present])
    jb = lambda k: [o for r, o in zip(res[k]["test"][0], res[k]["test"][1]) if r["task"] in JB]
    row("JevBench: всего 231", [np.mean(jb(k)) * 100 if jb(k) else None for k, _ in present])
    print("— уверенность (русские наборы + реалистичный, температура с общей калибровки) —")
    for metric in ["ECE", "автоматизация при 95%", "уверенные ошибки (p>0.9)"]:
        vals = []
        for k, _ in present:
            rows, ok, conf, _ = res[k]["test"]
            m = ru_mask(k)
            ok2 = np.concatenate([ok[m], res[k]["real"][1]]) if len(res[k]["real"][1]) else ok[m]
            cf2 = np.concatenate([conf[m], res[k]["real"][2]]) if len(res[k]["real"][1]) else conf[m]
            vals.append({"ECE": ece(cf2, ok2.astype(float)) * 100, "автоматизация при 95%": coverage(ok2, cf2) * 100,
                         "уверенные ошибки (p>0.9)": ((cf2 > 0.9) & ~ok2).mean() * 100}[metric])
        row(metric, vals)
    print("\nтемпература (общая калибровка):", {k: round(res[k]["T"], 2) for k, _ in present})
    json.dump({k: {"T": res[k]["T"], "test": res[k]["test"][3], "real": res[k]["real"][3]} for k, _ in present},
              open(R / "summary_ood.json", "w"), ensure_ascii=False, indent=1, default=float)


if __name__ == "__main__":
    main()
