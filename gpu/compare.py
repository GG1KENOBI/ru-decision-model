"""Сводное сравнение моделей на тесте v2: по группам задач, калибровка (T на val), доля автоматизации."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

R = Path(sys.argv[1] if len(sys.argv) > 1 else "results")
MODELS = [m for m in ["base", "tev1", "tev1_ours", "v2", "v3"] if (R / f"{m}_test.jsonl").exists()]
NAT = {"terra", "rcb", "danetqa", "reviews", "headlines", "massive"}
OLD_SYN = {"procurement", "expense", "routing"}
OLD_TR = {"t_vacation", "t_access", "t_credit"}


def load(m, s):
    p = R / f"{m}_{s}.jsonl"
    return [json.loads(l) for l in open(p)] if p.exists() else None


def probs(rows, T):
    out = []
    for r in rows:
        z = np.array(r["logits"], dtype=float) / T
        e = np.exp(z - z.max())
        out.append(e / e.sum())
    return out


def fit_T(rows):
    grid = np.exp(np.linspace(np.log(0.2), np.log(20), 300))
    return float(min(grid, key=lambda T: -np.mean([np.log(p[r["gold"]] + 1e-12) for p, r in zip(probs(rows, T), rows)])))


def ece(conf, ok, bins=10):
    conf, ok = np.asarray(conf), np.asarray(ok, float)
    tot = 0.0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            tot += m.mean() * abs(conf[m].mean() - ok[m].mean())
    return tot


def cov(ok, conf, target):
    o = np.argsort(-np.asarray(conf))
    acc = np.cumsum(np.asarray(ok)[o]) / np.arange(1, len(ok) + 1)
    g = np.where(acc >= target)[0]
    return (g.max() + 1) / len(ok) if len(g) else 0.0


def group(task):
    if task in NAT:
        return "открытые датасеты"
    if task in OLD_SYN:
        return "закупки/авансы/обращения (v1)"
    if task in OLD_TR:
        return "перенос v1 (отпуск/доступ/кредит)"
    if task.startswith("tx_"):
        return "перенос v2 + проверки кодом" if task.endswith("+код") else "перенос v2 (сырые)"
    return "24 политики (отложенные)"


def main():
    res = {}
    for m in MODELS:
        test, val = load(m, "test"), load(m, "val")
        T = fit_T(val) if val else 1.0
        P = probs(test, T)
        ok = np.array([p.argmax() == r["gold"] for p, r in zip(P, test)])
        conf = np.array([p.max() for p in P])
        res[m] = dict(test=test, ok=ok, conf=conf, T=T)

    groups = sorted({group(r["task"]) for r in res[MODELS[0]]["test"]})
    print(f"{'группа':38s} {'n':>5s} " + " ".join(f"{m:>10s}" for m in MODELS))
    for g in groups + ["ВСЕГО"]:
        line = None
        for m in MODELS:
            idx = [i for i, r in enumerate(res[m]["test"]) if g == "ВСЕГО" or group(r["task"]) == g]
            if line is None:
                line = f"{g:38s} {len(idx):5d} "
            line += f" {res[m]['ok'][idx].mean():9.3f}"
        print(line)

    print("\nКалибровка и автоматизация (весь тест, температура подобрана на val):")
    for m in MODELS:
        r = res[m]
        print(f"  {m:10s} T={r['T']:.2f}  acc={r['ok'].mean():.3f}  ECE={ece(r['conf'], r['ok']):.3f}  "
              f"автоматизация при 95%: {cov(r['ok'], r['conf'], 0.95):.0%}, при 90%: {cov(r['ok'], r['conf'], 0.90):.0%}")

    print("\nПо задачам:")
    tasks = sorted({r["task"] for r in res[MODELS[0]]["test"]})
    print(f"{'задача':22s} " + " ".join(f"{m:>10s}" for m in MODELS))
    for t in tasks:
        line = f"{t:22s} "
        for m in MODELS:
            idx = [i for i, r in enumerate(res[m]["test"]) if r["task"] == t]
            line += f" {res[m]['ok'][idx].mean():9.3f}"
        print(line)


if __name__ == "__main__":
    main()
