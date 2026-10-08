"""razvilka -> наш формат (state, question, options с буквами, answer). Типы choice / noul / score / ranking."""
import json, re, sys
from pathlib import Path
L = "ABCDEFGHIJKLMNOPQRSTUVWX"
src, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(exist_ok=True)
rows, skipped = [], 0
for line in open(src, encoding="utf-8"):
    it = json.loads(line); req = json.loads(it["request"]); gold = json.loads(it["gold"])
    (qid, q), = req["questions"].items()
    t, crit = q["type"], q["criteria"]
    if t in ("choice", "ranking"):
        items = list(crit.items()); keys = [k for k, _ in items]; descs = [str(v) for _, v in items]
        g = keys.index(str(gold))
    elif t == "noul":
        keys = ["true", "false"]
        descs = [("Да. " + crit["true"]) if isinstance(crit, dict) else "Да.", ("Нет. " + crit["false"]) if isinstance(crit, dict) else "Нет."]
        g = 0 if gold in (True, "true", "True") else 1
    elif t == "score":
        levels = crit if isinstance(crit, list) else list(crit.values())
        keys = [str(i) for i in range(len(levels))]; descs = [str(x) for x in levels]
        g = int(gold)
    else:
        skipped += 1; continue
    if len(keys) > len(L): skipped += 1; continue
    st = req["state"] if isinstance(req["state"], str) else json.dumps(req["state"], ensure_ascii=False)
    rows.append({"task": "razv_" + it["task"], "id": it["id"], "type": t, "split": "test", "state": st,
                 "question": q["instructions"], "max_chars": 12000,
                 "options": [{"label": L[i], "key": k, "description": d} for i, (k, d) in enumerate(zip(keys, descs))],
                 "answer": L[g], "_request": it["request"], "_gold": it["gold"]})
with open(out / "test.jsonl", "w", encoding="utf-8") as f:
    for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(rows), "skipped", skipped)
