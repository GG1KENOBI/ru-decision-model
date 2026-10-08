"""FRIDA-Decisions: razvilka (их запросы как есть) + наши тесты в нашем формате результатов (task, gold, logits=log p)."""
import json, math, sys, time
from pathlib import Path
from frida_decisions import Judge

J = Judge.from_pretrained("ai-forever/FRIDA-Decisions", device="cuda")
Path("results").mkdir(exist_ok=True)
L = "ABCDEFGHIJKLMNOPQRSTUVWX"


def probs_of(ans, keys):
    if ans.get("type") == "noul" and "noul" in ans:
        p = float(ans["noul"]); d = {"true": p, "false": 1 - p}
    else:
        d = {str(k): float(v) for k, v in ans["probabilities"].items()}
    return [math.log(max(d.get(k, 0.0), 1e-9)) for k in keys]


def run(rows, name, mk_request, keys_of, batch=16):
    out, t0 = [], time.time()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        reqs = [mk_request(r) for r in chunk]
        try:
            res = J.judge_batch(reqs)
        except Exception as e:
            res = []
            for q in reqs:
                try: res.append(J.judge(q))
                except Exception as e2: print("fail", str(e2)[:120], flush=True); res.append(None)
        for r, a in zip(chunk, res):
            keys = keys_of(r)
            lg = probs_of(next(iter(a["answers"].values())), keys) if a else [0.0] * len(keys)
            out.append({"task": r["task"], "gold": L.index(r["answer"]), "logits": lg, "id": r.get("id")})
    with open(f"results/{name}.jsonl", "w") as f:
        for o in out: f.write(json.dumps(o, ensure_ascii=False) + "\n")
    acc = sum(max(range(len(o["logits"])), key=o["logits"].__getitem__) == o["gold"] for o in out) / len(out)
    print(f"== {name}: {len(out)} за {time.time()-t0:.0f} с, acc {acc:.4f}", flush=True)


# 1) razvilka: запрос как есть
razv = [json.loads(l) for l in open("data_razv/test.jsonl", encoding="utf-8")]
run(razv, "frida_test_razv", lambda r: json.loads(r["_request"]), lambda r: [o["key"] for o in r["options"]])

# 2) наши тесты: один вопрос choice, критерии = описания вариантов
def ours(r):
    st = r["state"] if isinstance(r["state"], str) else json.dumps(r["state"], ensure_ascii=False)
    crit = {f"o{i}": (o.get("description") or o.get("key") or o["label"]) for i, o in enumerate(r["options"])}
    return {"state": st, "questions": {"q": {"type": "choice", "instructions": r["question"], "criteria": crit}}}
okeys = lambda r: [f"o{i}" for i in range(len(r["options"]))]
for data, split, name in [("data_ood", "calib", "frida_calib"), ("data_ood", "test", "frida_test"), ("data_ood", "realistic", "frida_realistic"),
                          ("data_v41", "val", "frida_val_v4set"), ("data_v41", "test", "frida_test_v4set"), ("data_v41", "test_en", "frida_test_en_v4set")]:
    rows = [json.loads(l) for l in open(f"{data}/{split}.jsonl", encoding="utf-8")]
    run(rows, name, ours, okeys)
print("=== FRIDA DONE", flush=True)
