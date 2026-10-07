"""Оценка на GPU: сохраняет логиты по вариантам в том же формате, что evaluate.py на Mac."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import LETTERS, gold_index, letter_ids, load_rows, render
from train_gpu import collate, option_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter")
    ap.add_argument("--style", default=None, choices=["ours", "tev1"], help="по умолчанию — стиль каждой записи")
    ap.add_argument("--data", default="data_v2")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--tag", default="", help="суффикс к имени файлов результатов")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--variants", action="store_true", help="складывать вероятности «A» и « A» (для моделей без дообучения)")
    ap.add_argument("--trust", action="store_true", help="trust_remote_code")
    ap.add_argument("--int8", action="store_true", help="загрузить в 8 бит (bitsandbytes) — запасной путь для больших моделей")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=a.trust)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lids = letter_ids(tok)
    extra = {}
    if a.int8:
        from transformers import BitsAndBytesConfig
        extra["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype="auto" if a.trust else torch.bfloat16, device_map={"": 0},
                                                 trust_remote_code=a.trust, **extra)
    var = []  # для каждой буквы — список её токенов: «A» и « A» (если « A» — один токен)
    for L, lid in zip(LETTERS, lids):
        sp = tok.encode(" " + L, add_special_tokens=False)
        var.append([lid] + (sp if a.variants and len(sp) == 1 and sp[0] != lid else []))
    if a.adapter:
        model = PeftModel.from_pretrained(model, a.adapter)
    model.eval()
    Path("results").mkdir(exist_ok=True)

    for split in a.splits.split(","):
        rows = load_rows(Path(a.data) / f"{split}.jsonl")
        for r in rows:
            r["_ids"] = render(tok, r, a.style)
        rows.sort(key=lambda r: len(r["_ids"]))
        out, t0 = [], time.time()
        with torch.no_grad():
            chunks, cur = [], []
            for r in rows:  # батч ограничен и числом задач, и суммой токенов (длинные документы JevBench)
                if cur and (len(cur) >= a.bs or (len(cur) + 1) * len(r["_ids"]) > 24000):
                    chunks.append(cur)
                    cur = []
                cur.append(r)
            if cur:
                chunks.append(cur)
            for chunk in chunks:
                ids, att, last, n_opt, gold = collate(chunk, pad, "cuda")
                sub, full = option_logits(model, ids, att, last, n_opt, lids)
                if a.variants:
                    lsm = torch.log_softmax(full, -1)
                    sub = torch.stack([torch.logsumexp(lsm[:, v], -1) for v in var], -1)
                for r, l in zip(chunk, sub.cpu().tolist()):
                    out.append({"task": r["task"], "gold": gold_index(r), "logits": l[:len(r["options"])]})
        sec = time.time() - t0
        with open(f"results/{a.name}_{split}{a.tag}.jsonl", "w") as f:
            for o in out:
                f.write(json.dumps(o) + "\n")
        acc = np.mean([int(np.argmax(o["logits"]) == o["gold"]) for o in out])
        print(f"== {a.name} / {split}: {len(out)} примеров за {sec:.0f} с, acc {acc:.4f}", flush=True)

    # задержка одного решения (батч 1)
    rows = load_rows(Path(a.data) / "test.jsonl")[:100]
    for r in rows:
        r["_ids"] = render(tok, r, a.style)
    ts = []
    with torch.no_grad():
        for r in rows:
            torch.cuda.synchronize(); t = time.time()
            ids, att, last, n_opt, _ = collate([r], pad, "cuda")
            option_logits(model, ids, att, last, n_opt, lids)
            torch.cuda.synchronize(); ts.append((time.time() - t) * 1000)
    print(f"Медианная задержка {a.name}: {np.median(ts[5:]):.0f} мс", flush=True)


if __name__ == "__main__":
    main()
