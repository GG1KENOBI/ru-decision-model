"""Сливает LoRA-адаптер с базовой моделью и проверяет, что ответы совпадают с адаптерной версией."""
import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import letter_ids, load_rows
from train_gpu import collate, option_logits
from common import render


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--adapter", default="out/v2/best")
    ap.add_argument("--out", default="out/merged")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lids = letter_ids(tok)
    rows = load_rows("data_v2/test.jsonl")[:64]
    for r in rows:
        r["_ids"] = render(tok, r)

    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, a.adapter).eval()
    with torch.no_grad():
        ref, _ = option_logits(model, *collate(rows, pad, "cuda")[:4], lids)
    merged = model.merge_and_unload().eval()
    with torch.no_grad():
        got, _ = option_logits(merged, *collate(rows, pad, "cuda")[:4], lids)
    same = (ref.argmax(-1) == got.argmax(-1)).float().mean().item()
    print(f"Совпадение ответов адаптер vs слитая модель: {same:.3f}, макс. расхождение логитов "
          f"{(ref - got).abs().max().item():.3f}")
    assert same > 0.98, "слитая модель отвечает иначе — не публикуем"

    out = Path(a.out)
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(out)
    print("Сохранено в", out)


if __name__ == "__main__":
    main()
