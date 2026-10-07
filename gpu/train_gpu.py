"""LoRA-дообучение Qwen3.5-4B как модели решений (CUDA, transformers + peft).

Потери: кросс-энтропия по буквам вариантов в последней позиции промпта (модель решений)
+ lm_weight × обычная кросс-энтропия по всему словарю на тот же токен (чтобы generate() отвечал буквой).
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from common import gold_index, letter_ids, load_rows, render


def batches(rows, bs, rng, max_tokens=6144):
    """Бакеты по длине; батч ограничен и числом примеров, и числом токенов (длинные английские политики)."""
    rows = sorted(rows, key=lambda r: len(r["_ids"]) + rng.random() * 40)
    chunks, cur = [], []
    for r in rows:
        width = max([len(x["_ids"]) for x in cur] + [len(r["_ids"])])
        if cur and (len(cur) >= bs or width * (len(cur) + 1) > max_tokens):
            chunks.append(cur)
            cur = []
        cur.append(r)
    if cur:
        chunks.append(cur)
    rng.shuffle(chunks)
    return chunks


def collate(chunk, pad, device):
    width = max(len(r["_ids"]) for r in chunk)
    ids = torch.full((len(chunk), width), pad, dtype=torch.long)
    att = torch.zeros((len(chunk), width), dtype=torch.long)
    for i, r in enumerate(chunk):
        ids[i, :len(r["_ids"])] = torch.tensor(r["_ids"])
        att[i, :len(r["_ids"])] = 1
    last = torch.tensor([len(r["_ids"]) - 1 for r in chunk])
    n_opt = torch.tensor([len(r["options"]) for r in chunk])
    gold = torch.tensor([gold_index(r) for r in chunk])
    return ids.to(device), att.to(device), last.to(device), n_opt.to(device), gold.to(device)


def option_logits(model, ids, att, last, n_opt, lids):
    """Логиты словаря только в последней позиции промпта (без логитов по всей последовательности)."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    h = base.model(input_ids=ids, attention_mask=att).last_hidden_state
    h = h[torch.arange(ids.shape[0], device=ids.device), last]
    full = base.lm_head(h).float()
    sub = full[:, lids]
    mask = torch.arange(len(lids), device=ids.device)[None, :] < n_opt[:, None]
    return sub.masked_fill(~mask, -1e9), full


@torch.no_grad()
def evaluate(model, rows, lids, pad, device, bs=32):
    model.eval()
    correct = 0
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids, att, last, n_opt, gold = collate(chunk, pad, device)
        sub, _ = option_logits(model, ids, att, last, n_opt, lids)
        correct += (sub.argmax(-1) == gold).sum().item()
    model.train()
    return correct / len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--data", default="data_v2")
    ap.add_argument("--out", default="out/v2")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lm-weight", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=400)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--init-adapter", help="продолжить с сохранённого адаптера")
    ap.add_argument("--start-step", type=int, default=0, help="с какого шага продолжать (батчи и расписание lr прокручиваются)")
    a = ap.parse_args()

    torch.manual_seed(0)
    rng = random.Random(0)
    device = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    lids = letter_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    cfg = LoraConfig(r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, target_modules="all-linear", task_type="CAUSAL_LM")
    if a.init_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, a.init_adapter, is_trainable=True)
    else:
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    train = load_rows(Path(a.data) / "train.jsonl")[: a.limit]
    val = load_rows(Path(a.data) / "val.jsonl")
    for r in train + val:
        r["_ids"] = render(tok, r)
    print(f"train {len(train)}, val {len(val)}, средняя длина {sum(len(r['_ids']) for r in train) / len(train):.0f} токенов",
          flush=True)

    steps = math.ceil(len(batches(train, a.bs, random.Random(1))) * a.epochs)
    total = min(steps, a.max_steps) if a.max_steps else steps
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(opt, int(0.03 * total), total)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    best, step, t0, run, skipped = -1.0, 0, time.time(), [], 0
    for _ in range(a.start_step):
        sched.step()
    model.train()
    while step < total:
        for chunk in batches(train, a.bs, rng):
            if step >= total:
                break
            if step < a.start_step:  # прокрутка уже пройденных батчей без вычислений
                step += 1
                if step == a.start_step:
                    t0 = time.time() - 1e-9
                    best = evaluate(model, val, lids, pad, device)
                    print(f"продолжаем с шага {step}, val acc {best:.4f}", flush=True)
                continue
            try:
                ids, att, last, n_opt, gold = collate(chunk, pad, device)
                sub, full = option_logits(model, ids, att, last, n_opt, lids)
                gold_tok = torch.tensor(lids, device=device)[gold]
                loss = F.cross_entropy(sub, gold) + a.lm_weight * F.cross_entropy(full, gold_tok)
                loss.backward()
            except torch.OutOfMemoryError:  # редкий слишком тяжёлый батч — пропускаем, а не падаем
                ids = att = sub = full = loss = None
                opt.zero_grad(set_to_none=True); torch.cuda.empty_cache()
                skipped += 1
                print(f"  OOM на шаге {step}: батч пропущен (всего {skipped})", flush=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            run.append(loss.item())
            step += 1
            if step % 25 == 0:
                el = time.time() - t0
                done = step - a.start_step
                print(f"шаг {step}/{total} loss {sum(run) / len(run):.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{el / done:.2f} с/шаг, осталось ~{el / done * (total - step) / 60:.0f} мин, "
                      f"VRAM пик {torch.cuda.max_memory_allocated() / 1e9:.1f} ГБ", flush=True)
                run = []
            if step % 400 == 0:
                model.save_pretrained(out / "last")
            if step % a.eval_every == 0 or step == total:
                acc = evaluate(model, val, lids, pad, device)
                print(f"  val acc ({len(val)}): {acc:.4f}{'  ← лучший' if acc > best else ''}", flush=True)
                if acc > best:
                    best = acc
                    model.save_pretrained(out / "best")
                    (out / "best" / "train_state.json").write_text(json.dumps({"step": step, "val_acc": acc, **vars(a)}))
    model.save_pretrained(out / "last")
    print(f"Готово за {(time.time() - t0) / 60:.1f} мин, лучший val acc {best:.4f}", flush=True)


if __name__ == "__main__":
    main()
