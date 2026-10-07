#!/bin/bash
# Обучение и оценка ru-decision-4b на одной RTX 4090 (24 ГБ): ~5,5 ч обучения + ~15 мин оценки.
set -e
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # без этого у нас на шаге ~3200 кончилась память
python -u train_gpu.py --data ../data/data_v41 --epochs 1 --eval-every 800 --out out/ru-decision-4b > train.log 2>&1
# если обучение упало, продолжить с последней сохранённой точки:
#   python -u train_gpu.py --data ../data/data_v41 --epochs 1 --eval-every 800 --out out/ru-decision-4b-resume \
#       --init-adapter out/ru-decision-4b/best --start-step <шаг из best/train_state.json>
python -u eval_gpu.py --name v41b_4b --adapter out/ru-decision-4b/best --data ../data/data_v41 --splits val,test,test_en --tag _v4set
python -u eval_gpu.py --name v41b_4b --adapter out/ru-decision-4b/best --data ../data/data_ood --splits calib,test,realistic
python cmp_ood.py
