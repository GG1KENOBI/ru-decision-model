#!/bin/bash
# Сборка всех наборов по порядку. Каждый следующий проверяет утечки против тестов предыдущих,
# поэтому порядок важен. Нужен интернет (Hugging Face datasets), ~15–30 минут на CPU.
#   TEV1_DIR     — путь к tev1/data (git clone https://github.com/togethercomputer/tev1, их README собирает records)
#   JEVBENCH_DIR — путь к jevbench/datasets/public (git clone https://github.com/fstandhartinger/jevbench)
set -e
cd "$(dirname "$0")"
python build_data.py        # v1: открытые русские наборы + первые регламенты      -> data/
python build_data_v2.py     # v2: 24 типа регламентов                              -> data_v2/
python build_data_v3.py     # v3: 5 стилей записи, контрастные пары                -> data_v3/
python build_data_v4.py     # v4: + Кинопоиск, RuSciBench, CEDR, MERA, «нужен расчёт» -> data_v4/
python build_ood.py         # нейтральный тест: 8 русских наборов + JevBench        -> data_ood/test, calib
python realistic_set.py     # 260 реалистичных задач, написанных вручную            -> data_ood/realistic
python build_data_v41.py    # итоговые данные: живой текст, Open-Jev, проверка утечек -> data_v41/
EN_SCALE=0.7 python add_english.py data_v41   # английская часть по рецепту tev1 (20,3 тыс.)
cp data_v4/test.jsonl data_v41/test.jsonl
echo "Готово: data_v41/train.jsonl, val.jsonl, test.jsonl, test_en.jsonl; data_ood/*"
