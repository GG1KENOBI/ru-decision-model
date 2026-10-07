"""Нейтральный тест: задачи, которых не видела ни одна из сравниваемых моделей.

data_ood/test.jsonl  — тест (русские наборы + JevBench public 231, английский);
data_ood/calib.jsonl — общая калибровочная выборка (другие примеры тех же русских наборов) для подбора температуры.
Источники и лицензии: Banking77-ru (DeepPavlov), kz-gov-complaints (Apache-2.0), ru-support-toxicity (MIT),
Georeview (MIT), RuCoLA (Apache-2.0), MERA RWSD / ruHateSpeech (MIT), JevBench public (MIT).
"""
import ast
import json
import random
from pathlib import Path

from datasets import load_dataset

from build_data import make

OUT = Path(__file__).parent / "data_ood"
import os
# git clone https://github.com/fstandhartinger/jevbench — путь к datasets/public
JEVBENCH = Path(os.environ.get("JEVBENCH_DIR", Path(__file__).parent / "jevbench" / "datasets" / "public"))
NONE = ("none", "Ни один из вариантов не подходит.")

BANKING_RU = {
    "activate_my_card": "Активировать карту", "age_limit": "Возрастные ограничения", "apple_pay_or_google_pay": "Apple Pay или Google Pay",
    "atm_support": "Банкоматы", "automatic_top_up": "Автопополнение", "balance_not_updated_after_bank_transfer": "Баланс не обновился после перевода",
    "balance_not_updated_after_cheque_or_cash_deposit": "Баланс не обновился после внесения наличных или чека",
    "beneficiary_not_allowed": "Нельзя добавить получателя", "cancel_transfer": "Отменить перевод", "card_about_to_expire": "Срок карты истекает",
    "card_acceptance": "Где принимают карту", "card_arrival": "Карта ещё не пришла", "card_delivery_estimate": "Сроки доставки карты",
    "card_linking": "Привязать карту", "card_not_working": "Карта не работает", "card_payment_fee_charged": "Комиссия за оплату картой",
    "card_payment_not_recognised": "Незнакомая оплата картой", "card_payment_wrong_exchange_rate": "Неверный курс при оплате картой",
    "card_swallowed": "Банкомат забрал карту", "cash_withdrawal_charge": "Комиссия за снятие наличных",
    "cash_withdrawal_not_recognised": "Незнакомое снятие наличных", "change_pin": "Сменить ПИН-код", "compromised_card": "Карта скомпрометирована",
    "contactless_not_working": "Не работает бесконтактная оплата", "country_support": "Поддерживаемые страны",
    "declined_card_payment": "Отклонена оплата картой", "declined_cash_withdrawal": "Отклонено снятие наличных",
    "declined_transfer": "Отклонён перевод", "direct_debit_payment_not_recognised": "Незнакомое автосписание",
    "disposable_card_limits": "Лимиты одноразовой карты", "edit_personal_details": "Изменить личные данные",
    "exchange_charge": "Комиссия за обмен валюты", "exchange_rate": "Курс обмена", "exchange_via_app": "Обмен валюты в приложении",
    "extra_charge_on_statement": "Лишнее списание в выписке", "failed_transfer": "Перевод не прошёл",
    "fiat_currency_support": "Поддерживаемые валюты", "get_disposable_virtual_card": "Получить одноразовую виртуальную карту",
    "get_physical_card": "Получить пластиковую карту", "getting_spare_card": "Дополнительная карта",
    "getting_virtual_card": "Получить виртуальную карту", "lost_or_stolen_card": "Карта потеряна или украдена",
    "lost_or_stolen_phone": "Телефон потерян или украден", "order_physical_card": "Заказать пластиковую карту",
    "passcode_forgotten": "Забыт код доступа", "pending_card_payment": "Оплата картой в обработке",
    "pending_cash_withdrawal": "Снятие наличных в обработке", "pending_top_up": "Пополнение в обработке",
    "pending_transfer": "Перевод в обработке", "pin_blocked": "ПИН-код заблокирован", "receiving_money": "Получение денег",
    "Refund_not_showing_up": "Возврат не поступил", "request_refund": "Запросить возврат",
    "reverted_card_payment?": "Оплата картой отменена", "supported_cards_and_currencies": "Поддерживаемые карты и валюты",
    "terminate_account": "Закрыть счёт", "top_up_by_bank_transfer_charge": "Комиссия за пополнение переводом",
    "top_up_by_card_charge": "Комиссия за пополнение картой", "top_up_by_cash_or_cheque": "Пополнение наличными или чеком",
    "top_up_failed": "Пополнение не прошло", "top_up_limits": "Лимиты пополнения", "top_up_reverted": "Пополнение отменено",
    "topping_up_by_card": "Пополнение картой", "transaction_charged_twice": "Двойное списание",
    "transfer_fee_charged": "Комиссия за перевод", "transfer_into_account": "Перевод на свой счёт",
    "transfer_not_received_by_recipient": "Получатель не получил перевод", "transfer_timing": "Сроки перевода",
    "unable_to_verify_identity": "Не удаётся подтвердить личность", "verify_my_identity": "Подтвердить личность",
    "verify_source_of_funds": "Подтвердить источник средств", "verify_top_up": "Подтвердить пополнение",
    "virtual_card_not_working": "Не работает виртуальная карта", "visa_or_mastercard": "Visa или Mastercard",
    "why_verify_identity": "Зачем подтверждать личность", "wrong_amount_of_cash_received": "Выдана неверная сумма наличных",
    "wrong_exchange_rate_for_cash_withdrawal": "Неверный курс при снятии наличных",
}


def banking(rng):
    names = load_dataset("legacy-datasets/banking77", split="test").features["label"].names
    d = load_dataset("DeepPavlov/banking77_ru", split="test").shuffle(seed=7)
    rows = []
    for ex in d.select(range(600)):
        gold = names[ex["label"]]
        distr = rng.sample([n for n in names if n != gold], 5)
        drop = rng.random() < 0.1
        keys = distr + ([] if drop else [gold])
        opts = [(k, BANKING_RU[k] + ".") for k in keys] + [NONE]
        rows.append(make("ood_banking77_ru", "test", f"Сообщение клиента банка: {ex['utterance']}",
                         "Какая тема обращения?", opts, "none" if drop else gold, rng))
    return rows[:450], rows[450:]


def gov(rng):
    d = load_dataset("Adilbai/kz-gov-complaints-data-kz-ru", split="train").shuffle(seed=3)
    cats = sorted(set(d["category"]))
    cat_desc = {c: f"Ведомство: {c}." for c in cats}
    rows_c, rows_u = [], []
    for ex in d:
        rows_c.append(make("ood_gov_category", "test", f"Обращение гражданина: {ex['text_ru']}",
                           "В какое ведомство направить обращение?", [(c, cat_desc[c]) for c in cats], ex["category"], rng))
        rows_u.append(make("ood_gov_urgency", "test", f"Обращение гражданина: {ex['text_ru']}",
                           "Какова срочность обращения?",
                           [("низкая", "Низкая."), ("средняя", "Средняя."), ("высокая", "Высокая.")], ex["urgency"], rng, shuffle=False))
    return rows_c[:400] + rows_u[:300], rows_c[400:500] + rows_u[300:400]


def toxicity(rng):
    d = load_dataset("Nelera/ru-support-toxicity-detection", split="test").shuffle(seed=4)
    opts = [("1", "Да, сообщение токсичное (оскорбления, грубость, угрозы)."), ("0", "Нет, сообщение не токсичное.")]
    rows = [make("ood_toxicity", "test", f"Сообщение: {ex['text'][:1200]}", "Является ли сообщение токсичным?",
                 opts, str(ex["label"]), rng, shuffle=False) for ex in d.select(range(500))]
    return rows[:400], rows[400:]


def georeview(rng):
    d = load_dataset("mteb/GeoreviewClassification")
    opts = [(str(i), f"{i + 1} из 5 — {t}.") for i, t in enumerate(["очень плохо", "плохо", "средне", "хорошо", "отлично"])]
    mk = lambda ex: make("ood_georeview", "test", f"Отзыв о месте: {ex['text'][:1500]}", "Какую оценку от 1 до 5 поставил автор отзыва?",
                         opts, str(ex["label"]), rng, shuffle=False)
    return ([mk(e) for e in d["test"].shuffle(seed=5).select(range(400))],
            [mk(e) for e in d["validation"].shuffle(seed=5).select(range(100))])


def rucola(rng):
    d = load_dataset("RussianNLP/rucola", revision="refs/convert/parquet", split="validation").shuffle(seed=6)
    opts = [("1", "Да, предложение корректно."), ("0", "Нет, в предложении есть ошибка.")]
    rows = [make("ood_rucola", "test", f"Предложение: {ex['sentence']}",
                 "Является ли предложение грамматически и по смыслу корректным на русском языке?", opts, str(ex["label"]), rng,
                 shuffle=False) for ex in d.select(range(500))]
    return rows[:400], rows[400:]


def as_dict(x):
    return x if isinstance(x, dict) else ast.literal_eval(x)


def mera(rng):
    test, calib = [], []
    d = load_dataset("MERA-evaluation/MERA", "rwsd")
    def rw(ex):
        x = as_dict(ex["inputs"])
        return make("ood_rwsd", "test", f"Текст: {x['text']}",
                    f"Относится ли «{x['span2_text']}» к «{x['span1_text']}»?", [("Да", "Да."), ("Нет", "Нет.")], ex["outputs"], rng)
    test += [rw(e) for e in d["validation"]]
    calib += [rw(e) for e in d["train"].shuffle(seed=1).select(range(100))]
    d = load_dataset("MERA-evaluation/MERA", "ruhatespeech", split="test").shuffle(seed=2)
    def hs(ex):
        x = as_dict(ex["inputs"])
        return make("ood_hatespeech", "test", f"Реплика: {x['replica'][:1200]}",
                    f"Какой из ответов на реплику токсичен по отношению к группе «{x['target_group']}»?",
                    [("1", x["reply_1"][:600]), ("2", x["reply_2"][:600])], str(ex["outputs"]), rng)
    rows = [hs(e) for e in d]
    return test + rows[:215], calib + rows[215:]


def jevbench(rng):
    rows = []
    for tier in ["original", "easy", "hard"]:
        for r in (json.loads(l) for l in open(JEVBENCH / f"{tier}.jsonl")):
            q = r["question"]
            crit = q["criteria"] if isinstance(q["criteria"], (dict, list)) else ast.literal_eval(q["criteria"])
            if q["type"] == "noul":
                opts = [("yes", crit.get("true", "Yes.")), ("no", crit.get("false", "No."))]
                gold = str(r["expected"]).lower()
            elif q["type"] == "score":
                levels = crit if isinstance(crit, list) else list(crit.values())
                opts = [(str(i), str(t)) for i, t in enumerate(levels)]
                gold = str(r["expected"])
            else:
                opts = list(crit.items())
                gold = r["expected"]
            row = make(f"jevbench_{tier}", "test", r["state"], q["instructions"], opts, gold, rng, shuffle=False)
            row["style"] = "tev1"
            row["jb_id"] = r["id"]
            row["max_chars"] = 40000
            rows.append(row)
    return rows, []


def main():
    rng = random.Random(2026)
    test, calib = [], []
    for fn in [banking, gov, toxicity, georeview, rucola, mera, jevbench]:
        t, c = fn(rng)
        test += t
        calib += c
        print(f"  {fn.__name__}: тест {len(t)}, калибровка {len(c)}")
    OUT.mkdir(exist_ok=True)
    for name, part in [("test", test), ("calib", calib)]:
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {}
        for r in part:
            by[r["task"]] = by.get(r["task"], 0) + 1
        print(name, len(part), by)


if __name__ == "__main__":
    main()
