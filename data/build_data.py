"""Сборка обучающих и тестовых данных для русской модели решений.

Каждая запись: state (данные), question (вопрос), options (варианты A, B, ...),
answer (буква правильного варианта), task (источник), split (train/val/test).
"""
import json
import random
from pathlib import Path

from datasets import load_dataset

OUT = Path(__file__).parent / "data"
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWX"
NONE_DESC = "Недостаточно данных для решения — передать человеку."


def make(task, split, state, question, options, correct_key, rng, shuffle=True):
    """options: list of (key, description). Ставит буквы, при желании перемешивает."""
    opts = list(options)
    if shuffle:
        rng.shuffle(opts)
    out = [{"label": LETTERS[i], "key": k, "description": d} for i, (k, d) in enumerate(opts)]
    answer = next(o["label"] for o in out if o["key"] == correct_key)
    return {"task": task, "split": split, "state": state, "question": question,
            "options": out, "answer": answer}


def rub(x):
    return f"{x:,.0f}".replace(",", " ") + " ₽"


# ---------------------------------------------------------------- открытые датасеты

def superglue(cfg):
    return load_dataset("RussianNLP/russian_super_glue", revision="refs/convert/parquet", data_dir=cfg)


def build_terra(rng):
    d = superglue("terra")
    val = d["validation"].shuffle(seed=1)
    parts = [("train", d["train"]), ("val", val.select(range(100))), ("test", val.select(range(100, len(val))))]
    for split, ds in parts:
        for ex in ds:
            yield make("terra", split, f"Текст: {ex['premise']}",
                       f"Следует ли из текста утверждение: «{ex['hypothesis']}»?",
                       [("entailment", "Да, утверждение следует из текста."),
                        ("not_entailment", "Нет, из текста это не следует.")],
                       ["entailment", "not_entailment"][ex["label"]], rng)


def build_rcb(rng):
    d = superglue("rcb")
    val = d["validation"].shuffle(seed=1)
    parts = [("train", d["train"]), ("val", val.select(range(60))), ("test", val.select(range(60, len(val))))]
    for split, ds in parts:
        for ex in ds:
            yield make("rcb", split, f"Текст: {ex['premise']}",
                       f"Как утверждение «{ex['hypothesis']}» соотносится с текстом?",
                       [("entailment", "Следует из текста."),
                        ("contradiction", "Противоречит тексту."),
                        ("neutral", "Из текста нельзя определить.")],
                       ["entailment", "contradiction", "neutral"][ex["label"]], rng)


def build_danetqa(rng):
    d = superglue("danetqa")
    val = d["validation"].shuffle(seed=1)
    parts = [("train", d["train"]), ("val", val.select(range(150))), ("test", val.select(range(150, 550)))]
    for split, ds in parts:
        for ex in ds:
            yield make("danetqa", split, f"Справка: {ex['passage']}",
                       f"Ответьте по справке: {ex['question']}",
                       [("yes", "Да."), ("no", "Нет.")],
                       ["no", "yes"][ex["label"]], rng)


def build_reviews(rng, n_train=2500):
    d = load_dataset("ai-forever/ru-reviews-classification")
    parts = [("train", d["train"].shuffle(seed=1).select(range(n_train))),
             ("val", d["validation"].shuffle(seed=1).select(range(200))),
             ("test", d["test"].shuffle(seed=1).select(range(400)))]
    opts = [("negative", "Негативный: клиент недоволен."),
            ("neutral", "Нейтральный или смешанный."),
            ("positive", "Позитивный: клиент доволен.")]
    for split, ds in parts:
        for ex in ds:
            yield make("reviews", split, f"Отзыв покупателя: {ex['text']}",
                       "Какова общая тональность отзыва?", opts, ex["label_text"], rng, shuffle=False)


def build_headlines(rng, n_train=1500):
    d = load_dataset("ai-forever/headline-classification")
    parts = [("train", d["train"].shuffle(seed=1).select(range(n_train))),
             ("val", d["validation"].shuffle(seed=1).select(range(200))),
             ("test", d["test"].shuffle(seed=1).select(range(400)))]
    topics = ["культура", "наука", "политика", "происшествия", "спорт", "экономика"]
    for split, ds in parts:
        for ex in ds:
            yield make("headlines", split, f"Заголовок новости: {ex['text']}",
                       "К какой рубрике относится новость?",
                       [(t, t.capitalize() + ".") for t in topics], ex["label_text"], rng)


INTENTS = {
    "alarm_query": "Узнать, какие будильники установлены.", "alarm_remove": "Удалить будильник.",
    "alarm_set": "Поставить будильник.", "audio_volume_down": "Сделать тише.",
    "audio_volume_mute": "Выключить звук.", "audio_volume_other": "Другая настройка звука.",
    "audio_volume_up": "Сделать громче.", "calendar_query": "Узнать о событиях в календаре.",
    "calendar_remove": "Удалить событие из календаря.", "calendar_set": "Добавить событие или напоминание.",
    "cooking_query": "Вопрос о готовке.", "cooking_recipe": "Найти рецепт.",
    "datetime_convert": "Перевести время между часовыми поясами.", "datetime_query": "Узнать дату или время.",
    "email_addcontact": "Добавить контакт.", "email_query": "Проверить почту.",
    "email_querycontact": "Найти данные контакта.", "email_sendemail": "Отправить письмо.",
    "general_greet": "Приветствие.", "general_joke": "Попросить шутку.",
    "general_quirky": "Странная или отвлечённая реплика.", "iot_cleaning": "Включить уборку (робот-пылесос).",
    "iot_coffee": "Сварить кофе.", "iot_hue_lightchange": "Изменить цвет освещения.",
    "iot_hue_lightdim": "Приглушить свет.", "iot_hue_lightoff": "Выключить свет.",
    "iot_hue_lighton": "Включить свет.", "iot_hue_lightup": "Сделать свет ярче.",
    "iot_wemo_off": "Выключить умную розетку.", "iot_wemo_on": "Включить умную розетку.",
    "lists_createoradd": "Создать список или добавить в него.", "lists_query": "Узнать содержимое списка.",
    "lists_remove": "Удалить из списка.", "music_dislikeness": "Сообщить, что музыка не нравится.",
    "music_likeness": "Сообщить, что музыка нравится.", "music_query": "Узнать, что играет.",
    "music_settings": "Настройки воспроизведения музыки.", "news_query": "Узнать новости.",
    "play_audiobook": "Включить аудиокнигу.", "play_game": "Запустить игру.",
    "play_music": "Включить музыку.", "play_podcasts": "Включить подкаст.",
    "play_radio": "Включить радио.", "qa_currency": "Узнать курс валют.",
    "qa_definition": "Узнать значение слова.", "qa_factoid": "Фактический вопрос.",
    "qa_maths": "Посчитать.", "qa_stock": "Узнать котировки акций.",
    "recommendation_events": "Посоветовать мероприятие.", "recommendation_locations": "Посоветовать место.",
    "recommendation_movies": "Посоветовать фильм.", "social_post": "Опубликовать в соцсети или пожаловаться компании.",
    "social_query": "Узнать новости из соцсетей.", "takeaway_order": "Заказать еду.",
    "takeaway_query": "Узнать о заказе еды или доставке.", "transport_query": "Узнать о транспорте.",
    "transport_taxi": "Вызвать такси.", "transport_ticket": "Купить билет.",
    "transport_traffic": "Узнать о пробках.", "weather_query": "Узнать погоду.",
}


def build_massive(rng, n_train=2000):
    d = load_dataset("mteb/amazon_massive_intent", "ru")
    parts = [("train", d["train"].shuffle(seed=1).select(range(n_train))),
             ("val", d["validation"].shuffle(seed=1).select(range(200))),
             ("test", d["test"].shuffle(seed=1).select(range(400)))]
    labels = list(INTENTS)
    for split, ds in parts:
        for ex in ds:
            gold = ex["label"]
            scen = gold.split("_")[0]
            near = [l for l in labels if l.startswith(scen) and l != gold]
            far = [l for l in labels if not l.startswith(scen)]
            distr = rng.sample(near, min(2, len(near)))
            distr += rng.sample(far, 4 - len(distr))
            drop_gold = rng.random() < 0.15
            keys = distr + ([] if drop_gold else [gold])
            opts = [(k, INTENTS[k]) for k in keys] + [("none", "Ни один из вариантов не подходит.")]
            yield make("massive", split, f"Реплика пользователя голосовому ассистенту: {ex['text']}",
                       "Чего хочет пользователь?", opts, "none" if drop_gold else gold, rng)


# ---------------------------------------------------------------- синтетические политики

CATEGORIES = ["канцтовары", "ИТ-оборудование", "мебель", "услуги клининга", "программное обеспечение",
              "маркетинговые услуги", "спецодежда", "ремонт помещений", "консалтинг", "обучение персонала"]
DEPTS = ["отдел продаж", "бухгалтерия", "ИТ-департамент", "склад", "HR", "маркетинг", "юридический отдел", "производство"]


def gen_procurement(rng, split):
    lim_fin = rng.choice([300_000, 500_000, 1_000_000, 1_500_000])
    lim_kp = rng.choice([100_000, 200_000, 250_000])
    need_kp = rng.choice([2, 3])
    amount = rng.choice([rng.randint(5, 99) * 1_000, rng.randint(100, 3_000) * 1_000])
    budget = amount + rng.randint(-400, 900) * 1_000
    budget = max(budget, 0)
    in_reg = rng.random() > 0.2
    kp = rng.randint(0, 4)
    missing = rng.random() < 0.08
    fields = {
        "Подразделение": rng.choice(DEPTS),
        "Категория": rng.choice(CATEGORIES),
        "Сумма заявки": rub(amount),
        "Остаток бюджета статьи": rub(budget),
        "Поставщик в реестре проверенных": "да" if in_reg else "нет",
        "Приложено коммерческих предложений": str(kp),
    }
    if missing:
        del fields[rng.choice(["Остаток бюджета статьи", "Поставщик в реестре проверенных",
                               "Приложено коммерческих предложений"])]
    policy = (f"Регламент закупок (правила проверяются по порядку, применяется первое подходящее): "
              f"1) поставщик не из реестра проверенных — отклонить; "
              f"2) сумма больше остатка бюджета — отклонить; "
              f"3) сумма свыше {rub(lim_kp)} и приложено меньше {need_kp} КП — запросить документы; "
              f"4) сумма свыше {rub(lim_fin)} — направить финансовому директору; "
              f"5) иначе — утвердить. Если в заявке нет данных для проверки — передать человеку.")
    if missing:
        gold = "none"
    elif not in_reg or amount > budget:
        gold = "reject"
    elif amount > lim_kp and kp < need_kp:
        gold = "docs"
    elif amount > lim_fin:
        gold = "cfo"
    else:
        gold = "approve"
    state = policy + "\nЗаявка: " + "; ".join(f"{k}: {v}" for k, v in fields.items())
    return make("procurement", split, state, "Какое решение принять по заявке на закупку?",
                [("approve", "Утвердить."), ("cfo", "Направить на согласование финансовому директору."),
                 ("docs", "Вернуть: запросить недостающие коммерческие предложения."),
                 ("reject", "Отклонить."), ("none", NONE_DESC)], gold, rng)


CITIES = {"Москва": "moscow", "Санкт-Петербург": "spb", "Казань": "other", "Екатеринбург": "other",
          "Новосибирск": "other", "Нижний Новгород": "other", "Сочи": "other", "Владивосток": "other"}
ROLES = ["специалист", "руководитель отдела", "директор"]


def gen_expense(rng, split):
    base = rng.choice([4000, 5000, 6000, 7000])
    lim = {"moscow": base + rng.choice([2000, 3000, 4000]), "spb": base + rng.choice([1000, 2000, 3000]), "other": base}
    mult = {"специалист": 1.0, "руководитель отдела": 1.3, "директор": 1.8}
    biz_roles = rng.choice([["директор"], ["директор", "руководитель отдела"]])
    role = rng.choice(ROLES)
    city = rng.choice(list(CITIES))
    limit = lim[CITIES[city]] * mult[role]
    hotel = int(limit * rng.uniform(0.6, 1.35) / 100) * 100
    ticket = rng.choice(["эконом", "эконом", "бизнес"])
    receipts = rng.random() > 0.15
    missing = rng.random() < 0.08
    fields = {"Сотрудник": role, "Город": city, "Гостиница за ночь": rub(hotel),
              "Класс авиабилета": ticket, "Чеки приложены": "да" if receipts else "нет"}
    if missing:
        del fields[rng.choice(["Гостиница за ночь", "Класс авиабилета", "Чеки приложены"])]
    policy = (f"Положение о командировках. Лимит гостиницы за ночь для специалиста: Москва — {rub(lim['moscow'])}, "
              f"Санкт-Петербург — {rub(lim['spb'])}, другие города — {rub(lim['other'])}. "
              f"Для руководителя отдела лимит ×1,3, для директора ×1,8. "
              f"Бизнес-класс разрешён только: {', '.join(biz_roles)}. "
              f"Порядок проверки: чеки, затем гостиница, затем билет. Если данных не хватает — передать человеку.")
    if missing:
        gold = "none"
    elif not receipts:
        gold = "no_docs"
    elif hotel > limit:
        gold = "hotel"
    elif ticket == "бизнес" and role not in biz_roles:
        gold = "ticket"
    else:
        gold = "ok"
    state = policy + "\nАвансовый отчёт: " + "; ".join(f"{k}: {v}" for k, v in fields.items())
    return make("expense", split, state, "Каков результат проверки авансового отчёта?",
                [("ok", "Соответствует положению, принять."), ("no_docs", "Вернуть: нет подтверждающих документов."),
                 ("hotel", "Вернуть: превышен лимит на гостиницу."), ("ticket", "Вернуть: недопустимый класс билета."),
                 ("none", NONE_DESC)], gold, rng)


TICKETS = {
    "billing": ["С карты дважды списали {sum} за {what}, прошу вернуть лишнее.",
                "Пришёл счёт на {sum}, хотя по договору должно быть меньше. Объясните начисление.",
                "Оплатил {what}, деньги ушли, а в личном кабинете оплата не отображается."],
    "logistics": ["Заказ {num} должен был приехать {day}, до сих пор нет. Где он?",
                  "Курьер привёз не тот товар по заказу {num}, нужен обмен.",
                  "Доставку {num} перенесли уже третий раз, это невозможно."],
    "tech": ["Не могу войти в личный кабинет, пишет «ошибка 500».",
             "Приложение вылетает при попытке {action}.",
             "Интеграция с 1С перестала выгружать документы с {day}."],
    "sales": ["Хотим подключить ещё {n} сотрудников, какие условия?",
              "Интересует переход на тариф «{plan}», пришлите предложение.",
              "Нужен счёт на годовую подписку для {n} пользователей."],
    "quality": ["Менеджер {name} нагрубил мне по телефону, требую разобраться.",
                "Третий раз обещают перезвонить и не перезванивают. Качество сервиса ужасное.",
                "Сотрудник {name} дал неверную консультацию, из-за этого мы потеряли деньги."],
}
TICKET_DESC = {"billing": "Биллинг и расчёты.", "logistics": "Логистика и доставка.", "tech": "Техническая поддержка.",
               "sales": "Отдел продаж.", "quality": "Служба качества (жалобы на сервис).",
               "km": "Персональный менеджер ключевого клиента."}
CHURN = [" Если не решите — расторгаем договор.", " Думаем уйти к конкурентам.", " Рассматриваем отказ от ваших услуг."]


def gen_routing(rng, split):
    dept = rng.choice(list(TICKETS))
    text = rng.choice(TICKETS[dept]).format(
        sum=rub(rng.randint(2, 90) * 1000), what=rng.choice(["подписку", "услуги связи", "лицензии", "обслуживание"]),
        num=f"№{rng.randint(10000, 99999)}", day=rng.choice(["в понедельник", "вчера", "15 числа", "на прошлой неделе"]),
        action=rng.choice(["выгрузить отчёт", "подписать документ", "открыть заказ"]), n=rng.randint(3, 200),
        plan=rng.choice(["Бизнес", "Корпоративный", "Про"]), name=rng.choice(["Иванов", "Ольга", "Сергей Петрович", "Смирнова"]))
    segment = rng.choice(["розница", "малый бизнес", "ключевой клиент"])
    churn = rng.random() < 0.3
    if churn:
        text += rng.choice(CHURN)
    rule_km = rng.random() < 0.7
    policy = "Правила маршрутизации обращений: направляйте в профильный отдел по сути обращения."
    if rule_km:
        policy += " Исключение: если клиент из сегмента «ключевой клиент» и угрожает уйти — сразу персональному менеджеру."
    gold = "km" if (rule_km and churn and segment == "ключевой клиент") else dept
    keys = list(TICKET_DESC)
    state = f"{policy}\nСегмент клиента: {segment}\nОбращение: {text}"
    return make("routing", split, state, "Куда направить обращение?", [(k, TICKET_DESC[k]) for k in keys], gold, rng)


# --- политики только для теста переноса (в обучении не встречаются)

def gen_vacation(rng, split):
    notice = rng.choice([7, 14, 30])
    days = rng.randint(3, 28)
    left = rng.randint(0, 35)
    ahead = rng.randint(1, 60)
    overlap = rng.random() < 0.25
    policy = (f"Политика отпусков: заявление подаётся не позднее чем за {notice} дней; нельзя брать больше дней, "
              f"чем осталось; если отпуск пересекается с отпуском другого сотрудника, который его замещает, — предложить перенос. "
              f"Проверка по порядку: остаток дней, срок подачи, пересечение.")
    fields = {"Дней отпуска": days, "Остаток дней": left, "Подано за, дней": ahead,
              "Пересечение с замещающим": "да" if overlap else "нет"}
    if days > left:
        gold = "no_days"
    elif ahead < notice:
        gold = "late"
    elif overlap:
        gold = "move"
    else:
        gold = "ok"
    state = policy + "\nЗаявление: " + "; ".join(f"{k}: {v}" for k, v in fields.items())
    return make("t_vacation", split, state, "Как ответить на заявление на отпуск?",
                [("ok", "Одобрить."), ("no_days", "Отказать: недостаточно дней отпуска."),
                 ("late", "Отказать: заявление подано позже срока."), ("move", "Предложить перенести даты."),
                 ("none", NONE_DESC)], gold, rng)


def gen_access(rng, split):
    systems = {"1С:Бухгалтерия": "high", "CRM": "low", "корпоративная почта": "low", "банк-клиент": "high", "HR-портал": "low"}
    sys_ = rng.choice(list(systems))
    boss = rng.random() > 0.2
    training = rng.random() > 0.25
    sec = rng.random() > 0.3
    policy = ("Регламент доступа: без согласования руководителя — отклонить; без пройденного курса по ИБ — "
              "направить на обучение; для систем с критичными данными (1С:Бухгалтерия, банк-клиент) дополнительно нужно "
              "согласование службы безопасности, иначе — направить в СБ. Иначе — выдать доступ.")
    fields = {"Система": sys_, "Согласовано руководителем": "да" if boss else "нет",
              "Курс по ИБ пройден": "да" if training else "нет", "Согласовано СБ": "да" if sec else "нет"}
    if not boss:
        gold = "reject"
    elif not training:
        gold = "train"
    elif systems[sys_] == "high" and not sec:
        gold = "sb"
    else:
        gold = "grant"
    state = policy + "\nЗаявка на доступ: " + "; ".join(f"{k}: {v}" for k, v in fields.items())
    return make("t_access", split, state, "Что сделать с заявкой на доступ?",
                [("grant", "Выдать доступ."), ("reject", "Отклонить."), ("train", "Направить на обучение по ИБ."),
                 ("sb", "Направить на согласование в службу безопасности."), ("none", NONE_DESC)], gold, rng)


def gen_credit(rng, split):
    lim_committee = rng.choice([5_000_000, 10_000_000])
    years = rng.randint(0, 15)
    overdue = rng.choice([0, 0, 0, 1, 2, 5, 12, 40])
    limit = rng.randint(2, 300) * 100_000
    revenue = rng.randint(5, 2000) * 1_000_000
    policy = (f"Кредитная политика по отсрочке платежа: просрочки свыше 30 дней за год — отказать; "
              f"компания работает меньше 2 лет — только по предоплате; запрошенный лимит больше 10% выручки или больше "
              f"{rub(lim_committee)} — на кредитный комитет; иначе одобрить. Проверка по порядку.")
    fields = {"Лет на рынке": years, "Макс. просрочка за год, дней": overdue, "Запрошенный лимит": rub(limit),
              "Выручка за год": rub(revenue)}
    if overdue > 30:
        gold = "reject"
    elif years < 2:
        gold = "prepay"
    elif limit > revenue * 0.1 or limit > lim_committee:
        gold = "committee"
    else:
        gold = "approve"
    state = policy + "\nКонтрагент: " + "; ".join(f"{k}: {v}" for k, v in fields.items())
    return make("t_credit", split, state, "Какое решение по лимиту отсрочки для контрагента?",
                [("approve", "Одобрить лимит."), ("prepay", "Работать только по предоплате."),
                 ("committee", "Вынести на кредитный комитет."), ("reject", "Отказать."), ("none", NONE_DESC)], gold, rng)


def synth(gen, rng, sizes):
    for split, n in sizes.items():
        for _ in range(n):
            yield gen(rng, split)


def main():
    rng = random.Random(42)
    rows = []
    for b in [build_terra, build_rcb, build_danetqa, build_reviews, build_headlines, build_massive]:
        rows += list(b(rng))
    for g in [gen_procurement, gen_expense, gen_routing]:
        rows += list(synth(g, rng, {"train": 1500, "val": 150, "test": 200}))
    for g in [gen_vacation, gen_access, gen_credit]:
        rows += list(synth(g, rng, {"test": 150}))
    OUT.mkdir(exist_ok=True)
    for split in ["train", "val", "test"]:
        part = [r for r in rows if r["split"] == split]
        rng.shuffle(part)
        with open(OUT / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by = {}
        for r in part:
            by[r["task"]] = by.get(r["task"], 0) + 1
        print(split, len(part), by)


if __name__ == "__main__":
    main()
