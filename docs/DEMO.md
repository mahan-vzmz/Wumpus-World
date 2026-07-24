# سناریوی دموی ۵ دقیقه‌ای

این سناریو برای ارائهٔ دانشگاهی یا معرفی پروژه در مصاحبه طراحی شده است.

## پیش از ارائه

```bash
python -m pip install -e ".[dev]"
python -m wumpus train
pytest
```

## دقیقهٔ ۰ تا ۱ — مسئله و قرارداد

- محیط `8×8`، شروع از `(1,1)` و خروج تعیین‌شده در ورودی است.
- همهٔ عامل‌ها از یک موتور قطعی استفاده می‌کنند.
- A\* نقشه را کامل می‌بیند؛ RuleAgent و MLAgent فقط observation محلی دارند.
- بنابراین A\* کران بالا است، نه رقیب هم‌شرایط عامل‌های آنلاین.

## دقیقهٔ ۱ تا ۲ — موتور و A\*

```bash
python -m wumpus validate --input data/maps/example.txt
python -m wumpus run --agent search --input data/maps/example.txt
```

نشان بده:

- event log ترتیب حرکت، کم‌شدن جان، طلا و خروج را ثبت می‌کند؛
- خروجی A\* شامل expanded nodes، peak frontier و زمان برنامه‌ریزی است؛
- تابع loss دقیقاً با score نهایی معادل است.

## دقیقهٔ ۲ تا ۳ — استدلال توضیح‌پذیر

```bash
python -m wumpus run --agent rules \
  --input tests/fixtures/golden2_pit.txt --trace
```

در trace یک نمونه از `NO_BREEZE`، استنتاج `SAFE` و انتخاب frontier را توضیح بده.

## دقیقهٔ ۳ تا ۴ — ML و جلوگیری از نشت داده

- featureها فقط از observation و KnowledgeBase ساخته می‌شوند؛
- split بر اساس `map_id` است، نه ردیف‌های تصادفی؛
- مدل روی ۱۰۰ نقشه آموزش و روی holdout جدا ارزیابی شده است؛
- legal mask و confirmed-hazard mask جلوی action نامعتبر یا خطر قطعی را می‌گیرند.

فایل‌های کلیدی:

- `data/processed/metadata.json`
- `artifacts/models/training_metrics.json`
- `results/benchmark_summary.json`

## دقیقهٔ ۴ تا ۵ — نتیجه و محدودیت

- A\*: نرخ برد ۱۰۰٪ با دید کامل؛
- RuleAgent: نرخ برد ۹۵٪ و بهترین عامل آنلاین؛
- Random Forest: نرخ برد ۷۵٪ روی holdout؛
- محدودیت اصلی ML، تقلید خبرهٔ full-map با ویژگی‌های partial-observation و عدم
  توازن کلاس‌های حرکت است.

در پایان نمودار `docs/assets/benchmark_win_rate.svg` را نشان بده و تأکید کن که
نتایج خام، seedها، SHA مدل و محیط اجرا ثبت شده‌اند.
