# Wumpus World AI — شبیه‌ساز و بنچمارک سه روش هوش مصنوعی

[![CI](https://github.com/mahan-vzmz/Wumpus-World/actions/workflows/ci.yml/badge.svg)](https://github.com/mahan-vzmz/Wumpus-World/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![Tests](https://img.shields.io/badge/tests-99%20passed-brightgreen)
![Coverage](https://img.shields.io/badge/core%20coverage-91%25-brightgreen)

یک پروژهٔ دانشگاهی مهندسی‌شده برای پیاده‌سازی و مقایسهٔ سه پارادایم متفاوت هوش مصنوعی در محیط `8×8` دنیای Wumpus:

- **A\*** با دید کامل از نقشه، به‌عنوان خبره و کران بالای عملکرد؛
- **استدلال قاعده‌محور** با دید ناقص و trace قابل‌توضیح؛
- **یادگیری نظارت‌شده** با Random Forest، ویژگی‌های صرفاً مشاهده‌پذیر و action masking.

دو baseline حریصانه و تصادفی نیز برای تفسیر بهتر نتایج اجرا می‌شوند. تمام عامل‌ها از موتور بازی، قرارداد امتیاز و مجموعهٔ نقشهٔ مشترک استفاده می‌کنند.

## نتایج نهایی روی مجموعهٔ holdout

مدل روی ۱۰۰ نقشهٔ تولیدی با seedهای `100..199` آموزش دیده و ارزیابی نهایی روی ۲۰ نقشهٔ جدا با seedهای `2000..2019` انجام شده است. این مجموعه در انتخاب مدل استفاده نشده است.

| عامل | میزان مشاهده | نرخ برد | میانگین امتیاز تشخیصی | میانگین گام | ورود به چاه | مرگ با غول |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A\* Search | Full | **100%** | **41.6** | 14.9 | 0 | 0 |
| Rule-based | Partial | **95%** | 25.3 | 18.8 | **2** | **0** |
| Greedy baseline | Partial | 80% | 16.1 | **12.8** | 12 | 3 |
| Random Forest | Partial | 75% | 14.3 | 13.9 | 12 | 2 |
| Random baseline | Partial | 0% | -0.5 | 32.4 | 13 | 4 |

> مقایسهٔ A\* با عامل‌های آنلاین هم‌شرایط نیست: A\* نقشهٔ پنهان را می‌بیند و فقط نقش خبره/کران بالا دارد. مقایسهٔ منصفانهٔ آنلاین میان RuleAgent، MLAgent و baselineها است.

نتایج خام و مشخصات اجرای ثبت‌شده در [`results/`](results/) قرار دارند.

![Holdout win rate by agent](docs/assets/benchmark_win_rate.svg)

## معماری

```text
src/wumpus/
├── domain.py              # مدل‌های دامنه و وضعیت‌های بازی
├── parser.py              # parser و اعتبارسنجی ورودی ۱۲ خطی
├── engine.py              # موتور قطعی و ترتیب رویدادها
├── observation.py         # breeze / stench / glitter / legal actions
├── runner.py              # حلقهٔ مشترک اجرا و مدیریت خطای عامل
├── search.py              # A* امتیازبهینه با terminal cost
├── knowledge.py           # پایگاه دانش و forward chaining
├── encoder.py             # بردار ویژگی ۳۹۷بعدی بدون hidden-map leakage
├── dataset.py             # تولید demonstration و split نقشه‌محور
├── ml.py                  # آموزش، معیارها، serialization و masking
├── agents/                # Search / Rule / ML / Greedy / Random
└── evaluation/            # تولید suite و benchmark قابل‌بازتولید
```

## نصب سریع

پیش‌نیاز: Python 3.11 یا جدیدتر.

```bash
git clone https://github.com/mahan-vzmz/Wumpus-World.git
cd Wumpus-World

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

## اجرای تست و کنترل کیفیت

```bash
pytest
ruff check .
pytest --cov=wumpus --cov-report=term-missing
```

- ۹۹ تست خودکار؛
- پوشش ۹۱٪ برای کد هسته؛
- lint و تست خودکار روی Python 3.11 و 3.12 در GitHub Actions.

## اجرای نمونه

```bash
# اعتبارسنجی نقشه
python -m wumpus validate --input data/maps/example.txt

# A* با دید کامل
python -m wumpus run --agent search --input data/maps/example.txt

# عامل قاعده‌محور با trace استدلال
python -m wumpus run --agent rules \
  --input tests/fixtures/golden2_pit.txt --trace

# baselineها
python -m wumpus run --agent greedy --input data/maps/example.txt
python -m wumpus run --agent random --input data/maps/example.txt --seed 42
```

## بازتولید چرخهٔ ML و benchmark

فایل باینری مدل عمداً داخل Git نگهداری نمی‌شود؛ دیتاست، تنظیمات، معیارها و دستور بازتولید ثبت شده‌اند.

```bash
# ۱) بازتولید ۱۰۰ نقشهٔ آموزشی متنوع و demonstrationهای A*
python -m wumpus dataset \
  --num-maps 100 --seed 100 --output-dir data/processed

# ۲) آموزش و ذخیرهٔ مدل و معیارهای validation/test
python -m wumpus train \
  --data-dir data/processed --output-dir artifacts/models

# ۳) اجرای benchmark نهایی روی holdout ثابت
python -m wumpus benchmark \
  --maps-dir data/maps/holdout_suite \
  --model artifacts/models/random_forest.joblib \
  --results-dir results
```

اگر فقط عامل‌های غیر ML مدنظر باشند:

```bash
python -m wumpus benchmark --skip-ml
```

در صورت نبود مدل، CLI به‌جای اجرای یک fallback خاموش با پیام روشن و exit code غیرصفر متوقف می‌شود.

## قالب ورودی

ورودی شامل ۸ سطر نقشه و چهار مقدار تنظیمات است:

```text
********
**D*****
*****G**
W***P***
********
********
********
********
100
25
-10
8 8
```

نمادها: `*` خانهٔ خالی، `P` چاه، `W` غول، `D` دیوار و `G` طلا. مختصات بیرونی یک‌مبنا و به‌شکل `(row, column)` هستند.

## بازتولیدپذیری و داده‌های ثبت‌شده

- [`data/processed/metadata.json`](data/processed/metadata.json): schema، تعداد نمونه‌ها، profileها و توزیع کلاس‌ها؛
- [`artifacts/models/training_metrics.json`](artifacts/models/training_metrics.json): معیارهای validation/test و confusion matrix؛
- [`data/maps/holdout_suite/suite_manifest.json`](data/maps/holdout_suite/suite_manifest.json): seed و تنظیمات مجموعهٔ holdout؛
- [`results/benchmark_results.csv`](results/benchmark_results.csv): ۱۰۰ ردیف خام اجرای نهایی؛
- [`results/benchmark_summary.json`](results/benchmark_summary.json): خلاصه، نسخهٔ Python و SHA-256 مدل.

## محدودیت‌ها

- برچسب‌های خبره از A\* با دید کامل می‌آیند، ولی MLAgent فقط observation ناقص دارد؛ بخشی از رفتار خبره ذاتاً از روی ویژگی‌های آنلاین قابل‌بازیابی نیست.
- دیتاست چهارکلاسه نامتوازن است و کلاس‌های `UP` و `LEFT` نمونه‌های کمتری دارند؛ بنابراین macro-F1 و recall هر کلاس در کنار accuracy گزارش شده‌اند.
- `glitter` طبق قرارداد پروژه وجود دارد، اما چون طلا هنگام ورود خودکار جمع می‌شود، در episode عادی سیگنال تصمیم‌گیری فعالی نیست.
- اعداد زمان اجرا به سخت‌افزار و نسخهٔ کتابخانه‌ها وابسته‌اند؛ نتیجه‌گیری اصلی بر win rate، score و معیارهای ایمنی است.
- مجموعهٔ holdout فعلی ۲۰ نقشه دارد؛ برای ادعای تعمیم قوی‌تر باید تعداد نقشه و seedهای مستقل افزایش یابد.

## مستندات

- [`LICENSE`](LICENSE): مجوز MIT برای استفاده و توسعهٔ پروژه؛
- [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md): قراردادها و دفتر تصمیم‌ها؛
- [`docs/SPEC.md`](docs/SPEC.md): مشخصات فنی و رفتاری؛
- [`docs/PROJECT_REPORT.md`](docs/PROJECT_REPORT.md): گزارش روش‌ها و تحلیل نتایج؛
- [`docs/TASKBOOK.md`](docs/TASKBOOK.md): وضعیت اجرایی و کارهای باقی‌مانده؛
- [`docs/DEMO.md`](docs/DEMO.md): سناریوی ارائهٔ ۵ دقیقه‌ای؛
- [`tests/fixtures/GOLDEN_EXAMPLES.md`](tests/fixtures/GOLDEN_EXAMPLES.md): مثال‌های دستی حرکت‌به‌حرکت.
