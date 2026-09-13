# تست وبکم با YOLOv8-Lite-S Face

این پروژه وبکم لپ‌تاپ را باز می‌کند و با مدل سبک **YOLOv8-Lite-S** چهره را تشخیص می‌دهد. برای هر چهره:

- Bounding Box صورت نمایش داده می‌شود.
- Confidence نمایش داده می‌شود.
- ۵ نقطه‌ی صورت نمایش داده می‌شود: چشم چپ، چشم راست، نوک بینی، گوشه‌ی چپ دهان و گوشه‌ی راست دهان.
- تعداد چهره‌ها و FPS نمایش داده می‌شود.

> این مدل اجزای صورت را با **۵ landmark** مشخص می‌کند؛ برای خودِ چشم/بینی/دهان Bounding Box جدا نمی‌سازد.

## نصب با Conda

داخل پوشه پروژه:

```bash
conda env create -f environment.yml
conda activate yolo-face-webcam
python app.py
```

در اولین اجرا مدل حدود 7.4 MB به‌صورت خودکار در پوشه `models` دانلود می‌شود.

اگر می‌خواهی فقط مدل را از قبل دانلود کنی:

```bash
python download_model.py
```

## نصب بدون environment.yml

```bash
conda create -n yolo-face-webcam python=3.11 -y
conda activate yolo-face-webcam
pip install -r requirements.txt
python app.py
```

## خروج

در پنجره وبکم کلید `Q` یا `Esc` را بزن.

## دوربین دوم

اگر دوربین شماره 0 باز نشد:

```bash
python app.py --camera 1
```

## تغییر Confidence

مثلاً:

```bash
python app.py --confidence 0.35
```

## macOS

اگر تصویر وبکم باز نشد، دسترسی Camera را برای Terminal / iTerm / Python فعال کن:

`System Settings > Privacy & Security > Camera`

سپس Terminal را ببند و دوباره باز کن.

## بدون حالت آینه‌ای

```bash
python app.py --no-mirror
```
