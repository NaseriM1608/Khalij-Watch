title_fa: آشکارسازی شناورها در تصاویر Sentinel-1 خلیج فارس با یادگیری عمیق و بررسی نامزدهای شناور تاریک
title_en: Deep Learning for Vessel Detection in Sentinel-1 Images of the Persian Gulf and Assessment of Dark Vessel Candidates
short_title: آشکارسازی شناور در تصاویر ماهواره‌ای
author_fa: محمود ناصری
author_en: Mahmoud Naseri
supervisor_fa: دکتر بحرالعلوم
supervisor_en: Dr. Bahrololoum
date_fa: پاییز ۱۴۰۵
date_en: Autumn 2026
keywords_fa: آشکارسازی شناور، رادار روزنه مصنوعی، یادگیری عمیق، Sentinel-1، برچسب‌گذاری خودکار، خلیج فارس، تلفیق AIS
keywords_en: Vessel detection, synthetic aperture radar, deep learning, Sentinel-1, automatic labels, Persian Gulf, AIS fusion
:::abstract_fa
این پروژه آشکارسازی شناورها در تصاویر رادار روزنه مصنوعی Sentinel-1 خلیج فارس، تنگه هرمز و آب‌های مجاور را بررسی می‌کند. هدف، ساخت آشکارسازی قابل اجرا روی پردازنده مرکزی و نمایش هر یافته همراه برش تصویری آن است. برای آموزش بدون برچسب‌گذاری دستی، خروجی CFAR با خانه‌های آشکارسازی Global Fishing Watch یا GFW تطبیق داده شد. مجموعه حاصل شامل ۱۱۲٬۹۱۷ برش از ۴۲۲ محصول Sentinel-1A در سال‌های ۲۰۲۴ و ۲۰۲۵ است. آشکارساز نقطه‌ای با رمزگذار EfficientNet-B0، رمزگشای U-Net و دو سرِ احتمال حضور و جابه‌جایی مرکز، از طرح‌های موجود اقتباس شد. تقسیم داده با هشِ نام محصول انجام شد؛ مکان‌ها و بسیاری از شناورها میان تقسیم‌ها مشترک‌اند، بنابراین آزمون اصلی بازدیدهای تازه از همان محدوده‌ها را می‌سنجد. مدل نهایی روی اعتبارسنجی انتخاب و ثابت شد. روی ۹۲ گذر-محدوده آزمون، F1 مدل و CFAR با قاعده همان خانه به‌ترتیب ۰٫۸۱۷ و ۰٫۸۰۸ بود؛ تفاضل ۰٫۰۰۸۷ و بازه ۹۵٪ با بازنمونه‌گیری بر حسب برداشت، ۰٫۰۰۲۳ تا ۰٫۰۱۵۷ است. بازه هر دو قاعده تطبیق در مقایسه اولیه ۴۰ گذری شامل صفر بود و برتری قاعده همسایه در آزمون گسترده نیز قطعی نشد. بازیابی آشکارسازی‌های GFW بدون AIS از ۰٫۵۰۴ در مدل نخست به ۰٫۸۱۳ رسید؛ هم‌زمان با تغییر برچسب‌ها، برش آموزشی نیز تغییر کرده است. آزمون ساحل ایران و تصاویر Sentinel-1C/1D افت انتقال را نشان داد. این امتیازها توافق با مرجع خودکار GFW هستند؛ مرجع از برچسب‌های مدل مستقل نیست و حقیقت زمینی محسوب نمی‌شود. تلفیق ثانویه با حضور ساعتی AIS، نامزدهایی با توافق متوازن حدود ۰٫۶۷ با GFW تولید کرد. کنسول فارسی، آشکارسازی‌های مدل و تصویر VV/VH هر یک را با زمان برداشت و محدودیت‌های سنجش نمایش می‌دهد. سهم پروژه، مطالعه کاربردی و تجربی این زنجیره و محدودیت‌های آن است، نه معرفی معماری تازه.
:::
:::abstract_en
This project investigates vessel detection in Sentinel-1 synthetic aperture radar imagery of the Persian Gulf, the Strait of Hormuz, and adjacent waters. Its main output is a CPU-deployable detector and a Persian web demonstrator that displays each model detection with its VV/VH image chip. Automatic labels were constructed by comparing CFAR detections with Global Fishing Watch (GFW) detection cells. The dataset contains 112,917 chips from 422 Sentinel-1A products acquired in 2024–2025. A point detector combines an EfficientNet-B0 encoder, a U-Net decoder, and objectness and offset heads, following existing designs.

Products were assigned to splits by a deterministic hash. Shared locations, vessels, and some acquisition groups limit the main test to new acquisitions of familiar areas. After validation selection, the frozen model achieved same-cell F1 of 0.817 versus 0.808 for CFAR on 92 area-passes. The difference was 0.0087, with a 95% acquisition-cluster bootstrap interval of 0.0023 to 0.0157. Both intervals on the original 40-area-pass comparison included zero; the neighbour-rule advantage on the expanded test remained inconclusive. Recall of GFW detections without AIS increased from 0.504 to 0.813 between the two runs, although labels and training crops changed together. Tests on the Iranian coast and Sentinel-1C/1D showed transfer limitations. These scores measure agreement with GFW, an imperfect reference also used for training labels. Secondary fusion with hourly AIS presence produced dark-vessel candidates, with balanced agreement of approximately 0.67 against GFW. The contribution is an applied empirical study and an inspectable demonstrator, rather than a new detection architecture or independently established vessel accuracy.
:::
:::abbreviations
AIS | سامانه شناسایی خودکار شناورها
SAR | رادار روزنه مصنوعی
CFAR | آشکارسازی با نرخ هشدار نادرست ثابت
GFW | سازمان Global Fishing Watch
GRD | محصول دامنه‌ای در فاصله زمینی
SLC | محصول مختلط تک‌نگاه
IW | حالت تصویربرداری پهنه تداخل‌سنجی
VV | ارسال عمودی و دریافت عمودی
VH | ارسال عمودی و دریافت افقی
DN | عدد رقومی ثبت‌شده در تصویر
ENL | تعداد نگاه‌های معادل
CNN | شبکه عصبی پیچشی
CPU | پردازنده مرکزی
GPU | پردازنده گرافیکی
ONNX | قالب تبادل شبکه‌های عصبی
GCP | نقطه کنترل زمینی
GNSS | سامانه جهانی ناوبری ماهواره‌ای
OPD | مجموعه داده سکوهای فراساحلی
PU | یادگیری از نمونه‌های مثبت و بی‌برچسب
:::
