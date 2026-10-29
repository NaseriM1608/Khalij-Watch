/** AIS-reported type; this is not a classification from the radar image. */
export function shipTypeLabel(code: number | null | undefined): string {
  if (code == null || !Number.isInteger(code)) return "گزارش نشده";
  if (code >= 70 && code <= 79) return "باری";
  if (code >= 80 && code <= 89) return "نفتکش";
  if (code >= 60 && code <= 69) return "مسافری";
  if (code >= 40 && code <= 49) return "شناور تندرو";
  if (code >= 20 && code <= 29) return "شناور اثر سطحی";
  if (code >= 90 && code <= 99) return "سایر شناورها";
  const specific: Record<number, string> = {
    30: "ماهیگیری", 31: "یدک‌کشی", 32: "یدک‌کشی بزرگ",
    33: "لایروبی یا عملیات زیرآبی", 34: "عملیات غواصی", 35: "عملیات نظامی",
    36: "بادبانی", 37: "تفریحی", 50: "راهنما", 51: "جست‌وجو و نجات",
    52: "یدک‌کش", 53: "خدمات بندری", 54: "مقابله با آلودگی",
    55: "اجرای قانون", 58: "پزشکی", 59: "شناور دولتی",
  };
  return specific[code] ?? "گزارش نشده";
}
