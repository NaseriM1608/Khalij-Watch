/**
 * Persian formatting for the console: Persian digits, Solar Hijri (Jalali)
 * dates, all times in UTC as the satellite data are. Built on the browser's
 * own Intl (fa-IR, Persian calendar); no library.
 */

const DATE = new Intl.DateTimeFormat("fa-IR-u-ca-persian", {
  day: "numeric",
  month: "long",
  year: "numeric",
  timeZone: "UTC",
});
const DAY_MONTH = new Intl.DateTimeFormat("fa-IR-u-ca-persian", {
  day: "numeric",
  month: "long",
  timeZone: "UTC",
});
const TIME = new Intl.DateTimeFormat("fa-IR", {
  hour: "2-digit",
  minute: "2-digit",
  hourCycle: "h23",
  timeZone: "UTC",
});

function moment(value: string | number): Date {
  return new Date(typeof value === "number" ? value : Date.parse(value));
}

/** A number with Persian digits and separators, ``digits`` decimals. */
export function faNumber(value: number, digits = 0): string {
  return value.toLocaleString("fa-IR", {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** e.g. «۵ مهر ۱۴۰۵، ۰۲:۱۴ UTC». */
export function faDateTime(value: string | number): string {
  const when = moment(value);
  return `${DATE.format(when)}، ${TIME.format(when)} UTC`;
}

/** e.g. «۵ مهر ۱۴۰۵». */
export function faDate(value: string | number): string {
  return DATE.format(moment(value));
}

/** e.g. «۰۲:۰۰» (UTC hour and minute). */
export function faTime(value: string | number): string {
  return TIME.format(moment(value));
}

/** e.g. «۳۰ شهریور» (timeline axis). */
export function faDayMonth(value: string | number): string {
  return DAY_MONTH.format(moment(value));
}

/** How long ago, e.g. «۲ ساعت و ۵ دقیقه پیش». */
export function faAge(seconds: number): string {
  if (seconds < 60) return `${faNumber(Math.round(seconds))} ثانیه پیش`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${faNumber(minutes)} دقیقه پیش`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) {
    const rest = minutes % 60;
    return rest
      ? `${faNumber(hours)} ساعت و ${faNumber(rest)} دقیقه پیش`
      : `${faNumber(hours)} ساعت پیش`;
  }
  const days = Math.floor(hours / 24);
  const rest = hours % 24;
  return rest
    ? `${faNumber(days)} روز و ${faNumber(rest)} ساعت پیش`
    : `${faNumber(days)} روز پیش`;
}

/** Metres below 1 km, kilometres with one decimal above; «نامعلوم» when unknown. */
export function faMetres(metres: number | null | undefined): string {
  if (metres == null) return "نامعلوم";
  return metres >= 1000 ? `${faNumber(metres / 1000, 1)} کیلومتر` : `${faNumber(Math.round(metres))} متر`;
}

/** Latitude, longitude in degrees with four decimals. */
export function faCoord(lat: number, lon: number): string {
  return `${faNumber(lat, 4)}°، ${faNumber(lon, 4)}°`;
}
