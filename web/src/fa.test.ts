// Persian formatting for the console (run: npm test; Node runs TypeScript natively).
import assert from "node:assert/strict";
import test from "node:test";

import {
  faAge,
  faCoord,
  faDate,
  faDateTime,
  faDayMonth,
  faMetres,
  faNumber,
  faTime,
} from "./fa.ts";

test("numbers use Persian digits and separators", () => {
  assert.equal(faNumber(1184), "۱٬۱۸۴");
  assert.equal(faNumber(0.51, 2), "۰٫۵۱");
  assert.equal(faNumber(7), "۷");
});

test("pass times are Solar Hijri dates in UTC", () => {
  assert.equal(faDateTime("2026-09-27T02:14:30Z"), "۵ مهر ۱۴۰۵، ۰۲:۱۴ UTC");
  assert.equal(faDateTime("2026-10-04T14:41:48+00:00"), "۱۲ مهر ۱۴۰۵، ۱۴:۴۱ UTC");
  assert.equal(faDate("2026-09-27T02:14:30Z"), "۵ مهر ۱۴۰۵");
  assert.equal(faDayMonth(Date.parse("2026-09-21T00:00:00Z")), "۳۰ شهریور");
  assert.equal(faTime("2026-09-27T02:00:00Z"), "۰۲:۰۰");
});

test("ages read as Persian durations", () => {
  assert.equal(faAge(30), "۳۰ ثانیه پیش");
  assert.equal(faAge(12 * 60), "۱۲ دقیقه پیش");
  assert.equal(faAge(125 * 60), "۲ ساعت و ۵ دقیقه پیش");
  assert.equal(faAge(5 * 86400 + 22 * 3600), "۵ روز و ۲۲ ساعت پیش");
});

test("distances and positions", () => {
  assert.equal(faMetres(800), "۸۰۰ متر");
  assert.equal(faMetres(2900), "۲٫۹ کیلومتر");
  assert.equal(faMetres(null), "نامعلوم");
  assert.equal(faCoord(26.58227, 53.999064), "۲۶٫۵۸۲۳°، ۵۳٫۹۹۹۱°");
});
