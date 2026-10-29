import assert from "node:assert/strict";
import test from "node:test";
import { shipTypeLabel } from "./identity.ts";

test("reported cargo and tanker codes remain distinct on the identity card", () => {
  assert.equal(shipTypeLabel(70), "باری");
  assert.equal(shipTypeLabel(89), "نفتکش");
  assert.equal(shipTypeLabel(30), "ماهیگیری");
  assert.equal(shipTypeLabel(52), "یدک‌کش");
});

test("missing, zero and reserved codes never invent a ship type", () => {
  for (const code of [null, undefined, 0, 1, 19, 100, -1, NaN]) {
    assert.equal(shipTypeLabel(code), "گزارش نشده");
  }
});
