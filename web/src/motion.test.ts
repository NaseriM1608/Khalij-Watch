import assert from "node:assert/strict";
import test from "node:test";
import { motionFeatures } from "./motion.ts";

const now = Date.parse("2026-10-29T11:00:00Z");
const vessel = {
  mmsi: 422123456, lon: 56.5, lat: 26.5,
  ts: "2026-10-29T10:59:00Z",
  estimate: {
    status: "estimated", reason: null, lon: 56.51, lat: 26.5,
    target_time: "2026-10-29T11:00:00Z", elapsed_seconds: 60,
    max_age_seconds: 300, method: "constant_sog_cog",
  },
};

test("estimate adds a separate point and observed-to-estimated path", () => {
  const features = motionFeatures([vessel], true, now).features;
  assert.equal(features.length, 2);
  assert.deepEqual(features[0].geometry, {type: "Point", coordinates: [56.51, 26.5]});
  assert.deepEqual(features[1].geometry, {type: "LineString", coordinates: [[56.5, 26.5], [56.51, 26.5]]});
  assert.equal(vessel.lon, 56.5);
});

test("historical radar playback has no estimated markers", () => {
  assert.equal(motionFeatures([vessel], false, now).features.length, 0);
});

test("a failed refresh cannot keep an expired estimate on screen", () => {
  assert.equal(motionFeatures([vessel], true, now + 301_000).features.length, 0);
});

test("unknown and unavailable estimates do not invent motion", () => {
  const unavailable = {...vessel, estimate: {...vessel.estimate, status: "unavailable"}};
  assert.equal(motionFeatures([unavailable, {...vessel, estimate: null}], true, now).features.length, 0);
});
