/** Estimated points never replace the observed coordinates or radar evidence. */
interface EstimateContact {
  mmsi: number;
  lon: number;
  lat: number;
  ts: string;
  estimate?: { status: string; lon: number | null; lat: number | null } | null;
}

export function motionFeatures(
  vessels: EstimateContact[],
  enabled: boolean,
  nowMs = Date.now(),
): GeoJSON.FeatureCollection {
  const features: GeoJSON.Feature[] = [];
  if (!enabled) return { type: "FeatureCollection", features };
  for (const vessel of vessels) {
    const estimate = vessel.estimate;
    const age = (nowMs - Date.parse(vessel.ts)) / 1000;
    if (!Number.isFinite(age) || age < 0 || age > 300 || estimate?.status !== "estimated"
      || estimate.lon == null || estimate.lat == null) continue;
    const properties = { mmsi: vessel.mmsi };
    features.push({
      type: "Feature", properties,
      geometry: { type: "Point", coordinates: [estimate.lon, estimate.lat] },
    });
    features.push({
      type: "Feature", properties,
      geometry: { type: "LineString", coordinates: [[vessel.lon, vessel.lat], [estimate.lon, estimate.lat]] },
    });
  }
  return { type: "FeatureCollection", features };
}
