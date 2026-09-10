/**
 * API client. The browser talks only to our backend, never to a feed provider
 * directly (aisstream forbids browser connections outright).
 */

export type CoarseClass =
  | "cargo_tanker"
  | "fishing"
  | "service_tug"
  | "passenger_other";

export interface Vessel {
  mmsi: number;
  lon: number;
  lat: number;
  ts: string;
  /** Explicit, so a stale contact can never be drawn as if it were live. */
  age_seconds: number;
  sog_kn: number | null;
  cog_deg: number | null;
  heading_deg: number | null;
  name: string | null;
  coarse_class: CoarseClass | null;
  source: string;
  plausibility_score: number | null;
  /** AIS here is adversarial; the UI must be able to show doubt. */
  is_suspect: boolean;
  anomaly_flags: AnomalyFlag[] | null;
}

export interface AnomalyFlag {
  code: string;
  penalty: number;
  detail: string;
}

export const ANOMALY_LABELS: Record<string, string> = {
  ON_LAND: "Position on land",
  TELEPORT: "Teleported between fixes",
  IMPOSSIBLE_SPEED: "Impossible speed",
  IMPOSSIBLE_TURN: "Impossible turn rate",
  SPEED_MISMATCH: "Speed contradicts track",
  DUPLICATE_IDENTITY: "Identity in two places",
  COLLINEAR_TRACK: "Track impossibly straight",
  MID_REGION_MISMATCH: "Flag inconsistent with position",
};

export interface VesselsResponse {
  generated_at: string;
  bbox: BBox;
  count: number;
  suspect_count: number;
  vessels: Vessel[];
  /** Feed licences require attribution to reach the display. */
  attribution: string[];
}

export interface BBox {
  min_lon: number;
  min_lat: number;
  max_lon: number;
  max_lat: number;
}

export interface SarScene {
  id: number;
  product_name: string;
  satellite: string;
  polarisation: string;
  orbit_direction: string | null;
  acq_time: string;
  /** Hours since acquisition. Radar is not live; this says how stale it is. */
  age_hours: number;
  status: string;
  footprint: [number, number][];
}

export interface ScenesResponse {
  count: number;
  scenes: SarScene[];
  /** When this area was last observed by radar at all. */
  last_look: string | null;
  satellites: string[];
  attribution: string;
}

export interface Health {
  status: string;
  vessels: number;
  positions: number;
  sar_scenes: number;
  live_feeds: string[];
  has_live_ais: boolean;
}

export interface TrackPoint {
  lon: number;
  lat: number;
  ts: string;
  sog_kn: number | null;
  source: string;
}

export interface Track {
  mmsi: number;
  name: string | null;
  coarse_class: CoarseClass | null;
  imo: number | null;
  flag: string | null;
  sanctioned: boolean;
  points: TrackPoint[];
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    const detail = await response.text();
    throw new Error(`${response.status} ${response.statusText}: ${detail}`);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => getJson<Health>("/api/health"),
  aoi: () => getJson<{ gulf: BBox; hormuz: BBox }>("/api/aoi"),
  vessels: (bbox?: BBox, maxAgeMinutes = 360) => {
    const params = new URLSearchParams({
      max_age_minutes: String(maxAgeMinutes),
    });
    if (bbox) {
      params.set(
        "bbox",
        `${bbox.min_lon},${bbox.min_lat},${bbox.max_lon},${bbox.max_lat}`,
      );
    }
    return getJson<VesselsResponse>(`/api/vessels?${params}`);
  },
  track: (mmsi: number, hours = 24) =>
    getJson<Track>(`/api/vessels/${mmsi}/track?hours=${hours}`),
  scenes: (days = 14) => getJson<ScenesResponse>(`/api/scenes?days=${days}`),
};

/** Colour a radar footprint by how long ago it was acquired. */
export function coverageColour(ageHours: number): string {
  if (ageHours < 12) return "#3ad6a8";
  if (ageHours < 48) return "#4aa3ff";
  if (ageHours < 120) return "#8a7de8";
  return "#5a6274";
}

/** Human-readable age. The UI states this rather than implying freshness. */
export function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.round(seconds)}s ago`;
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ${minutes % 60}m ago`;
  return `${Math.floor(hours / 24)}d ${hours % 24}h ago`;
}

export const CLASS_COLOURS: Record<CoarseClass | "unknown", string> = {
  cargo_tanker: "#4aa3ff",
  fishing: "#5fd08a",
  service_tug: "#f0a24a",
  passenger_other: "#c081e8",
  unknown: "#8a94a6",
};

export const CLASS_LABELS: Record<CoarseClass | "unknown", string> = {
  cargo_tanker: "Cargo / Tanker",
  fishing: "Fishing",
  service_tug: "Service / Tug",
  passenger_other: "Passenger / Other",
  unknown: "Unidentified",
};

export function classOf(vessel: Vessel): CoarseClass | "unknown" {
  return vessel.coarse_class ?? "unknown";
}
