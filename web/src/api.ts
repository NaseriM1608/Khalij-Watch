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
  imo?: number | null;
  callsign?: string | null;
  ship_type?: number | null;
  coarse_class: CoarseClass | null;
  source: string;
  plausibility_score: number | null;
  /** AIS here is adversarial; the UI must be able to show doubt. */
  is_suspect: boolean;
  anomaly_flags: AnomalyFlag[] | null;
  timestamp_basis?: string | null;
  received_at?: string | null;
  provider_source?: string | null;
  provider_reencoded?: boolean | null;
  estimate?: {
    status: string;
    reason: string | null;
    lon: number | null;
    lat: number | null;
    target_time: string;
    elapsed_seconds: number;
    method: string;
    max_age_seconds: number;
  } | null;
}

export interface AnomalyFlag {
  code: string;
  penalty: number;
  detail: string;
}

export const ANOMALY_LABELS: Record<string, string> = {
  ON_LAND: "موقعیت روی خشکی",
  TELEPORT: "جهش ناممکن میان دو گزارش",
  IMPOSSIBLE_SPEED: "سرعت ناممکن",
  IMPOSSIBLE_TURN: "نرخ چرخش ناممکن",
  SPEED_MISMATCH: "سرعت با مسیر نمی‌خواند",
  DUPLICATE_IDENTITY: "یک هویت در دو جا",
  COLLINEAR_TRACK: "مسیر به‌طرز ناممکنی مستقیم",
  MID_REGION_MISMATCH: "پرچم با موقعیت نمی‌خواند",
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

/** Something the radar saw. It says nothing about identity or AIS status. */
export interface SarDetection {
  id: number;
  scene_id: number;
  lon: number;
  lat: number;
  acq_time: string;
  /** Hours since the pass that saw it. A detection is never a live position. */
  age_hours: number;
  satellite: string;
  polarisation: string | null;
  length_m: number | null;
  width_m: number | null;
  /** Modulo 180: radar cannot tell bow from stern. */
  heading_deg: number | null;
  confidence: number;
  detector: string;
  detector_version: string | null;
  pixel_count: number | null;
  peak_dn: number | null;
  product_name: string;
  /** The UAE study box it lies in (trained, chosen and tested there); null elsewhere. */
  study_area?: string | null;
  /** A study or Iranian box where detectors were measured against GFW; null elsewhere. */
  measured_area?: string | null;
  /** Metres to the nearest land (capped at a few km); null if unknown. */
  shore_m?: number | null;
  /** Measured accuracy: a tested detector, in a measured box, at least 1 km from land. */
  accuracy?: DetectionAccuracy | null;
  /** Whether ``accuracy`` applies here. */
  tested?: boolean;
  /** The detection's satellite image on disk; null until it is cut. */
  chip?: ChipImage | null;
  /** Something was detected here on many earlier passes (core/persistence.py); marked, never hidden. */
  persistent?: PersistentSpot | null;
}

/** Precision and recall measured against GFW in the detection's region (core/areas.py). */
export interface DetectionAccuracy {
  area: string;
  region: "uae" | "iran";
  precision: number;
  recall: number;
}

/** One box of one pass a detector searched (/api/analysed). */
export interface AnalysedWindow {
  id: number;
  scene_id: number;
  product_name: string;
  satellite: string;
  acq_time: string;
  age_hours: number;
  detector: string;
  detector_version: string | null;
  polarisation: string | null;
  /** Detections stored now from this search; 0 is not proof that no ship was there. */
  detections: number;
  /** The box that was requested. */
  bbox: BBox;
  /** The part of the box the pass imaged, as a ring of [lon, lat]: what was searched. */
  searched: [number, number][];
  /** Share of the requested box that was searched (the rest lies beyond the swath). */
  searched_share: number;
  footprint_known: boolean;
  /** The study or Iranian box wholly containing the searched area; null otherwise. */
  measured_area: string | null;
  /** A detector configuration measured against GFW, searching inside a measured box. */
  tested: boolean;
}

export interface AnalysedResponse {
  generated_at: string;
  count: number;
  windows: AnalysedWindow[];
  study_areas: string[];
  measured_areas: string[];
  attribution: string;
}

export interface DetectionsResponse {
  generated_at: string;
  count: number;
  detections: SarDetection[];
  attribution: string;
}

export interface Health {
  status: string;
  vessels: number;
  positions: number;
  sar_scenes: number;
  detections: number;
  live_feeds: string[];
  has_live_ais: boolean;
  ais_worker_running?: boolean;
  fresh_ais_vessels: number;
  ais_status: "fresh" | "stale" | "no_observations";
  latest_ais_at: string | null;
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

export type ConfidenceTier = "high" | "medium" | "low";

/** An AIS vessel near a dark-vessel candidate: its hourly cells around the pass. */
export interface NearestAis {
  vessel_id: string;
  mmsi: number | null;
  imo: string | null;
  ship_name: string | null;
  flag: string | null;
  vessel_type: string | null;
  /** Distance from the detection to the vessel's hourly track, in metres. */
  distance_m: number;
  stationary: boolean;
  /** Seen in only one of the three hours: speed unknown. */
  sparse: boolean;
  cells: { hour: string; lon: number; lat: number }[];
  /** The polyline the distance was measured to: the cells, extended to the pass time. */
  track: [number, number][];
}

/** GFW's own radar verdict for the candidate's 0.01 degree cell. */
export type GfwCellState = "matched" | "unmatched" | "mixed" | "none" | "no_gfw_data" | "untrusted";

export type Polarisation = "vv" | "vh";

/** A radar chip cut for the card (core/fusion/chips.py): north up, square, at a fixed scale. */
export interface ChipImage {
  pols: Polarisation[];
  size_px: number;
  pixel_m: number;
  north_up: boolean;
  /** Fixed logarithmic stretch: the DN shown as black and as white, per polarisation. */
  stretch_dn: Partial<Record<Polarisation, [number, number]>>;
  /** When the chip file was written: versions the image, since event ids can be reused. */
  cut_at?: string | null;
  /** Share of the chip the radar did not image (transparent pixels); null if unknown. */
  no_data?: number | null;
}

/** Why a detection was called dark (core/fusion/evidence.py). */
export interface DarkEvidence {
  status: "dark";
  scene: string;
  acquired: string;
  detection: { lon: number; lat: number; score: number };
  /** CFAR's score is a fixed 1.0 ("passed its test"), not a probability. */
  detector: { name: string | null; version: string | null };
  gate: { margin_m: number; r_over_v_s: number; nearest_gap_m: number | null };
  /** The largest swept gate margin at which the detection is still dark. */
  dark_up_to_m: number;
  confidence: ConfidenceTier;
  /** GFW's own SAR detections in the cell; cell_state says which case a null flag stands for. */
  gfw: {
    cell_matched: boolean | null;
    corroborates: boolean | null;
    cell_state: GfwCellState | null;
  };
  /** Null when unknown (no land within reach of the box). */
  prefilter: { shore_m: number | null; edge_m: number | null };
  nearest_ais: NearestAis[];
  pass_flags: {
    gfw_artefact?: boolean;
    presence_gap?: boolean;
    no_presence?: boolean;
    /** False: too few GFW detections to check this pass's AIS feed for a gap. */
    presence_checked?: boolean;
    gfw_refs?: number;
  };
  datasets: string[];
  /** The detection in its pass; `image` once its chip was cut (scripts/cut_chips.py). */
  chip: { scene: string; lon: number; lat: number; image?: ChipImage };
  attribution: string;
}

/** One stored pass over a measured box (/api/timeline). */
export interface TimelinePass {
  scene_id: number;
  product_name: string;
  satellite: string;
  acq_time: string;
  /** The product model searched this box on this pass. */
  searched: boolean;
  /** Detections stored now from that search. */
  detections: number;
  /** The pass holds dark-vessel candidates in this box. */
  fused: boolean;
}

export interface TimelineArea {
  name: string;
  region: "uae" | "iran";
  lon: number;
  lat: number;
  /** Oldest first. */
  passes: TimelinePass[];
}

export interface TimelineResponse {
  generated_at: string;
  since: string;
  days: number;
  areas: TimelineArea[];
}

/** A spot where the model detected something on many archive passes (scripts/build_persistence.py). */
export interface PersistentSpot {
  area: string;
  passes_seen: number;
  passes_imaged: number;
  first: string;
  last: string;
}

export interface AnalystEvent {
  id: number;
  event_type: string;
  ts: string;
  /** Hours since the radar pass. Fusion runs days after a pass: never live. */
  age_hours: number;
  lon: number | null;
  lat: number | null;
  severity: string;
  detection_id: number | null;
  scene_id: number | null;
  run: string | null;
  acknowledged: boolean;
  evidence: DarkEvidence;
  /** Something was detected here on many earlier passes (core/persistence.py); marked, never hidden. */
  persistent?: PersistentSpot | null;
}

export interface EventsResponse {
  generated_at: string;
  count: number;
  events: AnalystEvent[];
  attribution: string;
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
  vessels: (bbox?: BBox, maxAgeMinutes = 360, estimates = true) => {
    const params = new URLSearchParams({
      max_age_minutes: String(maxAgeMinutes),
      estimates: String(estimates),
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
  /** ``asOf`` (ISO, from the timeline scrubber) looks from an earlier moment; null is now. */
  scenes: (days = 14, asOf: string | null = null) =>
    getJson<ScenesResponse>(`/api/scenes?days=${days}${asOfQuery(asOf)}`),
  /** The product model's detections only, unless another detector is asked for. */
  detections: (days = 14, detector = "product", asOf: string | null = null) =>
    getJson<DetectionsResponse>(
      `/api/detections?days=${days}&detector=${detector}${asOfQuery(asOf)}`,
    ),
  darkVessels: (asOf: string | null = null) =>
    getJson<EventsResponse>(`/api/events?type=DARK_VESSEL${asOfQuery(asOf)}`),
  analysed: (days = 14, detector = "product", asOf: string | null = null) =>
    getJson<AnalysedResponse>(`/api/analysed?days=${days}&detector=${detector}${asOfQuery(asOf)}`),
  /** Every stored pass over each measured box: the timeline scrubber's bands. */
  timeline: (days = 60) => getJson<TimelineResponse>(`/api/timeline?days=${days}`),
  /**
   * Served from disk; the browser never causes a read of the radar scene itself.
   * The query names the chip's content (scene, position, cut time), so an event
   * id that re-fusion gave to another detection can never show a cached image.
   */
  chipUrl: (eventId: number, pol: Polarisation, chip: DarkEvidence["chip"], attempt = 0) =>
    `/api/events/${eventId}/chip/${pol}?${chipQuery(chip.scene, chip.lat, chip.lon, chip.image, attempt)}`,
  /** A detection's satellite image, keyed by its content like a candidate's. */
  detectionChipUrl: (detection: SarDetection, pol: Polarisation, attempt = 0) =>
    `/api/detections/${detection.id}/chip/${pol}?${chipQuery(
      detection.product_name ?? "",
      detection.lat,
      detection.lon,
      detection.chip,
      attempt,
    )}`,
};

function asOfQuery(asOf: string | null): string {
  return asOf ? `&as_of=${encodeURIComponent(asOf)}` : "";
}

function chipQuery(
  scene: string,
  lat: number,
  lon: number,
  image: ChipImage | null | undefined,
  attempt: number,
): URLSearchParams {
  const params = new URLSearchParams({ v: `${scene}/${lat}_${lon}/${image?.cut_at ?? ""}` });
  if (attempt > 0) params.set("attempt", String(attempt));
  return params;
}

// As in core/geo.py, so overlays land where core/sar/display.py sampled.
const EARTH_RADIUS_M = 6_371_008.8;

/**
 * Where a position falls on a north-up chip centred on (lon0, lat0), in chip
 * pixels from the top-left corner (core/sar/display.py north_up_grid).
 */
export function chipPixel(
  lon: number,
  lat: number,
  lon0: number,
  lat0: number,
  pixelM: number,
  sizePx: number,
): [number, number] {
  const rad = Math.PI / 180;
  const east = (lon - lon0) * rad * EARTH_RADIUS_M * Math.cos(lat0 * rad);
  const north = (lat - lat0) * rad * EARTH_RADIUS_M;
  return [sizePx / 2 + east / pixelM, sizePx / 2 - north / pixelM];
}

/**
 * Dark-vessel candidates outrank everything on the map. The tier says how far
 * from any AIS track the detection lies: high stays dark even with a 2 km gate
 * margin, low only just misses an AIS vessel's gate.
 */
// A red-orange family, kept well clear of the radar rings' yellow.
export const DARK_COLOURS: Record<ConfidenceTier, string> = {
  high: "#ff3b30",
  medium: "#ff6a00",
  low: "#ff8f40",
};

export const DARK_TIER_LABELS: Record<ConfidenceTier, string> = {
  high: "زیاد: حتی با حاشیه ۲ کیلومتری تاریک",
  medium: "متوسط: با حاشیه ۰٫۵ کیلومتری تاریک، با ۲ کیلومتر نه",
  low: "کم: تا ۰٫۵ کیلومتری دروازه یک شناور AIS",
};

export function tierOf(event: AnalystEvent): ConfidenceTier {
  const tier = event.evidence.confidence;
  return tier === "high" || tier === "medium" ? tier : "low";
}

/** Boxes a detector searched: neutral, so they never read as contacts. */
export const ANALYSED_COLOUR = "#9fb3c8";

/** Radar detections share one colour: what matters is that radar saw them. */
export const RADAR_COLOUR = "#ffd84a";

/** Colour a radar footprint by how long ago it was acquired. */
export function coverageColour(ageHours: number): string {
  if (ageHours < 12) return "#3ad6a8";
  if (ageHours < 48) return "#4aa3ff";
  if (ageHours < 120) return "#8a7de8";
  return "#5a6274";
}

/** Human-readable age. The UI states this rather than implying freshness. */
export const CLASS_COLOURS: Record<CoarseClass | "unknown", string> = {
  cargo_tanker: "#4aa3ff",
  fishing: "#5fd08a",
  service_tug: "#f0a24a",
  passenger_other: "#c081e8",
  unknown: "#8a94a6",
};

export const CLASS_LABELS: Record<CoarseClass | "unknown", string> = {
  cargo_tanker: "باری / نفتکش",
  fishing: "ماهیگیری",
  service_tug: "خدماتی / یدک‌کش",
  passenger_other: "مسافری / سایر",
  unknown: "شناسایی‌نشده",
};

export function classOf(vessel: Vessel): CoarseClass | "unknown" {
  return vessel.coarse_class ?? "unknown";
}
