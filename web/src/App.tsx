/**
 * Operational console shell: filter rail, map, context panel.
 *
 * Two honesty rules drive this layout, both from the plan:
 *  - every contact states its age explicitly, and
 *  - the data's provenance is always visible, including a loud warning when
 *    what is on screen is synthetic rather than observed.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { VesselMap } from "./VesselMap";
import {
  ANOMALY_LABELS,
  AnalysedResponse,
  AnalysedWindow,
  AnalystEvent,
  BBox,
  ChipImage,
  CLASS_COLOURS,
  CLASS_LABELS,
  CoarseClass,
  ConfidenceTier,
  DARK_COLOURS,
  DARK_TIER_LABELS,
  DetectionsResponse,
  EventsResponse,
  Health,
  NearestAis,
  Polarisation,
  RADAR_COLOUR,
  SarDetection,
  ScenesResponse,
  Track,
  Vessel,
  api,
  chipPixel,
  classOf,
  coverageColour,
  formatAge,
  tierOf,
} from "./api";

const REFRESH_MS = 5000;
// Detections change only when a detection job runs, so poll gently.
const RADAR_REFRESH_MS = 60_000;
const RADAR_LOOKBACK_DAYS = 14;
// A stable empty list, so hiding the layer does not re-render the map each tick.
const NO_DETECTIONS: SarDetection[] = [];
const NO_EVENTS: AnalystEvent[] = [];
const NO_WINDOWS: AnalysedWindow[] = [];
const NO_NEAREST: NearestAis[] = [];
// The product model keeps peaks scoring at least this (run 2, chosen on validation).
const MODEL_THRESHOLD = 0.6;
const TIERS: ConfidenceTier[] = ["high", "medium", "low"];
// The feed lists this many candidates; the count above it says how many exist.
const FEED_LENGTH = 25;
// The chip's scale bar: about two large tankers end to end, a fifth of the chip.
const CHIP_SCALE_BAR_M = 500;
const ALL_CLASSES: (CoarseClass | "unknown")[] = [
  "cargo_tanker",
  "fishing",
  "service_tug",
  "passenger_other",
  "unknown",
];

export default function App() {
  const [vessels, setVessels] = useState<Vessel[]>([]);
  const [health, setHealth] = useState<Health | null>(null);
  const [aoi, setAoi] = useState<BBox | null>(null);
  const [selected, setSelected] = useState<number | null>(null);
  const [track, setTrack] = useState<Track | null>(null);
  const [hidden, setHidden] = useState<Set<string>>(new Set());
  const [maxAgeMinutes, setMaxAgeMinutes] = useState(360);
  const [error, setError] = useState<string | null>(null);
  const [attribution, setAttribution] = useState<string[]>([]);
  const [suspectCount, setSuspectCount] = useState(0);
  const [suspectOnly, setSuspectOnly] = useState(false);
  const [coverage, setCoverage] = useState<ScenesResponse | null>(null);
  const [showCoverage, setShowCoverage] = useState(true);
  const [radar, setRadar] = useState<DetectionsResponse | null>(null);
  const [radarError, setRadarError] = useState<string | null>(null);
  const [analysed, setAnalysed] = useState<AnalysedResponse | null>(null);
  const [analysedError, setAnalysedError] = useState<string | null>(null);
  const [showRadar, setShowRadar] = useState(true);
  const [dark, setDark] = useState<EventsResponse | null>(null);
  const [darkError, setDarkError] = useState<string | null>(null);
  const [showDark, setShowDark] = useState(true);
  const [selectedEventId, setSelectedEventId] = useState<number | null>(null);
  // Id and position: a re-run can give a detection's id to another one.
  const [selectedDetectionKey, setSelectedDetectionKey] = useState<{
    id: number;
    lon: number;
    lat: number;
  } | null>(null);
  const detectionsRef = useRef<SarDetection[]>([]);

  useEffect(() => {
    api.health().then(setHealth).catch(() => undefined);
    api
      .aoi()
      .then((r) => setAoi(r.gulf))
      .catch(() => undefined);
    api.scenes(14).then(setCoverage).catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await api.detections(RADAR_LOOKBACK_DAYS);
        if (cancelled) return;
        setRadar(response);
        setRadarError(null);
      } catch (err) {
        if (!cancelled) setRadarError((err as Error).message);
      }
    };
    load();
    const timer = setInterval(load, RADAR_REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  // Where the detector searched: changes only when a detection job runs.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await api.analysed(RADAR_LOOKBACK_DAYS);
        if (cancelled) return;
        setAnalysed(response);
        setAnalysedError(null);
      } catch (err) {
        if (!cancelled) setAnalysedError((err as Error).message);
      }
    };
    load();
    const timer = setInterval(load, RADAR_REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  const searchedCounts = useMemo(() => {
    const windows = analysed?.windows ?? [];
    return {
      tested: windows.filter((w) => w.tested).length,
      untested: windows.filter((w) => !w.tested).length,
    };
  }, [analysed]);

  // Fusion runs in batch days after a pass, so events change rarely.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await api.darkVessels();
        if (cancelled) return;
        setDark(response);
        setDarkError(null);
      } catch (err) {
        if (!cancelled) setDarkError((err as Error).message);
      }
    };
    load();
    const timer = setInterval(load, RADAR_REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await api.vessels(undefined, maxAgeMinutes);
        if (cancelled) return;
        setVessels(response.vessels);
        setAttribution(response.attribution);
        setSuspectCount(response.suspect_count);
        setError(null);
      } catch (err) {
        if (!cancelled) setError((err as Error).message);
      }
    };
    load();
    const timer = setInterval(load, REFRESH_MS);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [maxAgeMinutes]);

  useEffect(() => {
    if (selected == null) {
      setTrack(null);
      return;
    }
    api.track(selected).then(setTrack).catch(() => setTrack(null));
  }, [selected]);

  const visible = useMemo(
    () =>
      vessels.filter(
        (v) => !hidden.has(classOf(v)) && (!suspectOnly || v.is_suspect),
      ),
    [vessels, hidden, suspectOnly],
  );

  const counts = useMemo(() => {
    const tally = new Map<string, number>();
    for (const vessel of vessels) {
      const key = classOf(vessel);
      tally.set(key, (tally.get(key) ?? 0) + 1);
    }
    return tally;
  }, [vessels]);

  const toggle = useCallback((key: string) => {
    setHidden((current) => {
      const next = new Set(current);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  // Newest pass first, then the most confident: what an analyst should look at first.
  const feed = useMemo(
    () =>
      [...(dark?.events ?? [])]
        .sort(
          (a, b) =>
            b.ts.localeCompare(a.ts) || TIERS.indexOf(tierOf(a)) - TIERS.indexOf(tierOf(b)),
        )
        .slice(0, FEED_LENGTH),
    [dark],
  );

  const darkCounts = useMemo(() => {
    const tally = new Map<ConfidenceTier, number>();
    for (const event of dark?.events ?? []) {
      tally.set(tierOf(event), (tally.get(tierOf(event)) ?? 0) + 1);
    }
    return tally;
  }, [dark]);

  const selectVessel = useCallback((mmsi: number | null) => {
    setSelected(mmsi);
    if (mmsi != null) {
      setSelectedEventId(null);
      setSelectedDetectionKey(null);
    }
  }, []);
  const selectEvent = useCallback((id: number | null) => {
    setSelectedEventId(id);
    if (id != null) {
      setSelected(null);
      setSelectedDetectionKey(null);
    }
  }, []);
  const selectDetection = useCallback((id: number | null) => {
    const found = detectionsRef.current.find((d) => d.id === id);
    setSelectedDetectionKey(found ? { id: found.id, lon: found.lon, lat: found.lat } : null);
    if (found) {
      setSelected(null);
      setSelectedEventId(null);
    }
  }, []);

  const selectedVessel = vessels.find((v) => v.mmsi === selected) ?? null;
  const selectedEvent =
    (showDark && dark?.events.find((e) => e.id === selectedEventId)) || null;
  detectionsRef.current = radar?.detections ?? NO_DETECTIONS;
  const selectedDetection =
    (showRadar &&
      selectedDetectionKey &&
      radar?.detections.find(
        (d) =>
          d.id === selectedDetectionKey.id &&
          d.lon === selectedDetectionKey.lon &&
          d.lat === selectedDetectionKey.lat,
      )) ||
    null;
  const isSynthetic = attribution.length > 0 && attribution.every((s) => s === "synthetic");

  return (
    <div className="app">
      <aside className="rail">
        <header className="brand">
          <h1>khalij-watch</h1>
          <p>Persian Gulf · maritime domain awareness</p>
        </header>

        {isSynthetic && (
          <div className="banner banner-warn">
            <strong>Synthetic data</strong>
            <span>
              No live AIS feed is configured. Everything on this map is
              simulated and is <em>not</em> a real observation.
            </span>
          </div>
        )}

        <section className="panel">
          <h2>Reported position integrity</h2>
          <p className="hint">
            AIS is self-reported and, in this region, frequently falsified.
            Doubted contacts are kept and marked, never discarded &mdash; a
            fabricated track is itself evidence.
          </p>
          <label className="legend">
            <span
              className={`suspect-row ${suspectCount ? "active" : ""}`}
            >
              <input
                type="checkbox"
                checked={suspectOnly}
                onChange={() => setSuspectOnly((v) => !v)}
              />
              <span className="legend-label">Show only doubted contacts</span>
              <span className="legend-count">{suspectCount}</span>
            </span>
          </label>
        </section>

        <section className="panel">
          <h2>Radar coverage</h2>
          <p className="hint">
            Sentinel-1 is not live. It sees everything, including vessels that
            broadcast nothing &mdash; but only when a satellite passes. Where
            the map is unshaded, nothing has looked recently.
          </p>
          <label className="legend">
            <span className="suspect-row">
              <input
                type="checkbox"
                checked={showCoverage}
                onChange={() => setShowCoverage((v) => !v)}
              />
              <span className="legend-label">Show radar footprints</span>
              <span className="legend-count">{coverage?.count ?? 0}</span>
            </span>
          </label>
          <ul className="legend coverage-key">
            {[
              [12, "under 12h"],
              [48, "12h – 2d"],
              [120, "2 – 5d"],
              [999, "over 5d"],
            ].map(([hours, label]) => (
              <li key={String(label)}>
                <span
                  className="swatch"
                  style={{ background: coverageColour(Number(hours) - 1) }}
                />
                <span className="legend-label">{label}</span>
              </li>
            ))}
          </ul>
          {coverage?.last_look && (
            <p className="hint last-look">
              Last radar look{" "}
              {formatAge(
                (Date.now() - new Date(coverage.last_look).getTime()) / 1000,
              )}
              {coverage.satellites.length > 0 &&
                ` · ${coverage.satellites.join(", ")}`}
            </p>
          )}
        </section>

        <section className="panel">
          <h2>Vessels in satellite images</h2>
          <p className="hint">
            Yellow rings are the deep-learning model&rsquo;s detections in
            Sentinel-1 radar satellite images, whether or not they broadcast
            anything: most are vessels, some are false alarms. Click one to see
            it in its image. Each is a snapshot from one pass, not a live
            position; passes fused with AIS mark their dark-vessel candidates
            separately (below). The CFAR baseline and the superseded first model
            are not shown here.
          </p>
          <label className="legend">
            <span className="suspect-row">
              <input
                type="checkbox"
                checked={showRadar}
                onChange={() => {
                  setShowRadar((v) => !v);
                  setSelectedDetectionKey(null);
                }}
              />
              <span className="ring" style={{ borderColor: RADAR_COLOUR }} />
              <span className="legend-label">
                Show last {RADAR_LOOKBACK_DAYS} days
              </span>
              <span className="legend-count">{radar?.count ?? 0}</span>
            </span>
          </label>
          {analysedError ? (
            <p className="hint">
              The areas detectors searched could not be loaded, so the map
              cannot show where no ring means not searched.
            </p>
          ) : (
            <p className="hint">
              Rings come only from the outlined areas a detector searched, once
              per pass. Outside them, no ring means not searched. Inside, no
              ring is not proof of no ship either: detectors miss some vessels.
            </p>
          )}
          <ul className="legend searched-key">
            <li>
              <span className="box-swatch" />
              <span className="legend-label">
                Searched by the model, where it was tested (except within 1 km of land)
              </span>
              <span className="legend-count">{searchedCounts.tested}</span>
            </li>
            <li>
              <span className="box-swatch dashed" />
              <span className="legend-label">
                Searched by the model, accuracy unknown here (outside the 12 measured boxes)
              </span>
              <span className="legend-count">{searchedCounts.untested}</span>
            </li>
          </ul>
          {radarError && <p className="error">{radarError}</p>}
          {analysedError && <p className="error">{analysedError}</p>}
        </section>

        <section className="panel">
          <h2>Dark-vessel candidates</h2>
          <p className="hint">
            Radar detections that no AIS vessel&rsquo;s hourly track reaches,
            from Global Fishing Watch AIS presence. Fused in batch days after a
            pass. A candidate, not a verdict: reception gaps, jamming and the
            hourly sampling can make an AIS ship look dark. Only passes already
            fused carry candidates; elsewhere no red ring means not checked,
            not clear.
          </p>
          <label className="legend">
            <span className="suspect-row">
              <input
                type="checkbox"
                checked={showDark}
                onChange={() => {
                  setShowDark((v) => !v);
                  setSelectedEventId(null);
                }}
              />
              <span className="ring ring-dark" style={{ borderColor: DARK_COLOURS.medium }} />
              <span className="legend-label">Show candidates</span>
              <span className="legend-count">{dark?.count ?? 0}</span>
            </span>
          </label>
          <ul className="legend tier-key">
            {TIERS.map((tier) => (
              <li key={tier}>
                <span className="ring ring-dark" style={{ borderColor: DARK_COLOURS[tier] }} />
                <span className="legend-label">{DARK_TIER_LABELS[tier]}</span>
                <span className="legend-count">{darkCounts.get(tier) ?? 0}</span>
              </li>
            ))}
          </ul>
          {showDark && feed.length > 0 && (
            <ol className="dark-feed" aria-label="Dark-vessel candidates, newest pass first">
              {feed.map((event) => (
                <li key={event.id}>
                  <button
                    className={event.id === selectedEventId ? "selected" : ""}
                    onClick={() => selectEvent(event.id)}
                  >
                    <span
                      className="ring ring-dark"
                      style={{ borderColor: DARK_COLOURS[tierOf(event)] }}
                    />
                    <span className="feed-text">
                      <span>
                        {event.ts.slice(0, 16).replace("T", " ")} UTC ·{" "}
                        {formatAge(event.age_hours * 3600)}
                      </span>
                      <span className="feed-sub">
                        {event.lat?.toFixed(3)}°, {event.lon?.toFixed(3)}° ·{" "}
                        {event.evidence.gate.nearest_gap_m == null
                          ? "no AIS vessel near"
                          : `${Math.round(event.evidence.gate.nearest_gap_m)} m outside nearest AIS gate`}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          )}
          {(dark?.count ?? 0) > FEED_LENGTH && (
            <p className="hint">
              Showing the {FEED_LENGTH} newest of {dark?.count}; all are on the map.
            </p>
          )}
          {darkError && <p className="error">{darkError}</p>}
        </section>

        <section className="panel">
          <h2>Vessel classes</h2>
          <p className="hint">
            Class comes from AIS, not from imagery. Radar cannot tell a tanker
            from a bulk carrier at 10 m.
          </p>
          <ul className="legend">
            {ALL_CLASSES.map((key) => (
              <li key={key}>
                <label>
                  <input
                    type="checkbox"
                    checked={!hidden.has(key)}
                    onChange={() => toggle(key)}
                  />
                  <span
                    className="swatch"
                    style={{ background: CLASS_COLOURS[key] }}
                  />
                  <span className="legend-label">{CLASS_LABELS[key]}</span>
                  <span className="legend-count">{counts.get(key) ?? 0}</span>
                </label>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">
          <h2>Contact age</h2>
          <p className="hint">
            Older contacts fade on the map. Nothing here is a live radar
            picture.
          </p>
          <label className="slider">
            <input
              type="range"
              min={15}
              max={1440}
              step={15}
              value={maxAgeMinutes}
              onChange={(e) => setMaxAgeMinutes(Number(e.target.value))}
            />
            <span>
              showing last{" "}
              {maxAgeMinutes >= 60
                ? `${Math.floor(maxAgeMinutes / 60)}h`
                : `${maxAgeMinutes}m`}
            </span>
          </label>
        </section>

        <section className="panel">
          <h2>Status</h2>
          <dl className="stats">
            <div>
              <dt>Showing</dt>
              <dd>{visible.length.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Known vessels</dt>
              <dd>{health?.vessels.toLocaleString() ?? "—"}</dd>
            </div>
            <div>
              <dt>Positions</dt>
              <dd>{health?.positions.toLocaleString() ?? "—"}</dd>
            </div>
            <div>
              <dt>Radar scenes</dt>
              <dd>{health?.sar_scenes.toLocaleString() ?? "—"}</dd>
            </div>
            <div>
              <dt>Detections stored (all detectors)</dt>
              <dd>{health?.detections.toLocaleString() ?? "—"}</dd>
            </div>
            <div>
              <dt>Live feeds</dt>
              <dd>{health?.live_feeds.length ? health.live_feeds.join(", ") : "none"}</dd>
            </div>
          </dl>
          {error && <p className="error">{error}</p>}
        </section>

        <footer className="attribution">
          Source: {attribution.length ? attribution.join(", ") : "—"}
          <br />
          Contains modified Copernicus Sentinel data.
          {showDark && (dark?.count ?? 0) > 0 && (
            <>
              <br />
              Dark-vessel AIS: Powered by Global Fishing Watch (CC BY-NC 4.0).
            </>
          )}
        </footer>
      </aside>

      <main className="stage">
        <VesselMap
          vessels={visible}
          scenes={showCoverage ? (coverage?.scenes ?? []) : []}
          detections={
            showRadar ? (radar?.detections ?? NO_DETECTIONS) : NO_DETECTIONS
          }
          track={track?.points ?? null}
          selected={selected}
          aoi={aoi}
          onSelect={selectVessel}
          darkEvents={showDark ? (dark?.events ?? NO_EVENTS) : NO_EVENTS}
          selectedEvent={selectedEvent}
          onSelectEvent={selectEvent}
          analysed={showRadar ? (analysed?.windows ?? NO_WINDOWS) : NO_WINDOWS}
          selectedDetectionId={selectedDetection?.id ?? null}
          onSelectDetection={selectDetection}
        />
      </main>

      <aside
        className={`context ${selectedVessel || selectedEvent || selectedDetection ? "open" : ""}`}
      >
        {selectedEvent ? (
          <DarkVesselDetail event={selectedEvent} onClose={() => setSelectedEventId(null)} />
        ) : selectedDetection ? (
          <DetectionDetail
            detection={selectedDetection}
            onClose={() => setSelectedDetectionKey(null)}
          />
        ) : selectedVessel ? (
          <VesselDetail vessel={selectedVessel} track={track} onClose={() => setSelected(null)} />
        ) : (
          <div className="empty">
            <h2>No contact selected</h2>
            <p>
              Click a yellow ring to see the vessel in its satellite image, or a
              red or orange ring to see why it was called dark.
            </p>
          </div>
        )}
      </aside>
    </div>
  );
}

function VesselDetail({
  vessel,
  track,
  onClose,
}: {
  vessel: Vessel;
  track: Track | null;
  onClose: () => void;
}) {
  const key = classOf(vessel);
  return (
    <div className="detail">
      <button className="close" onClick={onClose} aria-label="Close">
        ×
      </button>
      <h2>{vessel.name ?? `MMSI ${vessel.mmsi}`}</h2>
      <p className="subtitle">
        <span className="swatch" style={{ background: CLASS_COLOURS[key] }} />
        {CLASS_LABELS[key]}
      </p>

      <div className="age-badge">Last reported {formatAge(vessel.age_seconds)}</div>

      <dl className="stats stacked">
        <div>
          <dt>MMSI</dt>
          <dd>{vessel.mmsi}</dd>
        </div>
        {track?.imo && (
          <div>
            <dt>IMO</dt>
            <dd>{track.imo}</dd>
          </div>
        )}
        <div>
          <dt>Position</dt>
          <dd>
            {vessel.lat.toFixed(4)}°, {vessel.lon.toFixed(4)}°
          </dd>
        </div>
        <div>
          <dt>Speed</dt>
          <dd>{vessel.sog_kn != null ? `${vessel.sog_kn.toFixed(1)} kn` : "not reported"}</dd>
        </div>
        <div>
          <dt>Course</dt>
          <dd>{vessel.cog_deg != null ? `${vessel.cog_deg.toFixed(0)}°` : "not reported"}</dd>
        </div>
        <div>
          <dt>Source</dt>
          <dd>{vessel.source}</dd>
        </div>
        <div>
          <dt>Track points</dt>
          <dd>{track?.points.length ?? 0}</dd>
        </div>
      </dl>

      {track?.sanctioned && <div className="banner banner-alert">Sanctioned vessel</div>}

      {vessel.is_suspect && vessel.anomaly_flags && (
        <div className="anomalies">
          <h3>
            Position doubted
            {vessel.plausibility_score != null && (
              <span className="score">
                confidence {(vessel.plausibility_score * 100).toFixed(0)}%
              </span>
            )}
          </h3>
          <ul>
            {vessel.anomaly_flags.map((flag) => (
              <li key={flag.code}>
                <strong>{ANOMALY_LABELS[flag.code] ?? flag.code}</strong>
                <span>{flag.detail}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      <p className="hint">
        Identity is self-reported over AIS and may be falsified. In this region
        a large share of transits broadcast fabricated positions.
      </p>
    </div>
  );
}

/** What GFW's own radar matching says about the candidate's cell. */
function gfwLine(evidence: AnalystEvent["evidence"]): string {
  switch (evidence.gfw.cell_state) {
    case "untrusted":
      return "Not usable on this pass: GFW's matching failed for the whole pass.";
    case "unmatched":
      return "GFW also saw a vessel here without AIS (corroborates).";
    case "matched":
      return "GFW matched a vessel here to AIS (contradicts).";
    case "mixed":
      return "GFW saw vessels here with and without AIS: no single verdict.";
    case "none":
      return "GFW has no detection in this cell.";
    case "no_gfw_data":
      return "GFW published no radar detections for this pass.";
    default:
      return "Not recorded.";
  }
}

/** CFAR's 1.0 means it passed a fixed test, not that it is certain. */
function detectorLine(evidence: AnalystEvent["evidence"]): string {
  const { name, version } = evidence.detector ?? { name: null, version: null };
  if (name === "cfar") return "CFAR (passed its fixed false-alarm test; no probability)";
  if (name === "pointdet") {
    return `model ${version ?? ""}, score ${evidence.detection.score.toFixed(2)}`.trim();
  }
  return name ? `${name} ${version ?? ""}`.trim() : "not recorded";
}

function formatMetres(metres: number | null): string {
  if (metres == null) return "unknown";
  return metres >= 1000 ? `${(metres / 1000).toFixed(1)} km` : `${Math.round(metres)} m`;
}

/** An AIS presence cell is 0.01 degree: about 1 km, where the vessel was during that hour. */
const AIS_CELL_DEG = 0.01;

// Global Fishing Watch reports no vessel within about 1 km of land (core/areas.py).
const NEAR_SHORE_M = 1000;
const REGION_NAMES = { uae: "UAE study box", iran: "Iranian coast box" } as const;

/** Whether the model's accuracy was measured where this detection lies, and what it was. */
function testedHere(detection: SarDetection): string {
  const accuracy = detection.accuracy;
  if (detection.tested && accuracy) {
    const tenths = (value: number) => Math.round(value * 10);
    return (
      `yes (${REGION_NAMES[accuracy.region]}, over 1 km from land): on Sentinel-1C/1D passes ` +
      `of September 2026 about ${tenths(accuracy.precision)} in 10 of its detections matched a ` +
      `vessel Global Fishing Watch saw, and it found about ${tenths(accuracy.recall)} in 10 of ` +
      `those (precision ${accuracy.precision.toFixed(2)}, recall ${accuracy.recall.toFixed(2)})`
    );
  }
  if (!detection.measured_area) {
    return "no: outside the boxes where it was measured; its false alarms and misses were never measured here";
  }
  if (detection.shore_m == null) {
    return "no: its distance to land is unknown, and within 1 km of land its accuracy was never measured";
  }
  if (detection.shore_m < NEAR_SHORE_M) {
    return (
      `no: ${Math.round(detection.shore_m)} m from land. Global Fishing Watch reports no ` +
      "vessels within about 1 km of land, so the model's accuracy there was never measured"
    );
  }
  return "no: this detector version was never measured against Global Fishing Watch";
}

function hourStart(iso: string): number {
  const time = new Date(iso).getTime();
  return time - (time % 3_600_000);
}

/**
 * What the radar saw around a candidate: the chip cut from its pass
 * (scripts/cut_chips.py), north up, with the evidence drawn on it - the
 * detection's ring and the nearest AIS vessels' hourly cells and tracks.
 * AIS is drawn in cyan, never white, so it cannot be taken for a radar echo.
 */
function SarChip({
  image,
  imageUrl,
  centre,
  ringColour,
  nearest,
  passTs,
}: {
  image: ChipImage | null | undefined;
  imageUrl: (pol: Polarisation, attempt: number) => string;
  centre: { lon: number; lat: number };
  ringColour: string;
  /** AIS vessels to draw (dark-vessel candidates); none for a plain detection. */
  nearest: NearestAis[];
  passTs: string;
}) {
  const [choice, setChoice] = useState<Polarisation>("vv");
  const [failedUrl, setFailedUrl] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);
  if (!image || !image.north_up || image.pols.length === 0) {
    return <p className="sar-chip-missing">No satellite image stored for this detection yet.</p>;
  }

  const pol = image.pols.includes(choice) ? choice : image.pols[0];
  const url = imageUrl(pol, attempt);
  const size = image.size_px;
  const toPx = (lon: number, lat: number) =>
    chipPixel(lon, lat, centre.lon, centre.lat, image.pixel_m, size);
  const bar = CHIP_SCALE_BAR_M / image.pixel_m;
  const acrossKm = ((size * image.pixel_m) / 1000).toFixed(2);
  const passHour = hourStart(passTs);
  const half = AIS_CELL_DEG / 2;
  return (
    <figure className="sar-chip">
      <div className="chip-frame">
        {failedUrl === url ? (
          <div className="chip-error">
            <p>The radar image could not be loaded.</p>
            <button onClick={() => setAttempt((n) => n + 1)}>Try again</button>
          </div>
        ) : (
          <>
            <img
              src={url}
              alt={`Radar image (${pol.toUpperCase()}) ${acrossKm} km across, north up, centred on the candidate`}
              onError={() => setFailedUrl(url)}
            />
            <svg viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
              {nearest.map((vessel) => {
                const name = vessel.ship_name ?? (vessel.mmsi ? `MMSI ${vessel.mmsi}` : "unnamed");
                const track = vessel.track ?? vessel.cells.map((c) => [c.lon, c.lat]);
                return (
                  <g key={vessel.vessel_id}>
                    {vessel.cells.map((cell) => {
                      const [x0, y0] = toPx(cell.lon - half, cell.lat + half);
                      const [x1, y1] = toPx(cell.lon + half, cell.lat - half);
                      const atPass = hourStart(cell.hour) === passHour;
                      const hour = new Date(cell.hour).toISOString().slice(11, 16);
                      return (
                        <rect
                          key={cell.hour}
                          className={`chip-ais-cell${atPass ? " pass-hour" : ""}`}
                          x={x0}
                          y={y0}
                          width={x1 - x0}
                          height={y1 - y0}
                        >
                          <title>
                            {`${name}: AIS cell at ${hour} UTC${atPass ? " (the pass hour)" : ""}`}
                          </title>
                        </rect>
                      );
                    })}
                    {track.length > 1 && (
                      <polyline
                        className="chip-ais-track"
                        points={track.map(([lon, lat]) => toPx(lon, lat).join(",")).join(" ")}
                      />
                    )}
                  </g>
                );
              })}
              <circle
                className="chip-ring"
                cx={size / 2}
                cy={size / 2}
                r={12}
                stroke={ringColour}
              />
              <g className="chip-scale" transform={`translate(10 ${size - 10})`}>
                <line x1={0} y1={0} x2={bar} y2={0} />
                <text x={bar / 2} y={-5}>
                  {CHIP_SCALE_BAR_M} m
                </text>
              </g>
              <g className="chip-north" transform={`translate(${size - 18} 8)`}>
                <path d="M5 0 L10 13 L5 10 L0 13 Z" />
                <text x={5} y={25}>
                  N
                </text>
              </g>
            </svg>
          </>
        )}
      </div>
      {image.pols.length > 1 && (
        <div className="chip-pols" role="group" aria-label="Radar polarisation">
          {image.pols.map((p) => (
            <button
              key={p}
              aria-pressed={p === pol}
              className={p === pol ? "selected" : ""}
              onClick={() => setChoice(p)}
            >
              {p.toUpperCase()}
            </button>
          ))}
        </div>
      )}
      <figcaption>
        The Sentinel-1 radar image, {acrossKm} km across, north up. Bright means a
        strong echo, as from a steel hull; the ring marks the detection.
        {nearest.length > 0 &&
          " Cyan squares are the nearest AIS vessels\u2019 hourly cells (about 1 km: the vessel was somewhere inside during that hour; solid for the pass hour)."}
        {image.no_data == null
          ? " Hatched areas, if any, were not imaged by the radar."
          : image.no_data > 0 && " Hatched areas were not imaged by the radar."}
      </figcaption>
    </figure>
  );
}

/** A vessel the deep-learning model found in a Sentinel-1 image, with the image beside it. */
function DetectionDetail({
  detection,
  onClose,
}: {
  detection: SarDetection;
  onClose: () => void;
}) {
  const passTime = detection.acq_time.slice(0, 16).replace("T", " ");
  return (
    <div className="detail">
      <button className="close" onClick={onClose} aria-label="Close">
        ×
      </button>
      <h2>Detection in a satellite image</h2>
      <p className="subtitle">
        <span className="ring" style={{ borderColor: RADAR_COLOUR }} />
        Likely a vessel: found by the deep-learning model in a Sentinel-1 radar image
      </p>

      <div className="age-badge">Radar pass {formatAge(detection.age_hours * 3600)}</div>

      <SarChip
        key={detection.id}
        image={detection.chip}
        imageUrl={(pol, attempt) => api.detectionChipUrl(detection, pol, attempt)}
        centre={detection}
        ringColour={RADAR_COLOUR}
        nearest={NO_NEAREST}
        passTs={detection.acq_time}
      />

      <dl className="stats stacked">
        <div>
          <dt>Position</dt>
          <dd>
            {detection.lat.toFixed(4)}°, {detection.lon.toFixed(4)}°
          </dd>
        </div>
        <div>
          <dt>Model score</dt>
          <dd>
            {detection.confidence.toFixed(2)} (kept from {MODEL_THRESHOLD.toFixed(2)}; not a
            probability)
          </dd>
        </div>
        <div>
          <dt>Pass</dt>
          <dd>
            {detection.satellite} · {passTime} UTC
          </dd>
        </div>
        <div>
          <dt>Model tested here</dt>
          <dd>
            {testedHere(detection)}
          </dd>
        </div>
        {detection.product_name && (
          <div>
            <dt>Scene</dt>
            <dd className="scene-name">{detection.product_name.replace(/_COG\.SAFE$/, "")}</dd>
          </div>
        )}
      </dl>

      <p className="hint">
        The model says where it sees a vessel, not what it is: it has no class,
        length or heading output, and at 10 m radar pixels a tanker and a bulk
        carrier look alike. Whether this object carried AIS is checked only on
        fused passes (dark-vessel candidates, red and orange).
      </p>
      <p className="hint">Contains modified Copernicus Sentinel data.</p>
    </div>
  );
}

function DarkVesselDetail({ event, onClose }: { event: AnalystEvent; onClose: () => void }) {
  const evidence = event.evidence;
  const tier = tierOf(event);
  return (
    <div className="detail">
      <button className="close" onClick={onClose} aria-label="Close">
        ×
      </button>
      <h2>Dark-vessel candidate</h2>
      <p className="subtitle">
        <span className="ring ring-dark" style={{ borderColor: DARK_COLOURS[tier] }} />
        {DARK_TIER_LABELS[tier]}
      </p>

      <div className="age-badge">Radar pass {formatAge(event.age_hours * 3600)}</div>

      <SarChip
        key={event.id}
        image={evidence.chip.image}
        imageUrl={(pol, attempt) => api.chipUrl(event.id, pol, evidence.chip, attempt)}
        centre={evidence.chip}
        ringColour={DARK_COLOURS[tier]}
        nearest={evidence.nearest_ais}
        passTs={event.ts}
      />

      <div className="why-dark">
        <h3>Why dark</h3>
        <p>
          No AIS vessel&rsquo;s hourly track passes within its gate.{" "}
          {evidence.gate.nearest_gap_m == null
            ? "No AIS vessel at all was near this pass."
            : `It lies ${formatMetres(evidence.gate.nearest_gap_m)} outside the nearest AIS vessel's gate (still dark at the ${formatMetres(evidence.dark_up_to_m)} step of the margin sweep).`}
        </p>
        {evidence.pass_flags.presence_checked === false && (
          <p className="caveat">
            This pass had too few GFW detections to check its AIS feed for gaps:
            a gap in the feed would look like dark vessels.
          </p>
        )}
      </div>

      <dl className="stats stacked">
        <div>
          <dt>Position</dt>
          <dd>
            {evidence.detection.lat.toFixed(4)}°, {evidence.detection.lon.toFixed(4)}°
          </dd>
        </div>
        <div>
          <dt>Detector</dt>
          <dd>{detectorLine(evidence)}</dd>
        </div>
        <div>
          <dt>GFW radar</dt>
          <dd>{gfwLine(evidence)}</dd>
        </div>
        <div>
          <dt>Distance to shore / box edge</dt>
          <dd>
            {formatMetres(evidence.prefilter.shore_m)} / {formatMetres(evidence.prefilter.edge_m)}
          </dd>
        </div>
        <div>
          <dt>Gate</dt>
          <dd>
            cell + R/V {evidence.gate.r_over_v_s} s × speed + margin{" "}
            {formatMetres(evidence.gate.margin_m)}
          </dd>
        </div>
        <div>
          <dt>Pass</dt>
          <dd className="scene-name">{evidence.scene.replace(/_COG\.SAFE$/, "")}</dd>
        </div>
      </dl>

      <div className="nearest-ais">
        <h3>Nearest AIS vessels (white on the map)</h3>
        <ul>
          {evidence.nearest_ais.map((vessel) => (
            <li key={vessel.vessel_id}>
              <strong>{vessel.ship_name ?? (vessel.mmsi ? `MMSI ${vessel.mmsi}` : "unnamed")}</strong>
              <span>
                {formatMetres(vessel.distance_m)} away
                {vessel.flag ? ` · ${vessel.flag}` : ""}
                {vessel.stationary ? " · anchored" : vessel.sparse ? " · seen 1 hour" : ""}
              </span>
              <span className="cells">
                {vessel.cells
                  .map((cell) => new Date(cell.hour).toISOString().slice(11, 16))
                  .join(", ")}{" "}
                UTC
              </span>
            </li>
          ))}
        </ul>
      </div>

      <p className="hint">
        A candidate, not a verdict. AIS here comes as one position per vessel
        per hour in about 1 km cells, so a ship can look dark when its hourly
        cell is far from where the radar caught it, and GNSS jamming moves AIS
        positions. Data: {evidence.datasets.join(", ") || "unknown"}.
      </p>
      <p className="hint">{evidence.attribution}</p>
    </div>
  );
}
