/**
 * Operational console shell: filter rail, map, context panel.
 *
 * Two honesty rules drive this layout, both from the plan:
 *  - every contact states its age explicitly, and
 *  - the data's provenance is always visible, including a loud warning when
 *    what is on screen is synthetic rather than observed.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { VesselMap } from "./VesselMap";
import {
  ANOMALY_LABELS,
  BBox,
  CLASS_COLOURS,
  CLASS_LABELS,
  CoarseClass,
  Health,
  Track,
  Vessel,
  api,
  classOf,
  formatAge,
} from "./api";

const REFRESH_MS = 5000;
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

  useEffect(() => {
    api.health().then(setHealth).catch(() => undefined);
    api
      .aoi()
      .then((r) => setAoi(r.gulf))
      .catch(() => undefined);
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

  const selectedVessel = vessels.find((v) => v.mmsi === selected) ?? null;
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
        </footer>
      </aside>

      <main className="stage">
        <VesselMap
          vessels={visible}
          track={track?.points ?? null}
          selected={selected}
          aoi={aoi}
          onSelect={setSelected}
        />
      </main>

      <aside className={`context ${selectedVessel ? "open" : ""}`}>
        {selectedVessel ? (
          <VesselDetail vessel={selectedVessel} track={track} onClose={() => setSelected(null)} />
        ) : (
          <div className="empty">
            <h2>No contact selected</h2>
            <p>Click a vessel on the map to see its identity and recent track.</p>
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
