/**
 * Operational console shell: filter rail, map, context panel.
 *
 * Two honesty rules drive this layout, both from the plan:
 *  - every contact states its age explicitly, and
 *  - the data's provenance is always visible, including a loud warning when
 *    what is on screen is synthetic rather than observed.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Timeline } from "./Timeline";
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
  PersistentSpot,
  SarDetection,
  ScenesResponse,
  TimelineResponse,
  Track,
  Vessel,
  api,
  chipPixel,
  classOf,
  coverageColour,
  tierOf,
} from "./api";
import { faAge, faCoord, faDate, faDateTime, faMetres, faNumber, faTime } from "./fa";
import { shipTypeLabel } from "./identity";

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
  const [showEstimates, setShowEstimates] = useState(false);
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
  // The moment the radar layers look from (the timeline scrubber); null is now.
  const [asOf, setAsOf] = useState<string | null>(null);
  const [timeline, setTimeline] = useState<TimelineResponse | null>(null);
  const [clockMs, setClockMs] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setClockMs(Date.now()), REFRESH_MS);
    return () => clearInterval(timer);
  }, []);
  const latestAisAge = health?.latest_ais_at ? (clockMs - Date.parse(health.latest_ais_at)) / 1000 : Infinity;
  const hasFreshAis = health?.has_live_ais && latestAisAge >= 0 && latestAisAge <= 300;

  useEffect(() => {
    api.health().then(setHealth).catch(() => undefined);
    api
      .aoi()
      .then((r) => setAoi(r.gulf))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    let cancelled = false;
    api
      .scenes(14, asOf)
      .then((r) => !cancelled && setCoverage(r))
      .catch(() => undefined);
    return () => {
      cancelled = true;
    };
  }, [asOf]);

  // Every pass per box for the timeline: changes only when scenes are stored.
  useEffect(() => {
    let cancelled = false;
    const load = () =>
      api
        .timeline()
        .then((r) => !cancelled && setTimeline(r))
        .catch(() => undefined);
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
        const response = await api.detections(RADAR_LOOKBACK_DAYS, "product", asOf);
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
  }, [asOf]);

  // Where the detector searched: changes only when a detection job runs.
  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await api.analysed(RADAR_LOOKBACK_DAYS, "product", asOf);
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
  }, [asOf]);

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
        const response = await api.darkVessels(asOf);
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
  }, [asOf]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const response = await api.vessels(undefined, maxAgeMinutes, asOf == null);
        if (cancelled) return;
        setVessels(response.vessels);
        setAttribution(response.attribution);
        setSuspectCount(response.suspect_count);
        setError(null);
        api.health().then((r) => !cancelled && setHealth(r)).catch(() => undefined);
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
  }, [maxAgeMinutes, asOf]);

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

  const darkAtSpots = useMemo(
    () => (dark?.events ?? []).filter((event) => event.persistent).length,
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
          <p>خلیج فارس · آگاهی از وضعیت دریایی</p>
        </header>

        {isSynthetic && (
          <div className="banner banner-warn">
            <strong>داده‌های ساختگی</strong>
            <span>
              موقعیت‌های این لایه AIS شبیه‌سازی‌شده‌اند و مشاهده واقعی نیستند.
              این پیام درباره تصاویر راداری ذخیره‌شده نیست.
            </span>
          </div>
        )}

        <section className="panel">
          <h2>گزارش AIS و تخمین کوتاه‌مدت</h2>
          <p className="hint">
            {health?.ais_worker_running ? "دریافت‌کننده AIS همراه برنامه فعال است." : "دریافت‌کننده خودکار AIS فعال نیست."}
          </p>
          <p className="hint">
            {hasFreshAis
              ? `${faNumber(health.fresh_ais_vessels)} شناور با گزارش کمتر از پنج دقیقه`
              : "گزارش تازه AIS دریافت نشده است؛ پوشش کامل منطقه تأیید نشده است."}
          </p>
          <label className="legend">
            <input type="checkbox" checked={showEstimates}
              onChange={() => setShowEstimates((v) => !v)} disabled={asOf != null} />
            <span className="swatch" style={{background: "#e6b5ff"}} />
            <span>نمایش موقعیت تخمینی</span>
          </label>
          <p className="hint">
            نقطه بنفش و خط‌چین: تخمین با فرض سرعت و مسیر ثابت، حداکثر پنج دقیقه
            پس از گزارش ارائه‌دهنده؛ دقت آن تضمین نشده است. نماد اصلی، آخرین مشاهده است.
            {asOf && " تخمین در مرور تاریخی خاموش است؛ لایه AIS مستقل از زمان رادار است."}
          </p>
        </section>

        <section className="panel">
          <h2>درستی موقعیت‌های گزارش‌شده</h2>
          <p className="hint">
            AIS را خود شناورها گزارش می‌کنند و در این منطقه اغلب جعل می‌شود.
            موقعیت‌های مشکوک نگه داشته و علامت‌گذاری می‌شوند، هرگز حذف
            نمی‌شوند &mdash; یک مسیر ساختگی خود یک شاهد است.
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
              <span className="legend-label">فقط موقعیت‌های مشکوک</span>
              <span className="legend-count">{faNumber(suspectCount)}</span>
            </span>
          </label>
        </section>

        <section className="panel">
          <h2>پوشش راداری</h2>
          <p className="hint">
            Sentinel-1 زنده نیست. همه چیز را می‌بیند، حتی شناورهایی که هیچ
            سیگنالی نمی‌فرستند &mdash; اما فقط هنگام گذر ماهواره. هر جا نقشه
            سایه ندارد، اخیراً تصویری گرفته نشده است.
          </p>
          <label className="legend">
            <span className="suspect-row">
              <input
                type="checkbox"
                checked={showCoverage}
                onChange={() => setShowCoverage((v) => !v)}
              />
              <span className="legend-label">نمایش محدوده تصاویر راداری</span>
              <span className="legend-count">{faNumber(coverage?.count ?? 0)}</span>
            </span>
          </label>
          <ul className="legend coverage-key">
            {[
              [12, "کمتر از ۱۲ ساعت"],
              [48, "۱۲ ساعت تا ۲ روز"],
              [120, "۲ تا ۵ روز"],
              [999, "بیش از ۵ روز"],
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
              آخرین تصویر راداری{" "}
              {faAge(
                ((asOf ? Date.parse(asOf) : Date.now()) - new Date(coverage.last_look).getTime()) /
                  1000,
              )}
              {asOf ? " (در لحظه خط زمان)" : ""}
              {coverage.satellites.length > 0 &&
                ` · ${coverage.satellites.join(", ")}`}
            </p>
          )}
        </section>

        <section className="panel">
          <h2>شناورها در تصاویر ماهواره‌ای</h2>
          <p className="hint">
            حلقه‌های زرد آشکارسازی‌های مدل یادگیری عمیق در تصاویر راداری
            ماهواره Sentinel-1 هستند، چه شناور سیگنالی بفرستد چه نه: بیشترشان
            شناورند و برخی هشدار نادرست. روی یکی کلیک کنید تا آن را در تصویرش
            ببینید. هر کدام تصویری لحظه‌ای از یک گذر است، نه موقعیت زنده؛ در
            گذرهایی که با AIS تلفیق شده‌اند نامزدهای شناور تاریک جداگانه
            علامت خورده‌اند (پایین‌تر). خط مبنای CFAR و مدل نخستِ کنارگذاشته‌شده
            اینجا نمایش داده نمی‌شوند.
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
                نمایش {faNumber(RADAR_LOOKBACK_DAYS)} روز اخیر
              </span>
              <span className="legend-count">{faNumber(radar?.count ?? 0)}</span>
            </span>
          </label>
          {analysedError ? (
            <p className="hint">
              ناحیه‌هایی که آشکارسازها جست‌وجو کرده‌اند بارگیری نشد، پس نقشه
              نمی‌تواند نشان دهد کجا نبودِ حلقه یعنی جست‌وجو نشده.
            </p>
          ) : (
            <p className="hint">
              حلقه‌ها فقط از ناحیه‌های قاب‌گرفته‌ای می‌آیند که آشکارساز در هر
              گذر یک بار جست‌وجو کرده است. بیرون از آن‌ها نبودِ حلقه یعنی جست‌وجو
              نشده. درون آن‌ها هم نبودِ حلقه اثبات نبودِ کشتی نیست: آشکارسازها
              برخی شناورها را از دست می‌دهند.
            </p>
          )}
          <ul className="legend searched-key">
            <li>
              <span className="box-swatch" />
              <span className="legend-label">
                جست‌وجو با مدل، جایی که آزموده شده (جز تا ۱ کیلومتری خشکی)
              </span>
              <span className="legend-count">{faNumber(searchedCounts.tested)}</span>
            </li>
            <li>
              <span className="box-swatch dashed" />
              <span className="legend-label">
                جست‌وجو با مدل، دقت اینجا نامعلوم (بیرون از ۱۲ محدوده سنجیده)
              </span>
              <span className="legend-count">{faNumber(searchedCounts.untested)}</span>
            </li>
          </ul>
          {radarError && <p className="error">{radarError}</p>}
          {analysedError && <p className="error">{analysedError}</p>}
        </section>

        <section className="panel">
          <h2>نامزدهای شناور تاریک</h2>
          <p className="hint">
            آشکارسازی‌های راداری که مسیر ساعتیِ هیچ شناور AIS به آن‌ها نمی‌رسد،
            بر پایه حضور AIS از Global Fishing Watch. تلفیق چند روز پس از هر گذر
            به‌صورت دسته‌ای انجام می‌شود. نامزد است، نه حکم: شکاف دریافت، اخلال و
            نمونه‌برداری ساعتی می‌توانند کشتیِ دارای AIS را تاریک نشان دهند. فقط
            گذرهای تلفیق‌شده نامزد دارند؛ جاهای دیگر نبودِ حلقه قرمز یعنی بررسی
            نشده، نه پاک.
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
              <span className="legend-label">نمایش نامزدها</span>
              <span className="legend-count">{faNumber(dark?.count ?? 0)}</span>
            </span>
          </label>
          <ul className="legend tier-key">
            {TIERS.map((tier) => (
              <li key={tier}>
                <span className="ring ring-dark" style={{ borderColor: DARK_COLOURS[tier] }} />
                <span className="legend-label">{DARK_TIER_LABELS[tier]}</span>
                <span className="legend-count">{faNumber(darkCounts.get(tier) ?? 0)}</span>
              </li>
            ))}
          </ul>
          {darkAtSpots > 0 && (
            <p className="hint">
              {faNumber(darkAtSpots)} مورد از آن‌ها در نقطه‌های پایدار است، جایی که در
              بسیاری از گذرهای پیشین چیزی آشکار شده: احتمالاً سازه‌های ثابت یا کشتی‌هایی
              که ماه‌ها لنگر انداخته‌اند (روی کارت‌هایشان علامت خورده‌اند).
            </p>
          )}
          {showDark && feed.length > 0 && (
            <ol className="dark-feed" aria-label="نامزدهای شناور تاریک، جدیدترین گذر نخست">
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
                        {faDateTime(event.ts)} ·{" "}
                        {faAge(event.age_hours * 3600)}
                      </span>
                      <span className="feed-sub">
                        {faNumber(event.lat ?? 0, 3)}°، {faNumber(event.lon ?? 0, 3)}° ·{" "}
                        {event.evidence.gate.nearest_gap_m == null
                          ? "هیچ شناور AIS نزدیک نیست"
                          : `${faMetres(event.evidence.gate.nearest_gap_m)} بیرون از نزدیک‌ترین دروازه AIS`}
                        {event.persistent ? " · نقطه پایدار" : ""}
                      </span>
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          )}
          {(dark?.count ?? 0) > FEED_LENGTH && (
            <p className="hint">
              {faNumber(FEED_LENGTH)} مورد جدیدتر از {faNumber(dark?.count ?? 0)} نمایش داده
              شده؛ همه روی نقشه هستند.
            </p>
          )}
          {darkError && <p className="error">{darkError}</p>}
        </section>

        <section className="panel">
          <h2>رده‌های شناور</h2>
          <p className="hint">
            رده از AIS می‌آید، نه از تصویر. رادار با تفکیک ۱۰ متر نفتکش را از
            فله‌بر تشخیص نمی‌دهد.
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
                  <span className="legend-count">{faNumber(counts.get(key) ?? 0)}</span>
                </label>
              </li>
            ))}
          </ul>
        </section>

        <section className="panel">
          <h2>عمر موقعیت‌ها</h2>
          <p className="hint">
            موقعیت‌های قدیمی‌تر روی نقشه کم‌رنگ می‌شوند. هیچ چیز اینجا تصویر
            راداری زنده نیست.
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
              نمایش{" "}
              {maxAgeMinutes >= 60
                ? `${faNumber(Math.floor(maxAgeMinutes / 60))} ساعت`
                : `${faNumber(maxAgeMinutes)} دقیقه`}{" "}
              اخیر
            </span>
          </label>
        </section>

        <section className="panel">
          <h2>وضعیت</h2>
          <dl className="stats">
            <div>
              <dt>در حال نمایش</dt>
              <dd>{faNumber(visible.length)}</dd>
            </div>
            <div>
              <dt>شناورهای شناخته‌شده</dt>
              <dd>{health ? faNumber(health.vessels) : "—"}</dd>
            </div>
            <div>
              <dt>موقعیت‌ها</dt>
              <dd>{health ? faNumber(health.positions) : "—"}</dd>
            </div>
            <div>
              <dt>تصاویر راداری</dt>
              <dd>{health ? faNumber(health.sar_scenes) : "—"}</dd>
            </div>
            <div>
              <dt>آشکارسازی‌های ذخیره‌شده (همه آشکارسازها)</dt>
              <dd>{health ? faNumber(health.detections) : "—"}</dd>
            </div>
            <div>
              <dt>خوراک‌های زنده</dt>
              <dd>{hasFreshAis ? "گزارش تازه دریافت شده" : "فاقد گزارش تازه"}</dd>
            </div>
          </dl>
          {error && <p className="error">{error}</p>}
        </section>

        <footer className="attribution" dir="ltr">
          Source: {attribution.length ? attribution.join(", ") : "—"}
          {attribution.includes("openwaters") && <>
            <br /><a href="https://openwaters.io/ais/" target="_blank" rel="noreferrer">Open Waters AIS</a>
            {attribution.includes("aishub") && <>
              {" · Data: "}<a href="https://www.aishub.net" target="_blank" rel="noreferrer">AISHub contributors</a>
            </>}
          </>}
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
        <div className="map-holder">
        <VesselMap
          vessels={visible}
          estimatesEnabled={showEstimates && asOf == null}
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
        </div>
        <Timeline
          data={timeline}
          asOf={asOf}
          onChange={(moment) => {
            // A card of a pass after the new moment would describe the future.
            setSelectedEventId(null);
            setSelectedDetectionKey(null);
            setAsOf(moment);
          }}
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
          <VesselDetail vessel={selectedVessel} track={track}
            estimatesEnabled={showEstimates && asOf == null} onClose={() => setSelected(null)} />
        ) : (
          <div className="empty">
            <h2>موردی انتخاب نشده</h2>
            <p>
              روی یک حلقه زرد کلیک کنید تا شناور را در تصویر ماهواره‌ای‌اش ببینید،
              یا روی حلقه قرمز یا نارنجی تا ببینید چرا تاریک خوانده شده است.
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
  estimatesEnabled,
  onClose,
}: {
  vessel: Vessel;
  track: Track | null;
  estimatesEnabled: boolean;
  onClose: () => void;
}) {
  const key = classOf(vessel);
  const reportedImo = vessel.imo ?? track?.imo;
  const [nowMs, setNowMs] = useState(Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNowMs(Date.now()), 1000);
    return () => clearInterval(timer);
  }, []);
  const ageSeconds = Math.max(0, (nowMs - Date.parse(vessel.ts)) / 1000);
  const canEstimate = estimatesEnabled && ageSeconds <= 300 && vessel.estimate?.status === "estimated";
  return (
    <div className="detail">
      <button className="close" onClick={onClose} aria-label="بستن">
        ×
      </button>
      <h2>{vessel.name ?? `MMSI ${vessel.mmsi}`}</h2>
      <p className="subtitle">
        <span className="swatch" style={{ background: CLASS_COLOURS[key] }} />
        {CLASS_LABELS[key]}
      </p>

      <div className="age-badge">آخرین گزارش {faAge(ageSeconds)}</div>

      {canEstimate && vessel.estimate && (
        <div className="banner" style={{borderColor: "#e6b5ff"}}>
          <strong>موقعیت تخمینی با فرض حرکت ثابت</strong>
          <span>{faCoord(vessel.estimate.lat!, vessel.estimate.lon!)} · {faDateTime(vessel.estimate.target_time)}</span>
          <span>بر پایه زمان ارائه‌دهنده، نه زمان مستقل تأییدشده GPS؛ حداکثر فاصله پنج دقیقه</span>
        </div>
      )}
      {ageSeconds > 300 && <p className="hint">گزارش قدیمی است؛ تخمین فعلی تولید نمی‌شود.</p>}

      <dl className="stats stacked">
        <div>
          <dt>MMSI</dt>
          <dd>{vessel.mmsi}</dd>
        </div>
        {reportedImo && (
          <div>
            <dt>IMO</dt>
            <dd>{reportedImo}</dd>
          </div>
        )}
        <div>
          <dt>شناسه فراخوان</dt>
          <dd>{vessel.callsign ?? "گزارش نشده"}</dd>
        </div>
        <div>
          <dt>نوع اعلام‌شده AIS</dt>
          <dd>{shipTypeLabel(vessel.ship_type)}</dd>
        </div>
        <div>
          <dt>موقعیت</dt>
          <dd>{faCoord(vessel.lat, vessel.lon)}</dd>
        </div>
        <div>
          <dt>سرعت</dt>
          <dd>{vessel.sog_kn != null ? `${faNumber(vessel.sog_kn, 1)} گره` : "گزارش نشده"}</dd>
        </div>
        <div>
          <dt>جهت حرکت</dt>
          <dd>{vessel.cog_deg != null ? `${faNumber(vessel.cog_deg)}°` : "گزارش نشده"}</dd>
        </div>
        <div>
          <dt>منبع</dt>
          <dd>{vessel.source}</dd>
        </div>
        <div>
          <dt>نقاط مسیر</dt>
          <dd>{faNumber(track?.points.length ?? 0)}</dd>
        </div>
      </dl>

      {track?.sanctioned && <div className="banner banner-alert">شناور تحریم‌شده</div>}

      {vessel.is_suspect && vessel.anomaly_flags && (
        <div className="anomalies">
          <h3>
            موقعیت مشکوک
            {vessel.plausibility_score != null && (
              <span className="score">
                اطمینان {faNumber(vessel.plausibility_score * 100)}٪
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
        هویت از طریق AIS خوداظهاری است و ممکن است جعل شده باشد. در این منطقه
        بخش بزرگی از عبورها موقعیت ساختگی پخش می‌کنند.
      </p>
    </div>
  );
}

/** What GFW's own radar matching says about the candidate's cell. */
function gfwLine(evidence: AnalystEvent["evidence"]): string {
  switch (evidence.gfw.cell_state) {
    case "untrusted":
      return "در این گذر قابل استفاده نیست: تطبیق GFW برای کل گذر ناموفق بوده است.";
    case "unmatched":
      return "GFW هم اینجا شناوری بدون AIS دیده است (تأیید می‌کند).";
    case "matched":
      return "GFW شناوری را اینجا با AIS تطبیق داده است (نقض می‌کند).";
    case "mixed":
      return "GFW اینجا شناورهایی با و بدون AIS دیده است: حکم واحدی ندارد.";
    case "none":
      return "GFW در این خانه هیچ آشکارسازی ندارد.";
    case "no_gfw_data":
      return "GFW برای این گذر هیچ آشکارسازی راداری منتشر نکرده است.";
    default:
      return "ثبت نشده.";
  }
}

/** CFAR's 1.0 means it passed a fixed test, not that it is certain. */
function detectorLine(evidence: AnalystEvent["evidence"]): string {
  const { name, version } = evidence.detector ?? { name: null, version: null };
  if (name === "cfar") return "CFAR (از آزمون ثابت هشدار نادرست گذشته؛ احتمال نیست)";
  if (name === "pointdet") {
    return `مدل ${version ?? ""}، امتیاز ${faNumber(evidence.detection.score, 2)}`;
  }
  return name ? `${name} ${version ?? ""}`.trim() : "ثبت نشده";
}

/** An AIS presence cell is 0.01 degree: about 1 km, where the vessel was during that hour. */
const AIS_CELL_DEG = 0.01;

/** Why a detection at a persistent spot is probably not a vessel that switched AIS off. */
function persistentNote(spot: PersistentSpot): string {
  return (
    `نقطه پایدار: در ${faNumber(spot.passes_seen)} گذر از ${faNumber(spot.passes_imaged)} ` +
    `گذر پیشین (${faDate(spot.first)} تا ${faDate(spot.last)}) چیزی در فاصله ۱۵۰ متری ` +
    "اینجا آشکار شده است. احتمالاً یک سازه ثابت \u2014 اسکله، مهاربند، بویه یا سکوی " +
    "ثبت‌نشده \u2014 یا کشتی‌ای که ماه‌ها لنگر انداخته است."
  );
}

// Global Fishing Watch reports no vessel within about 1 km of land (core/areas.py).
const NEAR_SHORE_M = 1000;
const REGION_NAMES = { uae: "محدوده مطالعاتی امارات", iran: "محدوده ساحل ایران" } as const;

/** Whether the model's accuracy was measured where this detection lies, and what it was. */
function testedHere(detection: SarDetection): string {
  const accuracy = detection.accuracy;
  if (detection.tested && accuracy) {
    const tenths = (value: number) => faNumber(Math.round(value * 10));
    return (
      `بله (${REGION_NAMES[accuracy.region]}، بیش از ۱ کیلومتر از خشکی): در گذرهای ` +
      `Sentinel-1C/1D از ۱۷ شهریور تا ۷ مهر ۱۴۰۵ حدود ${tenths(accuracy.precision)} از هر ` +
      `۱۰ آشکارسازی‌اش با شناوری که Global Fishing Watch دیده بود منطبق بود و حدود ` +
      `${tenths(accuracy.recall)} از هر ۱۰ مورد آن‌ها را یافت (دقت ` +
      `${faNumber(accuracy.precision, 2)}، بازیابی ${faNumber(accuracy.recall, 2)})`
    );
  }
  if (!detection.measured_area) {
    return "خیر: بیرون از محدوده‌هایی که سنجیده شده؛ هشدارهای نادرست و موارد ازدست‌رفته‌اش اینجا هرگز سنجیده نشده است";
  }
  if (detection.shore_m == null) {
    return "خیر: فاصله‌اش تا خشکی نامعلوم است و دقت تا ۱ کیلومتری خشکی هرگز سنجیده نشده است";
  }
  if (detection.shore_m < NEAR_SHORE_M) {
    return (
      `خیر: ${faMetres(detection.shore_m)} تا خشکی. Global Fishing Watch تا حدود ۱ کیلومتری ` +
      "خشکی هیچ شناوری گزارش نمی‌کند، پس دقت مدل آنجا هرگز سنجیده نشده است"
    );
  }
  return "خیر: این نسخه آشکارساز هرگز با Global Fishing Watch سنجیده نشده است";
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
    return <p className="sar-chip-missing">هنوز تصویر ماهواره‌ای برای این آشکارسازی ذخیره نشده است.</p>;
  }

  const pol = image.pols.includes(choice) ? choice : image.pols[0];
  const url = imageUrl(pol, attempt);
  const size = image.size_px;
  const toPx = (lon: number, lat: number) =>
    chipPixel(lon, lat, centre.lon, centre.lat, image.pixel_m, size);
  const bar = CHIP_SCALE_BAR_M / image.pixel_m;
  const acrossKm = faNumber((size * image.pixel_m) / 1000, 2);
  const passHour = hourStart(passTs);
  const half = AIS_CELL_DEG / 2;
  return (
    <figure className="sar-chip">
      <div className="chip-frame">
        {failedUrl === url ? (
          <div className="chip-error">
            <p>تصویر راداری بارگیری نشد.</p>
            <button onClick={() => setAttempt((n) => n + 1)}>تلاش دوباره</button>
          </div>
        ) : (
          <>
            <img
              src={url}
              alt={`تصویر راداری (${pol.toUpperCase()}) به عرض ${acrossKm} کیلومتر، شمال بالا، با مرکز آشکارسازی`}
              onError={() => setFailedUrl(url)}
            />
            <svg viewBox={`0 0 ${size} ${size}`} aria-hidden="true">
              {nearest.map((vessel) => {
                const name = vessel.ship_name ?? (vessel.mmsi ? `MMSI ${vessel.mmsi}` : "بی‌نام");
                const track = vessel.track ?? vessel.cells.map((c) => [c.lon, c.lat]);
                return (
                  <g key={vessel.vessel_id}>
                    {vessel.cells.map((cell) => {
                      const [x0, y0] = toPx(cell.lon - half, cell.lat + half);
                      const [x1, y1] = toPx(cell.lon + half, cell.lat - half);
                      const atPass = hourStart(cell.hour) === passHour;
                      const hour = faTime(cell.hour);
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
                            {`${name}: خانه AIS ساعت ${hour} UTC${atPass ? " (ساعت گذر)" : ""}`}
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
                  {faNumber(CHIP_SCALE_BAR_M)} متر
                </text>
              </g>
              <g className="chip-north" transform={`translate(${size - 18} 8)`}>
                <path d="M5 0 L10 13 L5 10 L0 13 Z" />
                <text x={5} y={25}>
                  ش
                </text>
              </g>
            </svg>
          </>
        )}
      </div>
      {image.pols.length > 1 && (
        <div className="chip-pols" role="group" aria-label="قطبش راداری">
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
        تصویر راداری Sentinel-1 به عرض {acrossKm} کیلومتر، شمال بالا. روشن یعنی بازتاب
        قوی، مانند بدنه فولادی؛ حلقه محل آشکارسازی را نشان می‌دهد.
        {nearest.length > 0 &&
          " مربع‌های فیروزه‌ای خانه‌های ساعتی نزدیک‌ترین شناورهای AIS هستند (حدود ۱ کیلومتر: شناور در آن ساعت جایی درون آن بوده است؛ پُر برای ساعت گذر)."}
        {image.no_data == null
          ? " نواحی هاشورخورده، اگر باشند، با رادار تصویربرداری نشده‌اند."
          : image.no_data > 0 && " نواحی هاشورخورده با رادار تصویربرداری نشده‌اند."}
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
  return (
    <div className="detail">
      <button className="close" onClick={onClose} aria-label="بستن">
        ×
      </button>
      <h2>آشکارسازی در تصویر ماهواره‌ای</h2>
      <p className="subtitle">
        <span className="ring" style={{ borderColor: RADAR_COLOUR }} />
        احتمالاً یک شناور: یافته مدل یادگیری عمیق در تصویر راداری Sentinel-1
      </p>

      <div className="age-badge">گذر راداری {faAge(detection.age_hours * 3600)}</div>

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
          <dt>موقعیت</dt>
          <dd>{faCoord(detection.lat, detection.lon)}</dd>
        </div>
        <div>
          <dt>امتیاز مدل</dt>
          <dd>
            {faNumber(detection.confidence, 2)} (از {faNumber(MODEL_THRESHOLD, 2)} به بالا نگه
            داشته می‌شود؛ احتمال نیست)
          </dd>
        </div>
        <div>
          <dt>گذر</dt>
          <dd>
            {detection.satellite} · {faDateTime(detection.acq_time)}
          </dd>
        </div>
        <div>
          <dt>مدل اینجا آزموده شده؟</dt>
          <dd>
            {testedHere(detection)}
          </dd>
        </div>
        {detection.product_name && (
          <div>
            <dt>تصویر (صحنه)</dt>
            <dd className="scene-name" dir="ltr">
              {detection.product_name.replace(/_COG\.SAFE$/, "")}
            </dd>
          </div>
        )}
      </dl>

      {detection.persistent && (
        <p className="caveat">{persistentNote(detection.persistent)}</p>
      )}

      <p className="hint">
        مدل می‌گوید کجا شناوری می‌بیند، نه این‌که چه شناوری است: خروجی رده، طول یا
        جهت ندارد و با پیکسل‌های ۱۰ متری رادار، نفتکش و فله‌بر شبیه هم دیده می‌شوند.
        این‌که این شیء AIS داشته یا نه فقط در گذرهای تلفیق‌شده بررسی می‌شود (نامزدهای
        شناور تاریک، قرمز و نارنجی).
      </p>
      <p className="hint" dir="ltr">Contains modified Copernicus Sentinel data.</p>
    </div>
  );
}

function DarkVesselDetail({ event, onClose }: { event: AnalystEvent; onClose: () => void }) {
  const evidence = event.evidence;
  const tier = tierOf(event);
  return (
    <div className="detail">
      <button className="close" onClick={onClose} aria-label="بستن">
        ×
      </button>
      <h2>نامزد شناور تاریک</h2>
      <p className="subtitle">
        <span className="ring ring-dark" style={{ borderColor: DARK_COLOURS[tier] }} />
        {DARK_TIER_LABELS[tier]}
      </p>

      <div className="age-badge">گذر راداری {faAge(event.age_hours * 3600)}</div>

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
        <h3>چرا تاریک</h3>
        <p>
          مسیر ساعتیِ هیچ شناور AIS از دروازه آن نمی‌گذرد.{" "}
          {evidence.gate.nearest_gap_m == null
            ? "اصلاً هیچ شناور AIS نزدیک این گذر نبود."
            : `${faMetres(evidence.gate.nearest_gap_m)} بیرون از دروازه نزدیک‌ترین شناور AIS قرار دارد (در گام ${faMetres(evidence.dark_up_to_m)} از پویش حاشیه هنوز تاریک است).`}
        </p>
        {evidence.pass_flags.presence_checked === false && (
          <p className="caveat">
            این گذر آشکارسازی‌های GFW کافی برای بررسی شکاف‌های خوراک AIS نداشت: شکاف
            در خوراک مانند شناورهای تاریک دیده می‌شود.
          </p>
        )}
        {event.persistent && <p className="caveat">{persistentNote(event.persistent)}</p>}
      </div>

      <dl className="stats stacked">
        <div>
          <dt>موقعیت</dt>
          <dd>{faCoord(evidence.detection.lat, evidence.detection.lon)}</dd>
        </div>
        <div>
          <dt>آشکارساز</dt>
          <dd>{detectorLine(evidence)}</dd>
        </div>
        <div>
          <dt>رادار GFW</dt>
          <dd>{gfwLine(evidence)}</dd>
        </div>
        <div>
          <dt>فاصله تا ساحل / لبه محدوده</dt>
          <dd>
            {faMetres(evidence.prefilter.shore_m)} / {faMetres(evidence.prefilter.edge_m)}
          </dd>
        </div>
        <div>
          <dt>دروازه</dt>
          <dd>
            خانه + R/V {faNumber(evidence.gate.r_over_v_s)} ثانیه × سرعت + حاشیه{" "}
            {faMetres(evidence.gate.margin_m)}
          </dd>
        </div>
        <div>
          <dt>گذر</dt>
          <dd className="scene-name" dir="ltr">
            {evidence.scene.replace(/_COG\.SAFE$/, "")}
          </dd>
        </div>
      </dl>

      <div className="nearest-ais">
        <h3>نزدیک‌ترین شناورهای AIS (سفید روی نقشه)</h3>
        <ul>
          {evidence.nearest_ais.map((vessel) => (
            <li key={vessel.vessel_id}>
              <strong>{vessel.ship_name ?? (vessel.mmsi ? `MMSI ${vessel.mmsi}` : "بی‌نام")}</strong>
              <span>
                {faMetres(vessel.distance_m)} فاصله
                {vessel.flag ? ` · ${vessel.flag}` : ""}
                {vessel.stationary ? " · لنگرانداخته" : vessel.sparse ? " · فقط ۱ ساعت دیده شده" : ""}
              </span>
              <span className="cells">
                {vessel.cells.map((cell) => faTime(cell.hour)).join("، ")} UTC
              </span>
            </li>
          ))}
        </ul>
      </div>

      <p className="hint">
        نامزد است، نه حکم. AIS اینجا برای هر شناور یک موقعیت در ساعت در خانه‌های
        حدوداً ۱ کیلومتری است، پس کشتی‌ای ممکن است تاریک به نظر برسد وقتی خانه
        ساعتی‌اش از جایی که رادار آن را دید دور است، و اخلال GNSS موقعیت‌های AIS را
        جابه‌جا می‌کند. داده‌ها: <bdi>{evidence.datasets.join(", ") || "نامعلوم"}</bdi>.
      </p>
      <p className="hint" dir="ltr">{evidence.attribution}</p>
    </div>
  );
}
