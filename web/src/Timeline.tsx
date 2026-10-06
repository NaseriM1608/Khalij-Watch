/**
 * The timeline scrubber (PLAN §10 rule 4): one band per measured box with a
 * tick for every stored pass over it, so the user sees how often each area
 * was actually observed. Clicking or dragging sets the moment the console
 * looks from ("as of"); the map then shows only what was known by then, and
 * every age is counted from it. Releasing near a pass snaps to just after
 * it, so that pass is included. "Live" returns to now.
 */

import { useMemo, useRef, useState } from "react";
import type { PointerEvent } from "react";
import { DARK_COLOURS, RADAR_COLOUR, TimelinePass, TimelineResponse } from "./api";

const AREA_LABELS: Record<string, string> = {
  fujairah: "Fujairah",
  khor_fakkan: "Khor Fakkan",
  sharjah_dubai: "Sharjah / Dubai",
  ras_al_khaimah: "Ras al Khaimah",
  hormuz_lanes: "Hormuz lanes",
  bandar_abbas: "Bandar Abbas",
  larak_hormuz: "Larak / Hormuz",
  kish: "Kish",
  assaluyeh: "Assaluyeh",
  bushehr: "Bushehr",
  kharg: "Kharg",
  jask: "Jask",
};
const IMAGED_COLOUR = "#5d6b80";
// Releasing within this many pixels of a pass snaps to it.
const SNAP_PX = 8;
// Within this of now counts as live.
const LIVE_MS = 60 * 60 * 1000;
// A snapped moment falls just after its pass, so the pass is included.
const AFTER_PASS_MS = 60 * 1000;
const DAY_MS = 24 * 60 * 60 * 1000;

function passColour(pass: TimelinePass): string {
  if (pass.fused) return DARK_COLOURS.high;
  if (pass.searched) return RADAR_COLOUR;
  return IMAGED_COLOUR;
}

function describe(area: string, pass: TimelinePass): string {
  const when = `${pass.acq_time.slice(0, 16).replace("T", " ")} UTC`;
  const what = pass.fused
    ? `searched (${pass.detections} detections), dark-vessel candidates`
    : pass.searched
      ? `searched by the model (${pass.detections} detections)`
      : "imaged, not searched";
  return `${AREA_LABELS[area] ?? area} · ${pass.satellite} ${when} · ${what}`;
}

function formatMoment(ms: number): string {
  return `${new Date(ms).toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

export function Timeline({
  data,
  asOf,
  onChange,
}: {
  data: TimelineResponse | null;
  asOf: string | null;
  onChange: (asOf: string | null) => void;
}) {
  const barRef = useRef<HTMLDivElement>(null);
  const [dragging, setDragging] = useState<number | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  const span = useMemo(() => {
    if (!data) return null;
    return { start: Date.parse(data.since), end: Date.parse(data.generated_at) };
  }, [data]);
  const passTimes = useMemo(
    () => (data?.areas ?? []).flatMap((a) => a.passes.map((p) => Date.parse(p.acq_time))),
    [data],
  );
  const days = useMemo(() => {
    if (!span) return [];
    const first = Math.ceil(span.start / DAY_MS) * DAY_MS;
    const marks = [];
    for (let t = first; t <= span.end; t += DAY_MS) {
      if (new Date(t).getUTCDay() === 1) marks.push(t); // Mondays
    }
    return marks;
  }, [span]);

  if (!data || !span) return null;
  const width = span.end - span.start;
  const share = (ms: number) => Math.min(1, Math.max(0, (ms - span.start) / width));

  const momentAt = (clientX: number, snap: boolean): number => {
    const box = barRef.current?.getBoundingClientRect();
    if (!box) return span.end;
    const ms = span.start + ((clientX - box.left) / box.width) * width;
    if (!snap) return ms;
    let best: number | null = null;
    for (const t of passTimes) {
      const px = (Math.abs(t - ms) / width) * box.width;
      if (px <= SNAP_PX && (best === null || Math.abs(t - ms) < Math.abs(best - ms))) best = t;
    }
    return best === null ? ms : best + AFTER_PASS_MS;
  };
  const commit = (ms: number) =>
    onChange(ms >= span.end - LIVE_MS ? null : new Date(ms).toISOString());

  const onPointerDown = (event: PointerEvent<HTMLDivElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDragging(momentAt(event.clientX, false));
  };
  const onPointerMove = (event: PointerEvent<HTMLDivElement>) => {
    if (dragging !== null) setDragging(momentAt(event.clientX, false));
  };
  const onPointerUp = (event: PointerEvent<HTMLDivElement>) => {
    if (dragging === null) return;
    setDragging(null);
    commit(momentAt(event.clientX, true));
  };

  const cursor = dragging ?? (asOf ? Date.parse(asOf) : span.end);
  const looking = asOf ? `As of ${formatMoment(Date.parse(asOf))}` : "Live";

  return (
    <section className={`timeline ${collapsed ? "collapsed" : ""}`} aria-label="Timeline of radar passes">
      <header className="timeline-head">
        <button className="link" onClick={() => setCollapsed((c) => !c)}>
          {collapsed ? "Show timeline" : "Hide timeline"}
        </button>
        <span className="timeline-title">
          Stored radar passes over each box, last {data.days} days: drag to look back in time
        </span>
        <span className="timeline-key">
          <i style={{ background: IMAGED_COLOUR }} /> imaged
          <i style={{ background: RADAR_COLOUR }} /> searched by the model
          <i style={{ background: DARK_COLOURS.high }} /> dark-vessel candidates
        </span>
        <strong className={asOf ? "as-of past" : "as-of"}>
          {dragging !== null ? formatMoment(dragging) : looking}
        </strong>
        {asOf && (
          <button className="live" onClick={() => onChange(null)}>
            Live
          </button>
        )}
      </header>
      {!collapsed && (
        <div className="timeline-body">
          <ul className="timeline-labels">
            {data.areas.map((area) => (
              <li key={area.name} className={area.region}>
                {AREA_LABELS[area.name] ?? area.name}
              </li>
            ))}
          </ul>
          <div
            ref={barRef}
            className="timeline-bar"
            onPointerDown={onPointerDown}
            onPointerMove={onPointerMove}
            onPointerUp={onPointerUp}
          >
            {days.map((t) => (
              <span key={t} className="timeline-day" style={{ left: `${share(t) * 100}%` }}>
                {new Date(t).toISOString().slice(5, 10)}
              </span>
            ))}
            {data.areas.map((area, row) => (
              <div key={area.name} className="timeline-row" style={{ top: `${row * 13 + 16}px` }}>
                {area.passes.map((pass) => (
                  <span
                    key={pass.scene_id}
                    className="timeline-tick"
                    style={{ left: `${share(Date.parse(pass.acq_time)) * 100}%`, background: passColour(pass) }}
                    title={describe(area.name, pass)}
                  />
                ))}
              </div>
            ))}
            <div className="timeline-future" style={{ left: `${share(cursor) * 100}%` }} />
            <div className="timeline-cursor" style={{ left: `${share(cursor) * 100}%` }} />
          </div>
        </div>
      )}
    </section>
  );
}
