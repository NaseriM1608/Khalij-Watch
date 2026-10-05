/**
 * The map. MapLibre GL with native layers.
 *
 * Vessels are drawn as heading-oriented triangles when a course is known and
 * as circles when it is not - the shape itself tells the operator whether the
 * vessel's direction is actually known, rather than inventing a heading.
 *
 * Opacity encodes age. A contact that has not reported in hours must not look
 * as solid as one reporting now.
 *
 * Radar detections are hollow rings. A filled symbol means someone told us a
 * vessel is there; a ring means the radar saw something there. That
 * distinction is the point of the product, so the two never share a style.
 *
 * Dark-vessel candidates - radar detections that no AIS vessel's hourly track
 * reaches - are the visual priority: thicker rings, red to amber by
 * confidence, drawn above everything and never clustered away.
 */

import { useEffect, useRef, useState } from "react";
import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  AnalystEvent,
  BBox,
  CLASS_COLOURS,
  DARK_COLOURS,
  RADAR_COLOUR,
  SarDetection,
  SarScene,
  Vessel,
  classOf,
  coverageColour,
  formatAge,
  tierOf,
} from "./api";

// CARTO's dark basemap needs no API key and suits an operational console.
const BASEMAP =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const VESSEL_SOURCE = "vessels";
const TRACK_SOURCE = "track";
const SCENE_SOURCE = "scenes";
const DETECTION_SOURCE = "detections";
const DARK_SOURCE = "dark-events";
const EVENT_AIS_SOURCE = "event-ais";
const VESSEL_LAYERS = ["vessel-arrows", "vessel-dots"];
const DETECTION_LAYER = "detection-rings";
const DARK_LAYER = "dark-rings";

interface Props {
  vessels: Vessel[];
  scenes: SarScene[];
  detections: SarDetection[];
  track: { lon: number; lat: number }[] | null;
  selected: number | null;
  aoi: BBox | null;
  onSelect: (mmsi: number | null) => void;
  darkEvents: AnalystEvent[];
  selectedEvent: AnalystEvent | null;
  onSelectEvent: (id: number | null) => void;
}

function toFeatureCollection(vessels: Vessel[]): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: vessels.map((vessel) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [vessel.lon, vessel.lat] },
      properties: {
        mmsi: vessel.mmsi,
        colour: CLASS_COLOURS[classOf(vessel)],
        icon: `arrow-${classOf(vessel)}`,
        // Rotation only means something when a course is actually reported.
        rotation: vessel.cog_deg ?? vessel.heading_deg ?? 0,
        hasHeading: vessel.cog_deg != null || vessel.heading_deg != null,
        ageSeconds: vessel.age_seconds,
        moving: (vessel.sog_kn ?? 0) > 0.5,
        suspect: vessel.is_suspect,
      },
    })),
  };
}

function toDetectionFeatures(
  detections: SarDetection[],
): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: detections.map((detection) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [detection.lon, detection.lat] },
      // Display strings are prepared here so the popup never has to
      // interpret missing values.
      properties: {
        id: detection.id,
        ageHours: detection.age_hours,
        pass: [detection.satellite, detection.polarisation]
          .filter(Boolean)
          .join(" · "),
        size:
          detection.length_m != null && detection.width_m != null
            ? `${Math.round(detection.length_m)} × ${Math.round(detection.width_m)} m`
            : "not measured",
        heading: headingLabel(detection.heading_deg),
        detector: [detection.detector, detection.detector_version]
          .filter(Boolean)
          .join(" "),
      },
    })),
  };
}

function toDarkFeatures(events: AnalystEvent[]): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: events
      .filter((event) => event.lon != null && event.lat != null)
      .map((event) => ({
        type: "Feature",
        geometry: { type: "Point", coordinates: [event.lon as number, event.lat as number] },
        properties: {
          id: event.id,
          colour: DARK_COLOURS[tierOf(event)],
          ageHours: event.age_hours,
        },
      })),
  };
}

/** The selected candidate's nearest AIS vessels: their hourly tracks and cells. */
function toNearestAisFeatures(event: AnalystEvent | null): GeoJSON.FeatureCollection {
  if (!event) return { type: "FeatureCollection", features: [] };
  const features: GeoJSON.Feature[] = [];
  for (const vessel of event.evidence.nearest_ais) {
    // The polyline the gate measured to, not just the hourly cells.
    const coordinates = vessel.track ?? vessel.cells.map((c) => [c.lon, c.lat]);
    if (new Set(coordinates.map((c) => c.join(","))).size > 1) {
      features.push({
        type: "Feature",
        properties: { kind: "track" },
        geometry: { type: "LineString", coordinates },
      });
    }
    for (const cell of vessel.cells) {
      features.push({
        type: "Feature",
        properties: { kind: "cell" },
        geometry: { type: "Point", coordinates: [cell.lon, cell.lat] },
      });
    }
  }
  return { type: "FeatureCollection", features };
}

/** Radar cannot tell bow from stern, so both directions are stated. */
function headingLabel(headingDeg: number | null): string {
  if (headingDeg == null) return "not measured";
  const axis = Math.round(headingDeg) % 180;
  return `${axis}° or ${axis + 180}°`;
}

function bboxToPolygon(bbox: BBox): GeoJSON.Feature {
  const { min_lon, min_lat, max_lon, max_lat } = bbox;
  return {
    type: "Feature",
    properties: {},
    geometry: {
      type: "Polygon",
      coordinates: [
        [
          [min_lon, min_lat],
          [max_lon, min_lat],
          [max_lon, max_lat],
          [min_lon, max_lat],
          [min_lon, min_lat],
        ],
      ],
    },
  };
}

export function VesselMap({
  vessels,
  scenes,
  detections,
  track,
  selected,
  aoi,
  onSelect,
  darkEvents,
  selectedEvent,
  onSelectEvent,
}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  // State rather than a ref: data that arrives before the style has loaded
  // must still reach the map once it has, so every data effect depends on it.
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    if (!container.current || map.current) return;

    const instance = new maplibregl.Map({
      container: container.current,
      style: BASEMAP,
      center: [54.5, 26.8],
      zoom: 6,
      attributionControl: false,
    });
    instance.addControl(new maplibregl.NavigationControl(), "bottom-right");
    instance.addControl(
      new maplibregl.AttributionControl({ compact: true }),
      "bottom-left",
    );
    instance.addControl(new maplibregl.ScaleControl({ unit: "nautical" }));

    instance.on("load", () => {
      for (const id of [
        VESSEL_SOURCE,
        TRACK_SOURCE,
        SCENE_SOURCE,
        DETECTION_SOURCE,
        DARK_SOURCE,
        EVENT_AIS_SOURCE,
        "aoi",
      ]) {
        instance.addSource(id, {
          type: "geojson",
          data: { type: "FeatureCollection", features: [] },
        });
      }

      // Radar coverage sits at the very bottom: it is context for everything
      // above it, and shows WHERE the platform can see at all.
      instance.addLayer({
        id: "scene-fill",
        type: "fill",
        source: SCENE_SOURCE,
        paint: { "fill-color": ["get", "colour"], "fill-opacity": 0.1 },
      });
      instance.addLayer({
        id: "scene-outline",
        type: "line",
        source: SCENE_SOURCE,
        paint: {
          "line-color": ["get", "colour"],
          "line-width": 0.9,
          "line-opacity": 0.55,
        },
      });

      instance.addLayer({
        id: "aoi-outline",
        type: "line",
        source: "aoi",
        paint: {
          "line-color": "#2f4a6d",
          "line-width": 1,
          "line-dasharray": [3, 3],
        },
      });

      instance.addLayer({
        id: "track-line",
        type: "line",
        source: TRACK_SOURCE,
        paint: {
          "line-color": "#ffffff",
          "line-width": 1.6,
          "line-opacity": 0.55,
        },
      });

      // Radar detections: above coverage, below AIS symbols, so a vessel that
      // is both reported and seen shows its symbol inside a ring.
      instance.addLayer({
        id: DETECTION_LAYER,
        type: "circle",
        source: DETECTION_SOURCE,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 3.5, 12, 10],
          "circle-color": "transparent",
          "circle-stroke-width": [
            "interpolate",
            ["linear"],
            ["zoom"],
            5,
            1,
            12,
            1.8,
          ],
          "circle-stroke-color": RADAR_COLOUR,
          "circle-stroke-opacity": passAgeOpacity(),
        },
      });

      // Contacts whose own report contradicts itself. Drawn first so the
      // halo sits behind the vessel symbol rather than obscuring it.
      instance.addLayer({
        id: "vessel-suspect",
        type: "circle",
        source: VESSEL_SOURCE,
        filter: ["get", "suspect"],
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 7, 12, 15],
          "circle-color": "#ff5c5c",
          "circle-opacity": 0.18,
          "circle-stroke-width": 1.2,
          "circle-stroke-color": "#ff5c5c",
          "circle-stroke-opacity": 0.8,
        },
      });

      // Stationary or heading-less contacts: a circle claims no direction.
      instance.addLayer({
        id: "vessel-dots",
        type: "circle",
        source: VESSEL_SOURCE,
        filter: ["!", ["get", "hasHeading"]],
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 2.5, 12, 6],
          "circle-color": ["get", "colour"],
          "circle-opacity": ageOpacity(),
          "circle-stroke-width": 0.6,
          "circle-stroke-color": "#0b1017",
        },
      });

      // Contacts with a known course: an arrow, rotated to it.
      instance.addLayer({
        id: "vessel-arrows",
        type: "symbol",
        source: VESSEL_SOURCE,
        filter: ["get", "hasHeading"],
        layout: {
          "icon-image": ["get", "icon"],
          "icon-rotate": ["get", "rotation"],
          "icon-rotation-alignment": "map",
          "icon-allow-overlap": true,
          "icon-size": ["interpolate", ["linear"], ["zoom"], 5, 0.5, 12, 1.1],
        },
        paint: {
          "icon-opacity": ageOpacity(),
        },
      });

      instance.addLayer({
        id: "vessel-selected",
        type: "circle",
        source: VESSEL_SOURCE,
        filter: ["==", ["get", "mmsi"], -1],
        paint: {
          "circle-radius": 11,
          "circle-color": "transparent",
          "circle-stroke-width": 2,
          "circle-stroke-color": "#ffffff",
        },
      });

      // The selected candidate's nearest AIS vessels: their hourly cells and
      // the track between them, which the gate measured the detection against.
      instance.addLayer({
        id: "event-ais-tracks",
        type: "line",
        source: EVENT_AIS_SOURCE,
        filter: ["==", ["get", "kind"], "track"],
        paint: {
          "line-color": "#ffffff",
          "line-width": 1.4,
          "line-opacity": 0.8,
          "line-dasharray": [2, 2],
        },
      });
      instance.addLayer({
        id: "event-ais-cells",
        type: "circle",
        source: EVENT_AIS_SOURCE,
        filter: ["==", ["get", "kind"], "cell"],
        paint: {
          "circle-radius": 3,
          "circle-color": "#ffffff",
          "circle-opacity": 0.85,
        },
      });

      // Dark-vessel candidates last, so they sit on top of everything.
      instance.addLayer({
        id: DARK_LAYER,
        type: "circle",
        source: DARK_SOURCE,
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 5, 12, 14],
          "circle-color": "transparent",
          "circle-stroke-width": ["interpolate", ["linear"], ["zoom"], 5, 1.8, 12, 3],
          "circle-stroke-color": ["get", "colour"],
          "circle-stroke-opacity": darkAgeOpacity(),
        },
      });
      instance.addLayer({
        id: "dark-selected",
        type: "circle",
        source: DARK_SOURCE,
        filter: ["==", ["get", "id"], -1],
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 9, 12, 20],
          "circle-color": "transparent",
          "circle-stroke-width": 2,
          "circle-stroke-color": "#ffffff",
        },
      });

      addArrowIcons(instance);
      setLoaded(true);

      // A dark-vessel candidate first, then a reported vessel; otherwise
      // explain the radar ring.
      const pick = (event: maplibregl.MapMouseEvent) => {
        const dark = instance.queryRenderedFeatures(event.point, {
          layers: [DARK_LAYER],
        });
        if (dark.length) {
          onSelect(null);
          onSelectEvent(dark[0].properties?.id as number);
          return;
        }
        onSelectEvent(null);
        const hits = instance.queryRenderedFeatures(event.point, {
          layers: VESSEL_LAYERS,
        });
        if (hits.length) {
          onSelect(hits[0].properties?.mmsi as number);
          return;
        }
        onSelect(null);
        const seen = instance.queryRenderedFeatures(event.point, {
          layers: [DETECTION_LAYER],
        });
        if (seen.length) showDetectionPopup(instance, seen[0]);
      };
      instance.on("click", pick);
      for (const layer of [...VESSEL_LAYERS, DETECTION_LAYER, DARK_LAYER]) {
        instance.on("mouseenter", layer, () => {
          instance.getCanvas().style.cursor = "pointer";
        });
        instance.on("mouseleave", layer, () => {
          instance.getCanvas().style.cursor = "";
        });
      }
    });

    map.current = instance;
    return () => {
      instance.remove();
      map.current = null;
      setLoaded(false);
    };
  }, [onSelect, onSelectEvent]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    const source = instance.getSource(VESSEL_SOURCE) as maplibregl.GeoJSONSource;
    source?.setData(toFeatureCollection(vessels));
  }, [vessels, loaded]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    const source = instance.getSource(SCENE_SOURCE) as maplibregl.GeoJSONSource;
    source?.setData({
      type: "FeatureCollection",
      features: scenes
        .filter((scene) => scene.footprint.length > 3)
        .map((scene) => ({
          type: "Feature",
          properties: {
            id: scene.id,
            colour: coverageColour(scene.age_hours),
            ageHours: scene.age_hours,
          },
          geometry: { type: "Polygon", coordinates: [scene.footprint] },
        })),
    });
  }, [scenes, loaded]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    const source = instance.getSource(
      DETECTION_SOURCE,
    ) as maplibregl.GeoJSONSource;
    source?.setData(toDetectionFeatures(detections));
  }, [detections, loaded]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    const source = instance.getSource(DARK_SOURCE) as maplibregl.GeoJSONSource;
    source?.setData(toDarkFeatures(darkEvents));
  }, [darkEvents, loaded]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    const source = instance.getSource(EVENT_AIS_SOURCE) as maplibregl.GeoJSONSource;
    source?.setData(toNearestAisFeatures(selectedEvent));
    instance.setFilter("dark-selected", ["==", ["get", "id"], selectedEvent?.id ?? -1]);
  }, [selectedEvent, loaded]);

  // The context panel narrows the map when it opens: bring a newly selected
  // candidate into view - once per selection, not on every events refresh.
  const selectedId = selectedEvent?.id ?? null;
  const selectedLon = selectedEvent?.lon ?? null;
  const selectedLat = selectedEvent?.lat ?? null;
  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded || selectedId == null || selectedLon == null || selectedLat == null) {
      return;
    }
    // Zoom in far enough to tell the candidate from its neighbours; never zoom out.
    instance.easeTo({
      center: [selectedLon, selectedLat],
      zoom: Math.max(instance.getZoom(), 11),
      duration: 600,
    });
    // Coordinates are read with the id; a refresh of the same event must not re-centre.
  }, [selectedId, loaded]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    const source = instance.getSource(TRACK_SOURCE) as maplibregl.GeoJSONSource;
    source?.setData(
      track && track.length > 1
        ? {
            type: "Feature",
            properties: {},
            geometry: {
              type: "LineString",
              coordinates: track.map((p) => [p.lon, p.lat]),
            },
          }
        : { type: "FeatureCollection", features: [] },
    );
  }, [track, loaded]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded) return;
    instance.setFilter("vessel-selected", [
      "==",
      ["get", "mmsi"],
      selected ?? -1,
    ]);
  }, [selected, loaded]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !loaded || !aoi) return;
    const source = instance.getSource("aoi") as maplibregl.GeoJSONSource;
    source?.setData(bboxToPolygon(aoi));
  }, [aoi, loaded]);

  return <div ref={container} className="map" />;
}

/**
 * Fade a contact as it ages: solid when fresh, faint after six hours.
 * This is the visual half of "every object shows its age".
 */
function ageOpacity(): maplibregl.ExpressionSpecification {
  return [
    "interpolate",
    ["linear"],
    ["get", "ageSeconds"],
    0,
    1.0,
    3600,
    0.85,
    6 * 3600,
    0.45,
    24 * 3600,
    0.25,
  ];
}

/**
 * Fade a radar detection with the age of its pass. Radar ages in days, not
 * minutes: a ring stays legible for the few days until the next pass.
 */
function passAgeOpacity(): maplibregl.ExpressionSpecification {
  return [
    "interpolate",
    ["linear"],
    ["get", "ageHours"],
    0,
    1.0,
    12,
    0.9,
    48,
    0.65,
    120,
    0.4,
    336,
    0.2,
  ];
}

/**
 * Dark-vessel candidates fade with age too, but stay legible: fusion runs days
 * after a pass, so every candidate is old, and they are the visual priority.
 * The panel states the exact age.
 */
function darkAgeOpacity(): maplibregl.ExpressionSpecification {
  return ["interpolate", ["linear"], ["get", "ageHours"], 0, 1.0, 336, 0.75, 1440, 0.55];
}

/** Explain a radar ring: when it was seen, by what, and what it is not. */
function showDetectionPopup(
  instance: MapLibreMap,
  feature: maplibregl.MapGeoJSONFeature,
): void {
  const properties = feature.properties ?? {};
  const coordinates = (feature.geometry as GeoJSON.Point).coordinates;
  const rows: [string, string][] = [
    ["Seen", formatAge(Number(properties.ageHours) * 3600)],
    ["Pass", String(properties.pass ?? "")],
    ["Size", String(properties.size ?? "")],
    ["Heading", String(properties.heading ?? "")],
    ["Detector", String(properties.detector ?? "")],
  ];

  // Built from DOM nodes with textContent, never from an HTML string.
  const root = document.createElement("div");
  root.className = "radar-popup";
  const title = document.createElement("strong");
  title.textContent = "Radar detection";
  root.append(title);
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    const name = document.createElement("span");
    name.textContent = label;
    const text = document.createElement("span");
    text.textContent = value;
    row.append(name, text);
    root.append(row);
  }
  const note = document.createElement("p");
  note.textContent =
    "This ring alone says nothing about AIS. Dark-vessel candidates are marked separately.";
  root.append(note);

  new maplibregl.Popup({ maxWidth: "280px" })
    .setLngLat([coordinates[0], coordinates[1]])
    .setDOMContent(root)
    .addTo(instance);
}

/**
 * One pre-coloured arrow per vessel class.
 *
 * MapLibre's `icon-color` only applies to SDF icons, and an SDF is a signed
 * DISTANCE FIELD, not a plain alpha mask - handing it a hard-edged shape makes
 * the symbol vanish. Baking the colour into a handful of icons avoids the
 * whole problem for the five classes actually in use.
 */
function addArrowIcons(instance: MapLibreMap): void {
  const size = 24;
  const dpr = 2; // render at 2x so the arrow stays crisp when scaled up

  for (const [key, colour] of Object.entries(CLASS_COLOURS)) {
    const name = `arrow-${key}`;
    if (instance.hasImage(name)) continue;

    const canvas = document.createElement("canvas");
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const ctx = canvas.getContext("2d");
    if (!ctx) continue;
    ctx.scale(dpr, dpr);

    // A chevron: clearly directional even at a few pixels across.
    ctx.beginPath();
    ctx.moveTo(size / 2, 3);
    ctx.lineTo(size - 5, size - 4);
    ctx.lineTo(size / 2, size - 9);
    ctx.lineTo(5, size - 4);
    ctx.closePath();

    ctx.fillStyle = colour;
    ctx.fill();
    // A dark outline keeps the icon readable against pale coastline fills.
    ctx.strokeStyle = "rgba(8, 11, 17, 0.85)";
    ctx.lineWidth = 1;
    ctx.stroke();

    const { data } = ctx.getImageData(0, 0, size * dpr, size * dpr);
    instance.addImage(
      name,
      { width: size * dpr, height: size * dpr, data: new Uint8Array(data) },
      { pixelRatio: dpr },
    );
  }
}
