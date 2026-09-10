/**
 * The map. MapLibre GL with native layers.
 *
 * Vessels are drawn as heading-oriented triangles when a course is known and
 * as circles when it is not - the shape itself tells the operator whether the
 * vessel's direction is actually known, rather than inventing a heading.
 *
 * Opacity encodes age. A contact that has not reported in hours must not look
 * as solid as one reporting now.
 */

import { useEffect, useRef } from "react";
import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import {
  BBox,
  CLASS_COLOURS,
  SarScene,
  Vessel,
  classOf,
  coverageColour,
} from "./api";

// CARTO's dark basemap needs no API key and suits an operational console.
const BASEMAP =
  "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

const VESSEL_SOURCE = "vessels";
const TRACK_SOURCE = "track";
const SCENE_SOURCE = "scenes";

interface Props {
  vessels: Vessel[];
  scenes: SarScene[];
  track: { lon: number; lat: number }[] | null;
  selected: number | null;
  aoi: BBox | null;
  onSelect: (mmsi: number | null) => void;
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
  track,
  selected,
  aoi,
  onSelect,
}: Props) {
  const container = useRef<HTMLDivElement>(null);
  const map = useRef<MapLibreMap | null>(null);
  const ready = useRef(false);

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
      instance.addSource(VESSEL_SOURCE, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      instance.addSource(TRACK_SOURCE, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      instance.addSource(SCENE_SOURCE, {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });
      instance.addSource("aoi", {
        type: "geojson",
        data: { type: "FeatureCollection", features: [] },
      });

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

      addArrowIcons(instance);
      ready.current = true;

      const pick = (event: maplibregl.MapMouseEvent) => {
        const hits = instance.queryRenderedFeatures(event.point, {
          layers: ["vessel-arrows", "vessel-dots"],
        });
        onSelect(hits.length ? (hits[0].properties?.mmsi as number) : null);
      };
      instance.on("click", pick);
      for (const layer of ["vessel-arrows", "vessel-dots"]) {
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
      ready.current = false;
    };
  }, [onSelect]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready.current) return;
    const source = instance.getSource(VESSEL_SOURCE) as maplibregl.GeoJSONSource;
    source?.setData(toFeatureCollection(vessels));
  }, [vessels]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready.current) return;
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
  }, [scenes]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready.current) return;
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
  }, [track]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready.current) return;
    instance.setFilter("vessel-selected", [
      "==",
      ["get", "mmsi"],
      selected ?? -1,
    ]);
  }, [selected]);

  useEffect(() => {
    const instance = map.current;
    if (!instance || !ready.current || !aoi) return;
    const source = instance.getSource("aoi") as maplibregl.GeoJSONSource;
    source?.setData(bboxToPolygon(aoi));
  }, [aoi]);

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
