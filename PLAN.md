# Persian Gulf Maritime Domain Awareness Platform — Implementation Plan

**Working title:** `khalij-watch` (Persian Gulf Vessel Detection & Classification Platform)
**Type:** Academic / thesis project + working demonstrator
**Repo root:** `E:\sea-vessel-detection-3`

---

## 1. Context

You want an OSINT platform that gives a live view of the Persian Gulf, detects vessels, and classifies them with deep learning, behind a real UI.

Research into the current (2026) state of the domain changed three things about how this should be built. These are the load-bearing findings — the rest of the plan follows from them:

**Finding 1 — There is no such thing as a "live view" of the Gulf, and the honest version is more interesting than the fantasy.**
No civilian sensor gives continuous video of 240,000 km² of water. What exists is a *layered* picture with wildly different refresh rates. The product is the **fusion** of those layers, not any one of them.

**Finding 2 — AIS in this specific region is actively adversarial, not merely incomplete.**
Two compounding problems. First, coverage: every free AIS network is terrestrial, and the Gulf has ~4 contributing shore stations (3 UAE, 1 Qatar) against a 200–300 km-wide basin — the middle of the Gulf and the entire Iranian side are simply not received. Second, and worse, the data that *does* arrive is substantially fabricated: 2026 saw widespread GNSS jamming and AIS spoofing around Hormuz, with reporting claiming **nearly two of every three logged Hormuz transits in August 2026 were false**, >1,100 ships affected within 24 hours of one jamming episode, and positions placed on land, at airports, and at a nuclear power plant.

A platform that renders AIS as ground truth will render fiction. **AIS plausibility filtering must be a first-class ingest stage, not an afterthought.**

**Finding 3 — Fine-grained vessel type classification from free satellite imagery does not work, and planning to do it will sink the thesis.**
Global Fishing Watch published the closest precedent to this project in *Nature* (Paolo et al. 2024). With Google-scale resources and a global Sentinel-1 archive they reported: object detection >97%, vessel **length** regression R²=0.84, and vessel classification only as **binary fishing / non-fishing at >90%**. They did not attempt tanker-vs-cargo-vs-tug from 10 m SAR pixels, because at 10 m a 200 m tanker is 20 pixels long and a fishing boat is 2. The discriminative signal for fine class simply is not in the data.

The resolution is a **division of labour** that becomes the project's central design idea:

> **Pixels tell you a vessel exists, how big it is, and which way it points. AIS and registries tell you what it is called and what type it is. Fusion tells you when those two disagree — and that disagreement is the actual product.**

So the deep learning does detection + geometry + coarse class. Identity and fine type come from AIS/registry joins. A SAR detection with no AIS match is a **dark vessel**, and that is the headline output.

**Finding 4 — the usual architecture is backwards for this AOI.** The instinctive design is "AIS map, with occasional satellite checks." Given Findings 2 and 3, invert it:

> **SAR is the truth layer. AIS is a label that may be lying.**

Every downstream design decision in this plan follows from that inversion.

**Intended outcome:** a reproducible, defensible thesis with a novel regional dataset and a working web demonstrator, built entirely on free data and free compute.

---

## 2. What "live" actually means — the sensor tiers

This table is the product's honest contract with the user, and should appear in the UI itself.

| Layer | Sensor | Latency | Refresh (measured) | Coverage | Sees dark vessels? |
|---|---|---|---|---|---|
| **A. Cooperative** | AIS (terrestrial only, free) | seconds | continuous | ⚠️ **Coastal only — mid-Gulf is blind** | ❌ No — by definition |
| **B. Radar truth** | Sentinel-1 SAR (C-band) | **0.75–4.6 h, median 2.5 h** | **~2 d @ Hormuz, ~3 d elsewhere** | 250 km swaths, all-weather, day/night | ✅ **Yes** |
| **C. Night lights** | NOAA VIIRS Boat Detection | ~1 day (free tier) | nightly | global, 750 m | ✅ Lit vessels only |
| **D. Optical** | Sentinel-2 | 4.5–6.5 h | ~4 days | ⚠️ dust-limited (see below) | ✅ When clear |
| **E. Context** | Registries, sanctions, ADS-B | static/daily | — | — | Enrichment |

**Sentinel-1 is far stronger here than the nominal spec suggests, and it is the backbone.** The constellation is **Sentinel-1C + 1D** (1A retired 29 June 2026) with a 6-day *exact repeat* — but ship detection does not need an exact repeat orbit, and at 26–30°N adjacent tracks overlap with both ascending and descending passes. Measured over a 30-day window (11 Aug – 10 Sep 2026):

| Point | Location | Acquisitions/30 d | Mean revisit |
|---|---|---|---|
| 56.5°E, 26.5°N | **Strait of Hormuz** | **15** | **~2.0 d** |
| 52.0°E, 27.0°N | Central Gulf | 10 | ~3.0 d |
| 48.5°E, 29.5°N | Northern Gulf (Kuwait) | 10 | ~3.0 d |

Every scene is `IW_GRDH_1S`, **dual-pol VV+VH** (`1SDV`) — 100% consistent. **VH is the better channel for ship detection** (low ocean backscatter → higher target-to-clutter contrast); having both enables polarimetric features.

⚠️ **Constellation caveat:** the S1C/S1D configuration has a one-day shift in the revisit pattern versus the old S1A/S1B pair. Any assumption inherited from pre-2026 code or literature is wrong. Re-measure periodically.

⚠️ **Free AIS over the Gulf is much weaker than expected.** AISHub's public station registry lists **3 stations in the UAE and 1 in Qatar — and zero in Iran, Oman, Bahrain, Kuwait, Saudi Arabia, or Iraq.** Shore receivers reach ~40–60 km reliably; the Gulf is 200–300 km wide. The free AIS layer will show port and anchorage activity at Jebel Ali, Fujairah, Khalifa, Ras Laffan and Hamad, and **almost nothing in the middle of the Gulf or on the Iranian side.** This is not a bug to fix — it is a structural fact that makes the SAR-first design the only honest one.

**Design consequence:** the map is a *time-aware* display, not a real-time one. Every object carries an explicit age. AIS contacts pulse live; SAR detections render as timestamped "last seen" markers with a confidence that decays. Never draw a 20-hour-old SAR detection as if it were live — that is the single biggest UX failure mode in this domain.

---

## 3. Hard constraints

**Local machine:** AMD Ryzen 7 3700U (4C/8T, 15 W mobile), 10 GB RAM, integrated Vega (no CUDA), Windows 11.
**Disk:** C: 0.4 GB free — E: 57.6 GB, F: 66.7 GB free (~124 GB usable).
**Training:** Kaggle free tier — 30 GPU-h/week, P100 16 GB or 2×T4, 12 h max session, 20 GB working disk, internet enabled.
**Budget:** $0. Free and open data only.

### Consequences that shape every later phase

1. **Split the workload.** Data prep + inference + platform run locally; all training runs on Kaggle. Weights come back as ONNX.
2. **Never load a full scene into RAM.** A Sentinel-1 IW GRD scene is ~25,000 × 17,000 px. As float32 that is ~1.7 GB per polarization — on a 10 GB box with a browser open, that is fatal. All raster access is **windowed** via `rasterio` block reads.
3. **Process-and-discard.** A full-Gulf coverage is ~16 scenes ≈ 25 GB raw. Download → preprocess → extract chips + detections → **delete the raw scene**. Persist only detections (rows) and chips (KBs). Without this, 124 GB is gone in a week.
4. **Redirect caches off C:.** One-time setup — `PIP_CACHE_DIR`, `TORCH_HOME`, `HF_HOME`, `TMP`/`TEMP` → `E:\`. PyTorch alone is ~2.5 GB and will otherwise fail to install.
5. **Focus the AOI for the DL work.** Full Gulf for the platform; **Strait of Hormuz + approaches** (lon 54.0–57.5, lat 24.5–27.5, ~2–4 scenes/pass) for training and evaluation. Highest-value area, tractable data volume.

---

## 4. Scope and thesis contribution

Four contributions, ordered by how defensible they are:

1. **An AIS-supervised auto-labelled Sentinel-1 dataset for the Persian Gulf.** Correlate AIS positions with SAR acquisition timestamps to generate labelled ship chips *at scale, for free, with no manual annotation* — including true length and true type from AIS static messages. No such regional dataset is published. This is the strongest novelty claim.
2. **A CPU-deployable dense point detector**, trained on free cloud GPU and quantized to INT8 for edge inference — benchmarked against a classical CFAR baseline and against Global Fishing Watch's published detections over the same scenes. The accuracy/latency trade-off on commodity hardware is a real result, and "train in cloud, deploy on edge CPU" is a clean thesis narrative.
3. **Dark vessel detection and quantification for the Gulf.** Measure the AIS-dark fraction over the study period and compare against the ~50% figure reported for Hormuz.
4. **An operational demonstrator** proving the pipeline runs end-to-end on commodity hardware.

**Explicitly out of scope** (state this in the thesis, do not attempt): fine-grained vessel type from SAR pixels alone; real-time full-Gulf SAR; vessel *identity* from imagery; anything requiring paid tasking.

---

## 5. System architecture

```
                        ┌────────────────────────────────────┐
                        │        INGESTION (local)           │
                        ├────────────────────────────────────┤
  aisstream.io  ──WS──▶ │ ais-ingest    decode → dedupe →    │
  (aiscast fallback)    │               track assembly       │
                        │                                    │
  CDSE OData/S3 ──────▶ │ sar-ingest    poll → byte-range   │
  (S1 IW_GRDH_1S-COG)   │               read AOI → tile      │
                        │                                    │
  NOAA VIIRS VBD ─────▶ │ vbd-ingest    nightly CSV pull     │
  GFW API / OFAC  ────▶ │ ref-ingest    registry + sanctions │
                        └───────────────┬────────────────────┘
                                        ▼
                        ┌────────────────────────────────────┐
                        │   PostgreSQL + PostGIS + Timescale │
                        │   ais_positions (hypertable)       │
                        │   vessels · sar_scenes             │
                        │   detections · fusion_matches      │
                        │   events (dark/STS/spoof) · zones  │
                        └───────────────┬────────────────────┘
                                        ▼
        ┌───────────────────────────────┴──────────────────────────┐
        ▼                                                          ▼
┌───────────────────────┐                          ┌───────────────────────────┐
│  INFERENCE (CPU)      │                          │   FUSION ENGINE           │
│  ONNX Runtime INT8    │                          │   extrapolate AIS to t₀   │
│  ① land + infra mask  │                          │   ANISOTROPIC azimuth-    │
│  ② CFAR prefilter     │──── detections ────────▶ │     corrected gate        │
│  ③ point detector     │                          │   Hungarian assignment    │
│    (stride-2 heatmap) │                          │   → MATCHED | DARK        │
└───────────────────────┘                          └─────────────┬─────────────┘
        ▲                                                        ▼
        │ ONNX weights                              ┌───────────────────────────┐
┌───────┴───────────────┐                           │  API — FastAPI            │
│  TRAINING (Kaggle)    │                           │  REST + WebSocket         │
│  P100 / 2×T4          │                           └─────────────┬─────────────┘
│  timm encoder +       │                                         ▼
│  U-Net decoder,       │                           ┌───────────────────────────┐
│  multi-head outputs   │                           │  WEB — React + MapLibre   │
└───────────────────────┘                           │  + deck.gl overlays       │
                                                    └───────────────────────────┘
```

**Stack:** Python 3.12 / FastAPI / PostGIS + TimescaleDB / ONNX Runtime · React + Vite + TypeScript / MapLibre GL JS + deck.gl / Tailwind + shadcn/ui · Docker Compose.

> Python 3.14 is installed locally. **Use 3.12** — the geospatial stack (rasterio, GDAL, pyproj) and PyTorch lag on 3.14, and Kaggle runs 3.11/3.12. Matching Kaggle's version avoids pickle/ONNX opset friction.

### 5b. Installed skills

Six skills were installed for this project. Invoke them at the phases noted:

| Skill | Source | Use at |
|---|---|---|
| `gdal` | isaaccorley (73★) | **Phase 2** — COG creation, windowed reads, reprojection, tiling |
| `maplibre-tile-sources` | **official MapLibre** (679) | **Phase 2 & 7** — serving SAR COG tiles + vessel data to the map |
| `maplibre-cartography` | **official MapLibre** (117) | **Phase 7** — label/symbol legibility *on imagery*, figure-ground for point icons, layer ordering. Directly addresses the hardest part of the UI: making vessel markers readable over SAR backscatter |
| `detect-objects` | isaaccorley (73★) | **Phase 3/5** — pretrained geospatial detectors, explicitly including **ships** |
| `computer-vision-opencv` | mindrally (3.5K) | **Phase 3** — morphology, connected components, `minAreaRect` in the CFAR stage |
| `pytorch-lightning` | orchestra-research (761) | **Phase 5** — training loop, checkpointing (critical for the 12 h Kaggle cap) |

Install counts are low in this niche, so these were selected on **source reputation** over popularity — two are from the official MapLibre organisation, two from a recognised remote-sensing ML researcher. Skills run with full agent permissions; review before relying on them.

Already available in-session and worth using: `postgis`, `fastapi`, `fastapi-templates`, `dataviz`, `docker-patterns`, `postgres-patterns`, `database-migrations`, `tdd-workflow`, `python-testing`, `frontend-design`, `web-design-guidelines`, `accessibility`, `localization-design` (for RTL/Persian).

---

## 6. Data sources

| Source | What it gives | Access | Latency | Cost | Verdict |
|---|---|---|---|---|---|
| **Copernicus CDSE** ⭐⭐ | Sentinel-1 `IW_GRDH_1S-COG` VV+VH; Sentinel-2 L2A | OData / STAC / **S3 byte-range** | **median 2.5 h** | Free, 12 TB/mo | **The backbone.** Cleanest licence in the whole stack |
| **aisstream.io** | Live AIS, pre-decoded JSON, bbox-filtered | `wss://stream.aisstream.io/v0/stream` | seconds | Free | ✅ Best free live AIS. ⚠️ Coastal only; **no published ToS/licence** |
| **aiscast** (openwaters.io) | aisstream-compatible open AIS | `wss://ais.openwaters.io/v0/stream` | seconds | Free | Drop-in fallback; propagate `source` attribution |
| **xView3-SAR** ⭐ | **991 S1 scenes, 1,400 gigapixels, 243,018 verified objects** | Public download | archive | Free | **Primary training set.** Purpose-built for this exact task |
| **OSSDD** | 15,197 S1 VV/VH patches, ~55,759 ships | HF `sylviaHoch/OpenSARShip-Ship-Detection-Dataset` | archive | Free | CC BY-NC-SA. Strong supplement |
| **Global Fishing Watch API** | AIS-derived events + **S1 SAR detections** (`length_m`, `matching_score`) | REST v3 + Bearer | AIS 96 h; SAR ~5 d | Free | ⭐ Free DL detections to benchmark against. ⚠️ CC BY-NC |
| **Umbra / Capella / ICEYE Open Data** | SAR **16 cm – 1 m** | S3 + STAC, **no auth** | archive | Free | ✅ **CC BY 4.0.** Best free high-res SAR chips for training |
| **OpenSanctions** | **3,700+ vessels with IMO numbers**, consolidated | REST + bulk | daily | Free (non-comm) | ⭐ Better than parsing OFAC by hand |
| **OFAC SDN raw** | Sanctioned vessels | XML/CSV, no key | weekday | Free | ✅ Always-safe public-domain fallback |
| **NOAA VIIRS VBD** | Nightly lit-vessel detections | `eogdata.mines.edu` CSV | ~1 day | Free | FINAL tier open; NRT is paid |
| **adsb.lol** | ADS-B, **unfiltered military** | REST | real-time | Free | ✅ ODbL. P-8/MQ-4C orbits flag interesting areas |
| **DMA / NOAA AIS archives** | Bulk historical AIS (Denmark / US) | Bulk CSV | archive | Free | 🟡 Wrong AOI, but excellent free corpus to develop and test the AIS logic |
| GSHHG / OSM coastline | Land mask | Shapefile | static | Free | Essential CFAR prefilter |
| ~~Sentinel Hub Process API~~ | Preprocessed tiles | — | — | 10k PU/mo | ❌ **Avoid for bulk** — see below |
| ~~MarineTraffic~~ | — | — | — | — | ❌ **Scraping contractually prohibited.** Do not |
| **Skylight** (Ai2) | Free multi-sat CV detections, >10M km²/day | REST API | NRT | Free | ❓ **Email them** — eligibility for a non-gov project is the highest-value unknown |

**Two decisive access findings:**

1. **CDSE publishes a Cloud-Optimized GeoTIFF product directly: `IW_GRDH_1S-COG` (~830 MB vs ~1.7 GB for the SAFE archive).** This means **byte-range reads against S3 for just the AOI water polygon** — no full-scene download, no local orthorectification pipeline, no SNAP install. This eliminates most of what would have been the hardest phase. Budget: ~12–15 Gulf passes/month × 2–3 scenes × 0.83 GB ≈ **25–40 GB/month against a 12 TB free quota — 0.3% utilisation.**

2. **Sentinel Hub's Process API is a trap and must not carry the pipeline.** 1 PU = 512×512 px × 3 bands. The Gulf's ~251,000 km² of water at 10 m, VV+VH, 16-bit ≈ **6,383 PU — or ~12,800 PU with orthorectification, exceeding the entire 10,000 PU/month free tier on a single mosaic.** Reserve PUs strictly for interactive browse tiles and thumbnails in the UI.

**Other CDSE quotas:** 50,000 req/month, 2,000 req/min (S3), **4 concurrent connections**, 20 MB/s per connection, 10-min access tokens / 60-min refresh.

**Licensing — draw the commercial line now.** Copernicus is the only genuinely permissive source here: free, worldwide, royalty-free, perpetual, **no commercial restriction, no non-EU-user restriction**, attribution `"Contains modified Copernicus Sentinel data [Year]"`. Almost everything else free is non-commercial (GFW, OpenSanctions, Maxar ODP, OpenSky, Planet E&R) or has **no published licence at all** (aisstream, AISHub — an open question to aisstream about commercial use has sat unanswered since April 2026). For a thesis all of this is fine. Architect GFW/OpenSanctions/Skylight as **clearly separable enrichment modules behind an interface**, so a future commercial version can drop them and still work on a Copernicus-only core.

---

## 7. Deep learning design — the core

### 7.1 Architecture — dense point detection, NOT bounding boxes

**The instinct to reach for an oriented-bounding-box detector (YOLO-OBB, MMRotate, LSKNet) is wrong for this sensor, and avoiding it removes most of the project's technical risk.**

At Sentinel-1 IW resolution (10 m pixel spacing, ~20 m true resolution) a 300 m VLCC is **30 × 5 px** and a 25 m dhow is **2–3 px**. There is no meaningful orientation to regress on most targets. The xView3-SAR winner (aggregate 0.617 on holdout, ~3× the government reference model) used **no boxes at all** — a CenterNet/U-Net hybrid emitting a **stride-2 dense heatmap** with separate heads. Every other top-5 solution is in the same family. His own ablation shows why stride matters more than box geometry:

| Output stride | Localization F1 ceiling |
|---|---|
| 16 | 0.9672 |
| 8 | 0.9948 |
| 4 | 0.9997 |
| **2** | **0.9999** |

**Adopted architecture:**

```
2048×2048 tile (VV+VH) → timm encoder (EfficientNet-B4/V2S or ConvNeXt)
                       → U-Net decoder → stride-2 dense output
   heads: objectness · sub-pixel offset · length · width · heading · is_vessel · coarse_class
```

This choice pays off four times over:
1. **No AGPL.** Ports from the xView3 winner's **MIT-licensed** repo. Your code, your weights.
2. **Trivial ONNX export.** A conv stack plus 3×3 max-pool NMS — no custom operators. Rotated NMS is a custom op that requires MMDeploy (itself frozen) and is a genuine engineering cost. Avoiding it is what makes the CPU-deployment story work.
3. **No tile-seam box-merge problem.** Accumulate a scene-level heatmap, NMS once at scene level.
4. **Simpler to train** on a constrained budget than an OBB detector.

**Where CFAR still earns its place — as a compute prefilter, not an accuracy mechanism.** Zero of the top-5 xView3 solutions used CFAR. But a Sentinel-1 scene is ~425 Mpx and >99.9% empty ocean; a two-parameter CFAR pass costs milliseconds and cuts CNN tiles by 1–2 orders of magnitude. Keep it for (a) throughput on your CPU, (b) an interpretable deterministic-Pfa baseline to compare against, and (c) cold-start pseudo-labelling before Gulf labels exist. **Build CNN-first; add CFAR when compute-bound.**

**Reserve OBB for a later, optional tier** on high-resolution imagery (Umbra/Capella/ICEYE open data at 0.16–1 m), where orientation and fine class actually exist in the pixels. If pursued, use **PP-YOLOE-R** (Apache-2.0, 78.28 single-scale DOTA) — not Ultralytics, and **not LSKNet**, which is CC BY-NC despite GitHub showing `NOASSERTION`.

### 7.1b Training recipe — port these details verbatim

From the xView3 winner (MIT). These are hard-won and non-obvious:

- **Normalization:** uint16 → [0,1] via a **tuned sigmoid** (not linear scaling), so sea/land saturate to 0/1 and ship vicinities land in ~[0.2, 0.8]. Missing-data regions → 0. Freeze whatever you pick — normalization is a silent train/serve skew source.
- **Two channels only — VV+VH.** Bathymetry, wind speed/direction, and ice mask **all failed to help**; the model infers wind implicitly from the SAR signal. This saves an entire data-fusion workstream.
- **Objectness:** reduced focal loss with a **fixed 3 px heatmap radius** (not size-dependent — this neatly solves unknown-length targets).
- **Classification heads:** weighted BCE, **label smoothing 0.05**, 2 px label radius (~40× more training samples than a 1 px radius).
- **Length/offset:** MSE, scale 0.1 and 0.001 respectively.
- **⚠️ Focal loss is unusable under label noise** — the winner says so explicitly. Balanced sampling + BCE worked better. Directly relevant to AIS-derived weak labels.
- **Shannon entropy regularization** on classification heads for unlabelled objects, pushing predictions toward 0/1. **Directly applicable to our AIS auto-labels**, where many detections have no AIS truth.
- **Augmentation must be constrained in SAR:** brightness/contrast, Gaussian noise, H/V flips, ±5° rotation, ±10% scale, minor elastic. Use **unclipped** Albumentations variants — the standard ones clip and corrupt the SAR distribution.
- **Inference tiling:** 2048×2048, **stride 1536** (25% overlap) → ~204 tiles/scene. Accumulate into a scene-level canvas, NMS once. Left-right flip TTA gives a measurable gain.
- Winner's training cost: 2× RTX 3090, ~6–9 h per fold. A single model without ensembling is well inside the Kaggle budget.

### 7.2 Model outputs — attributes are the deliverable, class is secondary

Two independent production systems converged on the same answer, and it is not "vessel type":

- **Global Fishing Watch** (*Nature* 2024, 67M tiles / 2 PB): object identification >97%, **length R² = 0.84**, infrastructure oil/wind/other >98% — but fishing-vs-non-fishing only >90%, and that is a *binary*.
- **Allen Institute Skylight** (`allenai/vessel-detection-sentinels`, Apache-2.0, in production at global scale) outputs **length (m), width (m), heading buckets, speed, fishing probability** — and deliberately stops there. No vessel type.

So the primary deliverable is **`is_vessel` + length + width + heading + a fishing/non-fishing binary, with calibrated confidences.** Coarse type is a *secondary, honestly-reported* head:

| Class | AIS type codes | Rationale |
|---|---|---|
| `cargo_tanker` | 70–79, 80–89 | Large, rectangular, 100–330 m |
| `fishing` | 30 | Small, 10–40 m |
| `service_tug` | 31, 32, 50–59 | Small–medium, distinct wake |
| `passenger_other` | 60–69, 36, 37, others | Remainder |

Do **not** split tanker from cargo — geometrically near-identical at 10 m. Derive finer type downstream from **length × width × AIS/registry priors × behaviour**, not from pixels.

⚠️ **The literature will mislead you here.** OpenSARShip reports 97.5% (3-class) and FUSAR-Ship 94.92%. Both are *chip-level* (ship already located and centred), on 3–6 classes, with OpenSARShip **~61% "Cargo"** so a constant predictor scores 61%. Worse, **FUSAR-Ship is Gaofen-3 at ~1 m, not Sentinel-1 at 20 m** — conflating the two is the single most common way to mislead yourself planning this project. The 2025 survey (arXiv 2503.11906) names data scarcity and non-standardised protocols as the field's core problems.

### 7.3 Datasets — with licences, because they differ sharply

| Dataset | Size | Sensor | Labels | Licence |
|---|---|---|---|---|
| **xView3-SAR** ⭐ | 991 scenes, ~1,422 Gpx, >220k annotations | **Sentinel-1 IW GRD** | `is_vessel`, `is_fishing`, `vessel_length_m`, `confidence`, `distance_from_shore_km` | ⚠️ reported CC BY-NC-SA 4.0 — **verify** |
| **LS-SSDD-v1.0** ⭐ | 15 full scenes (24000×16000) → 9,000 sub-images | **Sentinel-1 IW** | HBB, single class | ✅ **Apache-2.0** |
| **Official-SSDD** | 1,160 img / 2,358 ships | mixed 1–15 m | HBB + RBB + polygon | ✅ **Apache-2.0** |
| **HRSID** | 5,604 img / 16,951 ships | 0.5–3 m | HBB + instance seg | ⚠️ **GPL-3.0** |
| **SRSDD-v1.0** | 666 img / 2,884 ships | GF-3 @1 m | oriented quad, **6 classes** | Baidu pan; imbalanced (LawEnforce = **25** instances) |
| **OpenSARShip 2.0** | 34,528 chips, 17 AIS types | Sentinel-1 | type + AIS metadata | ❌ no SPDX; "research only" |
| **FUSAR-Ship** | >5,000 chips, 15 cat | **GF-3 @1 m** | type + AIS matchup | ❌ no SPDX |
| **SARDet-100K / RSAR** | 116k / 95k img | mixed | HBB / OBB | ⚠️ CC BY-NC 4.0 |
| **Persian Gulf platforms** ⭐ | **1,731 platforms (2025)** | Sentinel-1 archive | point locations | ✅ **CC BY 4.0** — arXiv 2603.19801 |

**xView3 is the ideal primary training set** — same sensor, same task, length + `is_vessel` + `is_fishing`, and its metric is the right evaluation protocol:
`aggregate = loc_fscore × (1 + length_acc + vessel_fscore + fishing_fscore + loc_fscore_shore) / 5`
Note detection F1 **multiplies** everything: get detection wrong and nothing else counts. Matching tolerance 200 m.

⚠️ **Two xView3 caveats, one of which may be a hard blocker:**
1. **Access is US export-control gated** at `iuu.xview.us` against the State Sponsors of Terrorism list. **Given this project's AOI and your location, verify you can actually register and download before planning around it.** Test this in Phase 0.
2. **Known defect:** all scenes cluster into ~42 geographic locations with **scene-level leakage across train/val/public-test/holdout**, and the `train` split is noisier than `validation` — the winner trained on validation only. Published leaderboard numbers are optimistic.

**Fallback if xView3 is inaccessible** (fully viable, all permissive): **LS-SSDD-v1.0** (Apache-2.0, full Sentinel-1 scenes — the cleanest licence/sensor match available) + **Official-SSDD** (Apache-2.0) for pretraining, **GFW's public Sentinel-1 detections over the Gulf** as pseudo-labels, and your own AIS auto-labels (Section 8) as the main training signal. This path is slower to good accuracy but has no legal or access ambiguity at all.

⭐ **The Persian Gulf offshore platform dataset is the highest-ROI free asset for this AOI.** 1,731 platforms as of 2025, derived from the Sentinel-1 archive, **CC BY 4.0, public**. Every platform is a bright persistent point target that a naive detector calls a vessel. This replaces the "build a persistence mask from scratch" plan with "download a published mask and extend it."

### 7.4 Training budget on Kaggle

| Run | Est. GPU-h | Notes |
|---|---|---|
| Point detector, pretrain (xView3 or LS-SSDD+SSDD) | 6–9 | 2048² tiles, EfficientNet-B4 + U-Net, stride-2 head |
| Fine-tune on Gulf auto-labels | 2–3 | transfer from above |
| Ablations (5 configs) | 6–8 | stride, normalization, loss weights, entropy reg, VV-only vs VV+VH |
| **Total** | **~15–20** | Inside the 30 h/week quota with headroom |

The winner trained this architecture in **6–9 h per fold on 2× RTX 3090**. A single fold on a P100, no ensembling, is comfortably within budget. Skip the 12-model ensemble — it bought 0.51→0.617 in a competition, but it is not what a thesis needs and it does not fit the CPU deployment story.

**Checkpoint to `/kaggle/working` every epoch** — the 12 h session cap will otherwise cut you off mid-run and lose everything.

---

## 8. AIS-supervised auto-labelling (highest-value component)

This is what makes the project novel and removes the need for manual annotation.

**Procedure, per SAR scene:**

1. Read scene footprint and acquisition time `t₀` (± ~10 s across the swath).
2. Query `ais_positions` for all tracks with a fix within ±10 min of `t₀` inside the footprint.
3. **Interpolate** each track to exactly `t₀` — a vessel at 15 kn moves 460 m in 60 s, which is 46 pixels. Naive nearest-fix matching will mislabel constantly.
4. **Correct SAR azimuth displacement — and do not underestimate it.** A target with radial (line-of-sight) velocity `v_r` is displaced *along-track* in the SAR image by `Δaz ≈ (R / V) · v_r`, where `R` is slant range and `V` effective sensor velocity. For Sentinel-1 IW, `R/V` is **on the order of 100 s** (slant range ~800–880 km, effective velocity ~7.5 km/s):

   > **≈ 100 m of azimuth displacement per 1 m/s of radial velocity.**

   A vessel at 12 kn (6.2 m/s) with a significant radial component is displaced **500–700 m** — that is **2–3× the 200 m tolerance xView3 uses for matching**, and several times a ship's own length.

   ⚠️ **This is the error that quietly destroys dark-vessel detection.** Gate AIS matching at a fixed 200–300 m radius and you will systematically fail to match *fast-moving* vessels, then report them as dark. Your headline metric becomes a false-positive generator, biased toward exactly the vessels of most interest.

   Three mitigations, use all three:
   - **Anisotropic gate** — wide along-track, narrow across-track, sized from AIS SOG/COG projected onto the range direction.
   - **Correct the AIS position** by the predicted `Δaz` before matching (SOG/COG from AIS, imaging geometry from product metadata).
   - **Compute `R/V` per sub-swath from scene metadata — never hardcode it.** It varies with sub-swath and incidence angle.
5. Project corrected lat/lon → pixel coordinates.
6. Emit a chip centred there, labelled with AIS truth: `length`, `beam`, `heading` (COG), `ship_type` → coarse class, `mmsi`, `imo`.
7. **Quality gates** — discard if: another AIS target is within 300 m (ambiguous), the vessel is within 500 m of coast (land clutter), inside a known infrastructure buffer, or the CFAR response at the predicted pixel is below threshold (AIS says a ship is there, radar disagrees → likely spoofed position, which is *itself* a labelled event worth keeping separately).

**Yield estimate:** the Gulf is dense — expect several hundred to a few thousand AIS-matched vessels per scene. Around 100 processed scenes should yield **50k–200k labelled chips**. At 128×128 × 2 pol × uint8 ≈ 32 KB/chip, 200k chips ≈ **6.4 GB** — fits Kaggle's 20 GB working disk and uploads as a private Kaggle Dataset.

**Negative mining is mandatory.** Without it the model calls every bright pixel a ship. Sample hard negatives from: the **published 1,731-platform Persian Gulf infrastructure dataset** (CC BY 4.0 — use it directly rather than deriving persistence from scratch, then extend with your own ≥5-scene persistence detections), CFAR responses with no AIS and no structure match, wind-roughened sea, azimuth ambiguities (ghost returns at predictable offsets), and land edges. Target ~1:3 positive:negative.

**Handling unlabelled detections — use entropy regularization.** Many CFAR/CNN detections will have no AIS truth at all (that is the point of the product). Rather than discarding them, apply **Shannon entropy regularization** on the classification heads for unlabelled objects, pushing predictions toward 0/1. This is the xView3 winner's technique for exactly this situation and it maps onto AIS weak labels perfectly.

**⚠️ Label-bias caveat — this is the thesis's central methodological threat, address it head-on.** Labels come from AIS, so the training set contains *only AIS-broadcasting vessels*. But the target class — dark vessels — is by definition absent from the labels, and skews smaller, older, and less cooperative. **The label pipeline systematically under-samples the very vessels the product exists to find.** With 50–80% of Gulf tanker transits reportedly running dark, this bias is large, not marginal.

Mitigations, in order of strength:
1. Validate on **xView3-SAR**, whose labels are human-verified and *do* include non-AIS vessels — the only clean unbiased check available. (Contingent on access; see §7.3.)
2. Hand-label a stratified held-out Gulf set (~200 chips) as an unbiased test set.
3. Use **GFW's public Sentinel-1 detections** over the Gulf as an independent second opinion — they include an AIS matched/unmatched flag, so unmatched GFW detections are dark-vessel candidates you did not label yourself.
4. Report every metric **stratified by vessel length**, so the small-vessel failure mode is visible rather than averaged away.

---

## 9. Fusion and event detection

**Matching — three escalating levels. Ship v1, plan for v2.**

- **v1 (build first):** extrapolate AIS to exact acquisition time, apply the **anisotropic azimuth-corrected gate** from §8 step 4, score on position/length/heading residuals, then solve globally with the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`). Greedy nearest-neighbour fails badly in the dense anchorages off Fujairah and Bandar Abbas — do not use it.
- **v2 (the rigorous version):** GFW's published probabilistic method — build a **probability field** for where each AIS vessel could be at scene time given messages before *and* after, then score {AIS, detection} pairs against it. Its clever part is **global threshold calibration**: set the acceptance threshold so the total matched count equals the expected number of AIS-equipped vessels in-scene. Reference implementation: `GlobalFishingWatch/paper-longline-ais-sar-matching`.
- **Corroborate independently:** reuse GFW's **`pipe-gaps`** (Apache-2.0) for the AIS-side "went dark" signal. A vessel flagged dark by *both* an unmatched SAR detection and an independent AIS gap analysis is a far stronger call than either alone.

**Events generated:**

| Event | Logic | Value |
|---|---|---|
| `DARK_VESSEL` | SAR detection, no AIS match, not infrastructure | ⭐ Headline output |
| `AIS_GAP` | Known vessel silent > N hours inside AOI | Correlates with dark detections |
| `POSITION_SPOOF` | AIS says vessel here; CFAR says empty water | Direct spoofing evidence |
| `STS_TRANSFER` | Two vessels < 500 m, near-zero speed, > 2 h, offshore | Sanctions-evasion signature |
| `ZONE_ENTRY` | Crosses user-drawn geofence | Analyst workflow |
| `IDENTITY_ANOMALY` | Reported length vs SAR-measured length mismatch | Cross-check |

Each event stores its evidence — the SAR chip, the AIS track segment, match scores — so an analyst can audit any call. **Never surface an unexplainable alert.**

---

## 10. UI/UX

**Principle: a dark-themed operational console, not a consumer map.** Three-pane layout — left filter rail, centre map, right context panel — with a bottom timeline scrubber.

**Map** — MapLibre GL JS with deck.gl overlays in interleaved mode:
- `IconLayer` — AIS vessels, rotated to heading, coloured by class
- `TripsLayer` — animated recent tracks (this is the "alive" feeling)
- `ScatterplotLayer` — SAR detections, styled **distinctly**: hollow rings, not filled icons
- `BitmapLayer` — SAR scene footprints and browse imagery
- `PolygonLayer` — geofences, EEZ, traffic separation schemes

**Non-negotiable UX rules for this domain:**
1. **Every object shows its age.** A subtle opacity ramp plus an explicit "last seen 4h 12m ago". Stale contacts must never look live.
2. **AIS and SAR are visually distinct.** Filled icon = someone told us. Hollow ring = radar saw something. The distinction is the whole point.
3. **Dark vessels are the visual priority** — amber/red ring, always on top, never clustered away.
4. **Timeline scrubber** — drag back through time; SAR footprints appear as bands showing when each area was actually observed. This teaches the user the coverage cadence better than any explanation.
5. **Coverage honesty overlay** — a toggle showing "last SAR look" as a heatmap. Where it is dark, the platform is blind, and saying so builds trust.
6. Vessel detail panel: identity, AIS history sparkline, sanctions flag, matched SAR chip **side-by-side with the model's prediction and confidence**.

At Gulf densities (~3k–8k active MMSIs/day) simple GeoJSON over WebSocket is fine — no vector tiles needed until ~50k objects. Do not over-engineer this.

Full Persian/Farsi RTL support is worth building in from the start if the audience is regional; it is painful to retrofit.

---

## 11. Phased delivery

Each phase ends with something that runs.

### Phase 0 — Foundation (½ day)
Redirect caches to E:. Python 3.12 venv. Docker Compose: PostGIS + TimescaleDB. Repo skeleton, `.env.example`, pre-commit, pytest.
**Done when:** `docker compose up` gives a healthy DB and `pytest` runs green on an empty suite.

### Phase 1 — AIS pipeline + live map (2–3 days) ⭐ *first visible slice*
`ais-ingest` worker on aisstream.io with aiscast fallback and reconnect/backoff. aisstream emits **pre-decoded JSON**, so no decoder is needed for it — keep `pyais` for raw-NMEA sources (aiscast UDP, archives).
**Hard API constraints to code against:** 3 subscribed connections/account, 3 per IP, **subscription must be sent within 3 s of connect**, ≤1 subscription update/second, ≤200 MMSIs per subscription, bounding box required, **the server drops messages if you consume too slowly** (so decouple socket read from DB write with a bounded queue), and browser connections are prohibited — the frontend must go through your API, never direct.
TimescaleDB hypertable + continuous aggregates. FastAPI REST + WebSocket.
**Develop against the free Danish (DMA) and NOAA archives first** — they are bulk, free, and realistic, letting you build and test track assembly and matching before depending on a live feed.
**Done when:** live vessels move on the map and the ingester survives 24 h unattended.

### Phase 1.5 — AIS plausibility filter (1–2 days) ⭐ *promoted to first-class*
Given that a reported ~2/3 of August 2026 Hormuz transits were false, this cannot be an afterthought. Flag and score, do not silently drop:
- position on land (GSHHG mask) · impossible speed jumps · impossible turn rates · teleports between fixes
- duplicate MMSI broadcasting from two places · MMSI MID prefix inconsistent with claimed flag
- suspiciously collinear "straight-line" tracks (a classic synthetic-track signature)
- MMSI of 0, latitude 91.0, NUL bytes in names — the feed carries a lot of malformed records
Every position gets a `plausibility_score` and `anomaly_flags`; the UI must be able to show or hide implausible tracks.
**Done when:** known-bad tracks are flagged and the flag is visible in the UI.

### Phase 2 — SAR ingestion (1–2 days) ← *much smaller than expected*
CDSE publishes `IW_GRDH_1S-COG` directly, which removes the entire local orthorectification pipeline. **Do not install SNAP. Do not build a preprocessing chain.**
- CDSE OAuth (10-min access token, 60-min refresh) + OData search by footprint/date/`productType`.
- **S3 byte-range reads via `rasterio`/GDAL `/vsis3/` against the COG**, windowed to the AOI water polygon. Respect the 4-concurrent-connection quota.
- Land mask from GSHHG buffered 500 m seaward.
- Persist scene metadata; never persist raw pixels beyond the chips you extract.
- Use `IW_GRDH_1S-COG`, **not** SLC (4× larger, complex-valued, only needed for interferometry or Doppler work).
**Done when:** an AOI window renders as a tile layer under the AIS vessels, with peak RAM under 2 GB and no full-scene download.

### Phase 3 — CFAR baseline + masks (2 days)
Two-parameter CA-CFAR with integral images, windowed via rasterio. Land mask (GSHHG) + **the published 1,731-platform Persian Gulf infrastructure mask (CC BY 4.0)**. Morphological cleanup, connected components, centroid + rough extent.
Serves three purposes: a zero-training working detector for the demo, a tunable-Pfa interpretable **baseline** for the thesis comparison, and later a **compute prefilter** that cuts CNN tiles by 1–2 orders of magnitude on your CPU.
**Done when:** CFAR detections appear on the map with a measured false-alarm rate.

### Phase 4 — Auto-labelling (3–4 days) ⭐ *the novel piece*
Track interpolation to `t₀`, **anisotropic azimuth-corrected** matching, Hungarian assignment, quality gates, chip extraction, negative mining (platform mask + hard negatives). Dataset packaging + manifest + train/val/test split **by scene** — never random; adjacent chips from one scene leak between splits and inflate results.
**Done when:** ≥20k labelled chips exist with a visual QA contact sheet confirming ships are centred.

### Phase 5 — Model training on Kaggle (4–5 days)
Port the xView3 winner's **MIT-licensed** CenterNet/U-Net point detector: timm encoder + U-Net decoder, **stride-2** dense output, heads for objectness / sub-pixel offset / length / width / heading / `is_vessel` / coarse class. Apply the §7.1b recipe verbatim — tuned sigmoid normalization, 3 px objectness radius, label smoothing 0.05, BCE not focal, entropy regularization for unlabelled objects, unclipped SAR augmentations.
Pretrain on xView3 (or the LS-SSDD + SSDD fallback), fine-tune on Gulf auto-labels. Export ONNX (opset 17) → INT8 dynamic quantization. Benchmark CPU latency locally.
**Done when:** the ONNX model runs on the laptop CPU and beats the CFAR baseline on held-out data.

### Phase 6 — Fusion + events (3 days)
Hungarian AIS↔SAR matcher, event engine, evidence storage, alert feed.
**Done when:** dark vessels are flagged with auditable evidence and a measured Gulf-wide dark rate.
**Revised 2026-10-04 (agreed with the user; v1 design in §17 "Phase 6 feasibility"):** dark-vessel candidates in the five boxes are flagged with evidence cards, and the dark rate is reported with a bootstrap interval as a curve over the gate margin, by region and period, with the operating margin and pre-filter settings chosen on validation passes only and v1's agreement with GFW's matched flag reported.

### Phase 7 — UI/UX build-out (4–5 days)
Timeline scrubber, coverage honesty overlay, vessel detail panel with SAR chip, filters, watchlists, geofence drawing, alert feed, sanctions badges, RTL support.
**Done when:** a non-expert can find a dark vessel unaided.
**Revised 2026-10-06 (the user's decision; §17 "Phase 7 closed"):** the console
built so far is enough for the bachelor's thesis. Phase 7 closes with the
Persian, right-to-left console as its last slice; watchlists, geofence
drawing, sanctions badges, the coverage heatmap and further filters move to
later work, and the non-expert usability test was not run.

### Phase 8 — Evaluation + writing (4–5 days)
Full benchmark tables, ablations, error analysis, figure generation, reproducibility pack.

**Total: ~5–6 weeks solo.** Phases 1–3 give a demoable system in the first two weeks — and note that Phase 2 shrank from 3–4 days to 1–2 once the CDSE `-COG` product removed the preprocessing chain, which roughly pays for the new Phase 1.5.

---

## 12. Evaluation protocol (thesis rigor)

- **Splits by scene, never random.** State this explicitly; it is the most common methodological error in SAR ship detection papers. ⚠️ Note that **xView3 itself violates this** — its ~42 locations leak across official splits, so its leaderboard numbers are optimistic and yours will not be comparable unless you say so.
- **Primary metric: the xView3 aggregate**, so results are comparable to a published leaderboard:
  `aggregate = loc_fscore × (1 + length_acc + vessel_fscore + fishing_fscore + loc_fscore_shore) / 5`
  Detection F1 multiplies everything. Reference: 1st place 0.617 holdout; government baseline ≈ 1/3 of that.
- **Detection:** P/R/F1 at the operating point, false alarms per 100 km². Report on the Gulf held-out set and whichever public benchmark is accessible.
  ⚠️ **Do not use SSDD/HRSID leaderboard position to justify architecture choices** — both are saturated (AP50 routinely >95–98%), single-class, and every method overfits them. Use them as smoke tests only.
- **Stratify every metric by vessel length** (<25 m, 25–100 m, >100 m). Small-vessel recall is where these systems fail and where the honest discussion lives.
- **Length regression:** R², MAE in metres. Target R² ≈ 0.84 (GFW parity).
- **Classification:** per-class P/R + confusion matrix. Expect `cargo_tanker` vs `fishing` to separate and `service_tug` vs `passenger_other` to be weak — report it rather than hiding it.
- **Efficiency:** seconds per scene end-to-end on the Ryzen CPU, FP32 vs INT8, peak RAM. Headline result, not an appendix.
- **Ablations:** output stride (2/4/8/16 — reproduces the winner's localization-ceiling result); **with/without azimuth correction** (expect a large effect — the single best figure in the thesis); sigmoid vs linear normalization; with/without entropy regularization; with/without negative mining; VV-only vs VV+VH; CFAR window sizes.
- **Baselines:** CFAR-only · the point detector · CFAR-prefilter + point detector · **GFW's public Sentinel-1 detections over the same scenes** (an independent published system — a strong external reference point).

---

## 13. Risks

| Risk | L | I | Mitigation |
|---|---|---|---|
| **xView3 download is US export-control gated** and may be unavailable given the AOI/your location | **High** | **High** | ⚠️ **Test registration in Phase 0, before designing around it.** Fallback is fully viable: LS-SSDD (Apache-2.0, Sentinel-1 full scenes) + Official-SSDD (Apache-2.0) + GFW pseudo-labels + your own AIS auto-labels |
| **Azimuth-shift mis-modelling manufactures fake dark vessels** | **High** | **High** | ~100 m per m/s radial → 500–700 m at 12 kn. Anisotropic gate + per-sub-swath `R/V` from metadata + AIS position correction. Ablate it and show the effect |
| **AIS label bias** — dark vessels absent from training labels by construction | **High** | **High** | Validate on human-verified xView3; hand-label ~200 held-out Gulf chips; cross-check GFW unmatched detections; report length-stratified |
| aisstream: coastal-only coverage + bandwidth limits + no published licence | High | Med | Tight bbox, only needed message types, bounded consumer queue; aiscast fallback; **frame the AIS layer as port/anchorage awareness, not Gulf-wide** |
| Widespread GNSS jamming/spoofing corrupts AIS truth | **High** | **High** | Phase 1.5 plausibility filter is mandatory, not optional. Never treat AIS as ground truth |
| Oil platform false positives (1,731 in the Gulf) | High | Med | Use the **published CC BY 4.0 platform dataset** as a static mask; extend with ≥5-scene persistence |
| Small wooden dhows near-invisible to C-band SAR | High | Med | **Stated limitation, not a bug.** Low RCS. Bound it with a length-stratified recall curve |
| Speckle filtering erases 2–5 px targets | Med | **High** | **Do not speckle-filter.** Feed near-raw values with tuned sigmoid normalization; let the CNN handle speckle |
| GFW SAR detections have a discontinuity (offline 3 Jul → 8 Sep 2026, S1C/S1D recalibration) | Med | Med | Do not compute baselines or train across that boundary; treat pre/post statistics as non-comparable |
| CFAR params tuned on S1A don't transfer to S1C/S1D | Med | Med | GFW hit exactly this. Re-tune per satellite; keep `satellite` as a first-class field |
| Kaggle 12 h session cuts a run | Med | Med | Checkpoint every epoch to `/kaggle/working`; resumable training |
| 10 GB RAM exhaustion | Med | High | Windowed reads only; never `.read()` a full scene; assert on array allocation size |
| Scope creep (this domain is a rabbit hole) | High | High | Phases 1–3 first; everything past Phase 6 is optional polish |
| CDSE / API access from your region | Med | High | **Verify in Phase 0** — test CDSE registration and aisstream before building on them |

---

## 14. Verification

- **Phase gates:** each phase's "Done when" is the acceptance test.
- **Unit:** CFAR on synthetic scenes with known targets; azimuth-shift maths against worked examples; Hungarian matcher on hand-built cases; AIS decode against `pyais` fixtures. Target 80% coverage on `core/`.
- **Integration:** full pipeline on one known scene, asserting detection count within tolerance of AIS ground truth.
- **Soak:** 48 h unattended AIS ingestion — check for memory growth and reconnect correctness.
- **Visual QA:** contact sheets of labelled chips at every dataset build. Look at the data. Most auto-labelling bugs are obvious to the eye and invisible in metrics.
- **End-to-end demo script:** ingest → detect → fuse → surface a dark vessel in the UI, run from a clean checkout.

---

## 15. Build prompt

Paste-ready. Self-contained — assumes no memory of this conversation. On approval this is also written to `E:\sea-vessel-detection-3\PROMPT.md`.

````markdown
# PROJECT: khalij-watch — Persian Gulf Vessel Detection & Classification Platform

## ROLE
You are building an OSINT maritime domain awareness platform as an academic thesis
project. It ingests live AIS and free Sentinel-1 SAR satellite imagery over the Persian
Gulf, uses deep learning to detect and characterise vessels in the radar imagery, fuses
those detections against AIS to identify "dark" vessels that broadcast no position, and
presents the result in an operational web console.

Work through the phases in order. Each phase must produce something that RUNS before you
move on. Do not scaffold all phases at once.

---

## NON-NEGOTIABLE CONSTRAINTS

**Hardware (deployment + inference):**
- AMD Ryzen 7 3700U, 4 cores / 8 threads, 15 W mobile CPU. No CUDA. Integrated Vega only.
- 10 GB RAM total.
- Windows 11. Disk: C: is FULL (0.4 GB). Use `E:\` and `F:\` (~124 GB free).
- Project root: `E:\sea-vessel-detection-3`

**Hardware (training):** Kaggle free tier — 30 GPU-h/week, P100 16 GB or 2×T4, 12 h max
session, 20 GB working disk, internet enabled.

**Budget: $0.** Free and open data sources only. Never introduce a paid dependency.

**These follow directly and are not negotiable:**
1. NEVER read a full SAR scene into memory. A Sentinel-1 IW GRD is ~25,000 × 17,000 px
   (~1.7 GB/polarization as float32). Use windowed `rasterio` block reads everywhere.
   Add an assertion that refuses any array allocation over 512 MB.
2. All training happens on Kaggle. Local machine does data prep and ONNX inference only.
   Never write local training code that assumes a GPU.
3. Process-and-discard for raw imagery: download → preprocess → extract detections and
   chips → DELETE the raw scene. Persist only rows and small chips.
4. Before installing anything, redirect caches off C:
   `PIP_CACHE_DIR`, `TORCH_HOME`, `HF_HOME`, `TMP`, `TEMP` → `E:\...`.
5. Use Python 3.12, NOT the locally installed 3.14 — rasterio/GDAL/PyTorch lag on 3.14
   and Kaggle runs 3.11/3.12. Version mismatch causes ONNX opset pain.
6. Run GDAL inside Docker. Do not fight native geospatial installs on Windows.
   **Do NOT install ESA SNAP — it is not needed (see PHASE 2).**

---

## DOMAIN BRIEFING — get these right

These are researched facts, several measured live against the Copernicus catalogue on
2026-09-10. Do not "improve" on them.

**"Live" is layered, not continuous.** No civilian sensor watches the whole Gulf
continuously. The product is the FUSION of layers with explicit ages, never a fake
real-time feed. Measured Sentinel-1 revisit: **~2.0 days at the Strait of Hormuz** (15
acquisitions/30 days), ~3.0 days central and northern Gulf. Measured publish latency:
**0.75–4.6 h, median ~2.5 h.** Constellation is **Sentinel-1C + 1D** (1A retired
2026-06-29); the revisit pattern is shifted one day from the old S1A/S1B pair, so any
assumption inherited from pre-2026 code or literature is wrong.
Every scene is `IW_GRDH_1S`, **dual-pol VV+VH** (`1SDV`). **VH is the better ship-detection
channel** — lower ocean backscatter, higher target-to-clutter contrast.

**Free AIS over the Gulf is coastal-only, and much of what arrives is fabricated.**
The public AIS station registry lists **3 stations in the UAE and 1 in Qatar — zero in
Iran, Oman, Bahrain, Kuwait, Saudi Arabia or Iraq.** Shore receivers reach ~40–60 km; the
Gulf is 200–300 km wide. So the AIS layer is **port and anchorage awareness, not Gulf-wide
coverage** — present it that way in the UI. On top of that, 2026 saw severe GNSS jamming
and AIS spoofing around Hormuz, with reporting claiming **nearly two of every three logged
August 2026 transits were false**. AIS is a claim, not a measurement. Never render it as
ground truth.

**Therefore the architecture is inverted from the obvious one:**
> SAR is the truth layer. AIS is a label that may be lying.

**Fine-grained vessel type CANNOT be inferred from 10 m SAR. Do not attempt it.**
Global Fishing Watch (Paolo et al., Nature 2024) — the closest published precedent — got
detection >97%, length regression R²=0.84, but classified vessels only as binary
fishing/non-fishing at >90%. At 10 m/px a 200 m tanker is 20 pixels long. Any model
claiming tanker-vs-cargo from these pixels is overfitting noise.

**The division of labour, which is the core design idea:**
> Pixels tell you a vessel EXISTS, how BIG it is, and which WAY it points.
> AIS and registries tell you its NAME and TYPE.
> Fusion tells you when they DISAGREE — and that disagreement is the product.

**Coarse classes only** (grouped by AIS ship-type code, chosen for geometric separability):
`cargo_tanker` (70–89) · `fishing` (30) · `service_tug` (31, 32, 50–59) ·
`passenger_other` (60–69, 36, 37, rest). Never split tanker from cargo.

**Persian Gulf specifics:**
- Dense with fixed oil/gas platforms → major SAR false-alarm source. Mask by persistence.
- Many small wooden dhows → very low radar cross-section, often invisible to C-band.
  This is a stated limitation, not a bug to fix.
- Very high vessel density in anchorages (Bandar Abbas, Fujairah) → greedy nearest-neighbour
  matching WILL fail. Use global assignment.
- AOI (full platform): lon 47.5–58.0, lat 23.5–30.5
- AOI (DL training/eval): Strait of Hormuz — lon 54.0–57.5, lat 24.5–27.5

---

## ARCHITECTURE

```
INGESTION (local)          STORAGE                 INFERENCE (local CPU)
 ais-ingest   ──┐                                   ONNX Runtime INT8
 sar-ingest   ──┼──▶  PostGIS + TimescaleDB  ──▶    ① land/infra mask
 vbd-ingest   ──┤     ais_positions, vessels,       ② CA-CFAR prefilter
 ref-ingest   ──┘     sar_scenes, detections,       ③ point detector
                      fusion_matches, events           (stride-2 heatmap)
                              │                           │
      TRAINING (Kaggle GPU)   │                           ▼
      timm encoder +          │                      FUSION ENGINE
      U-Net decoder,  ──ONNX──┘                      anisotropic azimuth-
      stride-2 multi-head                            corrected gate
                              │                      Hungarian assignment
                              │                      → MATCHED | DARK
                              ▼                           │
                    FastAPI (REST + WebSocket) ◀──────────┘
                              │
                              ▼
                    React + MapLibre + deck.gl
```

**Stack:** Python 3.12 · FastAPI · PostgreSQL + PostGIS + TimescaleDB · ONNX Runtime ·
React + Vite + TypeScript · MapLibre GL JS + deck.gl · Tailwind + shadcn/ui · Docker Compose.

---

## REPO LAYOUT

```
E:\sea-vessel-detection-3\
├── docker-compose.yml
├── .env.example
├── README.md
├── services/
│   ├── ais_ingest/        # WebSocket AIS consumer
│   ├── sar_ingest/        # CDSE download + preprocess
│   ├── vbd_ingest/        # NOAA VIIRS nightly pull
│   ├── ref_ingest/        # registries + sanctions
│   └── api/               # FastAPI
├── core/                  # pure library code — the testable heart
│   ├── ais/               # decode, track assembly, interpolation
│   ├── sar/               # windowed IO, CFAR, masking, tiling
│   ├── fusion/            # azimuth correction, Hungarian, events
│   ├── labeling/          # auto-label pipeline, chip extraction
│   └── inference/         # ONNX runners
├── notebooks/kaggle/      # training notebooks (upload to Kaggle)
├── web/                   # React frontend
├── models/                # exported .onnx weights
├── data/                  # gitignored — chips, COGs, masks
└── tests/
```

Keep `core/` free of I/O side effects and framework imports so it is unit-testable without
a database or network.

---

## DATA CONTRACTS

```sql
-- TimescaleDB hypertable, chunk by 1 day
ais_positions(ts, mmsi, geom(Point,4326), sog, cog, heading, nav_status, source,
              plausibility_score, anomaly_flags)   -- Phase 1.5; source needed for attribution
vessels(mmsi PK, imo, name, callsign, ship_type, length, beam, flag, last_seen,
        sanctioned bool, sanction_source)
sar_scenes(id PK, product_id, sensor, mode, acq_time, footprint(Polygon,4326),
           orbit_dir, cog_path, status, processed_at)
detections(id PK, scene_id FK, geom(Point,4326),
           length_m, width_m, heading_deg, confidence,
           is_vessel_prob, is_fishing_prob, pred_class, class_probs jsonb,
           chip_path, detector_version, satellite)   -- satellite: CFAR/model params differ per unit
fusion_matches(detection_id FK, mmsi, score, dist_m, length_err_m, heading_err_deg,
               method, matched_at)
events(id PK, type, ts, geom, severity, detection_id, mmsi, evidence jsonb, acknowledged)
zones(id PK, name, geom(Polygon,4326), rule jsonb, owner)
```

Index: GiST on all geometry, BRIN on `ais_positions.ts`, btree on `mmsi`.
Continuous aggregate: hourly vessel position rollup for the timeline scrubber.

---

## BUILD ORDER

### PHASE 0 — Foundation
- Redirect caches off C:. Create Python 3.12 venv on E:.
- `docker-compose.yml`: `postgis/postgis` + TimescaleDB extension, volume on E:.
- Alembic migrations for the schema above. Repo skeleton, `.env.example`, pytest, ruff, pre-commit.
- **CRITICAL FIRST TASK — three access checks. Do these before writing any other code.**
  Any of them failing changes the plan, so find out now, not in week three:
  1. A CDSE account can be created and an `IW_GRDH_1S-COG` product byte-range read.
  2. An aisstream.io API key connects and receives messages.
  3. **An xView3-SAR account can be registered and data downloaded at `iuu.xview.us`.**
     ⚠️ This is **US export-control gated against the State Sponsors of Terrorism list**.
     Given this project's AOI and deployment location it may be unavailable. If it fails,
     do NOT treat it as a blocker — switch to the fallback training set (LS-SSDD-v1.0,
     Apache-2.0, full Sentinel-1 scenes + Official-SSDD, Apache-2.0 + GFW pseudo-labels)
     and report the switch.
- ✅ **Done when:** `docker compose up` yields a healthy DB, `pytest` runs green, and all
  three access checks have a recorded pass/fail result.

### PHASE 1 — AIS pipeline + live map  ← first visible slice
- `ais_ingest`: connect `wss://stream.aisstream.io/v0/stream`, subscribe with the Gulf
  bbox and only needed message types (1,2,3,5,18,19,24). Exponential-backoff reconnect.
  Fallback to `wss://ais.openwaters.io/v0/stream` (aiscast — same protocol shape).
  **aisstream emits pre-decoded JSON — no decoder needed for it.** Keep `pyais` for
  raw-NMEA sources (aiscast UDP, bulk archives).
  Carry a `source` field on every record — aiscast requires per-source attribution
  to be propagated through to the UI.
- **Hard API limits to code against:** 3 subscribed connections/account, 3 per IP,
  **subscription must be sent within 3 s of connect**, ≤1 subscription update/second,
  ≤200 MMSIs per subscription, bounding box required, **the server drops messages if you
  consume too slowly** (decouple socket read from DB write with a bounded queue), and
  browser connections are prohibited — the frontend goes through your API, never direct.
- **Develop against the free Danish (DMA) and NOAA MarineCadastre bulk archives first.**
  They are free, large and realistic, and let you build track assembly and matching
  without depending on a live feed.
- Batch-insert positions; upsert static data into `vessels`.
- FastAPI: `GET /vessels?bbox=`, `GET /vessels/{mmsi}/track?from=&to=`,
  `WS /stream?bbox=` pushing viewport-filtered updates.
- React + MapLibre + deck.gl: `IconLayer` rotated to heading, coloured by class;
  `TripsLayer` for trails.
- Note: at Gulf density (~3k–8k active MMSI/day) plain GeoJSON over WebSocket is fine.
  Do NOT build vector tiles yet — premature below ~50k objects.
- ✅ **Done when:** live vessels move on the map and the ingester survives 24 h unattended.

### PHASE 1.5 — AIS plausibility filter  ← mandatory, not optional
Given that a reported ~2/3 of August 2026 Hormuz transits were false, this is a
first-class ingest stage. **Flag and score — never silently drop.**
- Position on land (GSHHG mask) · impossible speed jumps · impossible turn rates ·
  teleports between consecutive fixes
- Same MMSI broadcasting from two places · MMSI MID prefix inconsistent with claimed flag
- Suspiciously collinear "straight-line" tracks (classic synthetic-track signature)
- Malformed records: MMSI 0, latitude 91.0, NUL bytes in names. **Validate everything —
  the feed carries a lot of junk.**
- Every position gets `plausibility_score` + `anomaly_flags`; the UI can show/hide these.
- ✅ **Done when:** known-bad tracks are flagged and the flag is visible in the UI.

### PHASE 2 — SAR ingestion  ← much smaller than it looks
**CDSE publishes a Cloud-Optimized GeoTIFF product directly: `IW_GRDH_1S-COG`
(~830 MB vs ~1.7 GB for the SAFE archive). This removes the entire local preprocessing
chain. Do NOT install SNAP. Do NOT build an orthorectification pipeline.**
- CDSE OAuth (10-min access token, 60-min refresh) + OData search by footprint / date /
  `productType`.
- **S3 byte-range reads via `rasterio` / GDAL `/vsis3/` against the COG**, windowed to the
  AOI water polygon. Respect the 4-concurrent-connection quota, 20 MB/s per connection.
- Land mask from GSHHG buffered 500 m seaward.
- Use GRD, **not SLC** (4× larger, complex-valued, only needed for interferometry/Doppler).
- ⚠️ **Do NOT apply a speckle filter.** Lee / Refined Lee / Frost will smear or erase
  2–5 px targets — exactly the signal you need. Feed near-raw values and let the CNN
  handle speckle. (This is the opposite of standard SAR ARD advice, and it is correct here.)
- Persist scene metadata; never persist raw pixels beyond extracted chips.
- ✅ **Done when:** an AOI window renders as a tile layer under the AIS vessels, with peak
  RAM under 2 GB and no full-scene download.

### PHASE 3 — CFAR baseline + masks
- Two-parameter CA-CFAR using summed-area tables for O(1) window sums. Guard band +
  background band. Tune Pfa ~1e-6.
- **Infrastructure mask: download the published Persian Gulf offshore platform dataset**
  (1,731 platforms as of 2025, derived from the Sentinel-1 archive, **CC BY 4.0**,
  arXiv 2603.19801). Do not derive this from scratch. Extend it with your own persistence
  rule (same pixel detected across ≥5 scenes ⇒ fixed structure).
- Apply land + infrastructure masks BEFORE thresholding.
- Morphological opening → connected components → centroid + rough extent.
- Filter by plausible ship geometry: 8 m < length < 400 m, aspect ratio 1.5–12.
- ⚠️ CFAR parameters **do not transfer between satellites** — Global Fishing Watch's
  production detector needed full recalibration for S1C/S1D and was offline
  3 Jul → 8 Sep 2026 as a result. Keep `satellite` as a first-class field and tune per unit.
- CFAR serves three roles: a zero-training working detector, an interpretable
  tunable-Pfa **baseline** for the thesis, and later a **compute prefilter** (a scene is
  >99.9% empty ocean; CFAR cuts CNN tiles by 1–2 orders of magnitude on your CPU).
- ✅ **Done when:** CFAR detections render on the map with a measured false-alarm rate.

### PHASE 4 — AIS-supervised auto-labelling  ← the novel contribution
For each processed scene:
1. Get acquisition time `t₀` and footprint.
2. Query AIS tracks with a fix within ±10 min of `t₀` inside the footprint.
3. **Interpolate each track to exactly `t₀`.** A vessel at 15 kn moves 460 m in 60 s = 46
   pixels. Nearest-fix matching mislabels constantly. Interpolate position along COG/SOG.
4. **Correct SAR azimuth displacement. This is the highest-consequence detail in the
   entire project — get it wrong and the headline feature silently breaks.**
   A target with radial (line-of-sight) velocity `v_r` is displaced *along-track* by
   `Δaz ≈ (R / V) · v_r`, where `R` = slant range, `V` = effective sensor velocity.
   For Sentinel-1 IW, `R/V` is **on the order of 100 s** (slant range ~800–880 km,
   effective velocity ~7.5 km/s), i.e.:

   > **≈ 100 m of displacement per 1 m/s of radial velocity.**

   A vessel at 12 kn (6.2 m/s) with a significant radial component is displaced
   **500–700 m** — **2–3× the 200 m tolerance xView3 uses**, and several times the
   ship's own length.

   ⚠️ If you gate matching at a fixed 200–300 m radius, you will systematically fail to
   match FAST-MOVING vessels and then report them as dark. The product's headline output
   becomes a false-positive generator, biased toward exactly the vessels that matter most.

   Implement all three mitigations:
   - **Anisotropic gate** — wide along-track, narrow across-track, sized from AIS SOG/COG
     projected onto the range direction.
   - **Correct the AIS position** by predicted `Δaz` before matching.
   - **Compute `R/V` per sub-swath from scene metadata. NEVER hardcode it** — it varies
     with sub-swath and incidence angle.
5. Project corrected lat/lon → pixel coords. Extract a 128×128 chip (VV + VH).
6. Label from AIS truth: `length`, `beam`, `heading` (COG), coarse class, `mmsi`, `imo`.
7. **Quality gates — discard the chip if:** another AIS target within 300 m (ambiguous);
   within 500 m of coast; inside an infrastructure buffer; or CFAR response at the
   predicted pixel is below threshold. That last case (AIS claims a ship, radar sees empty
   water) is a likely POSITION SPOOF — persist it as a labelled event, do not just drop it.
8. **Negative mining is mandatory.** Without it the model calls every bright pixel a ship.
   Sample from: CFAR hits with no AIS and no persistence match; offshore platforms;
   wind-roughened sea; azimuth ambiguity ghosts; land edges. Target ~1:3 positive:negative.
9. Package with a manifest. **Split by SCENE, never randomly** — adjacent chips from one
   scene leak between splits and inflate results.
10. Emit a visual QA contact sheet every build. Look at it. Most labelling bugs are
    obvious to the eye and invisible in metrics.
- ✅ **Done when:** ≥20k labelled chips exist and the contact sheet shows centred ships.

### PHASE 5 — Model training (Kaggle notebooks)
Upload the dataset as a private Kaggle Dataset (~6 GB fits the 20 GB working disk).

**⚠️ Use dense point detection, NOT bounding boxes.** At 10 m pixel spacing a 300 m VLCC
is 30×5 px and a 25 m dhow is 2–3 px — there is no meaningful orientation to regress.
The xView3-SAR winner used **no boxes at all**, and so did every other top-5 solution.
Port his architecture; the repo is **MIT-licensed**
(`BloodAxe/xView3-The-First-Place-Solution`).

**Architecture:**
```
2048×2048 tile (VV+VH) → timm encoder (EfficientNet-B4/V2S or ConvNeXt)
                       → U-Net decoder → STRIDE-2 dense output
   heads: objectness · sub-pixel offset · length · width · heading
        · is_vessel · is_fishing · coarse_class
```
Stride matters more than box geometry — the winner's localization-F1 ceiling is 0.9672 at
stride 16 but **0.9999 at stride 2**.

This choice also: avoids AGPL entirely, makes **ONNX export trivial** (conv stack + 3×3
max-pool NMS, no custom ops — rotated NMS would need MMDeploy, which is frozen), and
eliminates the tile-seam box-merge problem.

**Training recipe — port these verbatim, they are non-obvious and hard-won:**
- **Normalization:** uint16 → [0,1] via a **tuned sigmoid** (NOT linear), so sea/land
  saturate to 0/1 and ship vicinities land in ~[0.2, 0.8]. Missing data → 0. Then FREEZE
  it — normalization is a silent train/serve skew source.
- **Two channels only, VV+VH.** Bathymetry, wind speed/direction and ice mask all FAILED
  to help; the model infers wind implicitly. Do not build a data-fusion workstream for them.
- **Objectness:** reduced focal loss, **fixed 3 px heatmap radius** (not size-dependent —
  this solves unknown-length targets).
- **Classification heads:** weighted BCE, **label smoothing 0.05**, 2 px label radius
  (~40× more training samples than 1 px).
- **Length / offset:** MSE, scale 0.1 and 0.001.
- ⚠️ **Focal loss is unusable under label noise** — the winner says so explicitly. Use
  balanced sampling + BCE. This matters directly for AIS-derived weak labels.
- **Shannon entropy regularization** on classification heads for UNLABELLED objects,
  pushing predictions toward 0/1. Essential here — most detections have no AIS truth.
- **Augmentation, constrained for SAR:** brightness/contrast, Gaussian noise, H/V flips,
  ±5° rotation, ±10% scale, minor elastic. Use **unclipped** Albumentations variants —
  the standard ones clip and corrupt the SAR distribution.
- **Inference tiling:** 2048×2048, **stride 1536** (25% overlap), ~204 tiles/scene.
  Accumulate into a scene-level heatmap canvas and **NMS once at scene level** — never
  per-tile. Left-right flip TTA gives a measurable gain.

- Pretrain on xView3 (or the LS-SSDD + Official-SSDD fallback), fine-tune on Gulf auto-labels.
- **Skip the 12-model ensemble.** It bought 0.51→0.617 in a competition; it does not fit a
  thesis narrative or a CPU deployment. One model, one fold.
- Checkpoint to `/kaggle/working` EVERY epoch — the 12 h cap will otherwise lose the run.
- Export ONNX (opset 17), then INT8 dynamic quantization. Benchmark CPU latency locally.
- Budget ~15–20 GPU-h total, inside the 30 h/week quota.
- ✅ **Done when:** the ONNX model runs on the laptop CPU and beats the CFAR baseline on
  held-out data.

### PHASE 6 — Fusion + event detection
- **v1:** extrapolate AIS to acquisition time, apply the **anisotropic azimuth-corrected
  gate** from Phase 4, score on position / length / heading residuals, solve globally with
  the **Hungarian algorithm** (`scipy.optimize.linear_sum_assignment`). Greedy
  nearest-neighbour fails in dense anchorages — do not use it.
- **v2 (if time allows):** GFW's probabilistic method — build a probability field for where
  each AIS vessel could be at scene time given messages before AND after, then calibrate the
  acceptance threshold globally so total matches equal the expected number of AIS-equipped
  vessels in-scene. Reference: `GlobalFishingWatch/paper-longline-ais-sar-matching`.
- **Corroborate independently** with GFW's `pipe-gaps` (Apache-2.0) for the AIS-side "went
  dark" signal. A vessel flagged by both an unmatched SAR detection and an independent AIS
  gap is a far stronger call than either alone.
- Events: `DARK_VESSEL` (detection, no AIS match, not infrastructure) · `AIS_GAP` ·
  `POSITION_SPOOF` · `STS_TRANSFER` (two vessels <500 m, near-zero SOG, >2 h, offshore) ·
  `ZONE_ENTRY` · `IDENTITY_ANOMALY` (reported vs measured length mismatch).
- Every event stores its evidence: SAR chip, AIS track segment, match scores.
  **Never surface an alert that cannot be explained.**
- ✅ **Done when:** dark vessels are flagged with auditable evidence.
- **Superseded 2026-10-04:** v1 is a cell-hour fusion on GFW hourly presence (no
  position-level AIS is available); design, limits and the revised done-when in
  §11 Phase 6 and §17 "Phase 6 feasibility".

### PHASE 7 — UI/UX build-out
Dark-themed operational console. Three panes — left filter rail, centre map, right context
panel — plus a bottom timeline scrubber.

**Non-negotiable UX rules:**
1. **Every object shows its age** — opacity ramp plus explicit "last seen 4h 12m ago".
   Stale contacts must never look live.
2. **AIS and SAR are visually distinct.** Filled icon = someone told us. Hollow ring =
   radar saw something. This distinction is the whole point of the product.
3. **Dark vessels get visual priority** — amber/red ring, always on top, never clustered away.
4. **Timeline scrubber** — drag through time; show SAR footprints as bands marking when
   each area was actually observed.
5. **Coverage honesty overlay** — toggleable "last SAR look" heatmap. Where it is dark, the
   system is blind. Saying so builds trust.
6. **Vessel detail panel** — identity, AIS sparkline, sanctions badge, and the matched SAR
   chip side-by-side with the model's prediction and confidence.
7. Build RTL/Persian support in from the start if the audience is regional — retrofitting is painful.

### PHASE 8 — Evaluation + write-up
See EVALUATION below. Generate all figures programmatically into `reports/`.

---

## EVALUATION PROTOCOL

- **Split by scene, never randomly.** State this explicitly — it is the most common
  methodological error in SAR ship detection literature. ⚠️ Note xView3 itself violates
  this: its ~42 locations leak across official splits, so its leaderboard numbers are
  optimistic. Say so when comparing.
- **Primary metric — use the xView3 aggregate**, so results sit against a published
  leaderboard:
  `aggregate = loc_fscore × (1 + length_acc + vessel_fscore + fishing_fscore + loc_fscore_shore) / 5`
  Detection F1 MULTIPLIES everything. Reference points: 1st place 0.617 holdout;
  government baseline ≈ 1/3 of that.
- **Detection:** P/R/F1 at operating point, false alarms per 100 km², on the Gulf held-out
  set plus whichever public benchmark is accessible.
  ⚠️ **Do NOT use SSDD/HRSID leaderboard position to justify architecture choices** — both
  are saturated (AP50 routinely >95–98%), single-class, and everything overfits them.
  Smoke tests only.
- **Stratify every metric by vessel length** (<25 m, 25–100 m, >100 m). Small-vessel recall
  is where these systems fail; that is where the honest discussion lives.
- **Length regression:** R² and MAE in metres. Target R² ≈ 0.84 (GFW parity).
- **Classification:** per-class P/R + confusion matrix. Expect `cargo_tanker` vs `fishing`
  to separate and `service_tug` vs `passenger_other` to be weak. Report it, don't hide it.
- **Efficiency:** seconds/scene on the Ryzen CPU, FP32 vs INT8, peak RAM. Headline result,
  not an appendix.
- **Ablations:** output stride (2/4/8/16 — reproduces the winner's localization ceiling) ·
  **with/without azimuth correction (expect a large effect — the best figure in the
  thesis)** · sigmoid vs linear normalization · with/without entropy regularization ·
  with/without negative mining · VV-only vs VV+VH · CFAR window sizes.
- **Baselines:** CFAR-only · point detector · CFAR-prefilter + point detector ·
  **GFW's public Sentinel-1 detections over the same scenes** (an independent published
  system — a strong external reference).
- **Address label bias explicitly — it is the central methodological threat.** Training
  labels come from AIS, so the set contains ONLY AIS-broadcasting vessels, while the target
  class (dark vessels) is absent by construction and skews smaller and older. With a
  reported 50–80% of Gulf tanker transits running dark, this bias is large, not marginal.
  Mitigate by (a) validating on xView3 (human-verified, includes non-AIS vessels),
  (b) hand-labelling ~200 held-out Gulf chips as an unbiased test set, (c) cross-checking
  GFW's unmatched detections as an independent second opinion, (d) reporting
  length-stratified performance throughout.

---

## CODING STANDARDS
- Immutable by default: return new objects, do not mutate arguments.
- Functions < 50 lines, files < 800 lines. Many small modules over few large ones.
- Early returns over nesting beyond 4 levels.
- Explicit error handling; never swallow exceptions silently.
- Validate all external input (AIS fields are frequently malformed — MMSIs of 0,
  latitudes of 91.0, NUL bytes in names). Never trust the feed.
- No hardcoded secrets. `.env` + `.env.example`.
- Named constants, not magic numbers — especially CFAR thresholds and match radii.
- Type hints throughout; `ruff` + `mypy` clean.
- Tests first for `core/` logic. Target 80% coverage there. AAA structure, descriptive names.
- Conventional commits (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`, `chore:`).

## TESTING
- Unit: CFAR on synthetic scenes with known injected targets; azimuth-shift maths against
  hand-worked examples; Hungarian matcher on constructed ambiguous cases; AIS decode against
  `pyais` fixtures.
- Integration: full pipeline on one known scene, asserting detection count within tolerance
  of AIS ground truth.
- Soak: 48 h unattended AIS ingestion; check for memory growth and reconnect correctness.
- Guard the RAM ceiling with an explicit assertion on array allocation size.

---

## WHAT NOT TO DO
- Do NOT claim or imply continuous real-time satellite coverage. It does not exist.
- Do NOT attempt fine-grained vessel type classification from SAR pixels.
- Do NOT treat AIS as ground truth — in this AOI much of it is fabricated.
- Do NOT use an oriented-bounding-box detector for the Sentinel-1 tier. Use dense point
  detection. (OBB is only justified later, on 0.16–1 m imagery.)
- Do NOT use Ultralytics YOLO — unnecessary here, and AGPL-3.0.
- Do NOT use LSKNet. GitHub shows `NOASSERTION` but the LICENSE file is **CC BY-NC 4.0**.
- Do NOT apply a speckle filter — it erases the 2–5 px targets you need.
- Do NOT install ESA SNAP. CDSE's `-COG` product removes the need.
- Do NOT read whole SAR scenes into memory.
- Do NOT write training code that assumes a local GPU.
- Do NOT use greedy nearest-neighbour for AIS↔SAR matching.
- Do NOT skip the azimuth-shift correction, and do NOT hardcode `R/V`.
- Do NOT use focal loss on the noisy AIS-derived labels — balanced sampling + BCE.
- Do NOT NMS per-tile — accumulate a scene-level heatmap and NMS once.
- Do NOT split train/val/test randomly across chips from the same scene.
- Do NOT regress heading as a raw angle — use (sin, cos).
- Do NOT scrape MarineTraffic; its ToS forbids it and there is no need.
- Do NOT introduce paid dependencies.
- Do NOT build vector tiles, Kubernetes, microservice meshes, or a message broker for
  ~5k vessels. Match the architecture to the actual data volume.
- Do NOT surface an alert whose evidence cannot be shown to the analyst.

## SKILLS AVAILABLE — invoke these rather than reinventing
Six skills are installed for this project. Use them at the phases indicated:
- **`gdal`** — PHASE 2. COG creation, windowed reads, reprojection, tiling.
- **`maplibre-tile-sources`** (official MapLibre) — PHASE 2 & 7. Serving SAR COG tiles and
  vessel data; covers rendering data without tiles vs. hosted/self-hosted tile services.
- **`maplibre-cartography`** (official MapLibre) — PHASE 7. Label and symbol legibility
  **on imagery**, figure-ground for point icons, layer ordering, accessibility. This
  directly addresses the hardest UI problem here: keeping vessel markers readable on top
  of SAR backscatter.
- **`detect-objects`** — PHASE 3/5. Pretrained geospatial detectors, explicitly including
  **ships**. Check what it offers before training from scratch.
- **`computer-vision-opencv`** — PHASE 3. Morphology, connected components, `minAreaRect`.
- **`pytorch-lightning`** — PHASE 5. Training loop and checkpointing — important given the
  12 h Kaggle session cap.

Also available in-session: `postgis`, `fastapi`, `fastapi-templates`, `dataviz`,
`docker-patterns`, `postgres-patterns`, `database-migrations`, `tdd-workflow`,
`python-testing`, `frontend-design`, `web-design-guidelines`, `accessibility`, and
`localization-design` (for RTL/Persian).

## KEY REFERENCES — read before writing the detector
- `BloodAxe/xView3-The-First-Place-Solution` — **MIT**. The architecture and training
  recipe to port. Prebuilt container: `ekhvedchenyateam/xview3:72`.
- `allenai/vessel-detection-sentinels` — **Apache-2.0**. Closest existing OSS to this
  product; production system behind Skylight. Outputs length/width/heading/speed/fishing —
  and deliberately no vessel type.
- `allenai/rslearn` — **Apache-2.0**, actively maintained. AI2's current framework; prefer
  it over their stale 2024 repo if building fresh.
- `GlobalFishingWatch/paper-longline-ais-sar-matching` — principled probabilistic AIS↔SAR
  matching with global threshold calibration.
- `GlobalFishingWatch/pipe-gaps` — **Apache-2.0**. AIS gap detection primitive.
- `obss/sahi` — **MIT**. Sliced inference, if you ever go box-based.
- Persian Gulf offshore platform dataset — arXiv 2603.19801, **CC BY 4.0**, 1,731 platforms.

## ATTRIBUTION & LICENSING
- **Copernicus/Sentinel data:** free, worldwide, royalty-free, perpetual, **no commercial
  restriction**. Attribute as `"Contains modified Copernicus Sentinel data [Year]"`.
  This is the only fully permissive source in the stack — keep the core working on
  Copernicus alone.
- **aiscast:** re-serves per-source licences — propagate the `source` field to the UI.
- **aisstream.io:** ⚠️ **no published ToS or licence at all.** Fine for a thesis; get
  written clarification before any commercial use.
- **GFW:** CC BY-NC 4.0. Attribute "Powered by Global Fishing Watch". Non-commercial only.
- **xView3:** reported CC BY-NC-SA 4.0 (verify) + US export-control gated access.
- **HRSID:** GPL-3.0. **LS-SSDD-v1.0 and Official-SSDD: Apache-2.0** — prefer these.
- **OpenSARShip / FUSAR-Ship:** no SPDX licence published; research use only.
- Keep GFW / OpenSanctions / Skylight as **separable enrichment modules behind an
  interface**, so a future commercial version can drop them without a rewrite.

## START HERE
Begin with PHASE 0, and specifically with the three access checks: CDSE, aisstream.io, and
**xView3 registration** (export-control gated — may fail from this location). Do these
before building anything on top of them. Report back after Phase 0 with the pass/fail
result of each, and say which training-data path you are taking as a result.
````

---

## 16. Research provenance

Three parallel research agents were run. Two completed and their findings are folded in
above; the third (frontend/geospatial stack) was killed by a rate limit, so **Section 10
(UI/UX) is written from general knowledge rather than fresh research.** It is the section
to re-verify if you want current library specifics — the architectural direction
(MapLibre + deck.gl, GeoJSON-over-WebSocket at this data volume) is sound regardless.

Several figures here were **measured live** against the Copernicus catalogue on 2026-09-10
rather than quoted: Sentinel-1 revisit over three Gulf points, publish latency, product
type and polarization. Re-measure in a few months — the S1C/S1D configuration only
stabilised in July 2026.

**Flagged as unverified**, and worth confirming before relying on:
- xView3's exact licence (reported CC BY-NC-SA 4.0) and whether its export-control gating
  blocks your registration.
- aisstream.io's redistribution terms (genuinely undefined, not merely hard to find).
- Persian Gulf cloud cover: NASA describes it as ~90% dust-obscured by July, while
  Sentinel-2 `cloudCover` metadata for Jun–Jul 2026 reports a 0.8% median. Both are
  probably "true" — Sen2Cor does not reliably flag airborne dust as cloud. **Do not trust
  `cloudCover` as a usability filter over the Gulf**; build your own scene-quality gate.
- The "50–80% of Hormuz transits run dark" figure is industry/press reporting, not
  peer-reviewed. Directionally credible, numerically unverified — cite it as such.
- GFW's exact SAR 4Wings dataset identifier; query `/v3/datasets` once you have a token.
- Whether Skylight (Ai2) will grant API access to a non-governmental academic project.
  Free, high-value, and unknown — worth an email early.


---

## 17. Decisions log (updated during implementation)

Deviations from the plan as written, with the reason. Anything here supersedes
the corresponding text above.

### 2026-09-10 — Phase 0

**Docker is not installed; database deferred to SQLite.**
The plan assumed `docker compose up` with PostGIS + TimescaleDB. Docker is not
present, C: had only 4.1 GB free, and the host has 10 GB RAM, so Docker Desktop
was a poor fit. The user chose to defer and use SQLite for now.

Accepted cost: spatial queries and time-series work will need revisiting on the
move to PostGIS. Mitigations applied to keep that contained:
- Geometry stored as plain `lon`/`lat` REAL columns, not a vendor geometry type.
- SQLite **R\*Tree** companion tables give a genuine spatial index, kept in sync
  by triggers so the index cannot drift. These become GiST indexes on PostGIS.
- All dialect-specific code confined to `core/db/session.py` and
  `core/db/spatial.py`; callers use `bbox_filter_sql()`.
- WAL journal, `synchronous=NORMAL`, `foreign_keys=ON`, 64 MB page cache — the
  cache deliberately modest given 10 GB total RAM.

Also worth noting: **TimescaleDB was over-specified.** At Gulf volumes (~3–8k
active vessels) plain partitioning plus BRIN/btree indexes is sufficient, so
nothing of substance is lost by not having it.

**Gulf AOI eastern edge corrected: 57.0 → 58.0 °E.**
Caught by a test asserting the Hormuz training AOI sits inside the platform AOI
— it did not. The Strait of Hormuz is near 56–57 °E and transits must be
observable on *both* sides, so the AOI has to reach into the Gulf of Oman
approaches. 57.0 was simply wrong.

**No SNAP, no preprocessing chain** (confirmed, per the CDSE `-COG` finding).

**Sentinel-1 coverage independently verified** against the live catalogue on
2026-09-10, matching the research: Hormuz 28 acquisitions/30 d (**2.0 d mean
revisit**), Central Gulf 20 (2.8 d), N Gulf 20 (3.2 d); constellation S1C+S1D
only; every acquisition `IW_GRDH_1S` dual-pol VV+VH.

### 2026-09-10 — Phase 1.5

**Plausibility filter calibrated against synthetic spoofing.** Four falsification
profiles modelled on Hormuz reporting (position on land, teleport, impossibly
straight track, speed contradicting track) are all detected, at a **0.04%
false-positive rate** on legitimate traffic (4 of 10,512 positions, all at
course-reversal points).

Three calibration lessons, each found by measurement rather than reasoning:

1. **Straightness alone is not evidence.** A vessel on autopilot in a traffic
   separation scheme genuinely travels a straight line for hours. With a 30 m
   tolerance the check flagged 6,398 legitimate positions. The fix was to make
   the test *"is this track mathematically impossible?"* rather than *"is this
   track straight?"* - tolerance 5 m, below the real GPS noise floor. Because
   the check became strong evidence, its penalty rose from 0.3 to 0.55.
2. **Reported speed is instantaneous; implied speed is an average.** Comparing
   them is only fair over a middle band of time gaps (60-900 s). Too short and
   position noise dominates; too long and a legitimate speed change looks like
   a lie.
3. **The filter caught two bugs in the seeder**, which is the best evidence it
   works: the simulator advanced vessels by step index rather than elapsed
   time, and wrapped them from one end of the shipping lane to the other.

**`JSON(none_as_null=True)`** - without it SQLAlchemy stores Python `None` as
the JSON literal `null`, which still satisfies `IS NOT NULL` and silently
breaks every anomaly query.

**Policy confirmed in code:** implausible positions are flagged and kept, never
dropped. A fabricated track identifies a vessel actively concealing movement,
so discarding it would destroy the signal the product exists to find.

### 2026-09-10 — Phase 2

**No Docker needed after all.** rasterio 1.5.1 / GDAL 3.12.4 install from
Windows wheels, which retires the plan's "SAR preprocessing is a Windows
install nightmare" risk entirely. Combined with the CDSE `-COG` product, the
whole SNAP/orthorectification chain is gone.

**Confirmed against live CDSE:** the `-COG` variant is real and is 1,131 MB
against 1,729 MB for the SAFE archive. `S3Path` gives the byte-range target,
`GeoFootprint` is GeoJSON, and `Online` distinguishes readable products from
ones parked in long-term archive.

**Two real bugs found by measurement:**

1. **The catalogue search was silently truncating.** CDSE caps a page and hands
   back an `@odata.nextLink`, which was ignored: 234 products existed, 200 came
   back, no error. For a discovery worker that means missing scenes and a
   coverage picture that is quietly wrong. Now paginated.
2. **The R\*Tree triggers broke on upsert.** SQLite fires the AFTER INSERT
   trigger even when a statement takes the `DO UPDATE` path, and an R\*Tree is a
   virtual table that does NOT honour `INSERT OR REPLACE` - it raises UNIQUE
   instead of overwriting. Phase 1 never hit this because AIS positions upsert
   with `DO NOTHING`, which fires no trigger. Triggers now delete-then-insert,
   and the DDL drops stale triggers first so existing databases self-heal.

**Sigmoid normalisation recalibrated.** The first parameters put open ocean at
0.39, spending contrast on empty water - the exact thing the sigmoid exists to
avoid. Re-solved so sea (~80 DN) maps to 0.10 and a ship return (~1200 DN) to
0.90. These values are now FROZEN: normalisation is a silent train/serve skew
source.

**Naming corrected:** a whole-AOI search reports the gap between passes
*anywhere* over the area (~0.5 d), which is not the revisit interval of a
single point (~2-3 d). The report says so explicitly.

**Status: partially complete.** Everything that does not need credentials is
done and verified against live data - 117 real Sentinel-1 scenes discovered and
rendered as a coverage overlay. Reading actual pixels needs CDSE credentials;
the read path is written and tested against a real COG on disk, but has not
been exercised against `/vsis3/`.

### 2026-09-11 — Phase 3

**CFAR detector built and calibrated on synthetic SAR.** Requested Pfa 1e-3
measures 9.95e-4; 1e-5 measures 1.04e-5. Throughput 14.9 Mpx/s, so a full
25000x17000 IW scene takes ~28 s single-threaded on this laptop.

Two modelling errors, both found by measurement rather than reasoning:

1. **The textbook CFAR threshold assumes single-look clutter.**
   `alpha = -ln(Pfa)` is derived for exponential (L=1) intensity. Sentinel-1
   IW GRDH is ~4.4 looks, whose Gamma-distributed intensity is far less
   heavy-tailed, so that formula sets the threshold **2.7x too high** (13.82
   against 5.05 at Pfa=1e-6). The detector would have run far more
   conservatively than its dial claimed, missing vessels while appearing
   correctly configured. Now solved from the Gamma model, verified to reduce
   to `-ln(Pfa)` exactly at L=1.
2. **The guard band must exceed a vessel's LENGTH, not its beam.** With a
   130 m guard, a 300 m tanker put its own pixels in its own training ring:
   19 of 341 target pixels detected, length estimated at 79 m. With a 490 m
   guard: 337 pixels, 310 m. This fails as fragments rather than a miss,
   which is why it is easy to overlook.

**Morphological opening does most of the false-alarm work**, not the geometry
filter as assumed. At a loose Pfa, ~7,900 scattered clutter hits reduce to
zero after erosion, because speckle spikes are isolated. Geometry is the third
line of defence, and Pfa has to be pushed to 2e-1 before it has anything left
to reject.

**Offshore infrastructure mask fetched and integrated.** OPD v1.0.0
(doi:10.5281/zenodo.19046824, **CC BY 4.0**) gives 2,314 Persian Gulf
platforms with per-structure lifespans. Measured against real scene
footprints already in the database: **up to 870 platforms fall inside a single
Sentinel-1 scene** over the Saudi/Qatari shelf, median 30 across 60 recent
scenes. Unmasked, every one would be reported as a stationary vessel on every
pass.

The mask is **time-aware**, which is a correctness property rather than a
nicety: a platform first seen in 2023Q4 must not mask a 2019 scene, or it
hides a real vessel. Each entry is applied only to imagery inside its lifespan.

**Credential finding:** a Sentinel Hub OAuth client (id starting `sh-`)
authenticates and queries the catalogue, but the download service refuses it
with `DAT-ZIP-609: Token audience not allowed`. Reading pixels needs **S3
credentials**, generated separately in the CDSE dashboard - and that is also
the only path supporting byte-range windowed reads.

### 2026-09-11 - Phase 3 on real imagery

**S3 keys verified end to end.** Listing, a ranged GET returning 206, and a
windowed read of a live Sentinel-1 scene all succeed. `scripts/check_access.py`
now reads real pixels through the production reader. It previously reported
PASS merely because the keys were set.

Real data exposed five problems that synthetic tests could not:

1. **The credential helper crashed on first real use.** rasterio refuses AWS
   keys passed as GDAL options (`EnvError`) and accepts them only through an
   `AWSSession`, which needs boto3. The unit tests checked the dictionary the
   helper returned and never handed it to rasterio, so they passed while the
   read path was broken. Credentials now travel through the session, and a
   test builds the real rasterio environment.
2. **Sentinel-1 GRD scenes are not map-projected.** The COG variant has no
   CRS, an identity transform and 210 ground control points. Windowing,
   georeferencing and platform masking all assumed a geotransform, which would
   have placed every pixel at longitude = column. They now go through
   `core/sar/georef.py`. This corrects the plan: the COG removes the SNAP
   install, but not the need to handle radar geometry.
3. **GDAL default control-point polynomial is off by up to 1.2 km** on this
   grid. A thin-plate spline reproduces the control points exactly, with a
   leave-one-out error of 26 m mean and 55 m p90. AIS matching needs the spline.
4. **The image is mirrored** on descending passes: columns run east to west.
   Headings are computed geographically.
5. **Sidelobes inflate vessel extent.** On a VH window over the Fujairah
   anchorage (192 vessels) the CFAR mask gave a median 289 x 135 m, with 89% of
   widths above 80 m. Measuring only pixels within 12 dB of each target peak
   gave 208 x 62 m, with 16% above 80 m. That threshold is a heuristic until
   Phase 4 AIS lengths calibrate it. Heading alignment with the azimuth axis was
   about 39% under both measurements, so headings follow hulls rather than
   sidelobes; ships at anchor lying to the current explain the clustering.

**Operational findings.** After a failed request, a process that used GDAL
can fail to exit, even through `os._exit`, which points at native library
teardown. The pixel check therefore runs in a child process that is terminated
once it has reported. DNS on this host also fails intermittently mid-session,
so the reader retries transport errors but never credential errors, which it
names specifically: unknown access key versus wrong secret.

**First real detection run.** A 23 x 23 km VH window over the Fujairah
anchorage read in 2.2 s and ran CFAR in 1.2 s. The detections sit on the vessel
grid in the annotated preview; a few long bright streaks beside very bright
targets were not detected. With hull-based extents the 180 candidates
measure 212 x 63 m at the median (lengths 121-324 m and widths 47-86 m from
p10 to p90; 16% of widths above 80 m, 3% above 120 m). The widest few are
consistent with two hulls merged, such as a bunker barge alongside a tanker,
which is common at Fujairah, but that has not been verified. VH stays the default: on the same window VV gave 282
components against 192, with far brighter returns (peak 14271 DN against 2911),
which is consistent with more sidelobe fragmentation but has not been checked
by eye.

### 2026-09-11 - Phase 3 detections stored, served and drawn

**Detections persist per window, and a re-run replaces them.**
`DetectionRepository.replace_in_bbox` replaces one scene's detections for one
polarisation and detector inside the window's box. Re-running a window, after
a threshold change or by accident, never doubles the vessel count, and VV and
VH never overwrite each other. On the real Fujairah VH window: 167 stored;
after a re-run, still 167 rows and 167 index entries. Detections outside the
requested box are no longer reported: on a rotated scene the pixel window
reaches past the box, and those 13 would have survived a re-run of an
overlapping box as duplicates. Each row records `detector_version` (such as
`L4.4-g24-t40-p1e-06-h0.25`: looks, guard, training, Pfa, hull fraction), the
channel, pixel count and peak DN. CFAR stores confidence 1.0, meaning only
"passed the test"; it must never be ranked against a learned probability.

**Additive schema changes migrate themselves.** `create_all` never alters an
existing table, so `_add_missing_columns` adds new nullable columns to an
existing SQLite database and refuses anything else. No Alembic before the
PostGIS move.

**The R\*Tree is not an exact filter.** It stores 32-bit floats rounded
outward. Measured on 20,000 random Gulf points: none were missed by a box
equal to the point, but 19,503 were returned by a query box 1 cm away from
them, so it matches up to ~0.4 m outside a box. Harmless for map queries. For
the replacement delete it removed a neighbouring window's detection lying that
close to the shared edge, which a test reproduced. The delete now compares the
stored doubles as well. A code review had flagged the rounding but predicted
the opposite failure, missed rows and duplicates, which the measurement rules
out.

**The console draws radar detections as hollow rings**, distinct from filled
AIS symbols, fading with the age of the pass over days rather than hours.
`GET /api/detections` joins each detection to its pass and returns
`age_hours`. Clicking a ring states when it was seen and by which satellite
and channel, its size, both possible headings (radar cannot tell bow from
stern), the detector version, and that it has not been matched against AIS.
Fixed on the way: data that reached the map before the basemap style loaded
was never drawn, because readiness was held in a ref rather than state.

**Land-mask bug found in Phase 1.5 code.** `_OMAN_INTERIOR` reached 25.6 N
between 57.2 and 58.8 E, which is the open Gulf of Oman, and covered Muscat's
port while missing the actual interior. Every AIS position in the eastern
approach to Hormuz would have been flagged `ON_LAND`, and CFAR detections there
dropped as inland. The synthetic calibration never saw it because the seeded
shipping lane stays west of 56.8 E. It surfaced when the false-alarm script
refused two open-water windows. Corrected, with regression points.

**GDAL exit hang, second trigger.** Three processes that opened a scene over
S3, computed a window and closed it without reading, refused by that land
check, each hung at exit until a 300 s timeout killed them. No request had
failed; previously the hang had only followed failed requests. Root cause
still unknown. The false-alarm script now refuses a window from its
coordinates, before opening anything.

**Measured CFAR false-alarm rate on real open-sea clutter** (the remaining
Phase 3 criterion). Real imagery has no truth, so `core/sar/false_alarms.py`
measures what can be measured honestly: the fraction of sea pixels the CFAR
test fires on, away from every object it found (250 m exclusion), across a Pfa
sweep that shares one clutter area per window. On synthetic Gamma clutter it
recovers the requested rate within 15%. Scene S1D 2026-09-10 02:06 UTC, two
open Gulf of Oman windows of about 1,210 km2 each, no platforms in either:

| Pfa requested | VH 57.90 E 24.80 N | VH 57.35 E 25.15 N | VV 57.90 E 24.80 N |
|---|---|---|---|
| 1e-3 | 0.78x (9,462 hits) | 0.88x (10,618) | 1.57x (19,018) |
| 1e-4 | 0.80x (975) | 0.89x (1,070) | 2.60x (3,143) |
| 1e-5 | 1.06x (128) | 1.01x (122) | 5.43x (658) |
| 1e-6 | 2.97x (36) | 2.07x (25) | 18.2x (220) |
| 1e-7 | 14.0x (17) | 3.3x (4) | 95x (115) |
| candidates at 1e-6 | 1 (0.08 per 100 km2) | 2 (0.17) | 7 (0.58) |

1. **In VH the Pfa setting holds to 1e-5, then under-promises**: 2-3x the
   requested pixel rate at the 1e-6 operating point. The extreme tail of real
   sea clutter is heavier than Gamma(L=4.4). The slight shortfall at loose
   settings may mean an effective number of looks above 4.4; not measured.
2. **Pixel false alarms do not become objects.** VH candidate counts barely
   move across four decades of Pfa (4 to 1, and 2 to 2), so opening and the
   geometry filter absorb the excess. At the operating point that is 0.08 and
   0.17 candidates per 100 km2 of open water: upper bounds on object false
   alarms, since some or all may be real vessels, which only AIS can settle.
3. **VV is far worse on clutter**: 18x the requested rate at 1e-6 on the same
   pass and window, and its candidates fall steadily with Pfa (18 to 6)
   instead of levelling off, the signature of clutter-driven detections. This
   is evidence for VH as the detection channel, not proof: VV could also be
   seeing small targets that VH misses.

At 1e-7 the VH hit counts (4 and 17) are too small for tight ratios. This is
one scene and one sea state; Phase 8 repeats it across passes and wind.

**Phase 3 definition of done is met**: detections render on the map with a
measured false-alarm rate. Still open from Phase 3: a real coastline mask, so
coastal windows stop reporting shoreline.

**The "missed bright streaks" were the geometry filter, not CFAR.** A
diagnostic on the Fujairah window found only 10 bright pixels (at least 12x the
sea median) outside every CFAR component, so CFAR missed nothing bright. Of 192
components after opening, 180 passed the geometry filter, 11 failed as too
square and 1 as too long; this settles the earlier assumption that 192 versus
180 was the filter. Crops showed the brightest rejects were vessels: two
single ships whose hull, the pixels within 12 dB of a very bright peak, had
collapsed to a near-square core, and two ships lying together. The hull-based
extent had caused it, since their whole components have ship proportions
(aspect 1.37-1.74). A component whose hull fails the geometry test is now
judged and measured on its whole shape, keeping the hull's position. The window
yields 170 detections in the box instead of 167, and both open-water
false-alarm windows are unchanged at every Pfa, so the change admitted no
clutter there. The two ships lying together are reported as one detection.
Still rejected: one 472 m azimuth-smeared vessel over the 450 m length limit.

**Coastline mask from OpenStreetMap, not GSHHG** (download approved by the
user). GSHHG's full-resolution shoreline dates from 2017 and 1990s sources,
and this coast has since gained Palm Jumeirah, the Dubai and Doha reclamations
and new ports; its home server also refused connections on 2026-09-11.
OpenStreetMap land polygons (osmdata.openstreetmap.de, ODbL, rebuilt daily)
were downloaded once (925.7 MB). The 2,135 polygons touching 44.5-61.0 E,
22.0-32.5 N, the union of every catalogued scene footprint, were extracted to a
5.6 MB `.npz` (478,297 vertices) and the download deleted. Spot checks:
Fujairah city and Deira read as land; open sea off Dubai, the open Gulf of Oman
and the Fujairah anchorage as water; 2,391 coastline vertices fall inside Palm
Jumeirah's box. Land and a seaward buffer are rasterised through the scene's
control points and masked before CFAR, so land brightness no longer enters any
clutter estimate either.

Measured on a 20 km VH window over the Fujairah coast, city, port and inshore
anchorage, on the same pass:

| Mask | Candidates in box | Per 100 km2 searched |
|---|---|---|
| Interior polygons only (before) | 644 | 161.0 |
| Coastline, no buffer | 87 | 40.4 |
| Coastline, 150 m buffer (default) | 46 | 22.1 |
| Coastline, 500 m buffer (plan value) | 32 | 16.2 |

Without the mask, hundreds of candidates sat on the city and the Hajar
foothills. With it, the offshore anchorage boxes are the same in all four
previews; the buffers differ only near the shore. Of the 87 unbuffered
candidates, 25 lie within 50 m of a coastline vertex, 47 within 150 m and 59
within 500 m. At 150 m, candidates remain on curved breakwaters that
OpenStreetMap draws as lines rather than land; 500 m removes them but also
blinds the port approach. The default stays 150 m, provisional: structures that
recur on every pass are better removed by a persistence mask from repeated
passes (section 8) than by a buffer that wide. The buffer is a square dilation,
so it reaches about 1.4 times as far along diagonals.

### 2026-09-25 - Phase 4 started: AIS coverage and the label source

**aisstream has no coverage east of 40 E** (measured 2026-09-24, about
20:15 UTC). The key streams normally elsewhere: 1,285 messages in 25 s
worldwide, and 126-145 in 10 s over the North Sea through our own
subscription code. A Gulf subscription returned only the subscription
confirmation in 41 s, the ingest worker stored nothing in 90 s, and a
30-75 E, 5-35 N subscription returned 88 messages in 90 s, all from the
eastern Mediterranean. A published hobby tracker recorded 384 vessels in four
days around UAE ports with the same feed, so those receivers have existed but
are silent now. The worker keeps running, detached and logging to
`data/logs/ais_ingest.log`, in case they return. It also exposed a stats bug:
the subscription confirmation was counted as a rejected message.

**Label source: Global Fishing Watch** (user decision). GFW's Sentinel-1
vessel detections, 2017 to about five days ago, are matched to satellite and
terrestrial AIS, which reaches open water. From the official
gfw-api-python-client: the 4Wings report on `public-global-sar-presence:latest`
can be grouped by `MMSI` or `VESSEL_ID` at `HIGH` spatial and `HOURLY`
temporal resolution over a GeoJSON polygon, each row carrying a cell position
and the matched vessel's identity. It is gridded, not per detection, so labels
will attach to our CFAR detections only where a cell holds exactly one
detection and one vessel. The Bulk Download API currently serves only
`public-fixed-infrastructure-data:latest`. The cell size, and how unmatched
detections appear, are to be confirmed on real responses once the token
exists. Non-commercial terms.

**Chips store raw uint16 amplitude, not uint8.** Section 8 sized chips as
normalised uint8. The frozen sigmoid was calibrated on VV, where sea sits near
80 DN; VH sea sits near 18 DN and many ships peak at a few hundred, so 8-bit
chips would bake a VV calibration into VH. A chip is 128 x 128 x 2 x uint16 =
64 KB, so 50k chips take about 3.2 GB. Normalisation is chosen and frozen at
training time.

**The chip pipeline was checked on real imagery.** On the S1D 2026-09-22
02:06 UTC pass over the Fujairah anchorage, CFAR gave 153 detections and 148
chips were cut; five lay within 64 px of the window edge. In the contact sheet
each of the 48 brightest is centred on a vessel, and 16 open-sea negatives are
empty speckle. The scene-level split is a salted SHA-256 of the product name,
giving 79.8 / 10.1 / 10.1 percent over 100,000 names.

**GFW labels, measured.** The first calls failed with connection errors that
this host's intermittent network explains (curl got an answer in 1 s), so
every GFW request now retries. On the 2026-09-10 Fujairah pass GFW reported
129 detections in 112 cells at 02:06:53 UTC. Its per-cell counts equal our CFAR
counts in 95 of 112 cells when the coordinates are read as cell centres (48 as
corners), an independent check of our detector and georeferencing. But none of
the 129 was matched to AIS. Matched and unmatched detections over the same box:

| Period | Matched | Unmatched |
|---|---|---|
| Sep 2024 | 596 | 8 |
| Sep 2025 | 583 | 3 |
| Jun 2026 | 112 | 600 |
| 8-25 Sep 2026 | 373 | 328 |

Whether 2026 reflects GFW's processing after its summer outage or the regional
AIS spoofing is not known, so labels come from passes before 2026. Grouping by
MMSI returns only the MMSI; grouping by VESSEL_ID returns name, IMO, flag and
class. `spatial_aggregation` is refused without a group-by. Sentinel-1C passes
in 2025 have no GFW detections at all, so they are skipped rather than
reported as "GFW saw nothing".

**First labelled dataset: Fujairah, September 2025.** Of 15 passes, the 8 with
GFW coverage (all S1A) gave 707 detections in the 20 km box: 410 labelled
(58%), 42 not seen by GFW, 1 seen but unmatched, and 238 ambiguous (176
because two of our detections share a 1.1 km cell, 60 near an edge with GFW
across it, 2 where GFW saw several objects), plus 128 open-sea negatives. A
flat 100 m edge rule had discarded a third of every cell; checking the cell
across the edge instead raised the labels from 295 to 410 on the same passes.
Label check: only 19 of the 266 labelled vessels have a registry length in
GFW's identity data, so length labels will be sparse. On those 35 chips the
registry and CFAR lengths correlate at r = 0.80 (median CFAR/registry 1.08, 29
of 35 within 25%), which wrong labels would not produce. The largest
disagreements repeat on the same two ships across passes, consistent with a
barge alongside merging into one detection.

### 2026-09-27 - Phase 4 scale-up: areas, and the label check at scale

**Areas were chosen from GFW, not guessed.** A LOW-resolution (0.1 degree)
4Wings grid over 54.0-57.6 E, 24.3-27.0 N for September 2024 and 2025, then
the exact 20 km boxes at HIGH resolution with the build's own fetch, plus the
catalogue passes whose footprint holds each box centre. September counts of
AIS-matched GFW detections (matched / unmatched):

| Area (centre, box) | Sep 2024 | Sep 2025 | S1A passes 2024 / 2025 |
|---|---|---|---|
| Fujairah 56.53 E 25.22 N, 20 km | 596 / 8 | 583 / 3 | 92 / 90 |
| Khor Fakkan 56.52 E 25.41 N, 20 km | 313 / 15 | 488 / 18 | 92 / 90 |
| Sharjah/Dubai 55.33 E 25.47 N, 20 km | 350 / 54 | 292 / 28 | 60 / 62 |
| Ras Al Khaimah 55.88 E 26.02 N, 20 km | 69 / 13 | 119 / 12 | 91 / 91 |
| Hormuz lanes 56.55 E 26.50 N, 40 km | 129 / 16 | 117 / 20 | 92 / 90 |
| Hormuz lanes, same centre, 20 km | 35 / 1 | 34 / 2 | 92 / 90 |
| Jebel Ali 54.87 E 25.12 N, 20 km | 108 / 7 | 59 / 3 | 60 / 62 |
| Abu Dhabi 54.20 E 24.62 N, 20 km | 72 / 4 | 101 / 8 | 120 / 119 |

Built (user choice): Fujairah 2024, and Khor Fakkan, Sharjah/Dubai, Ras Al
Khaimah and the 40 km Hormuz lanes box for 2024-2025. The traffic lanes are
sparse, so a 20 km box there would hardly pay; the 40 km box covers both lanes
and brings moving ships. Sharjah/Dubai replaces Jebel Ali, which has about a
third of its matched detections. Qeshm and Bandar Abbas are mostly unmatched
and useless for labels. The boxes do not overlap, and a pass covering two of
them lands in one split, since the split hashes the product name.

**Builds** (every S1A pass with GFW detections processed, none failed; S1C
passes skipped):

| Area, period | S1A passes | Detections | Labelled | Unmatched | Not seen | Ambiguous (crowded) | Negatives |
|---|---|---|---|---|---|---|---|
| Fujairah 2025 | 89 | 7,819 | 4,157 | 504 | 560 | 2,417 (1,783) | 1,424 |
| Fujairah 2024 | 92 | 8,341 | 4,916 | 25 | 789 | 2,438 (1,757) | 1,472 |
| Khor Fakkan 2024-25 | 181 | 13,141 | 6,741 | 494 | 2,092 | 3,687 (2,632) | 2,896 |
| Sharjah/Dubai 2024-25 | 120 | 9,078 | 4,290 | 479 | 1,264 | 3,031 (2,200) | 1,920 |
| Ras Al Khaimah 2024-25 | 180 | 2,615 | 1,407 | 133 | 247 | 803 (610) | 2,880 |
| Hormuz lanes 2024-25 (40 km) | 182 | 3,560 | 1,762 | 336 | 91 | 1,365 (1,120) | 2,912 |

**Phase 4 target met: 23,273 labelled chips** over 844 area-scenes (422
distinct passes, since one pass often covers several boxes) in five areas,
plus 20,755 unlabelled detections and 13,504 open-sea negatives (3.14 GB).
Vessel chips split 17,701 / 2,893 / 2,679 (76 / 12 / 12%). Contact sheets
checked per area: labelled chips are centred on vessels (smaller and fainter at
Ras Al Khaimah and Sharjah than at Fujairah; moving ships with wakes and
azimuth smear in the Hormuz lanes), and negatives are open-sea speckle with no
coast in them. Negatives are 0.58 per labelled chip, short of the ~1:3
positive:negative target in section 8, and all are random open sea; the hard
negatives section 8 asks for (platforms, azimuth ghosts, wind-roughened sea,
land edges) are still to be mined.

**CFAR splits large moving ships.** In the Hormuz box 1,120 of 3,560
detections (31%) were ambiguous because two of our detections shared a cell,
spread along the lanes rather than at an anchorage. Of the first 280, 260 have
another detection within 30 px (median 170 m), their median CFAR length is 108
m against 223 m for labelled ships, and their chips show one long ship, often
with its wake, with the detection on one segment of it. These fragments stay
unlabelled, so the labels are clean, but the same splitting would count one
ship twice in Phase 6 dark-vessel counts; detections along one ship's axis
should be merged first.

**The length check at scale is r = 0.6-0.75, not 0.80.** Only 7-15% of
labelled vessels have a registry length in GFW's identity data:

| Area | Chips paired | Pearson r | Within 25% | Median CFAR / registry |
|---|---|---|---|---|
| Fujairah 2024 | 462 | 0.63 | 68% | 1.12 |
| Fujairah 2025 | 411 | 0.62 | 69% | 1.11 |
| Khor Fakkan | 1,001 | 0.70 | 75% | 1.12 |
| Sharjah/Dubai | 794 | 0.60 | 63% | 1.17 |
| Ras Al Khaimah | 171 | 0.75 | 60% | 1.16 |
| Hormuz lanes | 153 | 0.56 | 54% | 1.16 |

The 0.80 above came from 35 chips. Taken apart over both Fujairah years (873
chips, Spearman 0.54), the outliers are measurement, not labelling: 719 of 873
chips (82%) lie within 1.5 times the registry length, and on those r = 0.93. CFAR is more
than 1.5 times too long on 136 chips, 125 of them from 16 vessels that repeat
across passes: tugs, supply boats and bunker barges (45-130 m) measured at
265-365 m. Their chips show a large ship with a second hull alongside, so the
label names the small vessel moored against an unmatched or unresolved big one.
CFAR is more than 1.5 times too short on 18 chips of 18 different vessels, all
very large ships (276-361 m) where the hull extent kept only the bright end and
the chip is centred there. Positions are therefore sound for detection
training; length and class targets need filtering (for example dropping pairs
outside 1.5 times, or service-vessel labels in bunkering anchorages) before
Phase 5 uses them, and the point label of a very large ship can sit 100 m or
more from its centre.

**summary.json fixes.** Processed scenes wrote their chip-skip count under
`skipped`, overwriting the scene's own skip reason; the count is now
`chips_skipped`. Each build also overwrote the single `summary.json` of an
output directory it shares with other periods; summaries are now
`summary_<since>_<until>.json`, and the Fujairah 2025 summary was converted.

**A DNS outage cost the first run.** From about 09:30 to 09:53 UTC this host
resolved no names: Khor Fakkan processed its first 30 passes (to 2024-04-27)
and failed the other 151 with the reader's network error, and the three later
builds died in uv, which could not reach pypi.org to resolve packages. The
runner now calls `uv run --offline` (cached packages, which also stops version
drift between areas), waits until CDSE and GFW answer before each build, and
re-runs an area, up to three times, while its summary reports failed scenes.
Failed scenes keep their old manifest records and a re-run replaces the records
of every scene it processes, so re-runs never duplicate chips.

**Power-offs cost two Hormuz runs.** The PC was shut down from the Start menu
during the 40 km build on 2026-09-27 (after sleeping) and again on 2026-09-28
(Windows System log events 42 and 1074), at 56 and 62 of 223 passes. The build
writes its manifest only at the end, so both runs were lost whole. The third
ran in eight quarterly chunks into the same directory: each chunk writes its
manifest records and its own summary, and the runner skips a chunk whose
summary has no failed scene, so a shutdown costs at most one quarter (about 12
minutes). The output is the same as one run, since passes, GFW hours, negative
seeds and splits are all per scene.

**GFW's vessel classes changed under us.** The same request (Fujairah box,
September 2025, HIGH / HOURLY, matched, grouped by VESSEL_ID) returned
`OTHER_NON_FISHING` / `TANKER` for 341 of the 410 labelled vessels on
2026-09-25 and `OTHER` / `OTHER` for the same vessel IDs on 2026-09-27; the
positions, vessel IDs, MMSIs and names are unchanged. An ungrouped LOW /
ENTIRE report on the morning of 2026-09-27 still said `TANKER`. So labels
fetched on the 27th (Fujairah 2024 and every new area) mostly carry no coarse
class (Khor Fakkan: 5,647 of 6,741), while Fujairah 2025 does. Detection labels
are unaffected; coarse classes must be re-derived for every label from one
source, keyed by GFW vessel ID, before they are used. The vessel identity API
shows where the tanker class went: for 50 vessels labelled on the 25th (39 of
them `TANKER` then), the combined ship type (source
`COMBINATION_OF_REGISTRY_AND_AIS_INFERRED_NN_INFO`) now reads `OTHER` for
tankers while `CARGO`, `BUNKER` and `PASSENGER` survive. The registry records
still say `TANKER` or `CARGO`, but only 25 of the 50 vessels have one; they
also carry gross tonnage where the length is missing, a possible size proxy.

**GFW allows one report at a time per token.** A probe run while a build was
fetching was refused with HTTP 429 ("not currently enabled to perform more
than one concurrent report"). The client raises `RateLimitError`, which
`is_transient` does not retry, so a report started alongside a build would
have killed it. Query GFW only while no build is fetching.

### 2026-09-29 - Hard negatives, the training index, and Phase 5 code

**Azimuth ghosts are not mined: they are not visible at 10 m.** From the
annotation XML (IW1/IW2/IW3 PRF 1,717/1,452/1,686 Hz, slant range 800-900 km,
orbit speed 7.6 km/s) a ghost should sit 470-580 px along azimuth from its
ship. Over the 29 brightest Fujairah ships the median azimuth profile peaks at
1.19x sea at +564 px, no higher than speckle at control offsets (1.15x at
+-170-210 px), and CFAR detections within 3 columns of 5,544 bright ships show
no excess in the ghost band (79/65/91 per 50 px against 78 before and 46
after).

**Hard negatives, and what the data offer.** `scripts/mine_negatives.py`
re-reads every built scene, re-runs CFAR so every target in the window is
avoided, and cuts chips on land edges (10-70% land, 300 m from any
detection), on sea at least 1.3 times the window's sea median with no point
target in the chip, at offshore structures standing on the acquisition date,
and on extra open sea (56 per scene, to reach the ~1:3 ratio of section 8).
Records go to one file per scene, written atomically, so a rerun resumes; the
area manifests are never touched. The five boxes hold little of the hard
kinds: Fujairah and Khor Fakkan are open water, the six Sharjah "platforms"
in the Offshore Platform Dataset each stood for one quarter in 2017-2019 or
2023, and Hormuz has two platforms standing in 2024-2025 plus the Musandam
coast. Unlabelled detections at 4 spots seen in at least 30% of an area's
scenes and never labelled are recast as persistent-structure negatives (185
chips) in the index. A DNS outage failed 69 Ras Al Khaimah scenes; the rerun
recovered all of them.

**The training index** (`scripts/build_index.py`, `data/datasets/index.jsonl`)
prefixes chip ids and shards with the area (19,938 ids had clashed), and
states per chip what it may teach: positive (AIS-matched vessel at the
centre), ignore (unidentified detection, maybe dark), full negative (open sea
or clutter sampled 1 km clear of every target), or centre negative (platform,
coast, persistent structure: CFAR masked the ground around them, so vessels
nobody detected may stand elsewhere in the chip). Each chip lists the other
known objects inside it, across areas, so no unlabelled detection is ever
trained as sea. `--verify` opens every shard.

**Normalisation frozen per band** (`core/inference/contract.py`), solved on
4,000 chips so the median sea maps to 0.1 and the median ship peak to 0.9:
VV sea 60 DN, ship peak 1,829 DN (scale 0.0025, offset 945); VH sea 36 DN,
ship peak 650 DN (0.0072, 343). The older VV-only curve (0.004, 640) sent half
of all ship peaks above 0.99.

**Targets and model.** `core/labeling/targets.py` encodes a fixed 3-cell
Gaussian per vessel at stride 2 with sub-pixel offsets, a zero loss weight
within 8 cells of every unlabelled detection, and supervision only within 8
cells of a centre negative; `core/inference/decode.py` inverts it (3x3
max-pool peaks), and an encode-decode round trip returns the input
coordinates. `training/` holds the timm EfficientNet-B0 + U-Net point
detector (objectness and offset heads only: length, heading and class wait
for trustworthy targets), reduced-focal or BCE objectness, L1 offsets,
weighted sampling (easy sea 0.15, unidentified 0.1), a resumable loop scored
by point F1 within 100 m on the val split, and ONNX opset 17 export with INT8
dynamic quantisation; ONNX matches PyTorch within 1e-3. Training runs on
Kaggle (user decision), from ImageNet weights; Kaggle GPUs and notebook
internet need a phone-verified account.

**Packages through a mirror, verified against PyPI.** files.pythonhosted.org
resets every connection from this host while pypi.org answers.
`training/requirements.in` is resolved through a public mirror (the TUNA
mirror refuses uv's JSON requests; Aliyun works) and `scripts/pin_from_pypi.py`
writes `training/requirements.txt` with the sha256 hashes pypi.org publishes
for all 37 packages. `uv pip install --require-hashes` into `.venv-train`
therefore accepts only files identical to PyPI's.

**The finished dataset** (`data/datasets/index.jsonl`, 112,917 chips from 422
passes, every record checked against its shard):

| Split | Positive | Ignore | Full negative | Centre negative |
|---|---|---|---|---|
| train | 17,701 | 15,896 | 48,853 | 4,612 |
| val | 2,893 | 2,424 | 7,647 | 655 |
| test | 2,679 | 2,250 | 6,767 | 540 |

Negatives: 60,592 open sea, 5,074 coast, 2,675 bright clutter, 548 platform,
185 persistent structure, 69,074 in all, 2.97 per positive. Only 8,482 are
hard; the weighted sampler keeps easy sea at 0.15.

**CPU smoke run and export, this laptop (Ryzen 7 3700U).** One epoch of 256
samples plus 256 val chips took 87 s, so a 40,000-sample epoch would take
over two hours here: training belongs on the GPU. The smoke run exposed a real
bug, now fixed: OneCycleLR fixes its total steps, so resuming under a changed
epoch count crashed; resuming now refuses a changed schedule with a clear
message. ONNX export on the smoke checkpoint: fp32 17.9 MB, within 2.4e-6 of
PyTorch, **6.2 s per 2048 px tile**. **INT8 dynamic quantisation, as section
7.1b asks, is 12.7 times slower (78.4 s per tile)** at 4.8 MB and 1.7e-3:
ONNX Runtime quantises the convolutions into integer kernels that this CPU
runs slowly. fp32 is the deployment default; if INT8 is wanted, it must be
static (QDQ) quantisation calibrated on real chips, measured the same way.
At 6.2 s a tile, a 20 km box costs about 6 s and a full scene about 20
minutes before CFAR prefiltering.

### 2026-10-01 - Phase 5: first trained point detector (kaggle-v1)

**Run.** Private Kaggle dataset `mahmoudnaseri/khalij-watch-chips` (uploaded
through the user's VPN: api.kaggle.com refuses this network otherwise) and
private notebook `mahmoudnaseri/khalij-watch-point-detector`. Kaggle assigned
a Tesla T4 although a P100 was requested. EfficientNet-B0 from ImageNet
weights, reduced focal loss, 15 epochs x 40,000 weighted samples, batch 64,
one-cycle lr 1e-3: about 8 minutes an epoch, 2 hours in all.

**Validation F1 by epoch** (100 m tolerance, unlabelled ground ignored):
0.450, 0.574, 0.692, 0.633, 0.728, 0.725, **0.910**, 0.768, 0.759, 0.899,
0.831, 0.794, 0.780, 0.810, 0.810, while the training loss fell steadily from
1.08 to 0.10. The last five epochs all peaked at threshold 0.10, then the
grid's floor; the grid now reaches 0.01. Re-scored locally with it, the final
epoch is genuinely worse (0.810 at 0.07), not capped, and it lost recall in
the dense anchorages (Fujairah 0.904 -> 0.748, Khor Fakkan 0.930 -> 0.756,
Ras Al Khaimah 0.923 -> 0.663) while Sharjah and Hormuz held or improved.
Hypothesis, untested: late training learns to suppress ships that look like
their unlabelled anchorage neighbours. The epoch-6 checkpoint was chosen on
validation F1 before any test number existed.

**Held-out test, epoch 6, threshold 0.45 fixed on validation:** F1 **0.937**
(precision 0.927, recall 0.947; 2,687 hits, 212 false alarms, 150 misses of
2,837 vessels). The best threshold on test itself gives 0.938, so the
validation threshold transfers. By area: Khor Fakkan 0.959, Fujairah 0.948,
Ras Al Khaimah 0.927, Sharjah/Dubai 0.904, Hormuz lanes 0.870 (recall 0.817;
moving ships and CFAR fragments). These are scores against our own
auto-labels (CFAR detections matched by GFW), not independent truth.

**Export.** fp32 ONNX matches PyTorch within 1.0e-5 and takes 5.4 s per 2048
px tile on this laptop: the model runs on the CPU, as Phase 5 requires. INT8
dynamic quantisation is unusable on the trained model, not just slow: logits
differ by up to 8.8 (on Kaggle and locally) and it takes 64 s a tile.

**Still open for Phase 5:** "beats the CFAR baseline on held-out data". Our
labels are CFAR detections, so CFAR cannot be scored fairly against them; the
comparison needs an independent reference, such as GFW's full SAR detections
(matched and unmatched) for the test passes.

### 2026-10-01 - Model vs CFAR against GFW: the model does not beat CFAR yet

**Inference tiles must match the training chips.** EfficientNet's
squeeze-excitation blocks average over the whole input. On 2048 px tiles of
calm, dark sea the model went blind: Fujairah 2024-05-16 (VV sea median 25 DN)
gave 0 detections on 2048 px tiles, 69 on 512, 84 on 256 and 128 on 128,
while a normal-sea pass (69 DN) gave 73-78 at every size. The runner
(`core/inference/runner.py`) therefore uses 128 px tiles with 32 px overlap:
6 s for a 20 km box on this laptop. Wired into `scripts/detect_window.py
--detector model`; on the S1D pass of 2026-09-27 over Fujairah it found 97
vessels (CFAR 102), stored and served by the API as `pointdet`.

**Comparison** (`scripts/compare_cfar_model.py`): 40 held-out test passes, 8
per area evenly spaced in 2024-2025, chosen before any result. The reference
is every GFW SAR detection in the box at the pass hour (1,669: 1,540 matched
to AIS, 129 unmatched), outside the shared land/coast/platform mask. GFW gives
counts per 0.01 degree cell, so both detectors are scored per cell, strictly
and with neighbouring cells allowed. Model threshold 0.45 (from validation).

| Rule | Detector | Precision | Recall | F1 | Recall, AIS-matched | Recall, unmatched |
|---|---|---|---|---|---|---|
| same cell | CFAR | 0.710 | 0.918 | **0.800** | 0.930 | **0.771** |
| same cell | model | 0.679 | 0.838 | **0.750** | 0.869 | **0.477** |
| neighbours | CFAR | 0.748 | 0.968 | **0.844** | 0.979 | **0.837** |
| neighbours | model | 0.730 | 0.901 | **0.807** | 0.932 | **0.535** |

**Correction (2026-10-03): the two subset-recall columns were first
published optimistic** (CFAR 0.931 / 0.775 and 0.981 / 0.891, model 0.869 /
0.488 and 0.935 / 0.628). `summarise()` scored the AIS-matched and unmatched
subsets each against all predictions, so one prediction could count as a hit
in both, mostly through the neighbour rule. They now come from the same joint
assignment as precision, recall and F1, which were right and are unchanged.
GFW gives counts, not identities, so in the 13 of 1,575 reference cells
holding both kinds a cell's hits are shared in proportion to their counts;
any other choice moves unmatched recall by at most 0.012 (CFAR 0.767-0.775,
model 0.465-0.488 same cell; model 0.527-0.543 neighbours). The neighbour
pass is greedy: an optimal matching would add 1 hit for CFAR and 3 for the
model out of about 1,600, so it is kept.

By area (neighbours, F1 CFAR / model): Fujairah 0.866 / 0.820, Khor Fakkan
0.819 / 0.792, Sharjah/Dubai 0.842 / 0.836, Ras Al Khaimah 0.834 / 0.727,
Hormuz lanes 0.854 / 0.740 (model recall 0.65 there). **Phase 5's done-when
("beats the CFAR baseline on held-out data") is not met.**

**Why: the label bias of section 8, now measured.** The model was trained to
reproduce CFAR detections that GFW matched to AIS; every detection GFW saw but
could not match was "ignore". It learned AIS-like vessels: on GFW detections
without AIS its recall is 0.48 against CFAR's 0.77 (same cell; 0.54 against
0.84 with neighbours), while on AIS-matched detections it trails CFAR by only
0.06 (0.05). A bootstrap over the 40 passes puts CFAR's lead on unmatched
detections at +0.30 (95% interval +0.16 to +0.43 same cell, +0.15 to +0.47
neighbours) and on matched ones at +0.06 (+0.05 to +0.08). The gap is not
uniform: it is largest where non-AIS traffic is common (Hormuz lanes 0.20
against 0.76 same cell, n = 41; Ras Al Khaimah 0.16 against 0.45, n = 22), and
nearly absent in Sharjah/Dubai (0.86 against 0.96, n = 52, about the matched
gap there). Both detectors' precision near 0.7 is partly GFW missing small
vessels, which this reference cannot separate from false alarms. The 0.937
F1 on our own labels and this result are consistent: the model is good at
the task it was given, and the task under-specifies dark vessels.

**Next run (proposed):** make the 1,603 training detections GFW saw without
AIS (`gfw_unmatched`) positives instead of ignored; add translation jitter so
positives are not always at the chip centre; keep 128 px inference; select
the checkpoint on this GFW comparison over validation passes, not on our own
labels; re-run this comparison on all 92 test passes.

### 2026-10-04 - Run 2: dark vessels as positives; the model now edges CFAR

**Run 2 (`kaggle-v2`).** Run 1's recipe with three changes: the 1,603
training detections GFW saw without AIS (`gfw_unmatched`) became positives
(run 1 ignored them); random 96 px crops of the 128 px chips, so vessels are
not always centred; weights kept every epoch. Tesla T4, 15 epochs of about
8.6 minutes. Validation F1 on our own labels rose to 0.97-0.975 from epoch 4
on, with no late collapse; it now counts `gfw_unmatched` as positive, so it
is not comparable with run 1's 0.910.

**The new positives look like vessels.** In a sheet of 64 random training
`gfw_unmatched` chips, about 57 are compact, ship-shaped returns (several
smeared like moving ships) and about 7 are small faint points, 5-8 dB above
the sea in VH, that may be small boats; none look like land, coast or bare
wakes. They are smaller and fainter than AIS-matched positives (median CFAR
length 143 m against 181 m). This is a visual judgement, not a measured
label precision.

**Choosing the checkpoint against GFW on validation.**
`scripts/compare_cfar_model.py` now reads each pass once and runs several
models on it, recording each model's weights (sha256) and tiling so runs
cannot be mixed, and adds a threshold sweep and a paired bootstrap over
passes. On 25 validation passes (5 per area) it scored 13 configurations of
runs 1 and 2:
- Tile size matters again, in the other direction from run 1's blindness on
  2048 px tiles: 256 px tiles beat 128 px for run 2 (and run 2.5 times
  faster); the 96 px training crop was worst. Small tiles double the
  near-duplicate peaks: 8-12% of detections lie within 100 m of another from
  the same configuration, against about 4% with 256 px tiles (CFAR 1.3%).
  Tile seams split one vessel's response into two maxima.
- Longer training costs dark vessels: at 128 px, neighbour recall on
  GFW-unmatched detections fell from 0.85 (epoch 5) to 0.68 (epoch 14); at
  256/64 epoch 5 also scored best (mean F1 0.848 against 0.835 for epoch 14).
- The selection rule was agreed with the user before any test number. The
  first, neighbour F1 only, picked a threshold at which the model was below
  CFAR on the same-cell rule, so it was amended, still before test, to the
  mean of same-cell and neighbour F1, matching Phase 5's "better under both
  rules": run 2 epoch 5, 256 px tiles with 64 px overlap, threshold 0.60
  (validation mean F1 0.846 against CFAR's 0.829; optimistic, since chosen on
  those passes). Two validation runs agreed exactly on CFAR, GFW and the
  configuration they shared.

**Held-out test: the same 40 passes as run 1, all three detectors in one
run.** CFAR and run 1 reproduce their 2026-10-01 counts exactly.

| Rule | Detector | Precision | Recall | F1 | Recall, AIS-matched | Recall, unmatched |
|---|---|---|---|---|---|---|
| same cell | CFAR | 0.710 | 0.918 | 0.800 | 0.930 | 0.771 |
| same cell | run 1 | 0.679 | 0.838 | 0.750 | 0.869 | 0.477 |
| same cell | run 2 | 0.720 | 0.931 | **0.812** | 0.943 | 0.787 |
| neighbours | CFAR | 0.748 | 0.968 | 0.844 | 0.979 | 0.837 |
| neighbours | run 1 | 0.730 | 0.901 | 0.807 | 0.932 | 0.535 |
| neighbours | run 2 | 0.756 | 0.978 | **0.853** | 0.990 | 0.829 |

Run 2 minus CFAR, 95% intervals from resampling passes: F1 +0.012 (-0.000
to +0.026) same cell, +0.009 (-0.003 to +0.022) neighbours; recall on
AIS-matched detections +0.013 (+0.004 to +0.023); recall on unmatched
detections +0.016 (-0.10 to +0.12) and -0.008 (-0.14 to +0.10). Run 1 minus
CFAR: F1 -0.050 (-0.072 to -0.029) and -0.038 (-0.059 to -0.016).

**Phase 5's done-when is met under the criterion agreed with the user, but
not significantly:** run 2's F1 exceeds CFAR's under both rules on held-out
passes, and both intervals include zero. The label-bias fix worked: recall on
detections without AIS went from 0.48 (run 1) to 0.79, level with CFAR's
0.77. By area (neighbours, CFAR / run 2): Hormuz lanes 0.854 / 0.902 and Ras
Al Khaimah 0.834 / 0.900, where non-AIS traffic is common and run 2 also has
far fewer false alarms; Sharjah/Dubai 0.842 / 0.858; Fujairah 0.866 / 0.872;
Khor Fakkan 0.819 / 0.798 (25 more false alarms). It takes about 10 s per
20 km box on this laptop.

Limits: 40 of 92 test passes; GFW is not ground truth, and precision near
0.75 for every detector mixes false alarms with vessels GFW missed; small
tiles' duplicate peaks are reduced, not removed (2.8% within 100 m at the
chosen setting). The code-review fixes of 2026-10-03/04
(`8d71fae`..`89acf8b`) changed none of the numbers above; runs 1 and 2 were
trained with two small label bugs since fixed (a half-pixel target offset on
1.8% of positives, and ignore discs that a later negative disc could
override).

**Supplement: all 92 test passes** (the user chose to firm up the evidence;
configuration frozen, nothing tuned on test; the 40 passes above resumed and
52 added; 3,900 GFW detections, 246 without AIS). F1 same cell: CFAR 0.808,
run 1 0.763, run 2 **0.817**; neighbours: CFAR 0.848, run 1 0.820, run 2
**0.854**. Run 2 minus CFAR: F1 +0.009 (+0.002 to +0.016) same cell, now
significant, and +0.006 (-0.000 to +0.013) neighbours, still touching zero;
recall on AIS-matched detections +0.008 (+0.003 to +0.015); recall on
detections without AIS 0.813 against 0.768 (+0.045, -0.03 to +0.11). Run 1
stays significantly below CFAR (F1 -0.045 and -0.028). By area
(neighbours, CFAR / run 2): Hormuz lanes 0.860 / 0.918, Ras Al Khaimah 0.813
/ 0.859, Sharjah/Dubai 0.832 / 0.849, Fujairah 0.873 / 0.873, Khor Fakkan
0.832 / 0.820. The pre-agreed comparison remains the 40-pass one; on all 92
passes the conclusion is the same, with the same-cell difference now
outside its interval.

**Product (2026-10-04).** Run 2 is the model `scripts/detect_window.py
--detector model` runs by default (256 px tiles, 64 px overlap, threshold
0.60), and `/api/detections` now returns one detector per pass by default
(the model's where it ran, CFAR's elsewhere), so the map no longer counts
vessels twice. Training and evaluation used S1A passes only; the product
also runs on S1C and S1D, which is untested.

### 2026-10-04 - Sentinel-1C/1D check: the product model holds

Runs 1 and 2 were trained and tested on Sentinel-1A passes of 2024-2025,
but the product now runs on Sentinel-1C and 1D (S1A retired in June 2026).
The frozen product configuration (run 2 epoch 5, 256/64 tiles, threshold
0.60) and CFAR were scored against GFW on every S1C/S1D pass over the five
boxes since GFW's recalibration for those satellites (2026-09-08 to 09-29,
47 area-passes, none seen before), and on an S1A control from January-June
2026 (25 area-passes, 5 left out because GFW reported nothing there: the
acquisitions of 2026-04-21 and 05-27). `compare_cfar_model.py --pairs`
compares listed passes, leaves out passes without GFW detections in the box,
and records each pass's median sea amplitude.

| Passes | GFW detections (without AIS) | CFAR per GFW detection | F1 same cell, CFAR / run 2 | F1 neighbours, CFAR / run 2 | Run 2 - CFAR, same cell |
|---|---|---|---|---|---|
| S1A 2024-25 test (92) | 3,900 (6%) | 1.29 | 0.808 / 0.817 | 0.848 / 0.854 | +0.009 (+0.002 to +0.016) |
| S1A 2026 Jan-Jun (20) | 854 (26%) | 1.23 | 0.833 / 0.827 | 0.874 / 0.869 | -0.006 (-0.022 to +0.008) |
| S1C/S1D 2026-09 (47) | 1,899 (43%; 20% without a GFW artefact, next entry) | 1.58 | 0.718 / 0.721 | 0.753 / 0.758 | +0.003 (-0.006 to +0.014) |

- **No calibration shift.** Median sea amplitude on S1C (VV 61, VH 28 DN)
  and S1D (56, 28) lies inside the S1A range (per-area medians of 2024-25
  sea chips: VV 58-72, VH 29-39; S1A 2026: 58, 39), so the model's frozen
  input normalisation sees the same sea.
- **The model works on S1C/S1D, level with CFAR.** On 2026 data, with
  either satellite, run 2 and CFAR are tied within about 0.015; the small
  2024-25 advantage is not reproduced in 2026.
- **Both detectors lose about 0.1 F1 against GFW on S1C/S1D, together.**
  They find 1.58 objects per GFW detection there, against about 1.25 on
  S1A in both years, and a visual check of an S1C pass (Fujairah, 2026-09-25)
  shows clear ships without a GFW cell: GFW's recalibrated S1C/S1D reference
  is thinner, so absolute scores against GFW are not comparable across
  satellites.
- **CFAR fires on S1C's azimuth streaks; the model does not.** On that pass,
  bright vertical streaks along azimuth carry chains of CFAR detections on
  empty water. CFAR detections with no model detection within 100 m: S1C
  10.1%, S1D 6.1%, S1A 4-6%; model detections with no CFAR within 100 m:
  3.4% on S1C/S1D.

No recalibration is needed; run 2 stays the product model. The share of GFW
detections without AIS rose from 6% (2024-25) to 26% (early 2026) and 43%
(September 2026). (Corrected in the next entry: the 43% includes a GFW
matching artefact on one orbit; without it, 20%.)

### 2026-10-04 - Phase 6 feasibility: AIS at pass times

Before any Phase 6 code, the user asked whether the planned fusion (§9 "v1":
AIS extrapolated to the acquisition time, the anisotropic azimuth-corrected
gate of §8, Hungarian assignment) can be fed with AIS we can actually get for
our Gulf passes. Desk research by three researchers and a synthesis (sources
inline, `data/qa/phase6_probes/phase6_desk_research_synthesis.md`) was
followed by API probes P1-P14 against Global Fishing Watch (raw responses and
`findings*.json` in `data/qa/phase6_probes/`; one-off scripts
`data/qa/phase6_probes*.py`, not committed). The synthesis predates the
probes; where they differ, the probes win. Probe numbers versus the
synthesis's list: P1-P4 and P6-P8 are its probes 1-4 and 6-8; P9 is its
probe 5 (the trend; P5 failed); P10 (a busy pass) also settles its probe 11;
P11 runs the non-fishing part of its probe 7, P12 the length-coverage part
of its probe 8 (its tanker-class check was not run); P13-P14 are new. Its probes 9
(GFW infrastructure), 10 (interaction endpoint), 12 (download-portal
metadata) and 13 (aisstream re-check) were not run. A multi-agent fact check
of this entry against the raw files corrected several numbers before commit.

**The only free AIS at our pass times is GFW's hourly presence.**
- Measured: `public-global-presence` (v4.0), read through 4Wings reports at
  0.01° cells (about 1.1 km), hourly, grouped by vessel, gives one row per
  vessel per hour, each worth exactly 1 hour, with the cell, GFW vessel id,
  MMSI, IMO, name, flag and GFW vessel type. No speed, no course, no time
  within the hour: entry and exit timestamps are truncated to the hour (all
  140 vessels of the busy pass below read ":00:00"). Hour-level query windows
  are honoured (a 3-hour window returns exactly those hours), so one small
  report per pass is enough (about 20 s for a 20 km box). Reports cover at
  most 366 days (HTTP 422 otherwise) and only one runs at a time per token
  (HTTP 429).
- Measured: GFW's `/v3/vessels/{id}/tracks` answers 403 for our token with
  the all-vessel tracks dataset (404 with the fishing-only one, for a
  non-fishing vessel). The endpoint is in the OpenAPI spec embedded in GFW's
  documentation page but not in its endpoint list or SDKs (desk research).
  So GFW gives no position-level AIS at our pass times; apart from a small
  Dubai sample (desk research, below), no free source does.
- Desk research, not verified here: presence latency about 96 h; GFW SAR
  detections about 5 days (one GFW page says 72 h), so fusion runs in batch
  at least 4-5 days after a pass. Licence CC BY-NC 4.0 with attribution
  ("Powered by Global Fishing Watch"). AISHub needs our own receiver; UNGP,
  EMSA and CMS are closed to a university researcher; aisstream keeps no
  history; position-level AIS is commercial (Datalastic area history, about
  EUR 199/month per a search snippet, UNVERIFIED; Kpler/Spire on quote). One
  small free position-level sample exists: about 43k terrestrial positions
  of 384 vessels off Dubai/Jebel Ali, March-April 2026 (Hugging Face
  `yasumorishima/hormuz-ais`, MIT); downloading it needs the user's consent.
- Measured earlier (§17, 2026-09-25): aisstream carried no Gulf traffic in
  about 3.5 h of ingest on 2026-09-24/25; not re-checked since.
- Desk research, not verified here: GNSS jamming and spoofing degrade any
  Gulf AIS (Windward: about 970 ships a day affected in mid-June 2025; about
  1,650 ships hit in total between 28 February and 7 March 2026; a sustained
  spoofing cluster at Hormuz already in November 2024-February 2025).

**What the probes measured at two passes.**
- Busy pass (S1C, Fujairah 20 km box, 2026-09-25 14:15 UTC): 142, 140 and
  140 AIS vessels present in the hours starting 13:00, 14:00 and 15:00,
  against 155 run 2 detections, 172 CFAR and 84 GFW SAR detections (69
  matched by GFW). Of the 127 vessels present in all three hours, 91 stayed
  in one cell (anchored). The anchorage is crowded at cell level: the 140
  vessels of the pass hour sit in 111 cells, 50 of them share a cell with
  another vessel (up to 4), and the median distance to the nearest other
  vessel is 1.0 km. Of GFW's 69 matched detections, 65 carry a vessel id
  (4 carry none; findings2.json and the handoff counted the blank id as a
  66th vessel). Of the 65, 56 are in the presence feed in the pass hour, 2
  more only at 13:00 or 15:00, and 7 not in the window. For the 56, GFW's SAR
  cell equals the presence cell for 49, lies within 1.1 km for 54, and is
  8.8 and 11.0 km away for 2. For the 58 found in the window, GFW's SAR
  position lies within 0.8 km of the vessel's hourly polyline (cells at
  13:00, 14:00, 15:00) for 53 and within 1.5 km for 57: a direct check of the
  gate proposed below. Types in the pass hour: 101 OTHER, 35 CARGO, 2 BUNKER,
  1 blank, 1 "NA"; no tanker class appears (which GFW classes tankers fall
  into was not checked).
- Quiet pass (S1C, Hormuz 40 km box, 2026-09-09 02:14 UTC): 1 AIS vessel in
  the pass hour and only 6 (12 vessel-hours, at most 2 at once) over the whole
  day, mostly small regional craft; GFW 4 SAR detections, 3 unmatched; our
  CFAR 5. GFW's SAR cell and the presence cell of the one matched vessel are
  5.4 km apart. Here almost every detection will look dark, and the rate
  measures AIS absence (traffic, jamming or reception), which v1 cannot tell
  apart.
- GFW SAR rows carry the acquisition hour and, in entryTimestamp, the scene
  time; matched rows join the presence feed on hour and vessel id.
- Events over the Gulf box. AIS gaps (GFW's rule, per its documentation:
  at least 12 h, starting at least 50 nm offshore; every returned event meets
  it): the query for 2024-01..2026-09 returned 42 events, 39 distinct, 36
  starting in the window, but only 15 distinct ones switched AIS off inside
  the box (7 in the Gulf of Oman near 24.7-24.8N 57.6-57.9E, none near our boxes;
  the latest started 2026-04-17). The other 24 do not switch AIS off in the
  box; 7 switch it back on inside it. No gap can start in our coastal boxes
  (50 nm rule), but two end inside them (Khor Fakkan 2024-10-13, Hormuz
  2024-11-14) and two within 7 km (Khor Fakkan, Fujairah). Encounters
  (P11): 1,490,740 for 2025-01..2026-09; filtered to FISHING-FISHING
  1,490,730, to NON_FISHING-NON_FISHING 0; the first 100 are fishing boats
  moored together about 1 km off Qatif (26.55N 50.03E) for days, starting
  in November-December 2024. So GFW publishes no tanker-tanker encounters
  here. Loitering: 67,557 in August-September 2026. All three queries
  return events that start before the requested window.
- Registry lengths (P12; needed for IDENTITY_ANOMALY): of the 140 vessels of
  the busy pass hour, 139 are known to GFW's Vessels API and 49 have a
  registry record, but only 4 have a registry length (116-182 m), although
  132 broadcast an IMO number. On the 2024-25 labels, 7-15% of vessels had a
  registry length (§17, 2026-09-27).

**The 2024-2026 trend (P9), and a GFW processing artefact (P13).**
Monthly per-cell reports over the Gulf box (47.5-58E, 23.5-30.5N; LOW 0.1°
cells; 99 reports, one per series and month, since a whole-year presence
report failed on GFW's side; `findings3.json`): AIS presence, and GFW SAR
detections matched and unmatched. East = east of 56.0°E (Strait of Hormuz,
UAE east coast, Gulf of Oman); west = the inner Gulf.

| Period | AIS vessel-hours a day | AIS vessels a month | GFW SAR detections a month | Matched share, all | west | east |
|---|---|---|---|---|---|---|
| 2024 | 125,863 | 14,880 | 12,411 | 0.71 | 0.71 | 0.69 |
| 2025 Jan-Sep | 132,275 | 19,458 | 13,566 | 0.70 | 0.72 | 0.66 |
| Oct 2025-Feb 2026 | 145,606 | 22,574 | 13,938 | 0.64 | 0.73 | 0.42 |
| Mar-Jun 2026 | 66,640 | 13,882 | 13,176 | 0.44 | 0.51 | 0.25 |
| Jul-Sep 2026 | 99,352 | 16,925 | 25,654 | 0.53 | 0.58 | 0.41 |

From October 2025 the east and "all" shares include the orbit artefact below
(confirmed for October-February and September 2026, presumed in between);
the west column is the clean trend. The 33 monthly rows are in
`findings3.json` and the handoff.

- **October 2025: a GFW artefact, not ships going dark.** The east's matched
  share fell from about 0.66 to 0.42 while east AIS presence, east vessel
  counts and SAR totals stayed flat. About half of the change sits in the UAE
  east-coast cells (24.9-25.7N, 56.0-56.7E: unmatched 159 -> 1,408, matched
  3,224 -> 1,934, July-September vs October-December 2025); Gulf of Oman
  cells to the south and east fell too (matched share 0.87 -> 0.39 and
  0.63 -> 0.31); the Hormuz cells did not. Per acquisition over our Fujairah
  and Khor Fakkan boxes (P13, hourly, September 2025-February 2026): from
  2025-10-02, every acquisition of one descending orbit (about 02:07 UTC,
  every 12 days: 10-02, 10-14, ... 02-23) has all its 132-165 GFW detections
  unmatched and none matched, while all other acquisitions stay at 0-5%
  unmatched. The same orbit was normal on 2025-09-08 and 09-20 (3 of 186 and
  6 of 151 unmatched, P13) and on 2025-08-03 (4 of 138, test passes). It is
  still broken on S1C/S1D: on the 02:06 UTC passes of 2026-09-10, 09-16 and
  09-22, all 553 GFW detections at Fujairah and Khor Fakkan (6 area-passes)
  are unmatched (the 3 Hormuz area-passes hold 6 detections, too few to
  judge; the same orbit's Hormuz frame stayed matched in 2025), the other
  September passes 17-38%. The AIS presence feed is intact at the artefact
  hours (P14: 243 vessels over the two boxes at 2026-09-16 02:00, against 253
  at the normal pass of 09-15 02:00): the fault is in GFW's SAR matching.
- **From March 2026: AIS presence fell together with the matched share.**
  Gulf-wide AIS vessel-hours fell from 142,132 a day (February) to 61,637
  (March) and distinct AIS vessels from 21,945 to 10,199, in the west as in
  the east, while GFW's SAR detections stayed level (13,211 -> 13,547): the
  ships were still there, far fewer carried AIS that GFW received. The west's
  matched share fell from 0.74 (February) to 0.51 (March). It is not a
  matching artefact of the October kind, since the presence feed itself fell;
  whether the fall comes from ships switching AIS off, reception loss,
  jamming or GFW's own AIS processing, these data cannot separate. The
  timing matches the regional conflict the desk research dates to
  28 February 2026 (UNVERIFIED here). The east's March-June shares probably
  still include the orbit artefact (not checked for those months; in June
  presence rose 51% on May while the Gulf-wide share stayed at 0.38).
- **July-September 2026: partial recovery.** AIS presence back to about 75%
  of 2025, the west's matched share 0.58. July and August hold GFW SAR
  detections at about the post-June level (July about 11% lower per day than
  August), so GFW's reported backfill from 30 June largely holds for this
  box; a short gap within July cannot be excluded at monthly resolution.
  Detections double because two satellites image.
- Spoofing to land is not visible in the presence feed: at most 0.5% of AIS
  hours fall in the confidently-on-land interior polygons of
  `core/ais/landmask.py` in any month (GFW may filter such positions;
  unverified).
- **Corrections to earlier entries.** (1) The 43% of GFW detections without
  AIS on the S1C/S1D passes (2026-09, table above in "Sentinel-1C/1D check")
  includes the 6 artefact area-passes; without them it is 20% (269 of 1,346;
  263 of 1,340 without the 3 sparse Hormuz area-passes of that orbit too).
  F1, recall and "CFAR per GFW detection" do not use the matched flag and are
  unaffected. (2) 727 of run 2's 1,603 `gfw_unmatched` training positives
  come from 14 artefact area-passes (Fujairah/Khor Fakkan, 02:07 UTC orbit,
  October-December 2025), each with no matched detection, where the
  neighbouring passes over the same anchorage are about 98% matched: they
  are very likely AIS ships. They are still vessels, so they remain valid
  positives, but only about 876 of the new positives are vessels GFW really
  saw without AIS. Two validation area-passes (2025-12-01) are affected but
  are not among the 25 used for selection; no test pass, no validation pass
  used, and no S1A 2026 pass is affected, so Phase 5's numbers stand. The 6%
  (2024-25) and 26% (S1A 2026) figures contain no artefact passes.

**Which events are feasible with GFW cell-hour AIS.**

| Event | With GFW hourly presence (v1) | What limits it |
|---|---|---|
| `DARK_VESSEL` | Feasible: one of our detections with no AIS vessel's hourly track inside the gate, after the pre-filters | The gate is about a kilometre or more; in lanes and in crowded anchorages a dark ship near an AIS ship is absorbed (rate biased down); reception gaps, jamming and spoofing bias it up |
| `ZONE_ENTRY` | Feasible: detections at pass times, presence cells every hour | Pass times only: about 2-3.5 a week per box with S1C+S1D (10 passes per box, 7 at Sharjah/Dubai, 8-29 September 2026), 1.2-1.8 with S1A alone |
| `STS_TRANSFER` | Conditional, from our SAR only: two detections about 100-300 m apart (or one long merged CFAR blob), more than a set distance offshore, persistent across passes, with paired presence cells; GFW encounters cannot corroborate (none non-fishing) | CFAR fragments of one large ship lie about 170 m apart and the model leaves 2.8% duplicate peaks within 100 m; bunkering alongside big ships at Fujairah gives 265-365 m CFAR blobs; the merged-blob branch needs CFAR length |
| `AIS_GAP` | Weak: a vessel absent from hourly presence while inside imaged coverage; GFW gap events corroborate where they exist; at our boxes only gaps that end there can (15 distinct events started in the Gulf box in 33 months, none near our boxes; 2 ended inside them) | Absence cannot separate switching AIS off from reception loss or jamming without positions |
| `POSITION_SPOOF` | Coarse candidates for stationary vessels only: same cell for 3 h, well inside the footprint, no detection within about 1-2 km; many vessels jumping together become an interference flag, not spoofing | Ships under way cannot be tested; missed small ships make false candidates |
| `IDENTITY_ANOMALY` | Not feasible with GFW alone: 4 of 140 vessels have a registry length | The product model measures no length; CFAR extents are rough (r 0.56-0.75 against registry lengths) and merged hulls at bunkering anchorages would raise false anomalies |

**An honest dark rate.**
- Unit: our own detections (the frozen product model, run 2) in imaged water
  of the five boxes, after (a) merging duplicate peaks and fragments along a
  ship's axis (about 100 m), (b) removing detections in the shore buffer,
  (c) within about 300 m of fixed infrastructure, using the time-aware OPD
  mask already in the pipeline (§17, 2026-09-11) plus multi-pass persistence
  (infrastructure is a minor confounder in these boxes: Fujairah and Khor
  Fakkan are open water, Hormuz has two platforms, §17 2026-09-29), (d)
  azimuth ghosts at the per-sub-swath offset from the annotation XML (about
  4.7-5.8 km; none seen at 10 m on S1A), and (e) near the footprint or box
  edge.
- Dark: no AIS presence vessel whose hourly polyline (cells at H-1, H, H+1)
  passes within the gate of the detection. A detection inside some vessel's
  gate that loses the one-to-one assignment is "ambiguous" (a candidate
  set), not dark.
- Reported as a curve over the swept gate margin (the margin added to the
  floor of cell half-diagonal plus azimuth allowance, from 0 to about
  10 km), with a bootstrap interval over passes, never as one number.
- Validation objective: on validation passes, our "has an AIS candidate"
  call against GFW's matched flag for our detections that coincide with a
  GFW detection (GFW matched with position-level AIS), excluding GFW
  artefact passes; that agreement, reported on test and 2026 passes, is the
  proxy for v1's accuracy as a matcher (GFW's matching has its own errors).
  Validation fixes the operating margin used for DARK_VESSEL alerts and the
  pre-filter thresholds; the rate is still reported over the whole sweep.
- Strata. Region: inner Gulf (Sharjah/Dubai, Ras Al Khaimah), Hormuz lanes,
  Gulf of Oman (Fujairah, Khor Fakkan). Period, with passes processed so
  far: before 28 February 2026 (92 test area-passes of 2024-25 and 10 of
  January-February 2026; not interference-free, given the reported Hormuz
  spoofing cluster of late 2024); 28 February-29 June 2026 (15 S1A
  area-passes processed, 10 with GFW detections for the comparison);
  30 June-7 September 2026 (S1C/S1D only after S1A's
  retirement; GFW SAR offline from 3 July, later backfilled; no pass
  processed yet); after GFW's 8 September recalibration (47). Period and
  satellite are confounded (S1A until June, S1C/S1D after, with GFW's
  thinner S1C/S1D reference). Motion is not a stratum of the dark rate: it is
  defined from AIS cells, which dark detections lack by definition, and the
  model has no heading or speed output.
- Length: the product model outputs no length, so a length stratum (at least
  50 m, roughly ships obliged to carry AIS) can only use a co-located CFAR
  detection's extent; v1 reports the rate without a length cut and with a
  CFAR-length stratum.
- Not Gulf-wide: we detect in five boxes. GFW's unmatched share over the same
  cell-hours is the external comparison (same radar, different detector and
  matcher), excluding its artefact passes; transit-based figures (e.g.
  42-70% of Hormuz tanker transits dark in late June-mid July 2026, Lloyd's
  List Intelligence via the desk research) are not comparable.
- Error sources: AIS reception gaps, jamming and GNSS displacement inflate
  the rate; SAR false alarms, duplicate peaks and split ships inflate it
  (run 2's precision against GFW is 0.72-0.76 on S1A test passes, and
  detectors find 1.58 objects per GFW detection on S1C/S1D, partly vessels
  GFW misses); hourly sampling and crowding (wide or shared gates) deflate
  it; missed small boats; GFW's own matching artefacts if GFW is used.

**Recommended v1. This supersedes §9 "v1" and the §15 PHASE 6 "v1"
bullet**: exact-time extrapolation, the AIS-velocity azimuth correction and
heading residuals need position-level AIS, which we do not have. For the
same reason the §12 "with/without azimuth correction" ablation and the §13
AIS-position-correction mitigation wait for upgrade (1), and the Phase 1.5
plausibility filter cannot run on cell-hour presence (no messages); the
interference flag below stands in for it. v1 is a cell-hour fusion, run in
batch at least 4-5 days after a pass:
1. Inputs per pass: our detections (run 2; CFAR for length); GFW presence
   for H-1..H+1 over the box buffered by about one hour of travel (20-35 km),
   so ships under way keep their hourly cells, in one report; GFW SAR
   matched and unmatched for the same cell-hours (comparison only); the OPD
   mask; identity of the present vessels; dataset versions recorded.
2. Pre-filters: duplicate and fragment merging, shore buffer,
   infrastructure and persistence, azimuth ghosts, edges.
3. AIS hypotheses: each present vessel's hourly polyline. Gate: distance from
   the detection to the polyline within a floor of the cell half-diagonal
   (about 0.75 km) plus an azimuth allowance (R/V per sub-swath from the
   annotation XML, about 105-118 s, times a radial speed bounded by the cell
   displacement: an allowance, not a correction, since direction within the
   hour is unknown), plus the swept margin. A vessel in one cell for 3 h gets
   the floor only.
4. Assignment: a gated cost matrix solved with
   `scipy.optimize.linear_sum_assignment` (pairs outside the gate at a large
   finite cost, dropped afterwards); the margin to the second-best option is
   kept; where several vessels fit one detection or one vessel several
   detections (common at Fujairah), a candidate set is emitted, not a forced
   one-to-one match.
5. Events with evidence cards: the SAR chip, scene and time, detection,
   gate used, candidate vessels with their hourly polylines (the "AIS track
   segment"), GFW's matched flag for the cell-hour, pre-filter results,
   dataset versions, a confidence tier. No alert that cannot be explained.
6. The dark rate as defined above.
7. Guards: a pass where GFW matched nothing while the presence feed holds
   many vessels in the pass hour is flagged as a GFW matching artefact and
   left out of every comparison with GFW (our own fusion uses presence, which
   P14 found intact at an artefact hour); a pass where many vessels change
   cell together between hours gets an interference flag (possible GNSS
   spoofing), shown on its events. All GFW calls server-side (the token never
   reaches the browser); attribution; non-commercial; methods re-implemented
   from the papers (GFW's longline matching repository has no licence).
§9's corroboration with GFW's `pipe-gaps` needs positions and is dropped from
v1; GFW's own gap events are used where they exist.

Limits of v1: candidate sets in lanes and crowded anchorages; AIS_GAP at
hourly resolution without a reception test; POSITION_SPOOF for stationary
vessels only; no IDENTITY_ANOMALY; retrospective (at least 4-5 days);
dependent on GFW's reception in a jammed region (AIS presence halved from
March 2026); five boxes, not the Gulf.

**Upgrades, in order of value.** (1) GFW permits position-level tracks (an
email; the endpoint is 403 now; the same email can report the orbit
artefact): exact-time extrapolation, the §8 anisotropic azimuth-corrected
gate and its §12 ablation, kinematic POSITION_SPOOF, `pipe-gaps` AIS_GAP.
(2) GFW's Data Download Portal per-detection file (time, position, length,
matching score per GFW's 2024 announcement; fields, 2026 coverage and
licence unverified, not probed; a download needs the user's consent): a
benchmark for our matcher and lengths. (3) Commercial or licensed
position-level AIS (a Datalastic test, a Kpler research licence, a
university licence): GFW's probabilistic v2 (§9). (4) The free Dubai
sample (March-April 2026; user's consent): a check of the cell-hour gate
against true positions on overlapping S1A passes. (5) aisstream coverage
returns: a live collector at pass times.

**Proposed Phase 6 done-when** (replaces "dark vessels are flagged with
auditable evidence and a measured Gulf-wide dark rate"; to be agreed with the
user): dark-vessel candidates in the five boxes are flagged with evidence
cards, and the dark rate is reported with a bootstrap interval as a curve
over the gate margin, by region and period, with the operating margin and
pre-filter settings chosen on validation passes only and v1's agreement with
GFW's matched flag reported.

### 2026-10-05 - Phase 6 v1: cell-hour fusion and first dark rates

The user chose to build v1 on GFW hourly presence (2026-10-04). Built test
first in `core/fusion/` (presence tracks and fetcher, gate, one-to-one
assignment with candidate sets, pre-filters, dark rate with bootstrap,
guards, evidence cards, events), `scripts/fuse_passes.py` (batch over the
processed passes), `scripts/fuse_scene.py` (one stored pass into DARK_VESSEL
events), an `events` table keyed by scene and run, and `GET /api/events`
(commits `815f1c3` to `40dfb7d`). Two multi-agent reviews found and fixed:
tracks seen only at H-1 and H (or H and H+1) not reaching the pass time (now
extended by the pass's fraction of the hour; on the busy Fujairah pass this
moved 3 of 9 dark detections at margin 0 into a gate); the presence dataset
version not kept; gap flags and bootstrap per area-pass although boxes of
one overpass share its AIS feed (now per acquisition); a read timeout
resending a 4Wings report GFW may still be computing; zero-width [0, 0]
intervals.

**Two GFW data faults handled.** (1) Presence is incomplete on some dates:
on 2026-09-09 02:14 the Fujairah box held 8 presence vessels while GFW matched
64 of its own detections there to AIS. A pass whose box holds fewer presence
vessels than half of the GFW detections expected to carry AIS (at least 10)
has a presence gap; the gap marks every box of that acquisition, and those
passes are left out of the dark rate and counted. Passes with too few GFW
detections to judge are counted as unguarded. (2) The orbit artefact (§17,
2026-10-04): passes where GFW matched none of at least 10 detections while at
least 20 AIS vessels were in the box are left out of the agreement with GFW
(our own fusion uses presence, which is intact at those hours); evidence
cards there carry no GFW corroboration.

**Batch.** GFW presence for H-1..H+1 over each box buffered by 25 km, for all
189 processed area-passes (validation 25, test 92, S1A 2026 25, S1C/S1D 47),
one report at a time: 189 fused, 0 failed, about 90 min. Our detections are
the frozen product model's (run 2, 256/64, 0.60) from the comparison runs;
pre-filters: duplicates within 100 m, 1 km from land (GFW's convention),
300 m from OPD platforms, 500 m from the box edge (all fixed a priori, not
tuned); R/V bound 118 s.

**Operating margin, chosen on the 25 validation passes only:** 0 m, the
margin with the highest balanced agreement between our "has an AIS
candidate" call and GFW's matched flag (0.656 at 0 m, 0.615 at 500 m, 0.524
at 2 km, 0.500 at 10 km).

| Set | Area-passes used (gap / unguarded) | Kept | Dark rate at 0 m [95% CI] | Inner Gulf / Hormuz / Gulf of Oman | Balanced agreement with GFW |
|---|---|---|---|---|---|
| Validation, S1A 2024-25 | 25 (0 / 3) | 1,288 | 4.0% [2.6, 5.8] | 6.5 / 9.2 / 2.2% | 0.656 |
| Test, S1A 2024-25 | 92 (0 / 8) | 4,747 | 4.0% [3.2, 5.3] | 7.0 / 16.8 / 1.8% | 0.668 |
| S1A 2026 Jan-Feb (part of the 21 below) | 4 acquisitions | 466 | 3.2% [2.2, 5.1] | | |
| S1A 2026 28 Feb-29 Jun (part of the 21 below) | 5 acquisitions | 535 | 49% [8.5, 90] | | |
| S1A 2026 Jan-Jun, all | 21 (4 / 10) | 1,001 | 27.8% [4.7, 61] | 27.3 / 16.7 / 28.5% | 0.765 |
| S1C/S1D 2026-09-08..29 | 42 (5 / 18) | 2,483 | 7.9% [6.3, 9.7] | 18.4 / 91 (10 of 11) / 5.4% | 0.669 |

Intervals resample whole acquisitions. The rate falls steeply with the margin
(test: 4.0% at 0 m, 2.1% at 500 m, 0.5% at 2 km, 0.0% at 10 km; S1C/S1D:
7.9, 3.5, 1.2, 0.4%). S1A 2026 used 21 of its 25 area-passes (4 in gap
acquisitions; 10 of the 21 unguarded).

**What the numbers can and cannot say.**
- v1 is conservative: on test it calls dark 70 of the 196 detections GFW
  left unmatched (36%) and 74 of the 3,600 GFW matched (2.1%). Half its dark
  calls on GFW detections (74 of 144) are ships GFW matched with its own,
  position-level AIS, which hourly presence lacks in those hours. So the v1
  dark rate is not a count of dark ships; it is a consistent measure for
  comparing periods and regions with the same method, and GFW's unmatched
  share remains the external comparison.
- The rise from 4.0% (2024-25) to 7.9% (September 2026) and the 49% of
  March-June 2026 follow the AIS collapse measured in the trend probe; the
  spring 2026 figure rests on 5 acquisitions and its interval spans 8-90%.
- Hormuz lanes has few AIS vessels in its box at most passes (10 of 11 kept
  detections dark in September 2026), so its rate mostly measures AIS absence.
- Unguarded passes (gap rule not judgeable) stay in: 8 of 92 on test, 18 of 42
  on S1C/S1D.
- Not done: azimuth ghosts and the footprint's no-data edge (no scene
  geometry cached), per-sub-swath R/V (118 s bound), a CFAR-length stratum
  (CFAR lengths are not in the comparison results), cutting the SAR chip for
  cards (Phase 7).

**Product path.** `scripts/fuse_scene.py` fused the newest stored Fujairah
pass old enough for GFW (S1D 2026-09-10 02:06, CFAR detections; run 1 rows of
later passes are ignored): 168 kept, 147 AIS vessels in the box, 9 DARK_VESSEL
events (8 low, 1 medium confidence), served by `/api/events` with their
evidence cards; the pass is a GFW artefact pass, flagged on every card.

**Phase 6 done-when** (agreed 2026-10-04): dark-vessel candidates in the five
boxes are flagged with evidence cards (yes); the dark rate is reported with a
bootstrap interval as a curve over the gate margin, by region and period
(yes, `models/runs/kaggle-v2/fusion_v1/summary.json`); the operating margin
chosen on validation passes only (yes) and the pre-filter settings fixed a
priori (never tuned, on test or otherwise); agreement with GFW's matched flag
reported (yes).
The user accepted Phase 6 v1 as done on 2026-10-05 and chose to move to
Phase 7; the refinements (pre-filter tuning, run 2 on stored passes, ghost and
footprint filters, per-sub-swath R/V) move to Phase 8.

### 2026-10-05 - Phase 7: the radar chip on the evidence card

Phase 7 so far: dark-vessel rings and evidence cards on the map (`86d48a9`)
and an alert feed in the rail (`618203b`). The user then chose the SAR chip
as the next slice (over the timeline scrubber): each DARK_VESSEL card now
shows what the radar saw, a north-up VV and VH image 2.56 km across (256
pixels of 10 m, the GRD pixel spacing) centred on the detection, with the
detection's tier ring, the nearest AIS vessels' hourly cells and gate
tracks, a 500 m scale bar and a north arrow drawn over it.

- Cut in batch, served from disk. `scripts/cut_chips.py` (also run by
  `scripts/fuse_scene.py` once it has stored a pass's events; a chip failure
  keeps the events) opens the pass's COG on CDSE once and reads one small
  window per candidate and polarisation (about 310 pixels a side). Chips go
  to `data/chips/<scene>/<lat>_<lon>_<pol>.png`, named by scene and position
  because re-fusing a pass replaces its events (re-fused events take up the
  chips already on disk without a CDSE read), and the card records
  `chip.image` (polarisations, size, pixel, stretch, cut time, share of no
  data); pixels the radar did not image are transparent and show as a
  hatch. `GET /api/events/{id}/chip/{vv|vh}` serves the file: a browser
  never causes a CDSE read, and the API still needs neither numpy nor
  rasterio.
- North up. GRD scenes are in radar geometry: the descending S1D pass of the
  events shows the sea mirrored east-west and turned by the track angle.
  Each chip pixel is located in the raster through the scene's control
  points (the thin-plate spline the detections already use) and sampled
  bilinearly; no-data pixels are never blended in.
- Fixed logarithmic stretch per polarisation (VV 30-1500 DN, VH 15-750 DN,
  set a priori from the sea levels measured in the Sentinel-1C/1D check;
  display only), not a per-image percentile stretch, so a faint return
  stays faint and chips of different passes compare.
- Checks. On the pass's real control-point grid, tests place a bright pixel
  500 m north, 500 m east, and 800 m south and 300 m west of a detection and
  find it where a north-up chip puts it (within 2 pixels). On the 9 real
  chips, the pass's other stored CFAR detections projected onto each chip
  fall on bright returns (median grey 224-255 of 255), while the same
  positions mirrored east-west read 19-154 and random points 20-49: the
  chips agree with the scene's own geolocation (not an independent check of
  it). Product run: 9 chips (VV and VH) for S1D 2026-09-10 Fujairah in about
  a minute, none failed. In the browser: the card, the overlay against the
  map, the VV/VH toggle, a missing chip file and a card without a chip.
- Review: a four-lens multi-agent review (geometry, robustness, UI, tests;
  each lens checked by an adversarial verifier) confirmed 16 findings (11
  distinct issues; 3 refuted, 1 uncertain). The 4 of medium severity are two
  issues with one cause, SQLite reusing event ids when a pass is re-fused.
  Fixed: the console's chip URL names the chip's content (scene,
  position, cut time) and the API sends no-cache; cards are written back only
  if unchanged (compare and set); a chip that cannot be written fails alone;
  AIS hours are drawn as their 0.01 degree cells in cyan (solid for the pass
  hour), not as white dots that look like radar echoes or precise positions;
  a failed image can be retried; out-of-range event ids get 422. Tests added
  for the half-pixel convention (exact, on a linear raster) and for each
  polarisation's stretch in the written PNGs; each new test was checked to
  fail on the bug it guards (mutation). Not fixed: a process that used GDAL
  can hang on exit on this host (documented in both scripts).
- Limits: the chip shows the pass's radar return at 10 m spacing (about
  20 m resolution), not identity; on this pass the events come from CFAR, so
  the chip shows what CFAR detected, not the product model.

### 2026-10-05 - Phase 7: the Iranian coast, and where the detector searched

The user noticed that the console showed no ships on the Iranian side. Nothing
excluded Iran (the land mask covers 44-62E, the platform mask includes South
Pars and Kharg, and 73 of the 119 stored scenes reach north of 27N): detection
had only been run on one Fujairah box for the console, and the five study
boxes are all on the UAE side or in the strait. The console's AIS is a stale
synthetic demo set (2026-09-10), so it shows no AIS ships anywhere. The user
chose to run the product model on Iranian-side boxes now and to make the
console say where detection actually ran.

- Product model (run 2, frozen) on seven 20 km boxes, newest pass each
  (`data/logs/run_detect_iran.sh`, about 4 min, CDSE only): Bandar Abbas
  anchorage 69 detections (S1C 2026-10-03 02:14), Bushehr 12 and Kharg 10
  (S1D 2026-10-04 14:41, 18 Kharg platforms masked), Larak/Hormuz 9 and Jask 1
  (S1D 2026-10-04 02:06), Kish 6 and Assaluyeh 4 (S1D 2026-10-02 02:22). The
  model was never validated on this coast. A visual check of three boxes
  (Bandar Abbas, Bushehr, Kharg; north-up quicklooks in `data/qa/iran/`) found
  every ring on a ship-like echo and none on the coast; ships alongside
  Kharg's jetties are masked with the platforms, and some faint small echoes
  are not ringed. Not a measurement: no GFW comparison was run there.
- Every detection run now records its search (`analysed_windows`: pass,
  detector, version, polarisation, box), also when it finds nothing.
  `GET /api/analysed` serves each search as the requested box clipped to the
  pass footprint, with the share of the box imaged: a box can reach beyond the
  swath. Measured on the stored passes: the Fujairah study box is only 52%
  imaged on the 02:14 UTC passes of 2026-09-27 and 10-03 (the swath ends near
  56.535E at 25.22N), the Jask box 78%. A search counts as tested only for a
  detector configuration measured against GFW (run 2 on VV+VH; CFAR on VH at
  Pfa 1e-6) searching wholly inside one study box; run 1, other versions or
  channels, and searches reaching outside a study box count as untested.
  `/api/detections` marks each ring the same way. Searches of fused passes are
  served whatever their age, as their dark-vessel candidates are: the
  2026-09-10 pass (CFAR, box not recorded) has a conservative search area, the
  extent of its stored detections, which the run certainly searched.
- The console outlines searched areas (solid: tested; dashed: accuracy
  unknown), fading with age, and says that outside them no ring means not
  searched and that inside them no ring is not proof of no ship (tested
  detectors found about 4 in 5 of the vessels GFW saw without AIS on test
  passes). Of the 304 rings shown, 302 lie inside a searched area of their own
  pass; 2 (2026-09-27, run 1) sit on the swath edge, within about 100 m of
  CDSE's simplified footprint polygon.
- Review: two lenses, each with an adversarial verifier (the UI verifier was
  stopped by an API overload; its findings were checked by hand). Confirmed and
  fixed: the whole requested box claimed as searched beyond the swath (high);
  the fused pass without a search area; "tested" by box centre and by place
  only (not by detector configuration); stale counts and replacement across
  polarisations; wording that implied "searched, no ship"; a hidden error when
  searches fail to load; counts and age not shown.

### 2026-10-05 - Phase 7: the model's detections, each with its satellite image

The user set the thesis focus: vessels the deep-learning model finds in
Sentinel-1 satellite images, each shown with its image beside it. (The
product model has no class, length or heading output: only objectness and
a sub-pixel offset, the other heads left out for lack of trustworthy labels,
`training/model.py`.) The user chose "model + image" over keeping CFAR on the
map or moving to optical imagery.

- Console: the product model's detections only (run 2, `kaggle-v2`); CFAR and
  run 1 leave the map and stay in the thesis comparisons. Clicking a detection
  opens a card with its north-up VV/VH image 2.56 km across, the model score
  (kept from 0.60), the pass and its age, and whether the model was tested
  there, worded as a detection (likely a vessel): in a study box, about 3 in 4
  of its detections matched a vessel GFW saw and it found over 9 in 10 of
  those (precision 0.72-0.76, recall 0.93-0.98 on the 40 test passes);
  elsewhere its false alarms and misses were never measured.
- Chips while detecting: `scripts/detect_window.py` cuts each stored
  detection's chip from the pixels it already read for detection; only chips
  reaching beyond them are read from the scene (the 2026-09-10 run: 168 of
  170 from memory, 2 read again near the box edge). `scripts/cut_chips.py
  --detections` cut the 207 existing product detections' chips (0 failed).
  Chips are named by scene and position, shared with the dark-vessel cards.
- The 2026-09-10 Fujairah pass, whose dark-vessel candidates had come from
  CFAR, was run with the product model (`--scene`, 170 detections) and fused
  again: 163 kept, 9 DARK_VESSEL events (8 low, 1 medium), at the CFAR
  candidates' positions to within a few metres, all with chips (reused from
  disk, no extra read).
- API: `/api/detections?detector=product`, with each detection's chip status
  from one folder listing per scene (about 40 ms for 377 detections) and
  fused passes included whatever their age; `GET
  /api/detections/{id}/chip/{pol}` (a stale content key gets 404).
- Review: two lenses with adversarial verification; all 10 confirmed findings
  fixed (high: the 14-day lookback hid the fused pass's 170 detections while
  its candidates and searched area stayed on the map). The final selection
  fix was checked by type-check and build only (browser pane hidden); the
  card itself was verified in the browser before it.

### 2026-10-05 - The product model on the Iranian coast: a held-out region

The console shows the product model's detections in seven Iranian-side boxes
(Bandar Abbas, Larak/Hormuz, Kish, Assaluyeh, Bushehr, Kharg, Jask), where it
was never trained, chosen or tested. The user chose to measure it there
against GFW, like the Sentinel-1C/1D check: S1A 2024-25 first (the region
effect alone), then every S1C/S1D pass of 2026-09-08..29 (the satellites the
console uses now). Nothing was chosen on these passes: run 2, epoch 5, 256/64
tiles, threshold 0.60, as frozen; CFAR at Pfa 1e-6 as the baseline.

- Passes, by a rule written down before any result (`scripts/list_pairs.py`,
  headers of `models/runs/kaggle-v2/pairs_iran_*.txt`): passes whose
  footprint holds the box centre; for S1A, frames of the train and val splits
  left out, then 12 per box spread in time (83 area-passes; Jask had 11 left);
  for S1C/S1D every pass (62). The rule leaves out frames, not whole passes:
  31 of the 83 S1A area-passes are frames of a pass whose UAE frame was in
  training (Bandar Abbas, Larak and Jask lie next to the UAE boxes). No pixel
  of an Iranian box was ever trained on.
- Analysis strata fixed before the results were read (`scripts/compare_strata.py`):
  GFW reports no detections within about 1 km of land, while the comparison
  masks only 150 m, and the Iranian boxes have far more coast (land 2-23% of
  a box against 0-3% in the UAE boxes). So every set is also scored on the
  0.01 degree cells at least 1 km from land, applied to detections and GFW
  cells alike; and the S1A passes whose datatake has no train/val frame are
  scored alone. On the UAE runs the strata reproduce the published tables and
  the 1 km rule moves F1 by under 0.01.
- Run: `data/logs/run_compare_iran.sh`, detached, 145 area-passes in about
  2 h 50 min, none failed. 4 S1A passes (Jask 2, Kish 2) had no GFW detection
  in the box and are left out as before; they held 11 model and 18 CFAR
  detections (as false alarms they would lower the model's precision by about
  0.004). The run first stopped at its second pass: the comparison reused one
  GFW client across passes, each of which runs its own event loop, so its
  pooled connection was dead (earlier runs got through on a retry; the
  request never left, so nothing was sent twice); fixed by a client per pass.

Same-cell rule (neighbour-cell F1 in brackets); GFW detections without AIS
in parentheses:

| Set | Passes | GFW detections | Scoring | CFAR P / R / F1 | Model P / R / F1 | Model - CFAR, F1 [95%] |
|---|---|---|---|---|---|---|
| Iran S1A 2024-25 | 79 | 1,184 (823) | standard | 0.533 / 0.914 / 0.674 (0.711) | 0.673 / 0.932 / 0.781 (0.821) | +0.108 [+0.085, +0.137] |
| | | 1,178 | >= 1 km from land | 0.680 / 0.914 / 0.780 (0.823) | 0.693 / 0.931 / 0.795 (0.836) | +0.015 [+0.002, +0.030] |
| Iran S1C/S1D 2026-09 | 62 | 517 (359) | standard | 0.342 / 0.872 / 0.491 (0.532) | 0.499 / 0.905 / 0.644 (0.684) | +0.152 [+0.113, +0.207] |
| | | 503 | >= 1 km from land | 0.494 / 0.871 / 0.631 (0.678) | 0.509 / 0.909 / 0.652 (0.691) | +0.022 [+0.004, +0.047] |
| UAE test S1A 2024-25 | 92 | 3,900 (246) | standard | 0.716 / 0.927 / 0.808 (0.848) | 0.723 / 0.938 / 0.817 (0.854) | +0.009 [+0.002, +0.016] |
| | | 3,900 | >= 1 km from land | 0.722 / 0.927 / 0.812 (0.852) | 0.724 / 0.938 / 0.817 (0.855) | +0.005 [-0.001, +0.013] |
| UAE S1C/S1D 2026-09 | 47 | 1,899 (822) | standard | 0.585 / 0.927 / 0.718 (0.753) | 0.591 / 0.924 / 0.721 (0.758) | +0.003 [-0.006, +0.014] |
| | | 1,893 | >= 1 km from land | 0.590 / 0.928 / 0.722 (0.757) | 0.591 / 0.924 / 0.721 (0.758) | -0.000 [-0.009, +0.011] |

Neighbour-cell differences at >= 1 km: Iran S1A +0.013 [+0.001, +0.028],
Iran S1C/S1D +0.013 [-0.004, +0.035]. S1A passes on datatakes with no
train/val frame (49, mostly the inner-Gulf boxes), >= 1 km: CFAR 0.729,
model 0.774 (+0.045 [+0.019, +0.079]).

Model F1 per box, S1A, >= 1 km (GFW detections): Larak/Hormuz 0.847 (168),
Kish 0.837 (46), Bandar Abbas 0.813 (734), Jask 0.740 (40), Bushehr 0.718
(97), Kharg 0.707 (57), Assaluyeh 0.621 (36; precision 0.48, CFAR 0.28).

**What it says.**
- The model transfers to the Iranian coast with a modest loss. On S1A, away
  from the coast, its F1 is 0.795 against 0.817 in the UAE test boxes (recall
  0.93 in both; precision 0.69 against 0.72), and it stays significantly above
  CFAR. On S1C/S1D it reaches 0.652 against 0.721 in the UAE boxes, mostly
  through precision (0.51 against 0.59), with GFW's thinner S1C/S1D reference
  (PLAN §17, 2026-10-04) applying here too.
- Off Iran most GFW detections carry no AIS (70% on S1A 2024-25, against 6%
  in the UAE test boxes), and the model finds 0.93 of them on S1A (UAE test:
  0.81).
- The model's large lead under the standard scoring (+0.11 to +0.15 F1) comes
  almost entirely from the band 150 m to 1 km from land, where GFW reports
  nothing: of their false alarms against GFW, 440 (CFAR) and 51 (model) lie
  in cells within 1 km of land on the S1A passes (S1C/S1D: 420 and 28). Whether those are coastal clutter or real ships
  moored near shore cannot be judged against GFW; the fair comparison is the
  >= 1 km one, and the console should say that detections within about 1 km
  of land are unmeasured.
- Limits: the bootstrap resamples area-passes, and neighbouring boxes often
  share a pass (20 of 63 S1A scenes and 21 of 41 S1C/S1D scenes serve two
  boxes), so the intervals are somewhat narrow; Bandar Abbas and Larak share
  one GFW cell at a corner; the per-box numbers rest on 9 to 734 GFW
  detections.

**Why precision is lower off Iran (checked 2026-10-06, offline).** Of the
model's false alarms against GFW in cells at least 1 km from land, 349 of
926 (38%) lie at spots (300 m grid) where a false alarm recurs on at least
3 passes of the two sets (CFAR: 346 of 955, 36%); by box Bushehr 57%,
Bandar Abbas 36%, Kharg 35%, Assaluyeh 34%, Kish 38%, Larak 26%, Jask 0%.
The five most persistent spots are detected again in the console's October
2026 passes (scores 0.79-0.99), and their chips show: at Kharg (1.4 km
offshore) a straight structure about 1 km long with bright spots on it, the
tanker jetty; at Assaluyeh (4.7 km out, 7 passes) and Bushehr (11 km out,
13 passes) a compact point echo, next to a bright cross-shaped target (a
large metal structure's sidelobes) at Bushehr. Fixed objects - jetties,
moorings, buoys, platforms missing from the OPD mask - are removed by GFW as
infrastructure but counted as our false alarms. Some recurring spots may
also be ships moored for months (floating storage is common off Iran),
which a persistence-based infrastructure filter also removes: these
objects cannot be told apart from GFW alone. The model's precision off Iran
is therefore a lower bound; a persistence mask built from earlier passes
(as GFW does) would remove most of them, at the risk of hiding long-moored
ships (not built; a plan change).

**In the console (`57f9a12`, the user's choice).** A detection counts as
tested in any of the 12 measured boxes when it lies at least 1 km from land
(each detection now stores its distance to land; the 770 stored rows were
backfilled: 4 of the 377 product detections lie within 1 km), and its card
quotes its region's Sentinel-1C/1D numbers from the >= 1 km scoring (UAE
precision 0.59, recall 0.92; Iran 0.51, 0.91); near shore, with an unknown
distance or outside the boxes, the card says why accuracy is unmeasured.

### 2026-10-06 - Dark-vessel candidates off Iran

The user chose dark-vessel candidates on the Iranian coast next. The newest
Iranian passes (2-4 October) were too young for GFW (presence about 96 h,
SAR detections about 5 days), so each box took its newest S1C/S1D pass at
least 5 days old (25-30 September): the product model on it
(`detect_window.py --scene`), then Phase 6 v1 fusion unchanged
(`fuse_scene.py --scene`: GFW hourly presence, margin 0 m chosen on UAE
validation passes, pre-filters fixed a priori), detached
(`data/logs/run_dark_iran.sh`, 17 min, none failed). One pass often covers
two boxes, and storing a box's events deleted the pass's other events: fixed
first (`7b484e7`, events are now replaced per box).

| Box (pass) | Detections | Kept | Dark | AIS vessels in box | GFW refs (AIS-matched) |
|---|---|---|---|---|---|
| Bandar Abbas (S1D 09-27 02:14) | 54 | 51 | 33 | 21 | 39 (14) |
| Larak/Hormuz (same pass) | 14 | 10 | 7 | 2 | 8 (1) |
| Kish (S1C 09-30 14:24) | 4 | 4 | 1 | 7 | 4 (1) |
| Assaluyeh (S1D 09-29 14:32) | 6 | 6 | 1 | 5 | 5 (4) |
| Bushehr (S1D 09-30 02:38) | 14 | 14 | 12 | 2 | 9 (1) |
| Kharg (same pass) | 15 | 5 | 2 | 8 | 4 (2) |
| Jask (S1C 09-25 14:15) | 2 | 0 | 0 | 15 | 2 (1) |

56 candidates (high 16, medium 28, low 12), each with its image on the
console's card and in the alert feed (checked in the browser). Only Bandar
Abbas had enough GFW AIS-matched detections to check the pass for an AIS gap
(none found); the other six are unguarded in v1's sense.

- GFW's own radar agrees on 32 (a vessel without AIS in the same cell), sees
  nothing in 19 cells and matched 5 to AIS.
- 24 of the 56 lie within 300 m of spots where the model raised a
  GFW-unmatched detection on at least 3 other passes of the validation sets
  (Bandar Abbas 13 of 33, Bushehr 6 of 12, Larak 3 of 7, Assaluyeh 1, Kharg
  1): likely fixed objects or ships moored for months, not vessels that
  switched AIS off. Of these 24, GFW sees nothing in 12 cells (consistent
  with GFW removing a fixed object), reports a vessel without AIS in 9, and
  matched 3 to AIS.
- As expected off Iran, where AIS is sparse, the candidate share is high (56
  of 90 kept detections); it mostly measures missing AIS, and v1 is
  conservative (hourly presence, PLAN §17 Phase 6 v1). A candidate is not a
  verdict.

### 2026-10-06 - Persistent spots: fixed objects and long-moored ships

The user asked for the next steps; the persistence mask came first (a plan
change offered with its risk: it can hide ships moored for months). Decided
before any result: the console marks detections and dark-vessel candidates
at persistent spots and hides nothing (floating storage off Iran is exactly
what an analyst wants to see); the thesis gets a held-out number for what
removing them would do.

- Rule (`core/persistence.py`): a spot is where the product model (score
  >= 0.60) detected something within 150 m on at least 3 passes and on at
  least 30% of its box's archive passes. Archive: the S1A passes before
  2026-07-01 that were compared (UAE validation, test and 2026 H1; Iran
  2024-25): 44 spots (Bandar Abbas 15, Sharjah 7, Bushehr 6, Fujairah 5,
  Kharg 3, Khor Fakkan 3, Assaluyeh 2, Kish 1, Larak 1, Ras al Khaimah 1;
  none at Jask or in the Hormuz lanes) from 9,350 detections on 225
  area-passes (`scripts/build_persistence.py`, `data/reference/persistent_spots.json`).
- Held out in time, on the September 2026 S1C/S1D passes (detections near a
  spot left out for both detectors, GFW references kept, so a hidden moored
  ship GFW reports costs recall; `compare_strata.py --spots`):

| Set, scoring | Model P / R / F1 | Masked | CFAR F1, masked |
|---|---|---|---|
| Iran, >= 1 km from land | 0.509 / 0.909 / 0.652 | 0.523 / 0.909 / 0.664 | 0.631 -> 0.645 |
| Iran, standard | 0.499 / 0.905 / 0.644 | 0.516 / 0.905 / 0.657 | 0.491 -> 0.506 |
| UAE, >= 1 km from land | 0.591 / 0.924 / 0.721 | 0.595 / 0.924 / 0.724 | 0.722 -> 0.724 |

  The mask removes some false alarms at no measured cost in recall
  (neighbour-cell recall off Iran 0.961 -> 0.959), but only about 6% of the
  model's false alarms off Iran, against the ~38% that recur at fixed spots
  within the September and 2024-25 passes together. Most recurring spots of
  2026 are over 1 km from any 2024-25 spot: objects or moorings that
  appeared later, or were seen on fewer than 30% of only 12 archive passes
  per Iranian box. A rolling archive of recent passes (as GFW does) would
  catch more; judging it needs passes after October 2026 (not done; the rule
  was not changed after seeing these results).
- Console (`677e9e8`): `/api/detections` and `/api/events` carry
  `persistent`; the detection and candidate cards say "Persistent spot:
  something was detected within 150 m of here on N of M earlier passes ...
  likely a fixed object ... or a ship moored for months", the feed tags it
  and the legend counts candidates at spots. Under the rule, 9 of 486 stored
  product detections and 1 of the 65 candidates are marked (checked in the
  browser).

### 2026-10-06 - Phase 7: the timeline scrubber

PLAN §10 rule 4 (`9408161`): a bar under the map with one row per measured
box (5 UAE, 7 Iran) and a tick for every stored pass whose footprint holds
the box centre: grey imaged, yellow searched by the product model, red with
dark-vessel candidates. Over the last 60 days the boxes were imaged 6 to 11
times each (stored scenes begin on 2026-08-28; the bar says "stored").
Clicking or dragging sets the moment the console looks from; releasing near
a pass snaps to just after it, the later part of the bar is shaded, and
"Live" returns to now.

- API: `as_of` on `/api/scenes`, `/api/detections`, `/api/analysed` and
  `/api/events` replaces "now" (clamped to now): only passes acquired by
  then count, fused passes too, and every age is measured from it, so the
  map's age styling and the rail's last radar look describe that moment.
  `/api/timeline` serves the bands.
- Checked in the browser: as of 2026-09-27 02:15 UTC the map holds 3
  footprints, 240 detections and the 49 candidates of the Fujairah, Bandar
  Abbas and Larak passes; as of 2026-09-10 02:07 the Fujairah pass's 170
  detections and 9 candidates; Live restores 486 and 65; hiding the bar
  gives the map its height back.
- Limits: AIS (a stale synthetic demo set) does not follow the moment;
  fusion ran days after each pass, so "as of" a pass time shows candidates
  that were only known later (the card gives the pass age and the run).

The 2-4 October Iranian passes are prepared for fusion
(`data/logs/run_dark_iran_newest.sh`, detections and chips already stored):
fusable from 7 October (Kish, Assaluyeh) to 9 October 14:42 UTC (Bushehr,
Kharg); the runner skips passes still too young and can be relaunched.

### 2026-10-06 - Phase 7 closed: the console in Persian, right to left

The user judged the project enough for a bachelor's thesis and asked for
the console in Persian and right to left, then Phase 8. Phase 7 is closed
by that decision (§11 Phase 7, revised).

- Built (`a708724`): every UI text in Persian; `dir="rtl"` with logical CSS
  properties, so the rail sits on the right and cards on the left; dates
  in the Solar Hijri calendar and numbers in Persian digits (the user's
  choice), all times in UTC as the satellite data are, through the
  browser's own Intl (`web/src/fa.ts`, tested with Node's built-in test
  runner: `npm test`); the timeline runs right to left. Scene names,
  vessel names and the Copernicus and Global Fishing Watch attributions
  stay left to right and in English, as their licences word them.
- Checked in the browser: the rail, the timeline with Jalali day marks, a
  searched-area popup and a dark-vessel card (image, Persian distances,
  GFW verdict, nearest AIS vessels).
- Limits: the basemap (Carto) labels places mostly in Latin script; the
  few Persian-script names would need MapLibre's RTL text plugin (a
  third-party script, not loaded); API error messages stay technical
  English.
- Phase 7 as delivered: dark-vessel rings and evidence cards, the alert
  feed, the radar image on every detection and candidate card, the Iranian
  coast with searched areas, per-region accuracy claims (not within 1 km of
  land), persistent-spot marks, the timeline scrubber, Persian RTL. Not
  built (later work): watchlists, geofence drawing, sanctions badges (the
  field exists, no list feeds it), a coverage heatmap, filters beyond layer
  toggles; the non-expert usability test of the original done-when was not
  run.
