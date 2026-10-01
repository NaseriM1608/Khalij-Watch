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

### Phase 7 — UI/UX build-out (4–5 days)
Timeline scrubber, coverage honesty overlay, vessel detail panel with SAR chip, filters, watchlists, geofence drawing, alert feed, sanctions badges, RTL support.
**Done when:** a non-expert can find a dark vessel unaided.

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
| same cell | CFAR | 0.710 | 0.918 | **0.800** | 0.931 | **0.775** |
| same cell | model | 0.679 | 0.838 | **0.750** | 0.869 | **0.488** |
| neighbours | CFAR | 0.748 | 0.968 | **0.844** | 0.981 | **0.891** |
| neighbours | model | 0.730 | 0.901 | **0.807** | 0.935 | **0.628** |

**Caveat (found after this entry was written): the two recall columns are
optimistic.** `summarise()` scores the AIS-matched and unmatched subsets each
against all predictions, so one prediction in a cell holding one of each
counts as a hit in both. Precision, recall and F1 come from one joint
assignment and stand. The subset recalls must be recomputed from a joint
assignment before they are quoted; this work was also done under time
pressure and is listed for review in `handoff.md` section 2.0.

By area (neighbours, F1 CFAR / model): Fujairah 0.866 / 0.820, Khor Fakkan
0.819 / 0.792, Sharjah/Dubai 0.842 / 0.836, Ras Al Khaimah 0.834 / 0.727,
Hormuz lanes 0.854 / 0.740 (model recall 0.65 there). **Phase 5's done-when
("beats the CFAR baseline on held-out data") is not met.**

**Why: the label bias of section 8, now measured.** The model was trained to
reproduce CFAR detections that GFW matched to AIS; every detection GFW saw but
could not match was "ignore". It learned AIS-like vessels: on GFW detections
without AIS its recall is 0.49 against CFAR's 0.78 (n = 129, so wide error
bars), and it loses most where non-AIS traffic is common (Hormuz, Ras Al
Khaimah). Both detectors' precision near 0.7 is partly GFW missing small
vessels, which this reference cannot separate from false alarms. The 0.937
F1 on our own labels and this result are consistent: the model is good at
the task it was given, and the task under-specifies dark vessels.

**Next run (proposed):** make the 1,603 training detections GFW saw without
AIS (`gfw_unmatched`) positives instead of ignored; add translation jitter so
positives are not always at the chip centre; keep 128 px inference; select
the checkpoint on this GFW comparison over validation passes, not on our own
labels; re-run this comparison on all 92 test passes.
