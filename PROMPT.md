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
- AOI (full platform): lon 47.5–57.0, lat 23.5–30.5
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
