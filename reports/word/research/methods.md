# khalij-watch: the methods as the code and run records show them (for rewriting thesis Chapter 3)

Repository root `E:\sea-vessel-detection-3` (HEAD `be9e81c`). Citations are `path:line` relative to the root,
or `file.json → key`. "VERIFIED" means I recomputed the value from data on disk in this session (scripts and
outputs in `scratchpad/research/`: `split_analysis*.py/.out`, `summaries.py/.out`, `cluster_boot.py/.out`).
Nothing in the repository was modified. "NOT FOUND" means nothing on disk establishes the value.

---

## 0. Headline findings (read first)

**Split and leakage**

1. **The split is by product (one Sentinel-1 GRD frame). Each product name is hashed (salted SHA-256), giving 80/10/10.**
   It is not split by datatake, area or date (`core/labeling/manifest.py:18-38`; called at
   `scripts/build_dataset.py:202` and `scripts/mine_negatives.py:196`). Result: 422 products, 329/51/42 (78/12/10%),
   which make 649/103/92 area-passes. All are S1A. No product reaches two splits (VERIFIED).
2. **All five boxes appear in all three splits, interleaved in time over 2024-01 to 2025-12.** The test set therefore
   measures new acquisitions of the same places, not new places (VERIFIED, §5).
   - Each test area-pass is 2.49–9.51 days from the nearest training pass of the same box (median 3.5).
   - 83.2% of test AIS-labelled positives (2,230 of 2,679) are vessels whose GFW vessel id also labels a training chip.
   - 79.2% of test positives lie within 150 m of some training positive in the same box.
3. **The same box at the same acquisition cannot appear in two splits.** `one_per_pass` keeps one frame per pass per box
   (`core/labeling/scenes.py:13-30`); VERIFIED 0 cases.
   - However, 66 of the 181 two-frame datatakes have their two frames in different splits.
   - **73 of the 92 test area-passes share their datatake** (same overpass, same minute) with a train/val frame that
     covered another box. 19 do not. Verified, together with the handoff figure (`handoff.md:185`,
     `compare_test-strata/strata.json → unseen_datatakes.scenes = 19`).
   - The closest pair of cross-split chips inside one datatake is 43.1 km apart (median of the per-datatake minimum
     55.7 km). So no pixels and no ship-at-the-same-instant are shared. This holds by geography; no code checks it.
4. **The bootstrap resamples area-passes, which are not independent.** The 92 test area-passes come from only
   42 scenes and 40 acquisitions; the 47 S1C/S1D area-passes from 24 scenes and 13 acquisitions. I re-ran it by
   acquisition (in memory) and it does **not** change any conclusion:
   - same-cell test +0.0087, interval [+0.0023, +0.0157] (area-pass bootstrap: [+0.0018, +0.0164]);
   - neighbour [−0.0001, +0.0130].
5. **GFW is independent of CFAR but not of the model's training labels.**
   - Positives are CFAR detections whose GFW cell holds exactly one AIS-matched vessel. Run 2 adds exactly one
     GFW-unmatched detection per cell (`core/labeling/gfw.py:159-173`, `core/labeling/index.py:39-48`).
   - Evaluation then scores against the same GFW product in the same boxes. "Recall without AIS" is scored on exactly
     the class run 2 added as positives. Chapter 3 calls GFW "the independent reference" (`03-method.tex:94-96`).
6. **Test was never used to choose run 2's epoch, tiling or threshold** (timestamps and code, §10). However:
   - Run 2's recipe (label-bias fix) was motivated by run 1's result on the same 40 test passes (`PLAN.md:1860-1880`).
   - Test was extended from 40 to 92 passes after the 40-pass result, whose intervals included zero
     (`PLAN.md:1961-1974`). The extension had been proposed on 2026-10-01, before run 2 existed (`PLAN.md:1880`).
   - The selection rule was amended once, after the first validation run was seen (`PLAN.md:1915-1919`).

**Biggest gaps between the thesis text and the code** (full list in §13)

- Chapter 3 never gives the CFAR baseline's parameters, the training hyper-parameters, the augmentations, the input
  normalisation, the evaluation details (GFW query, exclusions, matching algorithm, bootstrap unit) or the
  validation tie-break that actually picked threshold 0.60.
- The chapter 3 split text and the data-table caption say "by image, not random". The split is a pseudo-random hash per
  image.
- Chapter 4 claims the advantage does not come from familiarity with training passes. The cited stratum cannot show
  that, and for the UAE test set it was defined after the result was known.
- Chapter 4 omits the pre-agreed 40-pass interval.
- Chapter 4 and the conclusion call the fusion "consistent with GFW" (balanced agreement only 0.656–0.668).

---

## 1. Data acquisition

| Item | Value | Citation |
|---|---|---|
| Catalogue | CDSE OData `https://catalogue.dataspace.copernicus.eu/odata/v1/Products`, no auth | `core/sar/catalogue.py:36` |
| Search filter | `Collection/Name eq 'SENTINEL-1'` AND `OData.CSC.Intersects(area=geography'SRID=4326;<box WKT>')` AND `ContentDate/Start gt since` [AND `lt until`] AND `contains(Name,'IW_GRDH_1S')`; `$orderby ContentDate/Start desc`, `$top 1000`, nextLink followed ≤20 pages | `core/sar/catalogue.py:176-193, 211-252` |
| Product | IW GRDH, dual-pol 1SDV (VV+VH), COG variant only (`_COG.SAFE`), `Online` and with S3 path | `catalogue.py:43, 218, 248`; `scripts/build_dataset.py:84-92` |
| Pass selection per box | Footprint must contain the box centre (`point_in_polygon`). Then `one_per_pass`: earliest product of each pass; products of the same satellite < 5 min apart = same pass | `scripts/build_dataset.py:84-92`; `core/labeling/scenes.py:10-30` |
| Polarisations used | VV and VH (stack order vv, vh). CFAR uses VH only | `scripts/build_dataset.py:95-105`; `core/inference/contract.py:22`; `build_dataset.py:181` (`bands[1]`) |
| Study boxes (centre lon, lat, side) | fujairah (56.53, 25.22, 20 km); khor_fakkan (56.52, 25.41, 20); sharjah_dubai (55.33, 25.47, 20); ras_al_khaimah (55.88, 26.02, 20); hormuz_lanes (56.55, 26.50, **40**) | `core/areas.py:24-30` |
| Box geometry | Square of side `size_km` on the ground: ±half/110.54 km° lat, ±half/(111.32·cos lat) lon | `core/areas.py:70-74` |
| Iranian boxes (held-out region) | bandar_abbas (56.25, 27.03), larak_hormuz (56.45, 26.85), kish (53.98, 26.53), assaluyeh (52.55, 27.40), bushehr (50.72, 28.95), kharg (50.30, 29.25), jask (57.80, 25.60), all 20 km | `core/areas.py:35-43` |
| Dataset date range | 2024-01-01 to 2026-01-01 (end exclusive). Fujairah was built in two runs (2024, 2025); the others in one run; Hormuz in 8 quarterly chunks | `data/logs/build_runner.log` (START lines); `data/datasets/*/summary_*.json`; `PLAN.md:1652-1660` |
| Passes processed | S1A only in effect: S1C passes of 2025 were skipped as "no GFW detections for this pass". No satellite filter exists in the code. Per area (ok / skipped): fujairah 181 / 42 (41 S1C); khor_fakkan 181 / 42; sharjah_dubai 120 / 28; ras_al_khaimah 180 / 42; hormuz_lanes 182 / 41 → **844 area-passes, 422 products** | `scripts/build_dataset.py:165-171`; VERIFIED from summaries |
| Test-time sets | Validation and test: from index splits (`split_pairs`). Validation: 5 per area, test: 8 per area ("40 passes") or all 92. "Evenly spaced" = `int(i·len/per_area)` in name order | `scripts/compare_cfar_model.py:190-237`; runners `data/logs/run_compare_val_v2.sh`, `run_compare_test_v2.sh` |
| S1A 2026 control | "S1A passes 2026-01..06, 5 per area evenly spaced (spread); written 2026-10-04 by the S1C/S1D check". 25 area-passes, last 2026-06-01; 5 left out (no GFW in box). The generating code is NOT FOUND (rule only in the file header) | `models/runs/kaggle-v2/pairs_s1a_2026h1.txt:1` |
| S1C/S1D set | "every S1C/S1D pass 2026-09-08..09-29 (GFW recalibrated…)", 47 area-passes. Generator NOT FOUND (header only) | `pairs_s1cd_2026-09.txt:1` |
| Iranian sets | `scripts/list_pairs.py`, rule written before results. Passes holding the box centre; satellites; **products (frames) of train/val left out, other frames of the same pass kept**; S1A 2024–25: 12 per box spread (83; Jask 11); S1C/S1D 2026-09-08..09-30 (end excl.): all (62). Header gives per-box counts (e.g. bandar_abbas: 108 frames left out, 13 kept) | `scripts/list_pairs.py:33-57, 105-127`; `pairs_iran_*.txt` headers |
| Window reading | The box edges are sampled at 21 points/edge. They are mapped through a thin-plate-spline GCP transform (`GCPTransformer(tps=True)`), floor/ceil to an axis-aligned pixel window, clamped to the raster. VH is read first, then VV on the same window; the shapes must agree. Read from `/vsis3/` with retry on transport errors only | `core/sar/georef.py:50, 125, 158-179`; `scripts/build_dataset.py:95-105`; `core/sar/reader.py:207-299` |
| Radiometry | **None.** Raw GRD digital numbers (amplitude, uint16). No σ0 calibration, no dB, no thermal-noise removal, **no speckle filter** (deliberate) | `core/sar/windows.py:14-16`; `core/labeling/chips.py:3-8`; `core/sar/cfar.py:244-246` (CFAR squares DN to "intensity") |
| Input normalisation ("per-band sigmoid") | y = 1/(1+exp(−s·(DN−o))) per band. VV (s, o) = (0.0025, 945); VH (0.0072, 343). Solved so the median sea maps to 0.1 and the median ship peak to 0.9 (VV sea 60, peak 1,829 DN; VH sea 36, peak 650 DN). VERIFIED f(60) = 0.099, f(1829) = 0.901 (VV); f(36) = 0.099, f(650) = 0.901 (VH). DN 0 stays 0. Frozen. The sample ("4,000 chips"), its split and the definition of "ship peak" are NOT FOUND on disk | `core/inference/contract.py:7-39`; `PLAN.md:1722-1726` |
| Nodata | DN == 0. Masked if 0 in **any** band (`(bands == 0).any(axis=0)`). Chips with > 10% zeros in any band are skipped. Normalisation keeps 0 → 0. Training noise was added to nodata in runs 1/2 (fixed later; about 0.05% of chips) | `scripts/build_dataset.py:116`; `core/labeling/chips.py:20-21, 66-68`; `contract.py:38`; `git show a373c52` |
| Older sigmoid (not used by the model) | `core/sar/windows.py:163-191` (0.004, 640), VV-only display curve | — |

## 2. Masks

| Item | Value | Citation |
|---|---|---|
| Land source | OSM land polygons `land-polygons-split-4326.zip` (osmdata.openstreetmap.de), downloaded once on 2026-09-11 and clipped to 44.5–61.0 E, 22.0–32.5 N: 2,135 polygons, 478,297 vertices, `data/reference/osm_land_polygons_gulf.npz`. OSM snapshot date NOT recorded (file mtime only) | npz `meta`; `PLAN.md:1436-1448` |
| Rasterisation | Polygons are projected through the scene's TPS GCP transform into window pixels, rasterised, then **square dilation** of ceil(150 m / finer pixel spacing) px. That reaches 150 m along the axes and about 212 m on diagonals. Land is loaded within a 0.25·box span + 0.02° margin | `core/sar/coastline.py:51, 148-194`; `core/sar/morphology.py:13-35`; `scripts/detect_window.py:147-161` |
| Buffer | **150 m** (default; was 500 m in the plan). Chosen by measurement on one Fujairah window: 87 candidates unbuffered → 46 (150 m) → 32 (500 m) | `coastline.py:48-51`; `PLAN.md:1450-1469` |
| Where used | Masks CFAR input and clutter; model peaks on masked pixels dropped; GFW reference cells with masked centre dropped; label chips | `build_dataset.py:108-123`; `runner.py:94-106`; `compare_cfar_model.py:266-281` |
| Platform source | Offshore Platform Dataset v1.0.0 (Spanier et al. 2026, doi:10.5281/zenodo.19046824, CC BY 4.0), `data/reference/OPD_platforms.geojson`: 6,457 features (Persian Gulf 2,314) | `core/sar/infrastructure.py:9-12, 33-39` |
| Detection-mask radius | **250 m** disc per platform (radius in px = 250 / finer pixel spacing). All OPD regions (`region=None`) within box + 0.05° | `infrastructure.py:45, 186-235`; `scripts/detect_window.py:129-144` |
| Lifespan logic | A platform masks a scene if installed-quarter start ≤ acquisition date ≤ last day of its last-observed quarter. Open markers "17Q1 or before" / "25Q1 or still present" mean unbounded on that side | `infrastructure.py:48-49, 55-104, 156-158` |
| Fusion pre-filter radius | 300 m (distance test `covers`), same time-aware filter | `scripts/fuse_passes.py:77, 430-432`; `core/sar/infrastructure.py:179-184` |

## 3. CFAR baseline

| Item | Value | Citation |
|---|---|---|
| Algorithm | Two-parameter-free **cell-averaging CFAR on intensity** (I = DN²) using summed-area tables. Test: I > α·μ_ring | `core/sar/cfar.py:1-37, 236-269` |
| Clutter model | Gamma (multi-look intensity), shape L = **ENL 4.4** (assumed nominal IW GRDH value, not estimated per scene). α solves Q(L, L·α) = Pfa by bisection. **α = 5.045** at Pfa 1e-6 (VERIFIED); the single-look formula would give 13.8 | `cfar.py:61-63, 70-138`; `PLAN.md:1236-1243` |
| Window | Guard half 24 px (49×49 px box, 490 m); training half 40 px (81×81 px, 810 m); ring = 4,160 cells | `cfar.py:57-58, 172-183` |
| Pfa | **1e-6** (labels, comparisons, product) | `cfar.py:59`; `build_dataset.py:350`; `compare_cfar_model.py:346` |
| Polarisation | **VH** only | `build_dataset.py:181`; `compare_cfar_model.py:349`; `core/areas.py:61` |
| Edges / masks | Reflect padding at window edges. Masked pixels are replaced by the median intensity of the window's unmasked pixels before the clutter estimate, then removed from detections | `cfar.py:219-233, 256-268` |
| Morphology | Binary opening (erosion then dilation) with a 5-pixel cross (centre + 4-neighbours) | `core/sar/detect.py:103-127` |
| Clustering | 8-connected components (two-pass union-find) | `detect.py:130-191` |
| Hull | Component pixels ≥ 0.25 × the component's peak amplitude (−12 dB). Falls back to the whole component if < 2 px | `detect.py:65-68, 194-211` |
| Geometry | PCA axes; length/width = projected extent + 1 px, × 10 m; heading mod 180° | `detect.py:214-245` |
| Filters | Component ≥ 3 px; length 15–450 m; aspect 1.3–14 (width floored at 10 m). If the hull fails, the whole component is judged and measured instead | `detect.py:57-63, 248-255, 306-313` |
| Position | Mean row/col of the hull pixels. Georeferenced by the TPS GCP transform. Kept only if inside the lon/lat box | `detect.py:314-326, 329-376`; `build_dataset.py:185` |
| Version tag | `L4.4-g24-t40-p1e-06-h0.25` | `detect.py:258-267`; `core/areas.py:61` |
| Not done | No merging of fragments of one ship (31% of Hormuz detections fell in cells shared by two of ours) | `PLAN.md:1599-1607` |

## 4. Automatic labelling

**GFW query.** The official `gfw-api-python-client` 1.4.0 calls `fourwings.create_sar_presence_report` on dataset
`public-global-sar-presence:latest` (`.venv-eval/Lib/site-packages/gfwapiclient/resources/fourwings/resources.py:404`;
`training/requirements-eval.txt`). Parameters (`core/labeling/gfw_api.py:91-128`):

- `spatial_resolution="HIGH"` (0.01° cells) and `temporal_resolution="HOURLY"`;
- two reports per calendar month (`month_ranges`, `gfw_api.py:27-38`):
  - `filters=["matched='true'"]` with `group_by="VESSEL_ID"`;
  - `filters=["matched='false'"]` with no group-by;
- `geojson` = box polygon;
- 4 attempts, retrying transient errors (`gfw_api.py:23-24, 41-64`).

The served SAR dataset version is **not recorded** (gap).

**Per pass.** The pass uses cells from the hours floor(t_acq − 2 min) and floor(t_acq + 2 min)
(`core/labeling/gfw.py:39, 128-131`). A cell row is placed at its cell centre; the cell key is
(round(lat/0.01), round(lon/0.01)) (`gfw.py:221-223`).

**Matching a CFAR detection to GFW cells** (`gfw.py:134-174`). Rules are applied in this order:

1. Another of our detections shares the cell → `ambiguous` / `crowded_ours`.
2. The detection lies within `EDGE_MARGIN_DEG` = 0.001° (about 100 m) of a cell edge on either axis, diagonal included,
   and GFW has anything (a matched vessel or an unmatched count) in the cell across that edge → `ambiguous` / `edge`
   (`gfw.py:37, 226-247`).
3. Exactly 1 distinct matched vessel (deduplicated by vessel_id/MMSI) and 0 unmatched → `labelled`.
4. 0 matched and exactly 1 unmatched → `gfw_unmatched`.
5. 0 and 0 → `gfw_absent`.
6. Anything else → `ambiguous` / `crowded_gfw`.

Nothing short of `labelled` is ever a negative (`gfw.py:185-218`).

**Supervision in the index** (`core/labeling/index.py:39-59`):

- **positive** = `labelled`; in index v2 (run 2) also `gfw_unmatched`.
- **ignore** = other unlabelled detections (`gfw_absent`, `ambiguous`; in v1 also `gfw_unmatched`).
- **full_negative** = source sea or clutter.
- **centre_negative** = coast, platform or persistent.

**Persistent recast.** Unlabelled detections in about-50 m bins (0.0005°; spot = bin + 8 neighbours) seen in ≥ 30% of an
area's scenes and labelled ≤ 5% of the time become centre negatives. This found 4 spots (185 chips), using
**all splits' scenes** (`index.py:33-36, 128-166`; `scripts/build_index.py:48-58`).

**Chips**

| Item | Value | Citation |
|---|---|---|
| Chips | 128×128 px, bands (VV, VH), raw **uint16**, centred at floor(row+0.5), floor(col+0.5) | `core/labeling/chips.py:17-21, 35-80` |
| Skipped | Outside the window, or > 10% nodata in any band | `chips.py:62-68` |
| Sea negatives (build) | 16 per scene; chip wholly on unmasked pixels; ≥ 100 px (1 km) from every detection in the window **and every GFW cell of the pass**; non-overlapping; per-scene seed | `scripts/build_dataset.py:126-147, 351`; `core/labeling/negatives.py:18-64` |
| Hard negatives (`mine_negatives.py`) | CFAR re-run so all targets are avoided. Per scene: coast edges 24 (10–70% land, ≥ 30 px from targets); bright clutter 32 (chip mean ≥ 1.3× window sea median, peak ≤ 4× chip median, ≥ 100 px from targets); extra sea **56** (`--sea 56` in the runner); every platform standing on the date. Azimuth ghosts not mined (not visible at 10 m) | `scripts/mine_negatives.py:131-167, 253-255`; `core/labeling/hard_negatives.py:19-31`; `data/logs/run_hardneg.sh:36-37`; `PLAN.md:1687-1694` |
| Objects in chips | Other known objects in the same scene, from **every area** (a chip can reach past its box): [Δrow, Δcol, label]. Detections outside a box but inside its read window are not listed (caveat). Neighbours centred just outside a chip are not listed (handoff open item a, ~7% of chips) | `scripts/build_index.py:68-78, 125-131`; `index.py:95-111`; `handoff.md:57` |

**Counts by split** (index v2, `data/datasets/index.jsonl` = the Kaggle index-v2 file, sha256 `4f1808fb…`; VERIFIED):

| Split | Scenes | Area-passes | Positive AIS | Positive gfw_unmatched | Ignore | Full negative | Centre negative |
|---|---|---|---|---|---|---|---|
| train | 329 | 649 | 17,701 | 1,603 | 14,293 | 48,853 | 4,612 |
| val | 51 | 103 | 2,893 | 224 | 2,200 | 7,647 | 655 |
| test | 42 | 92 | 2,679 | 144 | 2,106 | 6,767 | 540 |

- Totals: 112,917 chips. Negatives by source: sea 60,592, coast 5,074, clutter 2,675, platform 548, persistent 185.
- Ignore by status: ambiguous 13,650, gfw_absent 4,949.
- Run 1's index (v1, `data/kaggle/khalij-watch-chips/index_meta.json`) had positive 17,701/2,893/2,679 and ignore
  15,896/2,424/2,250.
- `data/datasets/index_meta.json` says "v1" but holds v2 counts (`handoff.md:60`).

## 5. The split: facts (all VERIFIED with `scratchpad/research/split_analysis*.py`)

- **Unit.** Product (frame) name. `split_for(scene)`: fraction = first 8 bytes of SHA-256("khalij-watch-v1:" + name)/2⁶⁴,
  then train < 0.8 ≤ val < 0.9 ≤ test (`core/labeling/manifest.py:18-38`). Deterministic and pseudo-random. There is no
  seed beyond the salt, and no stratification by area or date.
  - Realised: 329/51/42 products; 649/103/92 area-passes (76.9/12.2/10.9%).
- **One product, one split.** Guaranteed structurally, because both the builder and the miner take the split from the
  product name (`build_dataset.py:202`, `mine_negatives.py:196`). VERIFIED 0 products in > 1 split. No assertion in
  `build_index.py`. Tests check only determinism and fractions (`tests/test_labeling.py:128-136`).
- **Same box in all splits.**

  | Box | Train | Val | Test |
  |---|---|---|---|
  | fujairah | 138 | 22 | 21 |
  | hormuz_lanes | 145 | 21 | 16 |
  | khor_fakkan | 138 | 22 | 21 |
  | ras_al_khaimah | 136 | 26 | 18 |
  | sharjah_dubai | 92 | 12 | 16 |

- **Interleaved in time.** Train 2024-01-02 to 2025-12-30; val 2024-01-05 to 2025-12-30; test 2024-01-10 to 2025-12-06.
  All splits are present in almost every month (test scenes per month 0–5).
- **Distance in time.** From each test area-pass to the nearest training pass of the same box: min 2.49 d, median 3.5 d,
  max 9.51 d (val: 2.49 / 2.49 / 12.0).
- **Same vessels.**
  - Test AIS-labelled positives whose GFW vessel id also labels a training chip: 2,230 of 2,679 (83.2%).
  - Distinct test vessels: 1,707, of which 1,322 also appear in training.
  - Test positives within 150 m of a training positive in the same box (any date): 2,235 of 2,823 (79.2%).
  - Same vessel in a training chip within 3 days and 300 m: 270 of 2,637 test AIS positives.
- **Consecutive frames of one datatake** (satellite + 6-hex datatake id):
  - 241 datatakes: 181 contribute 2 frames, 60 contribute 1.
  - 66 datatakes have frames in > 1 split.
  - One box never gets two frames of one pass (`one_per_pass`; VERIFIED 0 cases).
  - Frames of one datatake that land in different splits always cover different boxes. Nearest cross-split chips
    are 43.1 km apart (median of the per-datatake minimum 55.7 km), so no chip overlap and no detection pair within
    100 m. Fujairah and Khor Fakkan never land in different splits within one datatake (they are 21 km apart).
- **"19 of 92".**
  - Source: `scripts/compare_strata.py:63-76, 158-164` with `--index data/datasets/index.jsonl --splits train,val`, output
    `models/runs/kaggle-v2/compare_test-strata/strata.json → unseen_datatakes.scenes = 19`.
  - Meaning: 73 test area-passes were imaged in the same datatake (same overpass) as some train/val frame of another box.
    The 19 that were not: ras_al_khaimah 7, sharjah_dubai 6, fujairah 2, hormuz_lanes 2, khor_fakkan 2.
  - Test pairs by box (test box ← sibling train/val box): fujairah ← hormuz_lanes 19, ras_al_khaimah 10; hormuz_lanes ←
    fujairah 14, khor_fakkan 14, sharjah_dubai 6, ras_al_khaimah 5; etc. (`split_analysis.out`).
- **Prevention in code.** Same-box duplication is prevented. Cross-split datatake sharing is not prevented, only measured
  afterwards (`compare_strata`).
  - Iranian lists drop train/val **frames**, not datatakes (`list_pairs.py:42-44`). 31 of the 83 Iranian S1A area-passes
    share a datatake with a train/val frame: bandar_abbas 10, jask 10, larak_hormuz 10, kish 1 (VERIFIED).
  - The S1A 2026 and S1C/S1D lists share no datatake with train/val (VERIFIED 0).
- **Other uses of test data in building the training set** (minor, factual):
  - The persistent-spot recast counts scenes from all splits (`build_index.py:54`).
  - The sample used to solve the input sigmoid is unrecorded.
  - The persistence archive for the console mask includes the 92 test area-passes (`persistent_spots.json → results`);
    it was judged only on September 2026.

## 6. Model

| Item | Value | Citation |
|---|---|---|
| Encoder | timm `efficientnet_b0`, `features_only=True`, `in_chans=2`. Channels [16, 24, 40, 112, 320], reductions [2, 4, 8, 16, 32] (VERIFIED locally) | `training/model.py:50-63` |
| Pretrained | `pretrained=True` in both runs (checkpoint config). timm 1.0.26 on Kaggle. The weight tag is not logged; timm's default for `efficientnet_b0` is `timm/efficientnet_b0.ra_in1k` (checked in local timm 1.0.29; inference). First-conv adaptation to 2 channels is done by timm | checkpoint `config.pretrained`; Kaggle log line 31 |
| Decoder | U-Net. 4 blocks; each: bilinear upsample to the skip size, concatenate, 2 × (3×3 conv, BN, ReLU). Channels (128, 64, 48, 32) at strides 16, 8, 4, 2 | `model.py:23-38, 56, 68-73` |
| Output stride | 2 (128 px input → 64×64 output) | `model.py:64`; `core/inference/contract.py:25-27` |
| Heads | Objectness: 3×3 conv (32→32) + ReLU + 1×1 conv → 1 logit. Offset: same shape → 2. Objectness bias init −log(99) = −4.595 (1% prior). No length/width/heading/class heads | `model.py:19-20, 41-44, 74-78` |
| Size | 4,479,935 parameters (VERIFIED); fp32 ONNX 17.9 MB | local; `models/runs/kaggle-v2/run/epoch_05.onnx` size |
| Heatmap target | Fixed Gaussian per vessel, HEAT_RADIUS = 3 output px. σ = (2·3+1)/6 = 1.167 output px (2.33 input px, about 23 m), truncated to a 7×7 square, peak 1 at the cell holding the point. Chip coord p → output (p − 0.5)/2 | `core/labeling/targets.py:28-43, 84-103` |
| Offset target | Sub-cell offset (out − cell, in output px) at the peak cell only; `offset_mask` = 1 there | `targets.py:92-103` |
| Weights | Base 1 for positive / ignore / full_negative chips; base **0** for centre_negative chips, with a disc of radius 8 output px (160 m) set to 1 around the centre and around persistent objects. Ignore / unlabelled objects: disc radius 8 output px (160 m) set to 0. Positives set the 7×7 square to 1, drawn last | `targets.py:29-36, 72-81, 106-112` |
| Bugs in runs 1/2 (fixed after) | (a) Target centre used Python `round()` (half-to-even) while chips use floor(x+0.5): 447 of 25,244 positives (1.8%) were 1 px off. (b) Ignore discs were drawn in list order, so a later negative disc could re-enable an ignored detection. Confirmed by diffing the staged run-2 code `data/kaggle/khalij-watch-index-v2/code/core/labeling/targets.py` against the repo | `git show 0e4eb6a`; `PLAN.md:1956-1959` |
| Loss | Penalty-reduced focal loss (CenterNet), α = 2, β = 4. Positive pixels where heatmap ≥ 1 − 1e-6; per-pixel weight map multiplies. Normalised by Σ(peak·weight) (≥ 1). Plus L1 offset on peak cells / (2·Σmask), weight **1.0**. No class weighting besides weight maps and sampler | `training/losses.py:14-47`; `training/train.py:106-116`; checkpoint `config.offset_weight=1.0` |
| Input | 2×128×128 normalised (per-band sigmoid); run 2 trains on 2×96×96 random crops | `training/data.py:112-142` |

## 7. Training (from `last.pt`/`best.pt` configs, read with `.venv-train`, and the Kaggle logs)

| Item | Run 1 (`kaggle-v1`) | Run 2 (`kaggle-v2`) | Citation |
|---|---|---|---|
| Index | `khalij-watch-chips/index.jsonl` (v1; gfw_unmatched = ignore) | `khalij-watch-index-v2/index.jsonl` (gfw_unmatched = positive) | checkpoint `config.index` |
| Encoder / pretrained | efficientnet_b0 / ImageNet | same | config |
| Optimiser | AdamW, weight decay 1e-4 (decoupled), eps 1e-8, β2 0.999 | same | `train.py:221-223`; optimizer state |
| LR schedule | OneCycleLR, max_lr 1e-3, PyTorch defaults: pct_start 0.3, cosine anneal, div_factor 25 (start 4e-5), final 4e-9; β1 cycled 0.95 → 0.85 → 0.95; total 9,375 steps (stepped per batch) | same | `train.py:224-225`; scheduler/optimizer state in `last.pt` |
| Epochs × samples | 15 × 40,000 (`WeightedRandomSampler`, with replacement) | same | config |
| Batch | 64 (`drop_last`), 625 steps per epoch | same | config |
| Sampling weights | positive 1.0, centre_negative 1.0, full_negative 1.0 except source sea **0.15**, ignore **0.1**. Expected draw shares (train): positive 53.4%, sea 21.1%, coast 12.1%, clutter 6.7%, ignore 4.8%, platform 1.3%, persistent 0.4% | positive 55.8%, sea 20.2%, coast 11.6%, clutter 6.5%, ignore 4.1%, platform 1.3%, persistent 0.4% | `training/data.py:25-31, 50-53`; VERIFIED |
| Augmentation | Per-band gain U(0.8, 1.25) on raw DN; Gaussian noise σ 0.01 after normalisation (also on nodata, pre-fix); independent H and V flips p = 0.5 (offset sign flipped). No rotation, scale or elastic | same + **random 96 px crop** at even offsets, after flips | `data.py:33-34, 56-83, 112-142`; staged code diff |
| Precision / hardware | AMP autocast fp16 + GradScaler; Kaggle Tesla T4 (P100 requested); torch 2.10.0+cu128, timm 1.0.26; 4 workers | same | `train.py:226, 243`; log lines 27/31 |
| Time per epoch (incl. val) | 478–503 s | 507–535 s | logs |
| Seeds | seed 0 (`torch.manual_seed`, sampler generator 0, augmentation key (seed, index, per-worker draw counter); no worker id in the pre-fix code). cuDNN determinism not set → not bit-reproducible | same | `train.py:209, 79-84`; staged diff |
| Validation during training | All val chips, whole 128 px, F1 within 10 px (100 m) over thresholds {0.01, 0.02, 0.03, 0.05, 0.07, 0.10 … 0.90}; predictions on weight-0 ground ignored | same | `train.py:34-36, 119-176` |
| Saved | `best.pt` (by own-label val F1: epoch 6), `last.pt` (epoch 14) | `epoch_00`–`epoch_14.pt`, `best.pt` (= epoch 14), `last.pt` | `models/runs/*/run/` |
| Chosen weights | Epoch 6 (zero-based), threshold 0.45, both chosen on own-label val F1 | Epoch **5** (zero-based = 6th epoch), chosen against GFW on 25 validation passes | `PLAN.md:1795-1798`; `val_choice.json` |
| Own-label val F1 | 0.450 … 0.910 (epoch 6) … 0.810 | 0.569 … 0.9725 (epoch 5) … 0.9754 (epoch 14) | logs |

Differences between run 1 and run 2: the index (gfw_unmatched as positives, +1,603 training positives), `crop=96` and
`keep_epochs=True`. Everything else is the same config. The code is the same apart from the crop/keep-epochs feature
(commit `c590265`). The Kaggle notebooks are `notebooks/kaggle*/train_point_detector*.ipynb`.

## 8. Inference

| Item | Value | Citation |
|---|---|---|
| Input | The whole window (VV, VH), normalised. **Masks are not applied to the input** (the network sees land); peaks on masked pixels are dropped afterwards | `core/inference/runner.py:131-137`; `compare_cfar_model.py:358-363` |
| Tiling | Tile 256, overlap 64 (step 192). The last tile is aligned to an even start, (size − tile + 1)//2·2. Each tile is zero-padded to a multiple of 32. Both tile and step must be even | `runner.py:26-28, 46-58, 61-91` |
| Stitching | Per output cell, the **highest probability across overlapping tiles wins**; the offset is taken from the same tile (no averaging) | `runner.py:84-90` |
| Peaks | 3×3 local maxima (`heat ≥ max of 3×3`, −inf padding; ties all kept) with heat ≥ threshold | `core/inference/decode.py:18-45` |
| Sub-pixel | Position = (i + offset)·2 + 0.5 input px, then georeferenced by TPS | `decode.py:39-41`; `scripts/detect_window.py:362-372` |
| Threshold | **0.60** (product). Comparisons store peaks ≥ 0.05 and apply thresholds at summary time | `runner.py:25-26`; `compare_cfar_model.py:54` |
| NMS | None beyond the 3×3 max. Near-duplicates remain (validation: 2.8% at the chosen setting, 8.4% at 128/32, CFAR 1.3%, counted once per pair — VERIFIED from `compare_val*/` as half of the "detection has a neighbour ≤ 100 m" share: 5.55% / 16.5% / 2.66%). The definition is not recorded on disk | `handoff.md:79` |
| ONNX | opset 17, dynamic batch/H/W, `torch.onnx.export(dynamo=False)`; onnxruntime 1.30.0, CPUExecutionProvider | `training/export.py:24, 35-52`; `requirements-eval.txt` |
| fp32 vs INT8 | Dynamic quantisation (weights QInt8) rejected. Run 1: logits off by up to 8.8, 64 s vs 5.4 s per 2048 px tile. Run 2 (Kaggle export of `best.pt` = epoch 14): logit difference 74.2. fp32 parity 3.8e-6 (Kaggle). Product `epoch_05.onnx` exported locally; its parity (≤ 4.3e-6 per handoff) is not stored on disk | `models/runs/kaggle-v1/local_best.export.json`; `models/runs/kaggle-v2/export.json`; `handoff.md:76` |
| Product file | `models/point_detector_kaggle-v2.onnx` = `run/epoch_05.onnx`, sha256 `0535c82ddaa577f1…` (VERIFIED identical) | — |

## 9. Evaluation (`scripts/compare_cfar_model.py`)

- **Reference**
  - GFW SAR presence cells (same query as §4): `fetch_sar_cells` for UTC days
    [date(t − 2 min), date(t + 2 min) + 1) (`:240-244`), then `cells_for_pass` for hours floor(t ± 2 min).
  - Each cell row is kept as (lon, lat, detections, matched) if its centre is inside the box and not on a masked pixel
    (`:266-281, 371-373`).
  - Cells: 0.01° (`:53`).
  - Masked reference detections dropped: 3 of 3,903 (test); 9 (Iran S1A); 17 (Iran S1C/S1D) (VERIFIED).
- **Detections scored**
  - CFAR (VH, Pfa 1e-6, §3) candidates whose centroid is unmasked, georeferenced, inside the box (`:346-356`).
  - Model peaks ≥ store threshold, unmasked, inside the box (`:284-308`).
  - The **same mask** (nodata ∪ platform 250 m ∪ coast + 150 m) applies to both detectors and the reference
    (`:343, 247-253`). Input treatment differs, though: CFAR's clutter estimate excludes masked pixels; the model sees
    them.
- **Matching** (`match_cells`, `:143-175`)
  - Same-cell: per cell, hits = min(#predictions, #references).
  - Neighbour rule: then, for each prediction cell with leftovers (in order of first appearance), references are taken
    greedily from the 3×3 neighbourhood in fixed scan order (−1,−1) … (+1,+1). Not nearest-first, not optimal; optimal
    matching adds 1–3 hits per ~1,600 (`PLAN.md:1852-1853`).
- **Metrics** (`:491-518`, pooled micro over passes)
  - TP = Σhits; FP = unmatched predictions; FN = Σrefs − TP.
  - P = TP/(TP+FP); R = TP/Σrefs; F1 = 2PR/(P+R).
  - Recall with / without AIS: in mixed cells hits are shared in proportion to counts, with min/max bounds recorded
    (`:446-465`). The joint assignment fixes the earlier double-counting bug (`PLAN.md:1842-1853`).
- **Passes left out.** `gfw_in_box == 0` (no GFW detection in the box before masking: "GFW did not cover the pass").
  Their detection counts are recorded (`:576-582, 609-621`):
  - val 0, test 0;
  - S1A 2026: 5 (acquisitions of 2026-04-21 and 2026-05-27; e.g. the Fujairah 2026-05-27 pass had 85 CFAR / 88 model
    detections);
  - S1C/S1D 0;
  - Iran S1A 4 (Jask 2, Kish 2; 18 CFAR / 11 model detections);
  - Iran S1C/S1D 0.
- **Bootstrap** (`paired_bootstrap`, `:57-58, 543-564`)
  - Paired percentile bootstrap of model − CFAR.
  - Resampling unit = **area-pass result file**, with replacement, n = number of passes.
  - **10,000** resamples, `np.random.default_rng(0)`; the same draws for both detectors; metrics recomputed on the
    pooled resampled counts; 95% CI = 2.5/97.5 percentiles.
  - Also reported: recall AIS-matched/unmatched differences and each detector's own CI.
  - Units behind the area-passes (VERIFIED):

    | Set | Area-passes | Scenes | Acquisitions |
    |---|---|---|---|
    | test | 92 | 42 | 40 |
    | val | 25 | 18 | 18 |
    | S1C/S1D | 47 | 24 | 13 |
    | Iran S1A | 79 | 59 | 58 |
    | Iran S1C/S1D | 62 | 41 | 31 |

  - Cluster (acquisition) bootstrap, computed here: same conclusions everywhere (`cluster_boot.out`).
- **Strata** (`scripts/compare_strata.py:3-18`, fixed 2026-10-05 "before the Iranian-coast results were read")
  - `shore_1000m`: keep detections and references whose **0.01° cell centre** is ≥ 1,000 m from OSM land (unbuffered
    polygons, land loaded within 0.05°) (`:59-98`).
  - `unseen_datatakes`: passes whose (satellite, datatake id) has no frame in index train ∪ val (`:63-76, 158-164`).
  - `persistent_masked`: detections of both detectors within 150 m of a spot dropped; references kept (`:101-114`).
  - Every stratum is scored by the comparison's own `summarise`.
- **Verified headline numbers**
  - Test 92, same-cell F1: CFAR 0.808, run 1 0.763, run 2 0.817; run 2 − CFAR +0.0087 [+0.0018, +0.0164].
  - Test 92, neighbour: +0.0063 [**−0.0003**, +0.0134].
  - 40 passes: +0.0117 [**−0.0003**, +0.0256] same-cell; +0.0086 [−0.0030, +0.0218] neighbour.
  - Unseen-datatake test (19): +0.0162 [+0.0043, +0.0315].
  - All match `handoff.md` §3 (`summaries.out`).

## 10. Model selection

- **Validation passes.** `--split val --per-area 5` gives 25 area-passes, 5 per box (18 scenes)
  (`data/logs/run_compare_val_v2.sh`).
- **Configurations swept.** Thresholds 0.05–0.90 step 0.05 (18) (`compare_cfar_model.py:56`). Model configurations:
  - `compare_val/` (9): run 1 `local_best.onnx` @128/32 (epoch 6); run 2 e05/e08/e11/e14 @128/32; e14 @128/64,
    @96/32, @256/32, @256/64.
  - `compare_val_t256/` (4): e05/e08/e11/e14 @256/64.
  - That makes 12 distinct (weights, tiling) configurations × 18 thresholds; run 1 was in the pool
    (`compare_val/summary.json → models`; `compare_val_t256/summary.json → models`).
  - Epochs 5/8/11/14 were the ONNX exports; other epochs were not tried.
- **Rule as executed** (`data/logs/select_val_config.py:1-16, 27, 63-80`, uncommitted; `val_choice.json → rule`):
  1. Maximise the mean of same-cell and neighbour F1.
  2. Among candidates within **0.005** of the best, take the highest neighbour recall on GFW-unmatched detections.
  3. Then fewer model seconds, then higher mean F1.
  - The script refuses unless both validation runs hold the same 25 passes, the same CFAR result and identical numbers
    for the shared configuration.
- **Outcome.**
  - Best mean F1 was e05 @256/64 at threshold **0.65** (0.8478).
  - The tied set was {0.65, 0.70, 0.60, 0.75}; **0.60** won on unmatched recall (0.868).
  - Chosen: same-cell 0.8296, neighbour 0.8616, mean 0.8456; CFAR on validation 0.8074 / 0.8512 (`val_choice.json`).
- **"Amended once before test"** (`PLAN.md:1915-1922`; select script docstring).
  - First rule: neighbour F1 only. It "picked a threshold at which the model was below CFAR on the same-cell rule".
  - Amended to the mean of the two rules, to match Phase 5's "better under both rules", **after the first validation
    run (`compare_val/`) had been seen** and before any run-2 test number.
  - Timing: `compare_val/summary.json` mtime 2026-10-04 07:14 +03:30 (03:44 UTC); amendment and the 256/64 validation
    run followed.
- **Test not used for selection.**
  - `val_choice.json` written 2026-10-04 06:17:53 UTC.
  - Test runner started 06:17:54 UTC (`data/logs/compare_test_v2_runner.log`); first test result file 06:19:05 UTC
    (mtime 09:49:05 +03:30).
  - The test runner reads the choice from `val_choice.json` (`run_compare_test_v2.sh`).
  - Test summaries print the threshold sweep "for information only" (best on test 0.70/0.75, unused).
- **Caveats** (facts, not selection on test):
  - Run 1's test result on the same 40 passes (2026-10-01) motivated run 2's label change (`PLAN.md:1860-1880`).
  - The extension from 40 to 92 test passes was run after the 40-pass result (proposed 10-01, `PLAN.md:1880`;
    executed 10-04, `PLAN.md:1961-1974`).

## 11. Fusion (Phase 6 v1, `core/fusion/`, `scripts/fuse_passes.py`)

| Item | Value | Citation |
|---|---|---|
| Detections | The frozen product model's comparison peaks (sha `0535c82ddaa577f1`, 256/64, ≥ 0.60), already masked and in-box | `fuse_passes.py:65-68, 113-128` |
| Presence query | `fourwings.create_ais_presence_report`, dataset `public-global-presence:latest` (served v4.0 recorded per pass). HIGH, HOURLY, `group_by=VESSEL_ID`. Hours H−1, H, H+1 (start H−1, end H+2 exclusive, '.000Z' bounds). Box buffered by **25 km**. One report per pass, cached | `core/fusion/presence.py:31-32, 36-51, 162-167, 203-229`; `fuse_passes.py:72, 349-368`; record `presence_datasets` |
| Tracks | One per vessel: first fix per hour, ordered. Two-fix tracks (H−1, H) or (H, H+1) are extended along their direction by the pass's fraction of the hour. Speed = fastest cell-to-cell displacement / elapsed time; single-fix = sparse (no speed); stationary = same cell in all 3 hours | `presence.py:71-134, 170-187` |
| Gate | Distance from the detection to the track polyline ≤ cell half-diagonal (0.5·√((0.01·110,540)² + (0.01·111,320·cos lat)²), about 0.74–0.75 km at 25–29° N) + **R/V × speed** + margin. R/V = **118 s** (constant upper bound, not per sub-swath). Sparse tracks get no allowance | `core/fusion/gate.py:49-89`; `fuse_passes.py:78-81` |
| Assignment | `scipy.optimize.linear_sum_assignment` on cost = distance/gate (in-gate), 1e6 outside (pairs dropped). **MATCHED** only if the detection has exactly one candidate, that vessel is claimed by no other detection, and it was assigned. Otherwise inside a gate → **AMBIGUOUS**. No candidate → **DARK** | `core/fusion/assign.py:29, 48-94` |
| Pre-filters (first reason wins, in this order) | Duplicates ≤ **100 m** (strongest first, non-chaining) → shore ≤ **1,000 m** (OSM land, unbuffered; land loaded 20 km around) → infrastructure ≤ **300 m** (OPD, time-aware) → box edge ≤ **500 m**. Fixed a priori, never tuned. Not built: azimuth-ghost and footprint no-data edge filters | `core/fusion/prefilter.py:41, 73-163`; `fuse_passes.py:73-84`; `PLAN.md:2373-2379` |
| Margin sweep | 0, 250, 500, 1000, 2000, 5000, 10000 m | `fuse_passes.py:70` |
| Margin choice | On validation records only (`compare_val_t256`): the margin with the highest **balanced accuracy** = mean(GFW-unmatched called dark, GFW-matched called AIS[matched or ambiguous]). Counted over kept detections whose cell has a single GFW flag, on passes that are neither GFW artefacts nor presence gaps. Ties → smaller margin. Result **0 m** (0.656; 250 m 0.613, 500 m 0.615, 2 km 0.524, 10 km 0.500). It sits at the **lower edge** of the grid, and the agreement falls monotonically. It rests on 42 GFW-unmatched (14 dark + 28 AIS) and 993 GFW-matched detections | `fuse_passes.py:213-219, 256-259, 305-313`; `fusion_v1/summary.json` (VERIFIED) |
| Dark rate | Σdark / Σkept over usable area-passes. Bootstrap by **acquisition** (satellite + datatake), 10,000 resamples, seed 0, percentile. All-zero or all-dark → one-sided Clopper–Pearson 97.5% bound | `core/fusion/rate.py:17-63`; `fuse_passes.py:222-225, 262-274` |
| GFW artefact guard | refs ≥ 10 AND matched = 0 AND box presence vessels (pass hour, unbuffered box) ≥ 20 → pass left out of the GFW agreement; on cards the GFW cell state is "untrusted" | `core/fusion/guards.py:19-20, 51-59`; `events.py:115-121` |
| Presence-gap guard | expected = GFW-matched detections (all detections if artefact). If expected ≥ 10 and box presence vessels < 0.5 × expected → gap. A gap in any box flags every box of that acquisition; those passes are left out of the rate. Passes with expected < 10 = "unguarded" and are kept (test 8/92, S1C/S1D 18/42) | `guards.py:27-28, 62-78`; `fuse_passes.py:228-253, 277-302` |
| Confidence tiers | dark_up_to = largest swept margin m with m < (distance to nearest track − its margin-0 gate). **high ≥ 2,000 m, medium ≥ 500 m**, else low | `core/fusion/evidence.py:26-28, 36-70` |
| Evidence card fields | status, scene, acquired; detection {lon, lat, score}; detector {name, version}; gate {margin_m, r_over_v_s, nearest_gap_m}; dark_up_to_m; confidence; gfw {cell_matched, corroborates, cell_state}; prefilter {shore_m, edge_m}; nearest_ais (3 nearest tracks: vessel_id, mmsi, imo, ship_name, flag, vessel_type, distance_m, stationary, sparse, cells, track); pass_flags {gfw_artefact, presence_gap, no_presence, presence_checked, gfw_refs}; datasets; chip {scene, lon, lat} (later chip.image); attribution | `evidence.py:103-151`; `core/fusion/events.py:66-74` |
| Product path | `fuse_scene.py`: pass must be ≥ 5 days old; stored detections (`auto`: model rows of kaggle-v2 where it ran, else CFAR); GFW SAR references for the pass day; **no events** if presence gap or empty presence | `scripts/fuse_scene.py:43-88, 100-109`; `events.py:101-102` |
| Periods / regions | before 2026-02-28; 2026-02-28 to 06-29; 2026-06-30 to 09-07; from 09-08. inner_gulf = sharjah_dubai, ras_al_khaimah; hormuz = hormuz_lanes; gulf_of_oman = fujairah, khor_fakkan | `fuse_passes.py:92-105` |
| Verified results | Margin 0. Validation 51/1,288 = 4.0% [2.6, 5.8], BA 0.656. Test 192/4,747 = 4.0% [3.2, 5.3], BA 0.668 (dark on GFW-unmatched 70/196, dark on GFW-matched 74/3,600). S1A 2026 27.8% [4.7, 61.3]. S1C/S1D 7.9% [6.3, 9.7] | `fusion_v1/summary.json` |

## 12. Persistent spots

- **Rule** (fixed before results). A spot is where product-model detections (≥ 0.60) from ≥ **3** distinct passes lie
  within **150 m** of a seed detection, **and** those passes are ≥ **30%** of the box's archive passes.
  - Seeds are processed in order of most passes; each seed takes all untaken detections within 150 m; the spot is their
    mean position (`core/persistence.py:25-27, 45-97, 127-128`).
  - Marking/masking test: haversine ≤ 150 m to the nearest spot (`persistence.py:100-111`).
- **Archive** (`scripts/build_persistence.py:32-33, 51-67`; `data/reference/persistent_spots.json → results, passes`).
  - Comparison results acquired before **2026-07-01** holding the product model: `compare_val_t256` (25),
    `compare_test` (92), `compare_s1a_2026h1` (25), `compare_iran_s1a_2024-25` (83). That is 225 area-passes and all
    result files, including passes without GFW. In practice the last pass is 2026-06-01.
  - Passes per box: fujairah 31, khor_fakkan 31, ras_al_khaimah 28, hormuz_lanes 26, sharjah_dubai 26, each Iranian
    box 12 (Jask 11).
  - Result: 44 spots.
- **Use.** Console marks, never hides. Evaluation stratum `persistent_masked` on the September 2026 S1C/S1D sets
  (detections near spots dropped for both detectors, references kept) (`compare_strata.py:14-18, 101-114, 165-168`).

---

## 13. Discrepancies between the thesis text and the code / records

`03` = `reports/thesis/chapters/03-method.tex`, `04` = `04-results.tex`, tables in `reports/tables/`.

1. **Split described as "not random"** (`03:63-65`; `data.tex` caption).
   - Fact: a pseudo-random hash per product; no grouping by datatake, area or time; the same 5 boxes in all splits,
     interleaved in time (§5).
   - The chip-leakage claim is right. The text never says the test set is re-visits of the same places: 83% of test
     AIS-labelled vessels and 79% of test positive locations also appear in training; test passes are 2.5–9.5 days
     from a training pass of the same box.
2. **"So the advantage does not come from familiarity with the training passes"** (`04:57-59`).
   - The `unseen_datatakes` stratum removes only acquisitions shared with train/val. All 19 passes are still the same
     boxes, ships and anchorages.
   - For the UAE test set the stratum was defined on 2026-10-05, after the test results of 2026-10-04 (post hoc).
     `03:103` says both extra strata were defined before results; true only for the Iranian sets.
3. **Selection rule** (`03:88-90`): given as "maximum mean F1 of the two rules, agreed before any test number".
   - Missing: the amendment (neighbour-only, changed after seeing the first validation run).
   - Missing: the ≤ 0.005 tie-break by unmatched recall, which is what chose 0.60 over the best-mean 0.65. The
     `validation.tex` caption mentions the tie-break; Chapter 3 does not.
   - Also missing: the candidate pool (12 configurations incl. run 1, 18 thresholds; other tilings only for epoch 14),
     the 64 px overlap in `04:13`, and that "25 passes" are area-passes (18 scenes), 5 per box, evenly spaced.
4. **GFW called "the independent reference"** (`03:94-96`). It is independent of CFAR, but the model's positives were
   defined by the same GFW product in the same boxes. Run 2's extra positives are exactly the "GFW without AIS" class
   whose recall is then reported (§4, §0 item 5). This should be stated as a limitation that favours the model.
5. **CFAR baseline not described** anywhere in Chapter 3 (Chapter 2 `02-background.tex:29-36` is generic). Missing:
   - CA-CFAR on DN², Gamma ENL 4.4 (α = 5.05), guard 49 px / outer 81 px, Pfa 1e-6, VH only, masked pixels → median;
   - opening (5-px cross), 8-connected components, ≥ 3 px, length 15–450 m, aspect 1.3–14;
   - hull at −12 dB, hull-centroid position.

   The same CFAR generates the labels.
6. **Data section** (`03:44-52`). Missing:
   - S1A only for train/val/test; S1C passes of 2025 skipped for lack of GFW;
   - raw uncalibrated DN (no σ0/dB, no speckle filter); per-band sigmoid normalisation and its values;
   - nodata rule;
   - platform mask radius **250 m** (fusion uses 300 m; the chapter states only the 300 m).
7. **Labelling** (`03:56-65`). Missing:
   - GFW dataset/endpoint and the hour window (floor(t ± 2 min));
   - edge margin 0.001° (≈ 100 m), rule precedence, `gfw_absent` → ignore;
   - hard negatives (coast 24, clutter 32, platforms, extra sea 56 per scene) — only "empty sea" is mentioned;
   - negatives kept 1 km from all detections and GFW cells;
   - uint16 raw chips;
   - persistent recast (185 chips);
   - the open `chip_objects` margin issue (~7% of chips may train a neighbour's hull as sea).
8. **Training** (`03:77-83`). Missing:
   - AdamW (wd 1e-4) and the OneCycle details (cosine, 30% warm-up, 4e-5 → 1e-3 → 4e-9);
   - fp16 mixed precision;
   - sampler weights (ignore 0.1, easy sea 0.15 → positives ≈ 56% of draws);
   - augmentations (gain 0.8–1.25, noise σ 0.01, H/V flips; no rotation);
   - focal α = 2, β = 4; Gaussian σ 1.17 output px, 7×7; ignore radius 160 m; offset L1 weight 1;
   - seed 0, no determinism;
   - that "epoch 5" is zero-based (the 6th of 15 passes over 40,000 samples);
   - that runs 1/2 contain the two label bugs (1.8% positives 1 px off; ignore discs overridable) and noise on nodata;
   - own-label validation during training used whole 128 px chips while run 2 trained on 96 px crops.
9. **Inference** (`03:86-88`): "maps stitched" is vague. Missing:
   - per-cell **maximum** over overlapping tiles, offset from the winning tile;
   - zero padding to multiples of 32;
   - 3×3 local maxima ≥ 0.60 with sub-pixel offsets;
   - masks applied after the network (land is network input);
   - no further NMS (2.8% near-duplicates remain).
10. **Evaluation** (`03:94-107`). Missing:
    - GFW 4Wings SAR presence, HIGH/HOURLY, hours floor(t ± 2 min);
    - detections and reference cell centres restricted to the box;
    - reference cells with masked centres dropped;
    - **passes with no GFW detection excluded** (S1A 2026: 5; Iran S1A: 4);
    - same-cell hits = min(counts) per cell; neighbour pass greedy in fixed scan order (not optimal);
    - subset recalls shared proportionally in mixed cells;
    - bootstrap unit = area-pass (not scene or acquisition), 10,000 resamples, seed 0, percentile;
    - the correlation of boxes sharing one scene (92 test area-passes = 42 scenes / 40 acquisitions; the
      acquisition-level bootstrap gives the same conclusions, §0 item 4);
    - the shore stratum judges by **cell centre** distance to unbuffered OSM land.
11. **Pre-agreed 40-pass result not reported with its interval** (`04:44-56`; `test40.tex` has no CI).
    - Run 2 − CFAR on the pre-agreed 40 passes: +0.012 [−0.000, +0.026] same-cell; +0.009 [−0.003, +0.022] neighbour.
      Both include 0.
    - Significance appears only after the extension to 92 passes, done after the 40-pass result.
    - Intro (`01-intro.tex:45-46`) and conclusion (`05-conclusion.tex:14-15`, "small but significant") omit this.
    - `\testDiffNbCi` prints the lower bound as "0.000"; it is −0.0003 (`compare_test/summary.json`).
12. **0.85 → 0.68** (`04:12-13`): neighbour-rule unmatched recall at 128/32 tiles, threshold 0.45 (`compare_val/` e05
    vs e14). Same-cell is 0.78 → 0.59. Neither the rule nor the tiling is stated, and the adjacent figure plots
    same-cell values.
13. **Per-area claims** (`04:89-92`, "clearly better in Hormuz and RAK"): no interval; 16–18 area-passes per box
    (`areas_uae.tex`).
14. **Fusion "consistent with GFW's matching"** (`04:199-201`; `05-conclusion.tex:21-22`). Balanced agreement is only
    0.656 (validation) / 0.668 (test), and 74 of the 144 dark calls on GFW detections are GFW-matched. `PLAN.md:2400-2407`
    calls v1 "conservative" and "not a count of dark ships".
15. **Margin and tiers** (`03:119-121`). The 0 m margin is the smallest value in the swept grid (boundary optimum,
    monotone curve), chosen on only 42 GFW-unmatched validation detections. The medium tier (≥ 500 m) and the margin
    grid are not stated.
16. **Gate description** (`03:113-116`): "the vessel's presence cells, widened by half a cell diagonal" is not the
    algorithm. Actual:
    - distance to the polyline through the H−1/H/H+1 cell centres, with two-fix tracks extended by the pass fraction;
    - compared with half-diagonal + 118 s × fastest hourly cell speed (no allowance for single-fix tracks) + margin;
    - presence fetched over box + 25 km;
    - MATCHED / AMBIGUOUS / DARK definitions;
    - presence-gap rule (box vessels < 0.5 × GFW-matched, judged only if ≥ 10) and artefact rule (≥ 10 refs, 0 matched,
      ≥ 20 vessels);
    - rate bootstrap by acquisition (whereas the detector comparison bootstraps area-passes).
17. **Iranian "never trained, never chosen"** (`03:7-10`) is true for pixels. Not mentioned: 31 of 83 S1A Iranian
    area-passes are datatake siblings of UAE train/val frames; Larak/Hormuz and Bandar Abbas lie about 10–60 km from
    the Hormuz-lanes training box (same traffic). The Iranian unseen-datatake result (49 passes, ≥ 1 km: +0.045
    [+0.019, +0.079]) is in `strata.json` but not in Chapter 4.
18. **Efficiency** (`04:287-290`). The 8.4 s / 209 MB benchmark is a synthetic window: Gamma L = 4.4 clutter at VV 60 /
    VH 30 DN with 20 targets, model only (normalise + tile + decode), 3 repeats, no CDSE read or masking
    (`scripts/bench_inference.py:29-33, 90-121`; `bench.json`). The "14.4 s" mean includes the 40 km Hormuz boxes
    (`make_reports.py:524`).
19. **Terminology.** "گذر" (pass) is used for area-passes throughout Chapters 3–4 (25 / 92 / 47 …). One acquisition
    often counts in 2–3 boxes. Define «گذر-محدوده» consistently (it is used only in the persistence paragraph,
    `04:187`).
20. **PLAN §12 protocol items dropped without being declared as deviations** (`PLAN.md:470-483`):
    - xView3 aggregate, length-stratified metrics, false alarms per 100 km²;
    - ablations (stride, normalisation, VV-only, …) and the FP32 vs INT8 table;
    - the hand-labelled test set (`PLAN.md:348`).

    The thesis should say they were not done and why (no length labels, no hand labels by choice).

## 14. Reproducibility gaps (parameters that matter but are not recorded, or are recorded only outside version control)

1. **Training code of runs 1/2 differs from the repository** (pre-`0e4eb6a` / `a373c52`: round-half-even target centre,
   disc order, noise on nodata, no worker-id seeding).
   - Copies exist only in git-ignored staging folders `data/kaggle/khalij-watch-index-v2/code` (run 2) and
     `data/kaggle/khalij-watch-chips/code` (run 1).
   - Checkpoints store the config but no git hash.
2. **Pretrained weight tag** not logged (timm default assumed).
3. **Non-determinism.** No cuDNN determinism, fp16 AMP, 4 workers with per-worker draw counters. Re-training will not
   reproduce the weights bit for bit.
4. **Input sigmoid calibration.** The 4,000-chip sample, its split and the ship-peak definition are not recorded
   (`contract.py:7-13`).
5. **GFW SAR dataset version** ("latest") is not stored in label or comparison records; presence records store v4.0.
   GFW data changed during the project (vessel classes 09-25 → 09-27, `PLAN.md:1662-1677`; backfills). Re-queries can
   differ.
6. **OSM snapshot date** not recorded (file mtime 2026-09-11 only). OPD is versioned (v1.0.0).
7. **Pair-list generators** for `pairs_s1a_2026h1.txt` and `pairs_s1cd_2026-09.txt` are not in the repo; the rule is only
   in their headers. The CDSE catalogue (online/offline, reprocessing) may return other products later.
8. **Uncommitted scripts that produced reported numbers.**
   - `data/logs/select_val_config.py` (selection rule) and the `run_*.sh` runners (e.g. `--sea 56`, `--per-area 5/8`) sit
     under git-ignored `data/`.
   - The build arguments of the area datasets are only in `data/logs/build_runner.log`.
   - Scratch analyses quoted in the thesis are not on disk: duplicate-peak shares, recurring false alarms 349/926,
     24 of 56 candidates at spots (`handoff.md:46, 48`).
9. **Local ONNX exports** of epochs 5/8/11/14: fp32 parity not stored (`handoff.md:76`).
10. **Per-pass comparison JSONs** record model sha256 and tiling but not the mask inputs (coastline/OPD file hashes),
    the CFAR version string or the code commit.
11. **Assumed constants, never measured per scene:** ENL 4.4 (CFAR), R/V 118 s (fusion). Fixed a priori, never tuned:
    shore 150 m buffer, platform 250/300 m, pre-filters.
12. `index_meta.json` on disk is labelled v1 but describes v2 (`handoff.md:60`). The Kaggle dataset subtitle is stale.
13. **Run 1's threshold (0.45) and epoch (6)** came from own-label validation (`eval_val_best.json`), which is recorded.
    Run 2's own-label `best.pt` (epoch 14) is not the product model; the product is `epoch_05`.
