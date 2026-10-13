# Chapter 2 source notes: deep learning for SAR ship detection

Prepared 2026-10-11 for the Chapter 2 rewrite (Persian B.Sc. report, Shahid Bahonar University of Kerman).
Category covered: **deep learning for SAR ship detection** (datasets, detector families, xView3, surveys,
Sentinel-1 work at scale, challenges, gaps relevant to the Persian Gulf project).

## How the entries were checked

- Metadata (authors, title, venue, volume, issue, pages, DOI): Crossref REST API (`api.crossref.org/works/<DOI>`
  and `query.bibliographic`), arXiv abstract pages, the NeurIPS and PMLR proceedings pages, GitHub API/README.
- Content and numbers: read from the full text where it could be opened (xView3 arXiv PDF; Paolo 2024 and
  Cao 2026 via Europe PMC full-text XML; Li 2022, SSDD-2021, LS-SSDD, SAR-Ship-Dataset via the MDPI PDF mirror
  `mdpi-res.com`; Beukema 2023 arXiv PDF; the xView3 scoring code `reference/metric.py`; the BloodAxe README).
  Where only the abstract was reachable (Crossref or Semantic Scholar abstract field), only abstract numbers are given.
- Status labels: **VERIFIED** = citation metadata confirmed and every fact written in the note was seen in a primary
  source. **PARTIAL** = citation confirmed, but one named detail is not confirmed (the detail is flagged in the entry).
  **UNVERIFIED** = not used. No entry is wholly unverified.
- Not done: Persian-language databases (SID, Magiran, Civilica) were **not** queried directly (only through general
  web search). Full texts of IEEE papers (HRSID, DAPN, FBR-Net, CenterNet++, Javed 2024) were not opened; numbers
  for these come only from abstracts or are omitted.

**Counts: 35 entries = 29 VERIFIED, 6 PARTIAL, 0 UNVERIFIED.**
(7 confirmations of references already in the thesis; 18 new core references; 4 Persian Gulf / Iran search results;
2 ESA web pages; 4 optional extras.)

---

## Part A. References with notes

Numbering below is local to this file (E = already in the thesis, D = dataset, M = method, S = survey,
O = operational/Sentinel-1 at scale, R = regional search result, W = web page, X = optional extra).

### A.1 References already in the thesis (details confirmed, not to be duplicated)

**[E1]** F. Paolo, T.-T. T. Lin, R. Gupta, B. Goodman, N. Patel, D. Kuster, D. Kroodsma, and J. Dunnmon,
"xView3-SAR: Detecting dark fishing activity using synthetic aperture radar imagery," in *Advances in Neural
Information Processing Systems 35 (NeurIPS 2022), Datasets and Benchmarks Track*, 2022, pp. 37604–37616,
doi: 10.52202/068431-2726. (arXiv:2206.00897.)
Status: **VERIFIED**. Checked: api.crossref.org/works/10.52202/068431-2726; arxiv.org/abs/2206.00897 (v4 full text);
proceedings.neurips.cc (D&B track page).
Note: 991 full-size Sentinel-1 IW GRD scenes (VV+VH, 10 m pixel spacing, about 20 m resolution), on average
29,400 × 24,400 px (the abstract; Step 1 of Sec. 3 prints 29,400 × 22,400, an internal inconsistency), 243,018 objects
over more than 43.2 million km². Labels combine GFW's AIS-to-SAR matching with professional annotators (437 scenes,
about 176,000 manual labels); 39.1 % of labels have both sources, 33.4 % manual only, 27.5 % automated only; each label
has HIGH/MEDIUM/LOW confidence. Splits: train 554 scenes (automated GFW labels only), validation 50, public 150,
holdout 237 (holdout not released). Regions: Adriatic, Bay of Biscay, Iceland, Norway, Gulf of Guinea (Table 2;
North Sea is also named in the text) — no Middle East scenes. Metric, results and winner: see Part D.
Weakness acknowledged by the authors: not global; the training split has no manual labels.

**[E2]** E. Khvedchenya (BloodAxe), "Winning solution for xView3 challenge," GitHub repository, 2022. [Online].
Available: https://github.com/BloodAxe/xView3-The-First-Place-Solution
Status: **VERIFIED** (README read; author name and holdout score stated there; repository created 2022-01-08).
Note: single-stage point detector "CircleNet" inspired by CenterNet and U-Net: pretrained encoder (EfficientNet B4,
B5, V2-S) + U-Net decoder, output stride 2; heads for objectness, length, vessel and fishing. Objectness trained with
the CenterNet reduced focal loss but with a fixed 3 px Gaussian radius (ship size in pixels varies little at 10 m).
Ensemble of 12 models (3 architectures × 4 folds) + left-right flip TTA; inference on 2048 px tiles with 1536 px step.
Holdout aggregate 0.617 (public LB 0.603). The README reports that encoding/decoding the ground truth alone gives
localization F1 0.9672 at stride 16, 0.9948 at stride 8, 0.9997 at stride 4, 0.9999 at stride 2 — the argument for a
fine output map. It also reports: training labels were the limiting factor (not the input signal); only 284 near-shore
objects in the train split versus 6,171 in validation; scenes cluster into about 42 locations shared across splits
(geographic leakage); adding wind/bathymetry did not help.

**[E3]** O. Ronneberger, P. Fischer, and T. Brox, "U-Net: Convolutional networks for biomedical image segmentation,"
in *Medical Image Computing and Computer-Assisted Intervention – MICCAI 2015*, Lecture Notes in Computer Science,
pp. 234–241, 2015, doi: 10.1007/978-3-319-24574-4_28.
Status: **VERIFIED** (Crossref: title, authors, series, pages, DOI). The LNCS volume number is not given by Crossref
and was not checked elsewhere (DBLP was behind a bot check); copy it from the thesis's existing entry if present.

**[E4]** X. Zhou, D. Wang, and P. Krähenbühl, "Objects as points," arXiv:1904.07850, Apr. 2019.
Status: **VERIFIED** (arXiv abstract page: submitted 16 Apr 2019, v2 25 Apr 2019; no journal reference listed).

**[E5]** M. Tan and Q. V. Le, "EfficientNet: Rethinking model scaling for convolutional neural networks," in
*Proc. 36th Int. Conf. Machine Learning (ICML)*, PMLR vol. 97, 2019, pp. 6105–6114.
Status: **VERIFIED** (proceedings.mlr.press/v97/tan19a.html).

**[E6]** T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal loss for dense object detection," in
*Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 2999–3007, doi: 10.1109/ICCV.2017.324.
Status: **VERIFIED** (Crossref). (A journal version exists: IEEE TPAMI, vol. 42, no. 2, pp. 318–327, 2020,
doi: 10.1109/TPAMI.2018.2858826 — also confirmed on Crossref; cite only one.)

**[E7]** F. S. Paolo, D. Kroodsma, J. Raynor, T. Hochberg, P. Davis, J. Cleary, L. Marsaglia, S. Orofino, C. Thomas,
and P. Halpin, "Satellite mapping reveals extensive industrial activity at sea," *Nature*, vol. 625, no. 7993,
pp. 85–91, Jan. 2024, doi: 10.1038/s41586-023-06825-8.
Status: **VERIFIED** (Crossref; full text via Europe PMC, PMC10764273).
What model and validation they used (answers brief item 5): **detection itself is not deep learning.** Vessels are
detected in Sentinel-1 with a modified two-parameter CFAR (inner window 200 × 200 px, outer 600 × 600 px,
time-dependent thresholds because S1A/S1B statistics drift), run on Google Earth Engine. A ResNet-based ConvNet then
takes 80 × 80 px dual-band (VV, VH) tiles around each CFAR detection and outputs object presence (to reject CFAR false
alarms) and length — more than 62 million tiles at inference. Its training/test data are AIS-matched detections plus
expert annotations (three labellers, high-confidence samples kept), about 12,000 samples split 80/20; reported
F1 0.97 (accuracy 97.5 %) for presence and R² 0.84 (RMSE 21.9 m) for length. A second ConvNet (ConvNeXt) classifies
infrastructure and a third classifies fishing vs non-fishing from 11 environmental rasters plus length. Recall
against AIS vessels seen within 2 min of acquisition: above 95 % for vessels > 50 m, about 80 % for 25–50 m, dropping
steeply below 25 m; most vessels < 15 m are missed; detection is lower for vessels closer than 1 km to each other
(600 m CFAR window). A 1 km buffer from the shoreline is removed before analysis. Headline: 72–76 % of industrial
fishing and 21–30 % of transport and energy vessel activity is not publicly tracked (2017–2021, about 2 PB of imagery).
The Persian Gulf appears only as one of 14 oil-infrastructure regions. These are the "GFW detections" the thesis
evaluates against, so they inherit this pipeline's size floor and coastal buffer.

### A.2 Public SAR ship datasets

**[D1]** J. Li, C. Qu, and J. Shao, "Ship detection in SAR images based on an improved Faster R-CNN," in *Proc.
SAR in Big Data Era: Models, Methods and Applications (BIGSARDATA)*, Beijing, China, Nov. 2017, pp. 1–6,
doi: 10.1109/BIGSARDATA.2017.8124934.
Status: **VERIFIED** (Crossref). The original paper was not opened; facts below are taken from [D2] and [S1].
Note: introduced SSDD, the first public SAR ship detection dataset, together with an improved Faster R-CNN baseline
(four strategies on top of standard Faster R-CNN, per [D2]). The survey [S1] dates the start of the deep-learning period of SAR ship
detection from this paper (1 Dec 2017). Weakness: small, manually boxed chips with ambiguous usage rules (random
train/test splits, no inshore/offshore protocol), which [D2] later corrected.

**[D2]** T. Zhang, X. Zhang, J. Li, X. Xu, B. Wang, X. Zhan, Y. Xu, X. Ke, T. Zeng, H. Su, I. Ahmad, D. Pan, C. Liu,
Y. Zhou, J. Shi, and S. Wei, "SAR Ship Detection Dataset (SSDD): Official release and comprehensive data analysis,"
*Remote Sensing*, vol. 13, no. 18, Art. no. 3690, Sep. 2021, doi: 10.3390/rs13183690.
Status: **VERIFIED** (Crossref; full text via mdpi-res.com PDF).
Note: SSDD = 1160 images, 2456 ships, from RadarSat-2, TerraSAR-X and Sentinel-1, HH/VV/VH/HV, resolution 1–15 m,
scenes near Yantai (China) and Visakhapatnam (India), inshore and offshore. The authors count that 46.59 % of 161 public
reports used SSDD and that the original coarse labels and loose usage rules prevented fair comparison. The official
release relabels ships in three forms (BBox-, RBox-, PSeg-SSDD) and fixes rules: train/test split by file-name suffix
(1 or 9 → test; 928 train / 232 test), an inshore–offshore protocol, ship-size definitions. Only 19.8 % of test images are
inshore (186 of 232 are offshore), and the authors warn that this imbalance flatters offshore accuracy. Weakness: still
a small chip benchmark; every image contains ships. (Another paper, [D5], quotes SSDD as 2358 ships — itself an example
of how dataset statistics drift between papers.)

**[D3]** Y. Wang, C. Wang, H. Zhang, Y. Dong, and S. Wei, "A SAR dataset of ship detection for deep learning under
complex backgrounds," *Remote Sensing*, vol. 11, no. 7, Art. no. 765, Mar. 2019, doi: 10.3390/rs11070765.
Status: **VERIFIED** (Crossref; full text; GitHub CAESAR-Radi/SAR-Ship-Dataset README).
Note: 102 Gaofen-3 and 108 Sentinel-1 scenes (Sentinel-1 modes SM and IW) cut into 43,819 chips of 256 × 256 px with
59,535 ships (ship count from [S1] and [D5]); resolutions from 3 to 25 m, single/dual/full polarization. Chips were taken
with a sliding window at 50 % overlap over sub-images that contain ships, labelled by SAR experts (no AIS), then randomly
split 70/20/10. Baselines (mAP): SSD-300 88.32 %, SSD-512 89.43 %, Faster R-CNN 88.26 %, RetinaNet 91.36 %. The
GitHub update removed repeated chips (39,729 chips). Strengths: large, multi-sensor. Weaknesses: chips are cut
around ships, so false alarms over open sea and land are barely tested; random splitting of 50 %-overlapping chips can
place the same ship in train and test (our observation; the paper does not discuss it).

**[D4]** S. Wei, X. Zeng, Q. Qu, M. Wang, H. Su, and J. Shi, "HRSID: A high-resolution SAR images dataset for ship
detection and instance segmentation," *IEEE Access*, vol. 8, pp. 120234–120254, 2020,
doi: 10.1109/ACCESS.2020.3005861.
Status: **PARTIAL** — citation, abstract and README verified; the sensor breakdown (99 Sentinel-1B, 36 TerraSAR-X,
1 TanDEM-X scenes) is from the survey [S1], not the original paper; resolution is "1–5 m" in the abstract but
"0.5, 1, 3 m" on the GitHub README and in [D5].
Note: 136 panoramic scenes cut into 800 × 800 px chips with 25 % overlap, giving 5604 chips and 16,951 ships, split 65/35,
COCO format with polygon (instance-segmentation) labels; optical imagery was used to reduce wrong/missing labels near
harbours. The README adds inshore/offshore tags (18.4 % inshore) and a separate set of 400 pure-background images for
robustness tests — an admission that the main set, like other chip sets, lacks ship-free samples.

**[D5]** T. Zhang, X. Zhang, X. Ke, X. Zhan, J. Shi, S. Wei, D. Pan, J. Li, H. Su, Y. Zhou, and D. Kumar,
"LS-SSDD-v1.0: A deep learning dataset dedicated to small ship detection from large-scale Sentinel-1 SAR images,"
*Remote Sensing*, vol. 12, no. 18, Art. no. 2997, Sep. 2020, doi: 10.3390/rs12182997.
Status: **VERIFIED** (Crossref; full text).
Note: 15 full Sentinel-1 IW scenes (24,000 × 16,000 px, VV+VH, 5 × 20 m resolution) cut into 9000 sub-images of
800 × 800 px **including ship-free sub-images**; first 10 scenes for training, last 5 for testing; 6015 ships labelled by
SAR experts with the help of AIS and Google Earth. The paper argues explicitly that chip datasets discard pure
backgrounds and therefore produce bright-dot false alarms over land, and it reports results on the entire scenes.
Best entire-scene results among its baselines: 75.74 % mAP (Double-Head R-CNN), F1 0.79, detection probability 78.47 %;
Faster R-CNN reaches 46.76 % mAP inshore versus 89.99 % offshore (F1 0.59 vs 0.87). It notes SSDD results already exceed
95 % mAP, whereas LS-SSDD is far from saturated. Closest public analogue to the thesis setting (Sentinel-1 IW, small
ships, whole scenes), but only 15 scenes, Chinese/East-Asian waters, box labels.

**[D6]** L. Huang, B. Liu, B. Li, W. Guo, W. Yu, Z. Zhang, and W. Yu, "OpenSARShip: A dataset dedicated to
Sentinel-1 ship interpretation," *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 11, no. 1, pp. 195–208,
Jan. 2018, doi: 10.1109/JSTARS.2017.2755672.
Status: **PARTIAL** — citation and the figure of 11,346 AIS-integrated ship chips verified (Crossref; Monash research
portal abstract); "41 Sentinel-1 scenes, 17 AIS ship types, chips 30–120 px" come only from secondary summaries.
Note: ship chips centred on single AIS-matched ships, built for ship **classification/interpretation** and for quality
assessment of Sentinel-1, not for detection. OpenSARShip 2.0 has 34,528 chips (opensar.sjtu.edu.cn; also [E1] Table 7,
which says it mixes GRD and SLC). Relevant to the thesis only as an example of AIS-derived labels on Sentinel-1;
AIS-only labelling excludes dark vessels by construction.

### A.3 Detector families

**[M1]** S. Ren, K. He, R. Girshick, and J. Sun, "Faster R-CNN: Towards real-time object detection with region
proposal networks," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 39, no. 6, pp. 1137–1149, Jun. 2017,
doi: 10.1109/TPAMI.2016.2577031.
Status: **VERIFIED** (Crossref).
Note: two-stage detector — a region proposal network with predefined anchors, then a box classifier/regressor.
In SAR it is the baseline of SSDD [D1] and of the xView3 reference model [E1] (800 × 800 chips, 10-px boxes drawn around
point labels; holdout F1_D 0.4302, F1_S 0.1293, aggregate 0.1904 — explicitly not a tuned baseline). Two-stage models
are 26.7 % of the 177 papers surveyed in [S1]. Weakness for 10 m SAR: anchors and stride-16 features are coarse
relative to ships of a few pixels.

**[M2]** J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You only look once: Unified, real-time object
detection," in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2016, pp. 779–788,
doi: 10.1109/CVPR.2016.91.
Status: **VERIFIED** (Crossref).
Note: one-stage detector that regresses boxes and classes on a grid in one pass; later YOLO versions dominate recent
SAR papers, including the only Persian Gulf deep-learning study found [R1] and the Iranian study [R2]. Single-stage
models are 25.6 % of the papers in [S1]. Strength: speed. Weakness: grid/anchor resolution limits very small,
closely spaced targets; most reported SAR numbers are on chip benchmarks.

**[M3]** Z. Tian, C. Shen, H. Chen, and T. He, "FCOS: Fully convolutional one-stage object detection," in *Proc.
IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2019, pp. 9626–9635, doi: 10.1109/ICCV.2019.00972.
Status: **VERIFIED** (Crossref pages 9626–9635; the CVF open-access version paginates 9627–9636 — use the DOI version).
Note: anchor-free, proposal-free per-pixel detector with a centre-ness branch; 44.7 % COCO AP with
ResNeXt-64x4d-101 (abstract). In SAR it is the base of several anchor-free detectors (e.g. FBR-Net [X1]). With
CenterNet [E4] it represents the anchor-free family that [S1] singles out as particularly well suited to sparse small ships
but that made up only about 5 % of SAR papers by 2022.

**[M4]** Y. Wang, C. Wang, H. Zhang, Y. Dong, and S. Wei, "Automatic ship detection based on RetinaNet using
multi-resolution Gaofen-3 imagery," *Remote Sensing*, vol. 11, no. 5, Art. no. 531, Mar. 2019,
doi: 10.3390/rs11050531.
Status: **VERIFIED** (Crossref; abstract).
Note: RetinaNet (FPN + focal loss [E6]) on 86 Gaofen-3 scenes at 3, 5, 8 and 10 m, robustness tested on two more
Gaofen-3 scenes and one COSMO-SkyMed scene; reports more than 96 % mAP and better results than the other detectors
compared. Shows that focal loss addresses the ship/background imbalance in SAR. Weakness: chip-level mAP on
high-resolution Chinese imagery; not comparable with 10 m Sentinel-1 scene-level F1.

**[M5]** Z. Cui, Q. Li, Z. Cao, and N. Liu, "Dense attention pyramid networks for multi-scale ship detection in SAR
images," *IEEE Trans. Geosci. Remote Sens.*, vol. 57, no. 11, pp. 8983–8997, Nov. 2019,
doi: 10.1109/TGRS.2019.2923988.
Status: **VERIFIED** (Crossref; abstract). Numeric results not checked (full text not opened).
Note: DAPN densely connects a convolutional block attention module (CBAM) to every concatenated level of a top-down
feature pyramid so that small, low-contrast ships keep both resolution and semantic information. Evaluated on SSDD,
with the claim of outperforming other SSDD methods. Typical of SAR-specific "multi-scale + attention" adaptations
(14.2 % of papers in [S1] are multi-scale). Weakness: gains shown on one small chip dataset.

**[M6]** H. Guo, X. Yang, N. Wang, and X. Gao, "A CenterNet++ model for ship detection in SAR images," *Pattern
Recognition*, vol. 112, Art. no. 107787, Apr. 2021, doi: 10.1016/j.patcog.2020.107787.
Status: **VERIFIED** (Crossref; abstract). Numeric results not checked.
Note: keypoint (centre) detector built on CenterNet [E4] with a feature refinement module (multi-scale context for small
ships), a feature-pyramid fusion module and a head enhancement module (foreground/background balance against
complex onshore clutter). Tested on AIR-SARShip, SSDD and SAR-Ship-Dataset; claims state of the art on all three with
negligible extra time over CenterNet. Closest academic relative of the thesis model (centre heatmap), but box-based
and evaluated on chips.

**[M7]** Y. Mao, Y. Yang, Z. Ma, M. Li, H. Su, and J. Zhang, "Efficient low-cost ship detection for SAR imagery based
on simplified U-Net," *IEEE Access*, vol. 8, pp. 69742–69753, 2020, doi: 10.1109/ACCESS.2020.2985637.
Status: **VERIFIED** (Crossref; abstract).
Note: anchor-free detector on a simplified U-Net backbone (about 0.47 M weights; about 0.93 M in total) with a box
regression sub-net and a score-map (ship-centre likelihood) sub-net; 68.1 % AP and 67.6 % AR on SSDD. Shows that a
U-Net encoder–decoder with a dense centre map is a viable SAR ship detector — the same family as the thesis model.
Weakness: SSDD only; very small model chosen for cost rather than accuracy.

### A.4 Surveys

**[S1]** J. Li, C. Xu, H. Su, L. Gao, and T. Wang, "Deep learning for SAR ship detection: Past, present and future,"
*Remote Sensing*, vol. 14, no. 11, Art. no. 2712, Jun. 2022, doi: 10.3390/rs14112712.
Status: **VERIFIED** (Crossref; full text, 44 pages).
Note: by the SSDD authors; analyses 177 papers (2017–2022). Past: lists four CFAR shortcomings (guard window tied to ship
size, need for an accurate clutter model, unsupervised, pieced-together pipeline) and quotes an AIR-SARShip-1.0
comparison in which K-distribution CFAR gets 19.2 % AP and an optimal-entropy threshold 28.2 % versus 79.3 % for
Faster R-CNN and 74.3 % for SSD-512. Present: SSDD is used 83 times (62.4 %); AP50 on SSDD rose from 78.8 % (2017) to
97.8 % (2022); two-stage 26.7 %, single-stage 25.6 %, anchor-free about 5 %, multi-scale 14.2 % of papers. Future:
anchor-free detectors deserve attention, small-ship detection is a lasting open problem (anchors too large and discrete),
datasets should be merged with fixed splits, metrics and benchmarks. Weakness: centred on Chinese chip datasets and
AP; no mention of xView3 or of scene-level/operational evaluation. (Its ref. 88 misattributes [O1] to "Kang, K.M.".)

**[S2]** M. J. Er, Y. Zhang, J. Chen, and W. Gao, "Ship detection with deep learning: A survey," *Artificial
Intelligence Review*, vol. 56, no. 10, pp. 11825–11865, Oct. 2023, doi: 10.1007/s10462-023-10455-x.
Status: **PARTIAL** — citation verified on Crossref; the abstract is withheld by the publisher in Crossref/Semantic
Scholar and the Springer page could not be opened, so **the content is unverified**. Use only as a pointer to a broader
(not SAR-only) survey, or drop it.

### A.5 Sentinel-1 at scale, automatic labels, operational systems

**[O1]** J. Song, D.-j. Kim, and K.-m. Kang, "Automated procurement of training data for machine learning algorithm
on ship detection using AIS information," *Remote Sensing*, vol. 12, no. 9, Art. no. 1443, May 2020,
doi: 10.3390/rs12091443.
Status: **VERIFIED** (Crossref; abstract).
Note: builds training boxes automatically from AIS: AIS tracks are interpolated to the acquisition time with a Kalman
filter, the Doppler azimuth shift is compensated, and the box size comes from the AIS-reported ship dimensions.
From 18 Sentinel-1 scenes they obtain 7489 training samples with no manual work; a detector trained on them scores 0.7713
overall detection performance on 3 test scenes, higher than the same detector trained on visually interpreted labels,
and it avoids harbour structures and azimuth-ambiguity ghosts. Direct precedent for the thesis's automatic labels.
Weakness: AIS-derived labels cannot contain dark vessels, and the test set is small.

**[O2]** C. Dechesne, S. Lefèvre, R. Vadaine, G. Hajduch, and R. Fablet, "Ship identification and characterization in
Sentinel-1 SAR images with multi-task deep learning," *Remote Sensing*, vol. 11, no. 24, Art. no. 2997, Dec. 2019,
doi: 10.3390/rs11242997.
Status: **VERIFIED** (Crossref; abstract).
Note: one shared convolutional network feeding three task branches (detection, classification, length estimation)
on Sentinel-1, with training data from AIS–Sentinel-1 matching; classification overall accuracy 97.25 %, mean length
error 4.65 m ± 8.55 m. Shows multi-task heads on Sentinel-1 work with AIS supervision (as later in xView3 and [E7]).
Weakness: AIS-only supervision; detection metrics are not given in the abstract.

**[O3]** P. Beukema, F. Bastani, Y. Zheng, P. Wolters, H. Herzog, and J. Ferdinando, "Satellite imagery and AI: A new
era in ocean conservation, from research to deployment and impact (Version 2.0)," arXiv:2312.03207, Dec. 2023
(v2 May 2025). NeurIPS 2023 Computational Sustainability workshop (best paper, per the arXiv comments).
Status: **VERIFIED** (arXiv abstract page and full text).
Note: describes the Sentinel-1 vessel detector deployed in AI2's Skylight platform: a Faster R-CNN head on a custom
13-layer fully convolutional encoder with FPN, which also ingests aligned **historical** Sentinel-1 images so that
objects that never move (platforms, islands) are suppressed. Trained on 55,499 expert point labels (no overlap with
xView3); an early version placed 4th in xView3; internal held-out F1 rose from 70.1 % (xView3 version) to 82.7 %.
Precision is prioritised; detections on known marine infrastructure are suppressed and a 10 m coastline dataset is used
to remove rocky-island false alarms and keep near-shore vessels. Strength: real operational system. Weakness:
internal validation only; not a public benchmark.

### A.6 Persian Gulf / Iran search results

**[R1]** M. A. Javed, E. Naseer, and M. A. Siddique, "Deep learning on Sentinel-1 SAR imagery for ship detection using
YOLO-V8 model," in *Proc. 2024 Horizons of Information Technology and Engineering (HITE)*, Oct. 2024, pp. 1–6,
doi: 10.1109/HITE63532.2024.10777281.
Status: **VERIFIED** (Crossref; abstract via Semantic Scholar). Full text closed; affiliation not checked.
Note: YOLOv8 trained on a ship/background-balanced subset of LS-SSDD-v1.0: 85.8 % precision, 70.5 % recall, 77.6 % mAP
on the imbalanced test set. Generalisation tested on a custom set of Sentinel-1 images from the Gulf of Oman and the
Persian Gulf: after 20 epochs of fine-tuning, "accuracy" 90.4 % (metric definition, number of scenes and labels not
given in the abstract). **The only deep-learning ship detection result on Persian Gulf Sentinel-1 data found.**
Weakness: needed fine-tuning to transfer; regional data and labels not public; no scene-level F1.

**[R2]** F. Ashtari, T. Managhebi, and F. Samadzadegan, "Identification of coastal and non-coastal vessels in SAR
radar images using YOLOv8 deep learning network," *Earth Observation and Geomatics Engineering*, vol. 7, no. 2,
Art. no. e99028, Dec. 2023, doi: 10.22059/eoge.2024.381462.1160.
Status: **PARTIAL** — title, authors, volume/issue and abstract checked on the journal page
(eoge.ut.ac.ir/article_99028.html, University of Tehran); the DOI is printed there but **Crossref returns 404** for it.
Note: Iranian (University of Tehran) study; YOLOv8 on **SSDD**, reporting 99 % accuracy, 96 % precision, 95 % recall,
98 % mAP, 97 % F1. No Persian Gulf or Iranian imagery is used. Shows that Iranian work so far benchmarks on the
public Chinese chip set rather than on regional scenes.

**[R3]** Y. Cao, H. Wang, F. Su, D. Fu, F. Yan, X. Hong, W. He, N. Xu, Y. Ou, and C. Chen, "Satellite radar and AIS reveal
a 97% decline in shipping traffic through the Strait of Hormuz," *The Innovation*, vol. 7, no. 6, Art. no. 101367,
Jun. 2026, doi: 10.1016/j.xinn.2026.101367.
Status: **VERIFIED** (Crossref; full text via Europe PMC, PMC13237841).
Note: event-driven analysis of the Strait of Hormuz around 28 Feb 2026 using 258 Sentinel-1 IW GRD scenes
(22 Feb–7 Mar 2026) processed on Google Earth Engine. Detection is a **fixed −15 dB threshold** plus an ocean mask,
removal of clusters under 3 px and manual inspection — not deep learning; F1 96.0 % (precision 94.2 %, recall 97.9 %)
on 18 test images. Combined with AIS counts. Shows recent Persian Gulf SAR vessel monitoring still relies on simple
thresholds and small validation sets.

**[R4]** R. Spanier, T. Hoeser, J. Truckenbrodt, F. Bachofer, and C. Kuenzer, "Offshore oil and gas platform dynamics
in the North Sea, Gulf of Mexico, and Persian Gulf: Exploiting the Sentinel-1 archive," arXiv:2603.19801, Mar. 2026.
Status: **PARTIAL** — arXiv abstract verified; the abstract does not name the detector (a search-engine summary said
YOLOv10; **unconfirmed**).
Note: deep-learning object detection of offshore platforms in Sentinel-1 (2017–2025); 3,728 platforms in 2025, of which
1,731 are in the Persian Gulf. Not ship detection, but it quantifies the main fixed false-alarm source in the study area
and is a candidate mask for vessel detections; no accuracy figures in the abstract.

### A.7 Web sources (new satellites)

**[W1]** European Space Agency, "Double win for Europe: Sentinel-1C and Vega-C take to the skies," ESA, Dec. 2024.
[Online]. Available: https://www.esa.int/Applications/Observing_the_Earth/Copernicus/Sentinel-1/Double_win_for_Europe_Sentinel-1C_and_Vega-C_take_to_the_skies
Status: **VERIFIED** (page read). Sentinel-1C launched 5 Dec 2024 on Vega-C; Sentinel-1B's mission ended in Aug 2022
after a technical fault; Sentinel-1C carries an integrated AIS receiver (four antennas).

**[W2]** European Space Agency, "Ariane 6 launches Sentinel-1D," ESA multimedia, Nov. 2025. [Online]. Available:
https://www.esa.int/ESA_Multimedia/Images/2025/11/Ariane_6_launches_Sentinel-1D
Status: **VERIFIED** (page read). Launched 4 Nov 2025 (flight VA265); flies in the same orbit as Sentinel-1C, 180° apart;
both carry AIS instruments.

### A.8 Optional extras (verified citations, for the writer to use if needed)

**[X1]** J. Fu, X. Sun, Z. Wang, and K. Fu, "An anchor-free method based on feature balancing and refinement network
for multiscale ship detection in SAR images," *IEEE Trans. Geosci. Remote Sens.*, vol. 59, no. 2, pp. 1331–1344,
Feb. 2021, doi: 10.1109/TGRS.2020.3005151. **VERIFIED** (Crossref; abstract). FBR-Net: anchor-free (FCOS-style per
[S1]) with an attention-guided balanced pyramid and a feature refinement module; SSDD and AIR-SARShip-1.0; claims
state of the art. Numbers not checked.

**[X2]** Z. Cui, X. Wang, N. Liu, Z. Cao, and J. Yang, "Ship detection in large-scale SAR images via spatial
shuffle-group enhance attention," *IEEE Trans. Geosci. Remote Sens.*, vol. 59, no. 1, pp. 379–391, Jan. 2021,
doi: 10.1109/TGRS.2020.2997200. **PARTIAL** — citation verified on Crossref; content known only from [S1]: a
CenterNet-based detector with spatial shuffle-group enhance attention to suppress inshore/inland clutter, no anchors
or NMS. The datasets and numbers used were **not** checked.

**[X3]** J. Jiao, Y. Zhang, H. Sun, X. Yang, X. Gao, W. Hong, K. Fu, and X. Sun, "A densely connected end-to-end neural
network for multiscale and multiscene SAR ship detection," *IEEE Access*, vol. 6, pp. 20881–20892, 2018,
doi: 10.1109/ACCESS.2018.2825376. **VERIFIED** (Crossref; abstract). Faster R-CNN with densely connected multi-scale
proposals and a loss that down-weights easy examples to cut inshore false alarms.

**[X4]** Y.-L. Chang, A. Anagaw, L. Chang, Y. C. Wang, C.-Y. Hsiao, and W.-H. Lee, "Ship detection based on YOLOv2 for
SAR imagery," *Remote Sensing*, vol. 11, no. 7, Art. no. 786, Apr. 2019, doi: 10.3390/rs11070786. **VERIFIED**
(Crossref; abstract). YOLOv2 and a reduced YOLOv2 on SSDD (90.05 % "accuracy") and a diversified set (89.13 %),
faster than Faster R-CNN on GPU.

---

## Part B. Dataset comparison table

| Dataset | Sensor | Resolution | Chips or scenes | Label source | Size | Notes |
|---|---|---|---|---|---|---|
| SSDD [D1], official release [D2] | RadarSat-2, TerraSAR-X, Sentinel-1 | 1–15 m | Chips (about 500 px), every chip has ships | Manual (relabelled 2021; box, rotated box, polygon) | 1160 images, 2456 ships | Most used (62.4 % of papers in [S1]); fixed split 928/232 since 2021; test set 80.2 % offshore; AP50 saturated near 98 % |
| SAR-Ship-Dataset [D3] | Gaofen-3 (102 scenes), Sentinel-1 (108 scenes, SM/IW) | 3–25 m | 256 × 256 chips cut around ships, 50 % overlap | Manual (SAR experts, no AIS) | 43,819 chips, 59,535 ships (39,729 chips after de-duplication) | Random 70/20/10 split of overlapping chips; RetinaNet 91.36 % mAP |
| HRSID [D4] | Sentinel-1B, TerraSAR-X, TanDEM-X (breakdown from [S1]) | 0.5/1/3 m (README) or 1–5 m (abstract) | 800 × 800 chips from 136 scenes | Manual, checked against optical images; polygons | 5604 chips, 16,951 ships | 18.4 % inshore; 400 ship-free images released separately for robustness tests |
| LS-SSDD-v1.0 [D5] | Sentinel-1 IW, VV+VH | 5 × 20 m | 15 full scenes (24,000 × 16,000 px) tiled into 9000 sub-images, ship-free tiles kept | Manual with AIS and Google Earth | 6015 ships | Scene-level results; best 75.74 % mAP; inshore 46.76 % vs offshore 89.99 % mAP (Faster R-CNN) |
| OpenSARShip 1.0/2.0 [D6] | Sentinel-1 (GRD and SLC for 2.0) | Sentinel-1 native | Single-ship chips (classification) | AIS matching (ship type) | 11,346 chips (1.0); 34,528 chips (2.0) | Not a detection set; AIS-only labels exclude dark vessels |
| xView3-SAR [E1] | Sentinel-1 IW GRD, VV+VH (+ wind, bathymetry rasters) | about 20 m (10 m pixels) | 991 full scenes (≈29,400 × 24,400 px) | GFW AIS–SAR matching + professional annotators, HIGH/MEDIUM/LOW confidence; train split automated only | 243,018 objects, >43.2 million km² | Only large scene-level, point-labelled benchmark; Europe + Gulf of Guinea, no Middle East; holdout withheld |

Chip versus scene: SSDD, SAR-Ship-Dataset, HRSID and OpenSARShip are cut around ships, so they measure how well a
ship is boxed once it is in view; they say little about the false-alarm rate over a full Sentinel-1 swath of sea, coast and
land. LS-SSDD [D5] and xView3 [E1] keep whole scenes (including empty sea and land), and LS-SSDD states the problem
directly.

---

## Part C. Method comparison table

| Approach | Representative work | Data | Reported result (as published; not mutually comparable) | Strengths | Weaknesses |
|---|---|---|---|---|---|
| CFAR + CNN discrimination (hybrid) | Paolo et al. [E7] | Sentinel-1 IW, global coastal waters 2017–2021 | Presence classifier F1 0.97, length R² 0.84; recall >95 % (>50 m), ~80 % (25–50 m) | Scales to petabytes; CFAR is transparent; CNN removes CFAR false alarms | CFAR sets the recall floor: misses most <15 m vessels and close pairs; 1 km coastal buffer dropped |
| Pure CFAR / threshold | K-distribution CFAR (in [S1]); Cao et al. [R3] | AIR-SARShip-1.0; 258 Hormuz scenes | 19.2 % AP vs 79.3 % for Faster R-CNN (in [S1]); F1 96.0 % on 18 images [R3] | No training data needed | Clutter-model and window choices; land/structure false alarms; weak on small ships |
| Two-stage, anchor-based | Faster R-CNN [M1]; SSDD baseline [D1]; xView3 reference [E1]; Jiao et al. [X3] | SSDD, LS-SSDD, xView3 | LS-SSDD Faster R-CNN 74.80 % mAP (entire scenes); xView3 reference F1_D 0.4302, F1_S 0.1293 | Mature, accurate on chips | Anchors and stride-16 maps coarse for few-pixel ships; NMS merges close ships; slower |
| One-stage, anchor-based | YOLO [M2]; RetinaNet/focal loss [E6], [M4]; YOLOv8 [R1], [R2] | Gaofen-3, SSDD, LS-SSDD, Gulf of Oman/Persian Gulf | RetinaNet >96 % mAP [M4]; YOLOv8 77.6 % mAP on LS-SSDD [R1] | Fast; focal loss handles imbalance | Anchor tuning; most numbers from chip sets |
| SAR-specific multi-scale/attention | DAPN [M5]; CenterNet++ [M6]; SSE-CenterNet [X2] | SSDD, AIR-SARShip, SAR-Ship-Dataset | SOTA claims on SSDD-type chips (numbers not checked) | Target small ships and inshore clutter | Small-dataset gains; hard to compare across papers |
| Anchor-free (box) | FCOS [M3]; FBR-Net [X1]; simplified U-Net [M7] | SSDD, AIR-SARShip | 68.1 % AP / 67.6 % AR on SSDD with ~0.93 M weights [M7] | No anchor design; suits sparse small ships | Still regresses boxes that a 10 m SAR ship barely has; only ~5 % of papers [S1] |
| Keypoint / centre heatmap (point detection) | CenterNet [E4]; xView3 winner CircleNet [E2] | xView3 full scenes | Holdout aggregate 0.6177, F1_D 0.7702, F1_S 0.5310 [E1] | Matches point labels (AIS/GFW/xView3); stride-2 map keeps close ships apart; no anchors or box NMS | Heatmap radius and threshold choices; near-shore still weak; winning system is a 12-model ensemble |
| Operational DL with temporal context | Skylight [O3] | 55,499 expert point labels, Sentinel-1 VV+VH + past images | Internal F1 82.7 % (vs 70.1 % for its xView3 version) | Historical images reject fixed objects; infrastructure and coastline masks | Internal validation only |
| Multi-task with AIS supervision | Dechesne et al. [O2]; automatic AIS labels [O1] | Sentinel-1 + AIS | Class. OA 97.25 %, length error 4.65 ± 8.55 m [O2]; 0.7713 detection score, beats manual labels [O1] | No manual labelling; extra tasks | Labels exclude dark vessels; small test sets |

---

## Part D. xView3: metric, top results, winning approach (brief item 3)

- **Matching.** Predictions and ground truth are points (scene row/column). Per scene, the reference scorer builds a
  distance matrix in pixels × 10 m, solves a Hungarian assignment (`linear_sum_assignment`), and counts a pair as a
  true positive if the distance is **below 200 m** (`distance_tolerance=200` in `reference/metric.py`); unmatched
  predictions are false positives, unmatched ground truth false negatives. Predictions matched to LOW-confidence
  labels are removed before scoring, and only MEDIUM/HIGH labels count as ground truth [E1].
- **Sub-scores.** F1_D (all objects); F1_S (close-to-shore: ground truth within **2 km** of the shoreline, i.e. the 0 m
  bathymetry contour, and predictions within 2 km + 200 m); F1_V (vessel vs fixed infrastructure); F1_F (fishing vs
  non-fishing); PE_L (one minus the aggregate percent length error, capped at 500 m).
- **Aggregate.** M_R = F1_D × (1 + F1_S + F1_V + F1_F + PE_L) / 5 — poor detection caps the whole score.
- **Constraint.** Inference on a full scene in under 15 min on one V100; the winning model needed over 63 h of training
  and about 13 min per scene.
- **Holdout results (Table 1 of [E1]).** 1 BloodAxe 0.6177 (F1_D 0.7702, F1_S 0.5310); 2 selim_sef 0.6047 (0.7629,
  0.4768); 3 Tumen 0.5805 (0.7395, 0.4205); 4 Skylight at AI2 0.5777 (0.7322, 0.4674); 5 Kohei 0.5717 (0.7342, 0.4527);
  reference Faster R-CNN 0.1904 (0.4302, 0.1293).
- **What the organisers observed.** Top solutions were single-stage detectors tailored to tiny, tightly packed objects;
  all used SAR-specific augmentation (brightness/contrast, speckle-like noise). All models were much weaker near shore
  (F1_S 0.42–0.53 versus F1_D 0.73–0.77) because rocky coasts and multipath/scattering artefacts look like vessels, and
  no competitor built a method specifically for the shoreline. Small vessels were harder in the Adriatic (possibly
  non-metal hulls). The paper motivates the dataset by dark vessels (AIS-only labels would miss them; hence the manual
  labels), but it does **not** report detection scores split by AIS status — no "dark-vessel recall" number exists in it.
- **Winner's design** [E2]: CenterNet-style point detection, EfficientNet/U-Net encoder–decoder, stride-2 output,
  reduced focal loss with fixed 3 px radius, 2-channel VV/VH input, label smoothing and entropy regularisation for noisy
  or missing labels, 12-model ensemble. These map almost one-to-one onto the thesis model (single EfficientNet-B0 +
  U-Net, stride 2, penalty-reduced focal loss, offset head, VV+VH).

---

## Part E. Known challenges, with sources (brief item 6)

1. **Small targets versus speckle.** Most xView3 vessels are under 40 m, i.e. a few pixels at ~20 m resolution [E1];
   GFW's pipeline misses most vessels under 15 m and recall drops steeply below 25 m [E7]. [S1] calls small-ship
   detection a lasting open problem because anchors are too large and too discrete. The BloodAxe stride table [E2] shows that
   even perfect labels lose localisation F1 at coarse output strides (0.9672 at 16 vs 0.9999 at 2).
2. **Inshore and coastal clutter.** Near-shore F1 in xView3 is 0.24–0.32 below overall F1 for every top-5 team [E1];
   on LS-SSDD, Faster R-CNN gets 46.76 % mAP inshore versus 89.99 % offshore [D5]; SSDD's test set is only 19.8 %
   inshore [D2]. Operational systems sidestep the problem: GFW removes a 1 km coastal buffer [E7]; Skylight masks known
   infrastructure and uses a 10 m coastline [O3]. In the Persian Gulf, offshore platforms (1,731 in 2025 [R4]) add fixed
   bright targets.
3. **Scale variation.** A major theme of chip-dataset research (multi-scale papers 14.2 % in [S1]; DAPN [M5],
   FBR-Net [X1]), driven by mixing 0.5–25 m sensors. At a single Sentinel-1 IW resolution, ship size in pixels varies little
   — enough for the xView3 winner to use one fixed heatmap radius [E2].
4. **Label quality and label-source bias.** Manual labels are costly and subjective (vessels, rocks and structures look
   alike at medium resolution [E1]); SSDD needed a full relabelling [D2]. AIS-derived labels [O1], [O2], [D6] are cheap
   but contain no dark vessels and favour larger, cooperative ships. The xView3 train split (automated labels only) holds
   284 near-shore objects against 6,171 in validation [E2], so automatic labels under-represent the coast; the winner judged
   label quality, not input signal, to be the limit [E2]. Evaluation against an automatic reference ([E7]) inherits its size
   floor and coastal exclusion.
5. **Domain shift between sensors and regions.** Multi-sensor chip sets mix resolutions and polarizations [D3], [D4];
   Sentinel-1 statistics differ between S1A and S1B and over time, forcing time-dependent CFAR thresholds [E7]; detection
   difficulty differs by region (Adriatic) [E1]; xView3 scenes cluster in about 42 locations shared across splits, so
   reported scores may be optimistic for new regions [E2]; a model trained on LS-SSDD needed fine-tuning for Persian Gulf
   imagery [R1]. [S1] recommends merging datasets to avoid overfitting.
6. **Chip-level versus scene-level evaluation.** Chip datasets drop ship-free samples, which hides false alarms over land
   and open sea [D5]; HRSID ships ship-free images only as an optional add-on [D4]. Scene-level protocols (LS-SSDD [D5],
   xView3 [E1]) give much lower numbers than chip AP50 (~98 % on SSDD [S1]).
7. **Comparability of reported numbers.** Random splits, missing inshore protocols and coarse labels made early SSDD results
   incomparable [D2]; metrics differ (AP50, COCO AP, mAP, F1 at 200 m, "accuracy" [R1], [R2]); even dataset counts differ
   between papers (SSDD 2456 vs 2358 ships [D2], [D5]). Numbers in Part C must not be ranked against each other.

---

## Part F. Synthesis (state of the art and its limits)

Since SSDD was released in 2017, deep detectors have replaced CFAR-plus-discriminator pipelines in the academic
SAR ship detection literature: on the same data a K-distribution CFAR reaches about a quarter of the AP of Faster R-CNN
(19.2 % vs 79.3 %) [S1],
and AP50 on SSDD climbed from 78.8 % to 97.8 % in five years [S1]. Most of this progress was made by adapting generic
detectors — two-stage Faster R-CNN [M1], one-stage YOLO and RetinaNet [M2], [M4], and, less often, anchor-free FCOS and
CenterNet [M3], [E4] — with SAR-specific multi-scale and attention modules [M5], [M6], [X1] on chip datasets built from
high-resolution imagery and manual labels [D1]–[D4]. These benchmarks are now saturated and do not reflect the
operational problem: chips are cut around ships and drop empty sea and land, so they barely measure false alarms
[D5]; their splits and metrics have not been consistent [D2]; and their 0.5–15 m imagery differs from the free 10 m
Sentinel-1 IW data used for wide-area monitoring. Where evaluation is done on whole Sentinel-1 scenes, scores are far
lower — about 76 % mAP on LS-SSDD [D5] and F1_D of about 0.77 for the best xView3 entry [E1].

The scene-level evidence points to a different design. In xView3, the only large public benchmark with full Sentinel-1
scenes and point labels, the leading entries were single-stage detectors for tiny, densely packed objects, and the winner
was a CenterNet-style point detector with an EfficientNet/U-Net encoder–decoder and a stride-2 output [E1], [E2] —
a combination that the chip literature barely explores (anchor-free models were about 5 % of papers in [S1]; U-Net
centre-map detectors such as [M7] remain rare). The operational systems that map vessels at global scale still use a CFAR
front end with CNN filtering [E7], or a box detector with temporal context and coastline/infrastructure masks [O3].
Three limits remain across all of this work: near-shore detection stays clearly worse than offshore detection and no
method specifically addresses it [E1], [D5]; labels are either manual and scarce or AIS-derived and blind to dark
vessels, and automatic labels under-represent the coast [O1], [E2], [E7]; and every benchmark and operational
validation is regionally limited — none includes the Persian Gulf [E1], [D5], with the only regional deep-learning result
being a fine-tuned YOLOv8 with an undefined "accuracy" figure [R1].

---

## Part G. Gaps relevant to this project

- **No published deep-learning evaluation on Persian Gulf / Iranian-coast Sentinel-1 scenes was found.** Searches
  (Crossref, Semantic Scholar, web search in English and Persian) found only: [R1] YOLOv8 fine-tuned on an undisclosed
  Gulf of Oman/Persian Gulf set with an "accuracy" of 90.4 %; [R2] an Iranian (University of Tehran) YOLOv8 study that
  uses only SSDD; [R3] a 2026 Hormuz study that uses a −15 dB threshold, not deep learning, validated on 18 images;
  [R4] platform (not ship) detection. xView3 contains no Middle East scenes [E1]. Persian-language databases (SID,
  Magiran, Civilica) were not searched directly, so a Persian conference paper may have been missed.
- **Chip benchmarks versus operational scenes.** Almost all reported SAR numbers are chip AP on ships-only crops
  [D1]–[D4], [S1]; scene-level evaluation with point matching exists only in xView3 (and box-level in LS-SSDD). The
  thesis trains on 128 px chips but evaluates whole scenes against GFW with distance matching, i.e. on the xView3 side of
  this divide; Chapter 3 should state the matching distance and whether a near-shore sub-score is reported.
- **Labels without manual work, and their bias.** Automatic AIS labels can beat manual ones [O1], and the xView3 train
  split is automatic only [E1], but such labels miss dark vessels and the coast [E2]. The thesis labels (CFAR detections
  confirmed by GFW) inherit GFW's size floor (<15–25 m) and 1 km coastal exclusion [E7]: scene-level agreement with GFW
  measures consistency with GFW, not true recall, and "dark vessels" (detections without AIS) can only be as complete as
  this reference. No paper found quantifies how such a CFAR×reference labelling scheme biases a trained detector.
- **Near-shore detection is unsolved.** Every top-5 xView3 team lost 0.24–0.32 F1 near shore and none designed for it [E1]; the
  Persian Gulf has dense islands, ports and about 1,731 platforms [R4]. Temporal context [O3] and platform maps [R4] are
  untested in this region.
- **New satellites, new data.** Sentinel-1B ended in Aug 2022 [W1]; Sentinel-1C (Dec 2024) and Sentinel-1D (Nov 2025)
  restore the pair and carry onboard AIS receivers [W1], [W2], which could give time-coincident AIS labels. No
  deep-learning ship detection study on Sentinel-1C/1D imagery was found (web search; not exhaustive), so transfer from
  models trained on S1A/S1B is untested.
- **Point detection is under-studied academically.** The design that won the only scene-level benchmark (centre heatmap,
  stride 2, encoder–decoder) [E2] is rare in the chip literature [S1]; a single lightweight model of this type
  (EfficientNet-B0 rather than a 12-model ensemble) has not been reported on Sentinel-1 scenes outside xView3.

---

## Part H. Bridge to Chapter 3 (what each design choice answers)

| Thesis choice | Motivated by |
|---|---|
| Point (centre heatmap) detector, no boxes | Ships are a few pixels at 10 m [E1], [E7]; anchors fit small ships badly [S1]; xView3 and GFW labels are points [E1], [E7]; xView3 winner [E2]; CenterNet [E4] |
| Output stride 2 + sub-pixel offset head | Stride table in [E2]; offset head from [E4]; 200 m matching tolerance [E1] |
| EfficientNet encoder + U-Net decoder | xView3 winner used EfficientNet B4/B5/V2-S + U-Net [E2]; [E3], [E5]; U-Net centre-map detector on SSDD [M7] |
| Penalty-reduced focal loss | Ship/background imbalance [E6], [M4]; CenterNet loss [E4]; same loss in [E2] |
| VV + VH input | xView3 [E1], LS-SSDD [D5], GFW ConvNet [E7], Skylight [O3] all use both channels |
| Automatic labels (CFAR × GFW), no manual labels | Automatic AIS labels can match or beat manual ones [O1]; xView3 train split automated [E1]; risk of coast/dark-vessel bias [E2], [E7] |
| Scene-level evaluation against GFW, CFAR baseline | Chip metrics hide false alarms [D5]; xView3 scene-level point matching [E1]; CFAR is the classical baseline and GFW's own front end [S1], [E7] |
