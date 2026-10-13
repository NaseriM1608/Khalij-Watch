# Chapter 2 literature: SAR–AIS fusion and dark vessels, labels without manual annotation, evaluation practice

Prepared for the rewrite of Chapter 2 of "Ship detection in Sentinel-1 SAR images of the Persian Gulf with deep learning, and finding dark vessels".
Verification date: 2026-10-11.

## How I checked the sources

- **Metadata** (authors, title, venue, volume, issue, pages or article number, year, DOI) comes from the Crossref API (`https://api.crossref.org/works/<DOI>`) for every DOI below.
- **Content** (what the paper did and what it found) comes from the abstract (Crossref, Semantic Scholar, OpenAlex or Europe PMC) and, where marked "full text", from the open-access full text (Europe PMC XML or arXiv PDF).
- **VERIFIED** means both the metadata and the claims in my note were checked against those sources.
- **PARTLY VERIFIED** means the metadata was checked but one named detail was not. That detail is flagged in the entry and should not be used as stated.
- Nothing here is invented. When I could not reach a source (MARAD pages and MDPI HTML returned 403), I say so and do not rely on it.
- Reference keys (A1, B1, C1, ...) exist only for this file. Renumber them for the thesis bibliography.

### References already in the thesis: details confirmed, not duplicated

| Existing reference | Confirmed details (Crossref) |
|---|---|
| Kroodsma et al., "Tracking the global footprint of fisheries" | *Science*, vol. 359, no. 6378, pp. 904–908, 2018, doi: 10.1126/science.aao5646 |
| Paolo et al., "Satellite mapping reveals extensive industrial activity at sea" | *Nature*, vol. 625, no. 7993, pp. 85–91, 2024, doi: 10.1038/s41586-023-06825-8. An Author Correction exists: *Nature* 626, E15, doi: 10.1038/s41586-024-07123-7 |
| Kuhn, "The Hungarian method for the assignment problem" | *Naval Research Logistics Quarterly*, vol. 2, no. 1–2, pp. 83–97, 1955, doi: 10.1002/nav.3800020109 |
| Efron & Tibshirani, *An Introduction to the Bootstrap* | Chapman & Hall/Springer, 1993, doi: 10.1007/978-1-4899-4541-9 |
| GFW APIs page | Not re-checked (web page) |

Paolo et al. 2024 is useful beyond the headline numbers, from the full text checked in Europe PMC (PMC10764273):
- Its training and evaluation sets were built from "SAR detections that matched to AIS data with high confidence". This is the same AIS-derived label source that category B discusses.
- Its SAR–AIS matching uses the probability-raster method of Kroodsma et al. 2022 [A3], with one score per candidate pair.
- It warns that a vehicle moving across the satellite path can appear more than 1 km from the road. The matching code accounts for this movement-induced offset.

---

## Core references (18)

### Category A: SAR–AIS association, fusion, dark vessels, AIS reliability

**[A1]** Z. Zhao, K. Ji, X. Xing, H. Zou, and S. Zhou, "Ship surveillance by integration of space-borne SAR and AIS – Review of current research," *J. Navig.*, vol. 67, no. 1, pp. 177–189, Jan. 2014, doi: 10.1017/S0373463313000659.
- Status: **VERIFIED** (Crossref metadata and abstract). Checked: https://api.crossref.org/works/10.1017/s0373463313000659
- The review you asked about exists. It was published online in Oct. 2013 and appeared in the Jan. 2014 issue.
- It reviews research that combines space-borne SAR and AIS for ship surveillance. Its starting point is that SAR offers wide-swath, all-weather imaging, while AIS gives identity and position but "not all ships are equipped with or operate AIS". The two are therefore treated as complementary.
- It covers SAR ship detection, AIS basics and the integration of the two (association and fusion). A companion paper by the same authors continues it: "…– Further research," *J. Navig.* 67(2), pp. 295–309, 2014, doi: 10.1017/S0373463313000702 (Crossref-verified).
- For the thesis this is the standard entry point for the claim that SAR and AIS are complementary and that association is the central step. It predates Sentinel-1 at scale and deep learning, so it frames the problem but not the current methods.

**[A2]** R. K. Raney, "Synthetic aperture imaging radar and moving targets," *IEEE Trans. Aerosp. Electron. Syst.*, vol. AES-7, no. 3, pp. 499–505, May 1971, doi: 10.1109/TAES.1971.310292.
- Status: **PARTLY VERIFIED**. Metadata and abstract were checked (Crossref, OpenAlex); the full text was not accessible. Checked: https://api.crossref.org/works/10.1109/taes.1971.310292
- This is the classic analysis of how slowly moving targets appear in a coherent side-looking SAR image. The abstract describes the image of a moving reflector and the effect of radar parameters on it.
- The result used in ship work comes from this analysis: a target's radial (range) velocity adds a Doppler shift, so the target is imaged displaced in azimuth, with Δaz ≈ (R/V)·v_r. Later papers attribute this to Raney. **The explicit formula was confirmed only through secondary sources, not the 1971 text.**
- The paper is about airborne SAR, but the physics carries over to Sentinel-1. This is why a moving ship's SAR position can be hundreds of metres from its true (AIS) position, and why the project's gate adds an R/V·speed term.
- Rough own estimate, to check against the thesis value: for Sentinel-1 IW, R/V is about 800–950 km ÷ ~7.5 km/s ≈ 105–125 s. A radial speed of 5 m/s then gives about 0.5–0.6 km of displacement.

**[A3]** D. A. Kroodsma, T. Hochberg, P. B. Davis, F. S. Paolo, R. Joo, and B. A. Wong, "Revealing the global longline fleet with satellite radar," *Sci. Rep.*, vol. 12, Art. no. 21004, Dec. 2022, doi: 10.1038/s41598-022-23688-7.
- Status: **VERIFIED** (Crossref metadata, abstract, full text in Europe PMC). Checked: https://api.crossref.org/works/10.1038/s41598-022-23688-7 and PMC9722684
- This is the method behind GFW's "matched/unmatched" SAR detections, the same reference data the project uses.
- From one year of AIS it builds probability rasters of where a vessel is likely to be minutes before or after an AIS ping, by vessel type, speed and time gap. The before and after rasters are combined, and each SAR detection gets a match score.
- The authors report that this "far outperformed" simple interpolation from speed and course. Pairs are then assigned **greedily** from a score matrix (highest score first, then remove that row and column), with a score threshold calibrated against two independent analysts.
- Findings: non-broadcasting vessels made up about 35% of longline activity north of Madagascar and about 10% near French Polynesia and Kiribati's Line Islands.
- For the thesis, this is the state of the art when point-level AIS is available. The project has only cell-hour AIS presence, so it cannot use this method directly. It does inherit the GFW labels the method produces.

**[A4]** A. Galdelli, A. Mancini, C. Ferrà, and A. N. Tassetti, "A synergic integration of AIS data and SAR imagery to monitor fisheries and detect suspicious activities," *Sensors*, vol. 21, no. 8, Art. no. 2756, 2021, doi: 10.3390/s21082756.
- Status: **VERIFIED** (Crossref metadata, abstract, full text in Europe PMC PMC8070621). Checked: https://api.crossref.org/works/10.3390/s21082756
- Data and detection: Sentinel-1 IW GRDH images over the central Adriatic, with ships detected by the JRC SUMO CFAR detector.
- Point-to-point association: a SAR target is associated when it falls inside a buffer around the AIS ping closest in time. The buffer radius grows with the distance the vessel could travel, d = (v·ΔT)·1.2 (a 20% margin).
- Point-to-line association: where a vessel's AIS track has a gap ("blackout") straddling the image time, targets are compared against the interpolated line. AIS gaps near managed areas are flagged as "suspicious".
- When a target falls in several buffers, it goes to the nearest buffer edge. That is a local decision, not a global assignment.
- For the thesis, this is the closest published precedent for the project's gate (cell + R/V·speed + margin), a speed-proportional gate with a safety factor. Its limitation is that it needs individual AIS pings.

**[A5]** M. Rodger and R. Guida, "Classification-aided SAR and AIS data fusion for space-based maritime surveillance," *Remote Sens.*, vol. 13, no. 1, Art. no. 104, 2021, doi: 10.3390/rs13010104.
- Status: **VERIFIED** (Crossref metadata and abstract; the MDPI page returned 403, so the full text was not read). Checked: https://api.crossref.org/works/10.3390/rs13010104
- It frames SAR–AIS data association as the key difficulty and notes that it fails in dense shipping, where detections get linked to the wrong AIS reports.
- It trains a ship-type classifier on AIS, transfers it to SAR detections, and feeds the predicted class into a **rank-ordered assignment** between SAR and AIS.
- Test areas: the English Channel (wide-area) and the Solent (focused). Using class information made the association robust in dense traffic and raised confidence in the matches.
- For the thesis, it supports a global (one-to-one) assignment rather than independent nearest-neighbour matches, and it names dense traffic as the main failure mode. The project cannot use class information, because GFW hourly presence carries no per-vessel type at the point level, so its cost is distance only.

**[A6]** R. Pelich, M. Chini, R. Hostache, P. Matgen, C. Lopez-Martinez, M. Nuevo, P. Ries, and G. Eiden, "Large-scale automatic vessel monitoring based on dual-polarization Sentinel-1 and AIS data," *Remote Sens.*, vol. 11, no. 9, Art. no. 1078, 2019, doi: 10.3390/rs11091078.
- Status: **VERIFIED** (Crossref metadata and abstract). Checked: https://api.crossref.org/works/10.3390/rs11091078
- It detects vessels over whole Sentinel-1 scenes using the complex coherence between the two polarization channels, then cross-compares the detections with AIS at large scale. Test areas: the English Channel and the Pacific coast of Mexico.
- Detection rate was above 80% for vessels longer than 60 m, with "a decrease of detection rate up to 40%" for smaller vessels.
- It also reports "many SAR detections without corresponding AIS positions", which it reads as evidence for dark vessels.
- For the thesis, this is a Sentinel-1 + AIS study that uses AIS as the yardstick. It shows the size dependence that also limits the project, since small vessels are under-detected by both CFAR and GFW.

**[A7]** J. Park, J. Lee, K. Seto, T. Hochberg, B. A. Wong, N. A. Miller, K. Takasaki, H. Kubota, Y. Oozeki, S. Doshi, M. Midzik, Q. Hanich, B. Sullivan, P. Woods, and D. A. Kroodsma, "Illuminating dark fishing fleets in North Korea," *Sci. Adv.*, vol. 6, no. 30, Art. no. eabb1197, 2020, doi: 10.1126/sciadv.abb1197.
- Status: **VERIFIED** (Crossref metadata, abstract, full text PMC7455503). Checked: https://api.crossref.org/works/10.1126/sciadv.abb1197
- It combines four satellite sources: AIS, SAR, VIIRS night lights, and PlanetScope/SkySat optical imagery. A CNN trained on 3 m PlanetScope imagery identifies pair trawlers, and SAR from three satellites verifies the fleet's location and size.
- Findings: more than 900 vessels of Chinese origin in 2017 and more than 700 in 2018 fished illegally in North Korean waters, and about 3000 North Korean vessels fished in Russian waters.
- For the thesis, it shows that "dark" fleets can be large and are found only by combining sensors with AIS. The project's dark-vessel candidates (SAR detection without AIS presence) are a single-sensor version of the same logic.

**[A8]** H. Welch, T. Clavelle, T. D. White, M. A. Cimino, J. Van Osdel, T. Hochberg, D. Kroodsma, and E. L. Hazen, "Hot spots of unseen fishing vessels," *Sci. Adv.*, vol. 8, no. 44, Art. no. eabq2109, 2022, doi: 10.1126/sciadv.abq2109.
- Status: **VERIFIED** (Crossref metadata, abstract, full text PMC9629714). Checked: https://api.crossref.org/works/10.1126/sciadv.abq2109
- From more than 3.7 billion AIS messages from fishing vessels (2017–2019), it uses rules to find gaps likely caused by intentional disabling: more than 55,000 suspected events in waters more than 50 nmi from shore.
- More than 40% of fishing vessels there had suspected disabling events, which hid up to 6% (more than 4.9 M hours) of activity. Hot spots were near the EEZs of Argentina and West Africa and in the NW Pacific, near transshipment areas and contested EEZ boundaries.
- Only areas with good satellite reception (more than 10 positions/day) were analysed. This is a reminder that AIS gaps can also be reception artefacts.
- For the thesis, it establishes that switching AIS off is common and patterned, which is one cause of "SAR without AIS". It also shows that gaps must be separated from poor reception.

**[A9]** A. Harati-Mokhtari, A. Wall, P. Brooks, and J. Wang, "Automatic Identification System (AIS): Data reliability and human error implications," *J. Navig.*, vol. 60, no. 3, pp. 373–389, Sep. 2007, doi: 10.1017/S0373463307004298.
- Status: **PARTLY VERIFIED**. Metadata and abstract were checked (Crossref, Cambridge page). Checked: https://api.crossref.org/works/10.1017/s0373463307004298 and the Cambridge Core abstract page.
- It uses earlier research and three separate AIS studies to show that AIS data quality depends on human input, especially manually entered fields. It adapts the "Swiss cheese" model of system failure to AIS.
- **Do not cite** the percentages often attributed to this paper (for example "2% wrong MMSI" and "30% wrong navigational status"). I found them only in secondary sources and could not confirm them in the paper.
- For the thesis, cite it for the qualitative point that AIS is self-reported and error-prone, so it is not a ground truth.

**[A10]** A. Androjna, I. Pavić, L. Gucma, P. Vidmar, and M. Perkovič, "AIS data manipulation in the illicit global oil trade," *J. Mar. Sci. Eng.*, vol. 12, no. 1, Art. no. 6, 2024 (published online 19 Dec. 2023), doi: 10.3390/jmse12010006.
- Status: **PARTLY VERIFIED**. Metadata and abstract were checked (Crossref, Semantic Scholar); the MDPI full text returned 403. Checked: https://api.crossref.org/works/10.3390/jmse12010006
- It documents AIS position falsification by tankers carrying Russian crude in ship-to-ship transfers. The false positions were found through repeated mismatches between AIS-reported positions and satellite radar imagery, and the authors reconstruct the tankers' true movements.
- It places this in the wider picture of GNSS/AIS spoofing for both military and commercial purposes.
- **UNVERIFIED:** whether the paper discusses Persian Gulf or Iranian cases. The abstract covers only the Russian-crude cases.
- For the thesis, this is peer-reviewed evidence that AIS positions can be deliberately wrong and that SAR can expose this. For the project, a vessel with spoofed AIS may appear as "SAR without AIS" at its true position.

**[A11]** Joint Maritime Information Center (JMIC), Combined Maritime Forces, in consultation with UKMTO and MSCIO, "JMIC Advisory Note – Update, JMIC# 006: Electromagnetic interference," 10 Oct. 2025. [Online]. Available: https://mscio.eu/media/documents/006_JMIC_Advisory_Note_UPDATE_001.pdf. Also: "Update 014 – JMIC Advisory Note, JMIC# 008-25: Regional tension – impact on maritime security (Arabian Gulf, Strait of Hormuz, Northern Arabian Sea)," 23 Jun. 2025. [Online]. Available: https://mscio.eu/media/documents/008_JMIC_Advisory_Note__Update__014.pdf. Accessed: 11 Oct. 2026.
- Status: **VERIFIED as grey literature** (official advisory; I downloaded both PDFs and read the text). This is not peer-reviewed, so mark it as an official advisory in the thesis.
- The Oct. 2025 note relays UKMTO Advisory 034-25. It reports numerous GNSS interference reports from 3–7 Oct. 2025, "a significant increase" over previous weeks.
- It says vessels reported interference affecting systems that draw position from GPS, Galileo and GLONASS, "such as ECDIS, ARPA, AIS, and GMDSS station". It reports higher concentrations near the Port of Asaluyeh, among other places, and says that in the Arabian Gulf the interference is "likely due to increased military activities".
- The June 2025 update reports "persistently high levels of electronic interference, particularly affecting GNSS systems" across the region.
- For the thesis, this is the citable source for GNSS interference in the study area. Corrupted GNSS means corrupted AIS positions, so some AIS-matched vessels may look unmatched and some "dark" candidates may be spoofed vessels. This argues for wide gates and for calling outputs "candidates".
- Not used: the MARAD MSCI advisories (for example 2019-012) could not be retrieved (HTTP 403), so I do not cite their text.

### Category B: Labels without manual annotation

**[B1]** F. Paolo, T.-T. T. Lin, R. Gupta, B. Goodman, N. Patel, D. Kuster, D. Kroodsma, and J. Dunnmon, "xView3-SAR: Detecting dark fishing activity using synthetic aperture radar imagery," in *Advances in Neural Information Processing Systems 35 (NeurIPS 2022)*, 2022, pp. 37604–37616, doi: 10.52202/068431-2726. (arXiv:2206.00897.)
- Status: **VERIFIED** (Crossref metadata for the proceedings DOI; arXiv v4 full text read; evaluation code checked on GitHub). Checked: https://api.crossref.org/works/10.52202/068431-2726, https://arxiv.org/abs/2206.00897, and https://github.com/DIUx-xView/xview3-reference/blob/main/reference/metric.py
- I could not confirm whether it was in the "Datasets and Benchmarks" track, so the track is not named here.
- Dataset: 991 Sentinel-1 IW scenes (Europe and West Africa) with 243,018 labels.
- Labels come from GFW CFAR detections correlated with AIS by the probabilistic method of [A3], plus professional manual labelling of 437 scenes (about 176,000 detections). Each label gets a confidence level (high, medium or low).
- The **training set (554 scenes) contains only automated GFW labels**. The validation, public and holdout sets combine automated and manual labels.
- The authors state that "an all-automated annotation approach would miss, among other things, vessels that do not broadcast AIS", which is the label bias the project found.
- The data are split by whole scene (train 554, validation 50, public 150, holdout 237). It is also used in category C.
- For the thesis, this is the closest analogue to the project's labelling. Its weakness for the Persian Gulf is that it has no Persian Gulf scenes and its unbiased evaluation sets depend on manual labels the project does not have.

**[B2]** J. Bekker and J. Davis, "Learning from positive and unlabeled data: a survey," *Mach. Learn.*, vol. 109, no. 4, pp. 719–760, Apr. 2020, doi: 10.1007/s10994-020-05877-5.
- Status: **VERIFIED** (Crossref metadata; arXiv 1811.04820 full text, Sections 2–3 read). Checked: https://api.crossref.org/works/10.1007/s10994-020-05877-5
- It surveys PU learning, where only some positives are labelled and the unlabelled set mixes positives and negatives. It organises the field around seven research questions, including the labelling mechanism and estimating the class prior.
- Under the "selected completely at random" (SCAR) assumption, labelled positives are a uniform sample of all positives, Pr(s=1|x) = c·Pr(y=1|x). A classifier that treats unlabelled data as negative then still ranks examples correctly.
- Under the more general "selected at random" (SAR) assumption, which positives get labelled depends on their attributes. Learning is then biased unless the labelling mechanism is modelled.
- For the thesis, this is the formal language for the project's label problem. Positives labelled because they match AIS are a SAR sample, since labelling depends on AIS broadcasting, which correlates with size and traffic lanes, so unlabelled vessels are systematically different. This explains why the first model missed dark vessels, why GFW detections without AIS had to be added as positives, and why unlabelled CFAR detections are never treated as negatives.

**[B3]** B. Frénay and M. Verleysen, "Classification in the presence of label noise: A survey," *IEEE Trans. Neural Netw. Learn. Syst.*, vol. 25, no. 5, pp. 845–869, May 2014, doi: 10.1109/TNNLS.2013.2292894.
- Status: **VERIFIED** (Crossref and OpenAlex metadata and abstract; full-text copy checked for the taxonomy). Checked: https://api.crossref.org/works/10.1109/tnnls.2013.2292894
- It defines the sources and types of label noise with a statistical taxonomy (Fig. 1): noisy completely at random (NCAR), noisy at random (NAR, depends on the class), and noisy not at random (NNAR, also depends on the features).
- It discusses the consequences: lower accuracy, more complex models, and more training samples needed.
- It reviews three families of methods: label-noise-robust, label-noise cleansing, and label-noise-tolerant. It also covers how to design experiments with noisy labels.
- For the thesis, automatically built labels contain CFAR false alarms that survive the GFW cross-check (ambiguities, platforms, coastline clutter) and miss small vessels. That noise is feature-dependent (NNAR), the hardest case. This justifies the conservative label rules and the warning that measured precision and recall inherit label noise.

### Category C: Evaluation practice for spatial and remote-sensing data

**[C1]** D. R. Roberts, V. Bahn, S. Ciuti, M. S. Boyce, J. Elith, G. Guillera-Arroita, S. Hauenstein, J. J. Lahoz-Monfort, B. Schröder, W. Thuiller, D. I. Warton, B. A. Wintle, F. Hartig, and C. F. Dormann, "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, Aug. 2017, doi: 10.1111/ecog.02881.
- Status: **VERIFIED** (Crossref metadata and abstract). Checked: https://api.crossref.org/works/10.1111/ecog.02881
- It reviews the literature, runs simulations and presents case studies showing that ignoring temporal, spatial or hierarchical dependence in random cross-validation leads to "serious underestimation of predictive error". It also opens the door to overfitting with non-causal predictors.
- Block cross-validation, which splits data strategically rather than randomly, was "nearly universally more appropriate" when the goal is predicting new data. The authors recommend it whenever dependence exists, even if it is not visible in the residuals.
- They warn that blocking can force extrapolation, so the block design must match the intended use.
- For the thesis, this is the general justification for splitting by scene. Tiles from one Sentinel-1 scene share sea state, incidence angle, calibration and even the same vessels, so the scene is the natural block.

**[C2]** P. Ploton, F. Mortier, M. Réjou-Méchain, N. Barbier, N. Picard, V. Rossi, C. Dormann, G. Cornu, G. Viennois, N. Bayol, A. Lyapustin, S. Gourlet-Fleury, and R. Pélissier, "Spatial validation reveals poor predictive performance of large-scale ecological mapping models," *Nat. Commun.*, vol. 11, Art. no. 4540, 2020, doi: 10.1038/s41467-020-18321-y.
- Status: **VERIFIED** (Crossref metadata and abstract). Checked: https://api.crossref.org/works/10.1038/s41467-020-18321-y
- It reproduces a typical large-scale biomass-mapping workflow: a random forest on multispectral and environmental predictors, with an inventory of 11.8 million trees in central Africa.
- Standard non-spatial validation suggested the model explained more than half of the biomass variation. Spatial validation that accounts for spatial autocorrelation revealed "quasi-null predictive power".
- For the thesis, this is a striking, widely cited demonstration that random splits of autocorrelated remote-sensing data produce falsely high scores. It supports the project's decision not to split tiles at random.

**[C3]** T. Kattenborn, F. Schiefer, J. Frey, H. Feilhauer, M. D. Mahecha, and C. F. Dormann, "Spatially autocorrelated training and validation samples inflate performance assessment of convolutional neural networks," *ISPRS Open J. Photogramm. Remote Sens.*, vol. 5, Art. no. 100018, Aug. 2022, doi: 10.1016/j.ophoto.2022.100018.
- Status: **VERIFIED** (Crossref metadata; abstract via OpenAlex). Checked: https://api.crossref.org/works/10.1016/j.ophoto.2022.100018
- The paper exists with these details.
- Task: CNN segmentation of tree species in multiple, spatially separate drone acquisitions.
- Compared with spatially blocked hold-outs, random hold-outs overestimated performance by up to 28%, and the optimism grew with smaller training sets.
- Spatial autocorrelation was much higher within an acquisition than between acquisitions. The authors recommend testing on "multiple independent remote sensing acquisitions".
- For the thesis, this is direct CNN evidence for the project's protocol: an acquisition (a Sentinel-1 scene) is the unit of independence, so train, validation and test are split by scene.

**[C4]** G. M. Foody, "Assessing the accuracy of land cover change with imperfect ground reference data," *Remote Sens. Environ.*, vol. 114, no. 10, pp. 2271–2285, Oct. 2010, doi: 10.1016/j.rse.2010.05.003.
- Status: **VERIFIED** (Crossref metadata and abstract). Checked: https://api.crossref.org/works/10.1016/j.rse.2010.05.003
- Using simulated data with known properties, it shows that even small errors in the reference data can strongly distort accuracy estimates.
- With 10% reference error, producer's accuracy was underestimated by 18.5% when the errors were independent of the classifier's errors, but overestimated by 12.3% when they were correlated.
- The author recommends avoiding the term "ground truth" and illustrates corrections (algebraic, or latent-class analysis).
- For the thesis, this is the remote-sensing source for evaluating against an imperfect reference. The project's reference (GFW) and model share Sentinel-1 input and CFAR-type detection, and the model was trained on GFW-derived labels, so their errors are probably correlated. Foody's result says scores against GFW may then be **optimistic**, and should be reported as agreement with GFW, not accuracy.

---

## Optional extra references (all verified, notes shortened)

**[X1]** C. Santamaria, M. Alvarez, H. Greidanus, V. Syrris, P. Soille, and P. Argentieri, "Mass processing of Sentinel-1 images for maritime surveillance," *Remote Sens.*, vol. 9, no. 7, Art. no. 678, 2017, doi: 10.3390/rs9070678.
- Status: **VERIFIED** (Crossref and abstract). Checked: https://api.crossref.org/works/10.3390/rs9070678
- The JRC study ran the SUMO detector on more than 11,500 Sentinel-1A images of the Mediterranean (Oct. 2014–Sep. 2016) and compared the detections with AIS in the Blue Hub fusion platform.
- Use it as the large-scale operational precedent for Sentinel-1 + AIS comparison. The SUMO algorithm itself is in Greidanus et al., *Remote Sens.* 9(3), 246, 2017, doi: 10.3390/rs9030246 (Crossref-verified).

**[X2]** F. Mazzarella, M. Vespe, and C. Santamaria, "SAR ship detection and self-reporting data fusion based on traffic knowledge," *IEEE Geosci. Remote Sens. Lett.*, vol. 12, no. 8, pp. 1685–1689, Aug. 2015, doi: 10.1109/LGRS.2015.2419371.
- Status: **VERIFIED** (Crossref and abstract via OpenAlex; full text not read).
- It improves SAR/AIS fusion by exploiting historical vessel-position knowledge (traffic patterns), tested on the Dover Strait. Use it as an example of predicting AIS positions to image time with prior traffic knowledge rather than straight-line interpolation.

**[X3]** A. A. Kurekin, B. R. Loveday, O. Clements, G. D. Quartly, P. I. Miller, G. Wiafe, and K. A. Agyekum, "Operational monitoring of illegal fishing in Ghana through exploitation of satellite earth observation and AIS data," *Remote Sens.*, vol. 11, no. 3, Art. no. 293, 2019, doi: 10.3390/rs11030293.
- Status: **VERIFIED** (Crossref and abstract).
- The INSURE system runs CFAR on Sentinel-1/2 and matches detections to AIS. It achieved 91% detection of AIS-registered vessels, but 75% of SAR detections over 17 months had no AIS counterpart.
- It shows how large the "unmatched" share can be in a weakly monitored region, and that unmatched detections are not automatically illegal.

**[X4]** L. Huang, B. Liu, B. Li, W. Guo, W. Yu, Z. Zhang, and W. Yu, "OpenSARShip: A dataset dedicated to Sentinel-1 ship interpretation," *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 11, no. 1, pp. 195–208, Jan. 2018, doi: 10.1109/JSTARS.2017.2755672.
- Status: **VERIFIED** (Crossref and abstract).
- It provides 11,346 Sentinel-1 ship chips "integrated with automatic identification system messages", meaning AIS supplies the labels. It is a pure AIS-label dataset, so by construction it contains only AIS-broadcasting ships. This is an example of the bias discussed in [B2].

**[X5]** C. Elkan and K. Noto, "Learning classifiers from only positive and unlabeled data," in *Proc. 14th ACM SIGKDD Int. Conf. Knowl. Discov. Data Min. (KDD)*, 2008, pp. 213–220, doi: 10.1145/1401890.1401920.
- Status: **VERIFIED** (Crossref and abstract; its role is confirmed by [B2], which cites it as the source of the SCAR-based "non-traditional classifier" results).
- Cite it as the original source if Chapter 2 states the SCAR result Pr(s=1|x) = c·Pr(y=1|x).

**[X6]** J. B. Reitsma, A. W. S. Rutjes, K. S. Khan, A. Coomarasamy, and P. M. Bossuyt, "A review of solutions for diagnostic accuracy studies with an imperfect or missing reference standard," *J. Clin. Epidemiol.*, vol. 62, no. 8, pp. 797–806, Aug. 2009, doi: 10.1016/j.jclinepi.2009.02.005.
- Status: **VERIFIED** (Crossref and the Europe PMC abstract).
- It groups solutions into four families: (1) impute or adjust for a missing reference; (2) correct accuracy estimates for a known-imperfect reference; (3) build a composite or consensus or latent-class reference; (4) give up the accuracy paradigm and validate against other relevant data.
- It is a general methodological source showing that "no gold standard" is a recognised problem with known options. The project uses option (4) in a weak form, by checking candidates against independent evidence rather than claiming accuracy.

**[X7]** C. A. Field and A. H. Welsh, "Bootstrapping clustered data," *J. R. Stat. Soc. Ser. B Stat. Methodol.*, vol. 69, no. 3, pp. 369–390, Jun. 2007, doi: 10.1111/j.1467-9868.2007.00593.x.
- Status: **VERIFIED** (Crossref and abstract).
- It shows that the cluster bootstrap (resampling whole clusters) gives consistent variance estimates under both the transformation model and the random-effect model.
- It supports the project's resampling of whole satellite passes rather than individual detections, because detections in one pass share conditions and are not independent. Use it alongside Efron & Tibshirani 1993.

**[X8]** G. Soldi *et al.*, "Space-based global maritime surveillance. Part II: Artificial intelligence and data fusion techniques," *IEEE Aerosp. Electron. Syst. Mag.*, vol. 36, no. 9, pp. 30–42, Sep. 2021, doi: 10.1109/MAES.2021.3070884.
- Status: **VERIFIED** (Crossref, abstract, arXiv 2011.11338 text).
- A recent review of SAR/optical/AIS fusion seen as multi-target tracking: measurement-origin uncertainty, JPDA/MHT, and sum-product (factor-graph) algorithms that fuse asynchronous satellite AIS with SAR detections.
- It is useful as a modern companion to [A1] and shows the probabilistic alternative to one-shot gating plus assignment.

---

## Synthesis for each category, with gaps that motivate the project

### A. SAR–AIS association and dark vessels

SAR and AIS complement each other [A1], [X8]. SAR sees every sufficiently large metal hull regardless of cooperation, while AIS adds identity and kinematics but only for vessels that broadcast. The literature splits association into four steps.
1. **Bring AIS to the image time.** Classic approaches interpolate or dead-reckon from the nearest pings. Galdelli et al. [A4] use a speed-proportional buffer d = 1.2·v·ΔT, and Mazzarella et al. [X2] add historical traffic knowledge. GFW's method [A3] replaces interpolation with empirical probability rasters of vessel displacement, which it reports to perform far better.
2. **Account for SAR geometry.** A target's radial velocity shifts it in azimuth by about (R/V)·v_r [A2]. Paolo et al. (2024) note that this offset can exceed 1 km for fast-moving targets, and their matching code allows for it.
3. **Gate.** Candidate pairs are restricted to a physically plausible distance.
4. **Assign.** Options range from greedy highest-score-first [A3] and nearest-buffer rules [A4] to rank-ordered or optimal one-to-one assignment [A5]. The latter is solvable with the Hungarian method (Kuhn 1955).

Dense traffic is the recognised failure mode, where local rules link detections to the wrong AIS reports [A5].

Large Sentinel-1 studies use AIS as the yardstick. They find high detection rates for large ships but much lower rates for small ones [A6], [X1], and a large share of SAR detections with no AIS counterpart: "many" in [A6] and 75% off Ghana [X3]. Global studies show that this share is real and large.
- AIS-based mapping [Kroodsma 2018] misses non-broadcasting fleets.
- Multi-sensor work uncovered hundreds of illegal pair trawlers [A7].
- Intentional AIS disabling hides up to 6% of fishing activity far from shore [A8].
- Non-broadcasting longliners reach about 35% of activity in some regions [A3].
- 72–76% of industrial fishing vessels are not publicly tracked [Paolo 2024].

At the same time, AIS itself is unreliable. It is self-reported and prone to human error [A9], can be deliberately falsified, with SAR exposing the false positions [A10], and in the study area is degraded by GNSS interference that official advisories say affects AIS. The interference is concentrated near Asaluyeh in Oct. 2025 [A11]. "SAR without AIS" therefore has several causes: switched-off AIS, poor AIS reception, spoofed or corrupted AIS position, small vessels without AIS, or a SAR false alarm.

**Gaps and challenges that motivate the project:**
- **No free point-level AIS for the Persian Gulf, hence cell-hour fusion.** Every association method reviewed ([A3]–[A6], [X1]–[X3]) assumes individual AIS pings with time, position and speed. The project has only GFW's hourly presence per 0.01° cell. Published point-matching methods therefore cannot be applied, and a cell-hour formulation is needed. The gate (cell + R/V·speed + margin) is a cell-level analogue of the speed-proportional buffer in [A4], with an extra worst-case azimuth-shift term [A2] because per-ship heading and radial speed are unknown.
- **Global versus local assignment.** Greedy or nearest-edge rules ([A3], [A4]) can double-count or mis-assign in dense traffic [A5]. A one-to-one Hungarian assignment (Kuhn 1955) is the principled choice when the cost is distance only. The class-aided costs of [A5] are not possible without per-vessel type.
- **"Unmatched" is not "illegal".** Because of AIS gaps [A8], human error [A9], spoofing [A10] and regional GNSS interference [A11], unmatched detections must be reported as **dark-vessel candidates**, not confirmed dark vessels. GNSS interference in the Gulf may inflate the candidate count.
- **Regional evidence gap.** My search (Crossref and web) found no peer-reviewed SAR–AIS dark-vessel study focused on the Persian Gulf. Global products [Paolo 2024] include it but give no regional analysis. This is search-based, not exhaustive, so phrase it carefully in the thesis.

### B. Labels without manual annotation

Manual annotation of SAR vessels is slow, subjective (rocks, platforms and ambiguities look alike), and cannot recover attributes such as vessel type [B1]. Large SAR datasets therefore take labels from AIS or from another detector.
- OpenSARShip pairs Sentinel-1 chips with AIS messages [X4].
- The xView3 training set uses only automated GFW labels, CFAR detections correlated with AIS [B1].
- Paolo et al. (2024) train on high-confidence AIS matches.

The cost is selection bias. In PU-learning terms [B2], positives labelled because they match AIS are not a uniform (SCAR) sample of all vessels but an attribute-dependent (SAR) one: broadcasting vessels are larger, on shipping lanes, and of particular types. A model trained on them, especially one that treats unlabelled detections as negatives, learns "AIS vessel" rather than "vessel". The xView3 authors say this directly: an all-automated labelling approach would miss vessels that do not broadcast AIS, which is why they added manual labels [B1].

Automatic labels are also noisy [B3]. CFAR false alarms that pass the cross-check, and vessels missed by both sources, create label noise that depends on the features (NNAR: near shore, around platforms, small targets). That is the hardest noise class. Robust, cleansing and tolerant strategies exist, but all of them need some clean data or assumptions about the noise.

**Gaps and challenges that motivate the project:**
- **No manual labels for the Persian Gulf.** xView3 covers Europe and West Africa only [B1], and the project has no budget for professional annotators. Labels must be built automatically, from CFAR detections cross-checked against GFW SAR detections.
- **AIS-only positives are biased against dark vessels.** The literature predicts the project's first-model failure ([B1], [B2]). The fix of adding GFW detections without AIS as positives brings the labelled set closer to the full positive distribution. It still inherits GFW's own detection limits (small vessels, near shore).
- **Unlabelled is not negative.** Following the PU view [B2], unmatched CFAR detections are never used as negatives. Because the class prior and the labelling mechanism are unknown, model scores are not calibrated probabilities. Some of the model's "false positives" against GFW may be real vessels GFW missed.
- **Feature-dependent label noise** [B3] near coasts, platforms and ambiguities means that error analysis should be broken down by distance to shore, as xView3 does with its close-to-shore F1 [B1].

### C. Evaluation practice for spatial and remote-sensing data

Samples that are close in space or time are not independent. Random splits of such data underestimate error [C1]. In remote sensing they can make a model with almost no predictive power look good [C2], and they inflated CNN scores by up to 28% [C3].

Kattenborn et al. [C3] also found autocorrelation much stronger within an acquisition than between acquisitions, so the acquisition is the natural unit of independence. xView3 likewise splits by whole scene [B1].

For point-like objects, detection is scored by matching predictions to references within a tolerance. xView3 computes F1 after Hungarian matching with a 200 m tolerance, plus a separate F1 for objects within 2 km of shore [B1]. This is confirmed in its `metric.py`, which defaults to `distance_tolerance=200` and uses `linear_sum_assignment`.

When the reference itself is imperfect, accuracy estimates can be biased in either direction. With errors correlated between the map and the reference, they are biased upward [C4], and the medical literature offers formal options for this case [X6].

Finally, detections from one satellite pass share conditions, so confidence intervals should resample passes, not detections. This is the cluster bootstrap [X7], built on Efron & Tibshirani (1993). Comparing two models on the same resampled passes gives a paired bootstrap.

**Gaps and challenges that motivate the project:**
- **Leakage, hence a scene-level split.** Random tile splits would put tiles of the same scene, including the same sea state, calibration and vessels, in both train and test ([C1]–[C3]). Splitting by scene follows [C3] and the xView3 practice [B1].
  - Residual risk: scenes of the same relative orbit over the same Gulf are still related. The test measures generalisation within the region, not to other seas. A region- or time-blocked split would be stronger [C1].
- **No manual ground truth, so GFW is a reference, not the truth.** All scores are agreement with GFW. Because the model was trained on GFW-derived labels and both use Sentinel-1 CFAR-like detection, their errors are probably correlated. Foody [C4] shows that correlated reference errors bias accuracy upward. Report results as agreement and avoid the term "ground truth" [C4].
- **Cell-level matching instead of a fixed distance.** The GFW reference is binned to 0.01° cells. That is about 1.1 km north–south and about 1.0 km east–west at 26–27°N (my own calculation). Point matching at 200 m as in xView3 is therefore not possible. Cell matching is more tolerant of misplacement and can over-count true positives in dense clusters, so state the tolerance explicitly.
- **Dependent detections, hence a paired bootstrap over passes.** Detections in one pass are not independent, so confidence intervals come from resampling passes [X7]. Model comparisons use the same resampled passes (paired), which removes pass-to-pass variation from the difference.

---

## Map from project choices to literature (for linking Chapter 2 to Chapter 3)

| Chapter 3 choice | Literature that motivates it |
|---|---|
| CFAR ∩ GFW automatic labels | [B1], [X4], Paolo 2024 (AIS- or detector-derived labels); [B3] (noise) |
| Adding GFW detections without AIS as positives | [B2] (SAR-type selection bias), [B1] (automated labels miss non-AIS vessels) |
| Never treating unlabelled detections as negatives | [B2], [X5] (PU learning) |
| Split by scene | [C1], [C2], [C3], [B1] |
| Cell-level matching against GFW | [B1] (matching-based F1, 200 m), GFW cell binning |
| Calling results "agreement with GFW" | [C4], [X6] |
| Paired bootstrap over passes | Efron & Tibshirani 1993, [X7] |
| Cell-hour gate = cell + R/V·speed + margin | [A2] (azimuth shift), [A4] (speed-proportional buffer), [A3] (displacement uncertainty) |
| Hungarian one-to-one assignment | Kuhn 1955, [A5] (global assignment in dense traffic) vs [A3]/[A4] (greedy/local) |
| Output named "dark-vessel candidates" | [A8], [A9], [A10], [A11], [X3] |

## Verification tally

- **Core (18):** 14 fully VERIFIED, 3 PARTLY VERIFIED, 1 VERIFIED grey-literature advisory.
  - PARTLY VERIFIED: [A2] Raney (the explicit Δaz formula was confirmed only through secondary sources); [A9] Harati-Mokhtari (do not use the commonly quoted error percentages); [A10] Androjna (whether Persian Gulf cases are covered is unknown).
  - Grey literature: [A11] JMIC (official, not peer-reviewed).
- **Optional (8):** all VERIFIED. [X2] content is from the abstract only.
- **Existing thesis references confirmed:** 4 (Kroodsma 2018, Paolo 2024, Kuhn 1955, Efron & Tibshirani 1993).
- **Fully UNVERIFIED references:** 0.
- **Unresolved details:** the NeurIPS track name for [B1]; the MARAD advisory text (403, not used).
