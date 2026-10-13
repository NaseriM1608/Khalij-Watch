# Chapter 2 literature: classical and statistical SAR ship detection, and operational systems

Scope: the CFAR family, sea-clutter models, fixes for mixed clutter and for several targets in one window, polarimetric and dual-pol detectors, operational Sentinel-1 systems, surveys, and the known failure modes of all of these.
Checked on 2026-10-11.

How the checks were done:
- Every DOI was resolved on the Crossref API (`https://api.crossref.org/works/<DOI>`). Title, authors, venue, volume, issue, pages and date below are copied from Crossref.
- Abstracts came from Crossref, or from OpenAlex (`https://api.openalex.org/works/https://doi.org/<DOI>`) when Crossref had none.
- For the open-access MDPI papers, the UPC repository copy of Stasolla et al., and the Cao et al. paper, the full text was read and the numbers were taken from it.
- "VERIFIED" means the bibliographic record was confirmed online. Any detail I could not confirm is flagged inside the note.

---

## 1. Reference list (14 core, 2 optional, 2 already in the thesis)

### Surveys

**[C1]** K. El-Darymli, P. McGuire, D. Power, and C. R. Moloney, "Target detection in synthetic aperture radar imagery: a state-of-the-art survey," *J. Appl. Remote Sens.*, vol. 7, no. 1, Art. no. 071598, pp. 1–35, Mar. 2013, doi: 10.1117/1.JRS.7.071598.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1117/1.JRS.7.071598, plus the abstract on Crossref.
- Note: This survey proposes a taxonomy of SAR target detectors and states the assumptions behind each family. The names of the taxonomy's groups were not checked against the full text, so do not quote them from this note. It compares representative methods in a table and discusses how well the multiplicative SAR clutter models fit real data. Its key insight is that CFAR is "a suboptimal one-class classifier": one-parameter CFAR is a Euclidean-distance classifier, and two-parameter CFAR is a quadratic discriminant with a missing term. It is a review, so it reports no new experiments. It is also not ship-specific, since it covers SAR automatic target recognition in general. In Chapter 2 it gives the theoretical reason why a two-class detector that learns both ship and sea appearance can do better than CFAR, which models only the sea.

**[E1] (already in the thesis)** D. J. Crisp, "The state-of-the-art in ship detection in synthetic aperture radar imagery," Defence Science and Technology Organisation (DSTO), Australia, Res. Rep. DSTO-RR-0272, May 2004. DTIC accession ADA426096.
- Status: VERIFIED (report number, author, title and May 2004 date). Checked: https://apps.dtic.mil/sti/pdfs/ADA426096.pdf, read through the Exa fetch because a direct download returned 403. The report has no DOI.
- Note: This review was written to choose algorithms for DSTO's Analysts' Detection Support System. It describes ship detection as four stages: land masking, preprocessing, prescreening (usually CFAR) and discrimination. It recommends a basic system for single-channel imagery at low to medium resolution, and leaves wake detection aside. Use it in Chapter 2 as the frame for the classical pipeline. Chapter 3's pipeline maps onto it directly: land mask, then CFAR baseline or learned detector, then peak extraction.

### The CFAR family and multiple targets in the reference window

**[C2]** H. Rohling, "Radar CFAR thresholding in clutter and multiple target situations," *IEEE Trans. Aerosp. Electron. Syst.*, vol. AES-19, no. 4, pp. 608–621, Jul. 1983, doi: 10.1109/TAES.1983.309350.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/TAES.1983.309350; abstract via OpenAlex.
- Note: This paper introduces ordered-statistic CFAR (OS-CFAR). The clutter level is estimated from a single ranked sample of the reference window rather than from the window mean, as cell-averaging CFAR (CA-CFAR) does. It is an analytical and simulation study of radar detection, not SAR imagery. Its abstract says OS-CFAR beats CA-CFAR when more than one target sits in the reference window, or when the window crosses a clutter edge. Both situations are common for SAR ships in shipping lanes, at anchorages and near coasts. The usual weakness, which I did not check numerically in this paper, is a small detection loss in uniform clutter, plus the need to sort the window at every pixel.

**[C3]** P. P. Gandhi and S. A. Kassam, "Analysis of CFAR processors in nonhomogeneous background," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 24, no. 4, pp. 427–445, Jul. 1988, doi: 10.1109/7.7185.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/7.7185; abstract via OpenAlex.
- Note: The paper compares five CFAR schemes analytically: cell-averaging (CA), greatest-of (GO), smallest-of (SO), ordered-statistics (OS) and trimmed-mean (TM). It tests them in uniform clutter and in two mixed cases: several targets in the window, and a clutter transition. Detection performance is compared through the average detection threshold, using an exponential noise model so that closed-form expressions exist. The abstract confirms this scope. The well-known trade-off is that GO limits false alarms at clutter edges but hides closely spaced targets, SO does the opposite, and OS/TM handle interfering targets well. That conclusion is the standard reading of the paper; I did not check it against the full text. The weakness is the exponential (Rayleigh-intensity) clutter assumption, which heavy-tailed SAR sea clutter breaks.

**[C4]** G. Gao, L. Liu, L. Zhao, G. Shi, and G. Kuang, "An adaptive and fast CFAR algorithm based on automatic censoring for target detection in high-resolution SAR images," *IEEE Trans. Geosci. Remote Sens.*, vol. 47, no. 6, pp. 1685–1697, Jun. 2009, doi: 10.1109/TGRS.2008.2006504.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/TGRS.2008.2006504; abstract via OpenAlex.
- Note: An adaptive global threshold first marks pixels that might be targets. Those pixels are then censored (excluded) from the local sliding window before the clutter is modelled with the G0 distribution. This keeps target pixels from inflating the background estimate, and a fast implementation cuts the cost of the sliding window. The authors report good CFAR behaviour in uniform regions, at clutter edges and with several targets, through theoretical analysis and experiments on "typical real SAR images". The sensor and the numbers were not confirmed from the abstract. The weakness is that it still depends on a parametric model and on a hand-set pre-screening threshold, and it is a single-channel method.

### Sea-clutter distribution models

**[C5]** A. C. Frery, H.-J. Müller, C. C. F. Yanasse, and S. J. S. Sant'Anna, "A model for extremely heterogeneous clutter," *IEEE Trans. Geosci. Remote Sens.*, vol. 35, no. 3, pp. 648–659, May 1997, doi: 10.1109/36.581981.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/36.581981; abstract via OpenAlex.
- Note: This paper derives the G family of distributions from the multiplicative speckle model, for intensity and amplitude data, single-look and multilook. The K distribution is a special case. The G0 member has as many parameters as K but also fits extremely heterogeneous clutter that K cannot. One of its parameters measures how homogeneous the clutter is. The paper validates the model on land scenes (urban, primary forest, deforested) from two sensors, not on sea. G0 later became a standard background model for SAR ship CFAR, as in [C4]. Its weaknesses are that parameter estimation in small windows is unstable, and that sea scenes were not part of the original validation.

**[C6]** X. Qin, S. Zhou, H. Zou, and G. Gao, "A CFAR detection algorithm for generalized gamma distributed background in high-resolution SAR images," *IEEE Geosci. Remote Sens. Lett.*, vol. 10, no. 4, pp. 806–810, Jul. 2013, doi: 10.1109/LGRS.2012.2224317.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/LGRS.2012.2224317; abstract via OpenAlex.
- Note: This paper models the background with the generalized gamma distribution (GΓD), estimates its parameters by the method of log-cumulants, and derives a closed-form threshold through the inverse incomplete gamma function. It is compared with CFAR detectors that use Weibull, K_A and G_A0 backgrounds. GΓD is reported to hold the false-alarm rate better and to run efficiently, on "an actual high-resolution SAR image". The sensor and the numbers were not confirmed from the abstract. In Chapter 2 this paper is the reference for the point that no single distribution (K, G0, Weibull, log-normal, GΓD) fits sea clutter in every case. Its weakness is that it was validated on one image. SUMO's authors also say they have not checked whether GΓD beats K operationally [C11].

### Polarimetric and dual-pol detectors

**[C7]** L. M. Novak, M. C. Burl, and W. W. Irving, "Optimal polarimetric processing for enhanced target detection," *IEEE Trans. Aerosp. Electron. Syst.*, vol. 29, no. 1, pp. 234–244, 1993, doi: 10.1109/7.249129.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/7.249129; abstract via OpenAlex. Crossref gives no month, so none is shown.
- Note: The polarimetric whitening filter (PWF) combines HH, HV and VV into one intensity image with minimum speckle, using adaptive and non-adaptive versions. A two-parameter CFAR detector then runs on the filtered image. The test data are real targets in clutter from the Lincoln Laboratory 35 GHz airborne SAR, which means land targets, not ships at sea. The non-adaptive PWF beat both the adaptive PWF and single-channel HH. The weakness is that it needs fully polarimetric complex data and covariance estimates. Sentinel-1 GRD provides only two intensity channels (VV, VH), so the PWF can only be approximated there.

**[C8]** A. Marino, "A notch filter for ship detection with polarimetric SAR data," *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 6, no. 3, pp. 1219–1232, Jun. 2013, doi: 10.1109/JSTARS.2013.2247741.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/JSTARS.2013.2247741; abstract via OpenAlex.
- Note: The polarimetric notch filter models the sea's polarimetric signature and flags anything whose polarimetric behaviour differs from it: ships, icebergs, buoys. It is built on perturbation analysis in the target space, and the paper also designs a dual-pol version. It was tested on TerraSAR-X quad-pol data, with "significant agreement" with ground truth, and through Monte Carlo estimates of detection and false-alarm probabilities. The detector is reported to be largely independent of sea conditions. The weakness is that it needs complex covariance data (SLC), not GRD intensities, and that buoys, platforms and icebergs still pass the filter because they are also "not sea".

**[C9]** R. Pelich, M. Chini, R. Hostache, P. Matgen, C. Lopez-Martinez, M. Nuevo, P. Ries, and G. Eiden, "Large-scale automatic vessel monitoring based on dual-polarization Sentinel-1 and AIS data," *Remote Sens.*, vol. 11, no. 9, Art. no. 1078, May 2019, doi: 10.3390/rs11091078.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.3390/rs11091078; full text read at https://www.mdpi.com/2072-4292/11/9/1078.
- Note: The detector works on the complex VV–VH coherence of Sentinel-1 IW SLC data. Because coherence falls from near range to far range, the threshold adapts per IW sub-swath. The baseline is CFAR on GRD VH intensity. Validation is a large-scale comparison with AIS over the English Channel and the Pacific coast of Mexico; one statistic uses about 1,400 AIS vessels from 10 images. Detection rate is above 80% for vessels longer than 60 m, with "a decrease of detection rate up to 40%" for smaller vessels. Visual checks found that at least half of the extra detections from VH-CFAR were false alarms from clutter and land ambiguities near the coast. The authors also note that many SAR detections had no AIS match, which points to dark vessels. Weaknesses: the method needs SLC data, so it cannot be applied to the GRD products used in this thesis; the per-sub-swath threshold is hand-designed; and small vessels stay hard to detect.

### Azimuth ambiguities

**[C10]** D. Velotto, M. Soccorsi, and S. Lehner, "Azimuth ambiguities removal for ship detection using full polarimetric X-band SAR data," *IEEE Trans. Geosci. Remote Sens.*, vol. 52, no. 1, pp. 76–88, Jan. 2014, doi: 10.1109/TGRS.2012.2236337.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/TGRS.2012.2236337; abstract via OpenAlex.
- Note: Azimuth ambiguities are "ghost" copies of bright scatterers, caused by aliasing of the Doppler phase history. The paper notes they are especially visible at low wind speed, in coastal areas and in harbours. Their sources are ships, oil platforms and bright structures on land near the coast. The paper separates them from real targets using their different signature in the cross-polarized channels of a two-channel PolSAR system, and detects targets with a generalized-K CFAR. The data are TerraSAR-X quad-pol from the dual-receive-antenna campaign of April–May 2010, validated against collocated AIS. Its importance for Chapter 2 is that it shows classic CFAR mistakes ambiguities for ships. The weakness is that the fix relies on a quad-pol experimental mode that Sentinel-1 does not offer.

### Operational systems on Sentinel-1 and comparable sensors

**[C11]** H. Greidanus, M. Alvarez, C. Santamaria, F.-X. Thoorens, N. Kourti, and P. Argentieri, "The SUMO ship detector algorithm for satellite radar images," *Remote Sens.*, vol. 9, no. 3, Art. no. 246, Mar. 2017, doi: 10.3390/rs9030246.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.3390/rs9030246; full text read at https://www.mdpi.com/2072-4292/9/3/246.
- Note: SUMO is the EU Joint Research Centre's (JRC) pixel-based CFAR detector, developed over more than 15 years for L-, C- and X-band SAR at 1–100 m resolution.
  - Pipeline: a land mask from GSHHG or OpenStreetMap with a seaward buffer (typically about 100 m); tile-based clutter estimation with a corrected K distribution; thresholding at a nominal false-alarm probability; clustering of target pixels; fusion of the polarization channels; geometric flagging of azimuth ambiguities (m = ±1, 2, plus a Sentinel-1-specific ghost feature); and multi-image removal of recurrent fixed targets.
  - Threshold: with PFA = 1e-7 the authors expect about 2.5 false alarms per 10,000 × 10,000-pixel image, "a compromise between too many false alarms and too many undetected small ships". An empirical adjustment factor *f* must be raised at high sea state, with icebergs, or under an inhomogeneous wind field.
  - Remaining false alarms, illustrated on a Sentinel-1 IW scene of Spain: coastal infrastructure outside the land mask, aquaculture, ambiguities of ships outside the image, oil platforms without AIS, and the cross-pol signature of ship wakes.
  - On dual-pol: cross-pol gives very high ship-to-sea contrast, but some ships have low cross-pol backscatter, and cross-pol becomes weak at far range.
  - Scale: batches of more than 10,000 images ran in under two hours.
  - The paper cites benchmark tests but gives no single summary detection probability. Weaknesses: thresholds tuned by hand, false-alarm handling built from rules, and an operator still recommended to confirm detections.

**[C12]** C. Santamaria, M. Alvarez, H. Greidanus, V. Syrris, P. Soille, and P. Argentieri, "Mass processing of Sentinel-1 images for maritime surveillance," *Remote Sens.*, vol. 9, no. 7, Art. no. 678, Jul. 2017, doi: 10.3390/rs9070678.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.3390/rs9070678; full text read at https://www.mdpi.com/2072-4292/9/7/678.
- Note: SUMO was run on 11,573 Sentinel-1A IW images of the Mediterranean, covering October 2014 to September 2016 and excluding 74 images with radio-frequency interference (RFI). Processing ran on the JRC JEODPP platform, with AIS fusion in Blue Hub.
  - Settings: PFA 1e-7, threshold adjustments of 10.0 for co-pol and 2.0 for cross-pol. A detection matches an AIS position within 500 m, or possibly within 5,000 m.
  - Detections: 667,746 in total. 59,420 were given the lowest reliability, mostly azimuth ambiguities, leaving 608,326 reliable ones. In dual-pol products, 98% of targets were found in the cross-pol channel and 54% in co-pol.
  - AIS match: 366,081 detections matched AIS. 40% of detections stayed unmatched.
  - Recurrence: 20% of detections were recurrent. Of all reliable detections, 15% were fixed structures and 5% ambiguities, found by a multi-image analysis of recurrence.
  - Weaknesses: RFI had to be flagged by hand; thresholds were set high, trading away small ships, because no operator was in the loop; and the evaluation relies on AIS, which cannot separate dark ships from false alarms.

**[C13]** M. Stasolla, J. J. Mallorqui, G. Margarit, C. Santamaria, and N. Walker, "A comparative study of operational vessel detectors for maritime surveillance using satellite-borne synthetic aperture radar," *IEEE J. Sel. Topics Appl. Earth Observ. Remote Sens.*, vol. 9, no. 6, pp. 2687–2701, Jun. 2016, doi: 10.1109/JSTARS.2016.2551730.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1109/JSTARS.2016.2551730. Numbers taken from the author copy at https://upcommons.upc.edu/bitstreams/ee6eea0d-fa6f-4543-8560-6a4c027a4ab3/download and the IEEE page text returned by Exa.
- Note: Four operational detectors were compared on five RADARSAT-2 and COSMO-SkyMed images, taken off West Africa (March 2013) and over the Mediterranean (June 2013): JRC SUMO, UPC-WT (wavelet-based), GMV SIDECAR and eOsphere R&G.
  - Test design: the scenes were chosen for difficult conditions (ambiguities, strong sea clutter, irregular shorelines) and checked against 232 verified targets. Metrics were probability of detection (PoD), false-alarm rate (FAR) and size accuracy.
  - Overall PoD: SUMO about 90%, UPC-WT 85%, R&G 75%, SIDECAR just above 50%. FARs were around 6e-9 to 1e-8.
  - Typical errors:
    - SUMO: 67 of its 77 false alarms came from poor coastal masking.
    - SIDECAR: mostly sea clutter.
    - UPC-WT: large ships split into several detections.
    - R&G: recognised none of the ambiguities as such, so they counted as targets.
  - One scene was dominated by small fishing boats, and the best PoD there was only 52%.
  - A majority vote of the four detectors reached about 87% PoD with only 9 false alarms.
  - The authors' conclusion is that "there is no perfect detector". Weaknesses: only five scenes, and no Sentinel-1 data.

**[C14]** Y. Cao *et al.* (Y. Cao, H. Wang, F. Su, D. Fu, F. Yan, X. Hong, W. He, N. Xu, Y. Ou, and C. Chen), "Satellite radar and AIS reveal a 97% decline in shipping traffic through the Strait of Hormuz," *The Innovation*, vol. 7, no. 6, Art. no. 101367, Jun. 2026, doi: 10.1016/j.xinn.2026.101367.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1016/j.xinn.2026.101367; text read at https://www.cell.com/the-innovation/fulltext/S2666-6758(26)00114-1 and https://pmc.ncbi.nlm.nih.gov/articles/PMC13237841/.
- Note: This is the only recent Sentinel-1 ship-detection study I found for the Persian Gulf itself. It analyses 22 Feb – 7 Mar 2026, around the blockade announced on 28 Feb 2026.
  - Method: 258 Sentinel-1 IW GRD scenes were preprocessed in Google Earth Engine and converted to σ0 in dB. Pixels above a fixed empirical threshold of −15 dB were kept, a high-precision ocean mask was applied, clusters smaller than 3 pixels were removed, and the results were inspected by hand.
  - Result: on 18 test images covering ports and open water, F1 was 96.0% (precision 94.2%, recall 97.9%). Combined with AIS, the detections showed a drop of about 97% in transits.
  - Weaknesses, as the authors state them: the fixed revisit and "signal confusion in congested waters". The fixed global threshold has no clutter model, and manual inspection was needed to remove artefacts. The paper is a short communication; the parts I read do not describe how offshore platforms were handled.
- Relevance: it shows that a simple threshold detector plus AIS is used in practice for exactly this region and sensor. This makes it a natural regional reference next to the thesis CFAR baseline.

### Optional (verified; useful for the "fixed structures" challenge in the Persian Gulf)

**[O1]** R. Spanier, T. Hoeser, J. Truckenbrodt, F. Bachofer, and C. Kuenzer, "Offshore oil and gas platform dynamics in the North Sea, Gulf of Mexico, and Persian Gulf: exploiting the Sentinel-1 archive," *Big Earth Data*, pp. 1–27, published online 15 Jun. 2026, doi: 10.1080/20964471.2026.2679328.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1080/20964471.2026.2679328; abstract via Crossref and OpenAlex. Crossref has not yet assigned a volume or issue (online-first article).
- Note: The authors applied deep-learning object detection to the Sentinel-1 archive and built quarterly time series of platform locations for 2017–2025. They identified 3,728 platforms in 2025, of which **1,731 were in the Persian Gulf**, more than in the Gulf of Mexico (1,641) or the North Sea (356). More than 2,700 platforms were installed or moved during the period, and mobile units such as jack-ups and drillships are growing in number. For Chapter 2, this puts a number on the biggest fixed-structure false-alarm source in the study area, and it shows that a static platform mask goes out of date.

**[O2]** B. A. Wong, C. Thomas, and P. Halpin, "Automating offshore infrastructure extractions using synthetic aperture radar & Google Earth Engine," *Remote Sens. Environ.*, vol. 233, Art. no. 111412, Nov. 2019, doi: 10.1016/j.rse.2019.111412.
- Status: VERIFIED. Checked: https://api.crossref.org/works/10.1016/j.rse.2019.111412; abstract via Crossref.
- Note: This is a classical method that maps offshore oil platforms in the Gulf of Mexico from Sentinel-1 on Google Earth Engine. A later paper (Ma et al., IEEE JOE 2023, https://strathprints.strath.ac.uk/86868/) describes it as a median composite over time, difference of Gaussians, thresholding and morphology. That description comes from the secondary source and was not checked in the original. Reported results: PoD 98.70%, overall accuracy 96.09%, commission error 2.68% and omission error 1.30%. When transferred to offshore wind farms in China and the UK, overall accuracy was 97.00%. In Chapter 2 it shows how stationary structures can be separated from ships through multi-temporal composites, a step a single-image CFAR cannot take.

### Already in the thesis (confirmed only)

**[E2]** R. Torres *et al.*, "GMES Sentinel-1 mission," *Remote Sens. Environ.*, vol. 120, pp. 9–24, May 2012, doi: 10.1016/j.rse.2011.05.028. VERIFIED: https://api.crossref.org/works/10.1016/j.rse.2011.05.028.

### Cross-category pointer (not counted; probably covered by the deep-learning / dark-vessel reviewer)

Paolo, F. S. *et al.*, "Satellite mapping reveals extensive industrial activity at sea," *Nature*, vol. 625, no. 7993, pp. 85–91, 2024, doi: 10.1038/s41586-023-06825-8. VERIFIED: https://api.crossref.org/works/10.1038/s41586-023-06825-8.
- This is the GFW study behind the reference detections used in the thesis. Its abstract confirms that it combines satellite imagery, vessel GPS data and deep learning, and that 72–76% of industrial fishing vessels are not publicly tracked. I did not check how its detection stage works internally, so the thesis should not describe it as CFAR-based on the strength of this note.

---

## 2. Synthesis: what classical methods achieve and where they fail

Classical SAR ship detection follows the chain Crisp described [E1]: land masking, prescreening by an adaptive threshold, discrimination, and clustering. Its core is the CFAR principle. The local sea background is estimated in a reference window around each pixel, a statistical clutter model is fitted, and the threshold is set so that the expected share of sea pixels above it equals a chosen false-alarm probability. Since CA-CFAR, the research has gone in two directions:
- **Better background estimation** when the window is contaminated: OS-CFAR [C2]; the GO/SO/OS/TM comparison [C3]; automatic censoring of target pixels [C4].
- **Better clutter models** for heavy-tailed sea clutter: K, the G/G0 family [C5], Weibull, log-normal and the generalized gamma [C6]. As the survey points out, the best choice changes with resolution, sea state and scene [C1].

Polarimetry adds a further gain by improving the contrast between ships and the sea before thresholding: the polarimetric whitening filter [C7], the polarimetric notch filter [C8] and dual-pol coherence [C9]. These methods are mature and fast enough for operations. SUMO processes more than 10,000 images in under two hours [C11], and two years of Mediterranean Sentinel-1 data were processed automatically with AIS fusion [C12]. On large ships in open water, operational detectors reach PoD of about 85–90% at FAR around 1e-8 [C13]. Sentinel-1 detection rates exceed 80% for vessels longer than 60 m [C9]. Even a fixed −15 dB threshold has been reported at F1 = 96% in the Persian Gulf [C14].

The failures are consistent across this literature, and each one is patched by hand.
- **Mixed clutter and crowded targets.** These break the assumption of uniform, single-target statistics in the reference window [C2], [C3], [C4]. A parametric model that fits one scene does not carry over to the next [C6], [C11].
- **Coasts and land.** These are the largest single error source. In a four-detector comparison, 67 of SUMO's 77 false alarms came from poor coastal masking [C13]. SUMO needs a buffered land mask, and still detects coastal infrastructure and aquaculture outside it [C11].
- **Instrument artefacts.** Azimuth and range ambiguities, Sentinel-1 ghosts and RFI all look like ships to a CFAR detector. The fixes are geometric rules [C11], quad-pol modes Sentinel-1 does not have [C10], or multi-image recurrence analysis [C12]. In the Mediterranean, 20% of reliable SUMO detections were recurrent artefacts or fixed structures [C12]. This is especially important in the Persian Gulf, where about 1,731 oil and gas platforms stood in 2025 [O1].
- **Small vessels.** Their echoes are close to speckle level, so lowering the threshold to catch them inflates false alarms. SUMO's PFA is explicitly "a compromise" [C11]. Operational settings raise the co-pol threshold so far that 98% of detections come from cross-pol [C12]. PoD falls to about 52% in scenes dominated by small fishing boats [C13], and Sentinel-1 detection rates drop sharply below 60 m [C9].
- **Sea state, wind and incidence angle.** These require manual threshold adjustment [C11] or thresholds per sub-swath [C9]. Ambiguities stand out most in low wind [C10], while cross-pol contrast fades at far range [C11].
- **Polarimetry on GRD.** The strongest polarimetric detectors need complex SLC data [C7], [C8], [C9]. On Sentinel-1 GRD only VV and VH intensities remain, and operational systems usually detect in each channel separately and merge afterwards [C9], [C11].

Finally, AIS-based evaluation cannot tell a dark ship from a false alarm: 40% of SUMO detections stayed unmatched in [C12]. Every reported accuracy number therefore depends partly on its reference data.

---

## 3. Comparison table (draft)

| Method [ref] | Data | Evaluation | Strengths | Weaknesses |
|---|---|---|---|---|
| OS-CFAR [C2] | Radar theory and simulation | Analytical comparison with CA-CFAR | Robust to several targets in the window and to clutter edges | Small loss in uniform clutter; sorting cost; no SAR-specific clutter model |
| CA/GO/SO/OS/TM-CFAR analysis [C3] | Exponential noise model (analytical) | Average detection threshold in uniform, multi-target and clutter-edge cases | Clear trade-off map between CFAR variants | Rayleigh-intensity assumption does not fit heavy-tailed sea clutter |
| G0 model [C5] | Multilook SAR, land scenes, two sensors | Model fit | Fits extremely heterogeneous clutter; contains K as a special case | Validated on land; unstable estimation in small windows |
| GΓD-CFAR [C6] | One high-resolution SAR image (sensor not confirmed) | False-alarm-rate control and speed vs. Weibull, K, G0 CFAR | Flexible model; closed-form threshold | One test image; still a parametric assumption |
| Automatic-censoring G0-CFAR [C4] | Real high-resolution SAR (sensor not confirmed) | Theory and experiments in uniform, edge and multi-target cases | Censors targets out of the window; fast | Hand-set pre-threshold; single channel |
| PWF + two-parameter CFAR [C7] | Lincoln Lab 35 GHz airborne quad-pol SAR (land targets) | Detection vs. adaptive PWF and HH-only | Speckle-minimizing combination of channels | Needs fully polarimetric complex data; not ships |
| Polarimetric notch filter [C8] | TerraSAR-X quad-pol; dual-pol version designed | Ground-truth agreement; Monte Carlo PD/PFA | Largely independent of sea conditions; physically interpretable | Needs complex data; platforms and buoys also flagged |
| Dual-pol coherence detector [C9] | Sentinel-1 IW SLC (English Channel, Mexico) | Large-scale AIS comparison; PD vs. ship length | >80% PD for ships >60 m; fewer false alarms than VH-CFAR | SLC only; hand-designed per-sub-swath threshold; small ships missed |
| Azimuth-ambiguity removal [C10] | TerraSAR-X quad-pol (DRA campaign 2010) + AIS | Discrimination of ambiguities from ships | Removes a major false-alarm source | Relies on quad-pol, unavailable on Sentinel-1 |
| SUMO [C11] | L/C/X-band SAR, 1–100 m resolution, including Sentinel-1 IW | Benchmarks; illustrated false-alarm types | Mature, multi-sensor, very fast; rules for ambiguities and recurrence | Manual threshold factor; coastal and fixed-structure false alarms; operator advised |
| Sentinel-1 mass processing [C12] | 11,573 S1A IW images, Mediterranean, 2 years | AIS correlation (500 m / 5 km); recurrence analysis | Shows operational scale; 20% recurrent false alarms quantified | High thresholds miss small ships; manual RFI flagging; 40% unmatched |
| Operational detector comparison [C13] | 5 RADARSAT-2 / COSMO-SkyMed scenes, 232 verified targets | PoD, FAR, size error | Head-to-head on hard cases (coast, ambiguities, clutter) | Small sample; best PoD 52% on small fishing boats; no single detector wins |
| Fixed-dB threshold + AIS [C14] | 258 S1 IW GRD scenes, Strait of Hormuz / Persian Gulf, 2026 | F1, precision, recall on 18 test images | Very simple; same region and sensor as this thesis | Fixed threshold, no clutter model; manual artefact removal; congested-water confusion |

---

## 4. Gaps and challenges that motivate a learned detector

- **The clutter model never quite fits, and CFAR models only the sea.** Every CFAR variant depends on a parametric background model and a clean reference window. Real Persian Gulf scenes break both conditions: mixed coastal clutter, shipping lanes with many targets, anchorages [C2], [C3], [C4], [C6]. Operational systems answer with empirical correction factors [C11]. CFAR is formally a suboptimal one-class classifier [C1]. A detector trained on both ship and non-ship examples can learn the decision boundary directly instead of assuming a distribution, which is the role of the EfficientNet–U-Net heatmap detector in Chapter 3.
- **False alarms need context that a per-pixel test does not have.** Ambiguities, ghosts, wakes, aquaculture, coastal infrastructure and platforms are removed by geometric rules, quad-pol tricks or multi-image recurrence [C10], [C11], [C12]. They still make up a large share of detections: 20% of reliable SUMO detections were recurrent [C12], and coastal masking caused 67 of 77 SUMO false alarms in [C13]. With about 1,731 platforms in the Persian Gulf [O1], a detector that learns spatial context from image chips is a reasonable way to suppress such patterns within a single image. Recurrence or platform masks [O2] remain a complement.
- **Small vessels and the threshold trade-off.** Lowering a CFAR threshold to catch small, speckle-level ships inflates false alarms. Operational settings therefore give up small ships [C11], [C12]. Reported detection drops sharply below about 60 m [C9], and PoD falls to about 52% in small-boat scenes [C13]. A learned detector that pools evidence from neighbouring pixels and from both channels is meant to improve this trade-off. The thesis should measure that claim against the CFAR baseline rather than assume it.
- **Dual-pol information is under-used on GRD.** The strongest polarimetric detectors need complex data (PWF [C7], notch filter [C8], coherence [C9]). On Sentinel-1 GRD, operational detectors threshold VV and VH separately and merge the results [C9], [C11]. SUMO itself notes that cross-pol gives strong contrast but fails for some ships and at far range [C11]. Feeding VV and VH to the network together is a pragmatic way to learn their combination from data.
- **Reference data and evaluation are the weak point.** Classical systems are evaluated against AIS with matching radii [C9], [C12], [C14]. Unmatched detections mix dark ships with false alarms (40% unmatched in [C12]), and small benchmarks are sensitive to single misses [C13]. This justifies the thesis design: labels from CFAR cross-checked with GFW, evaluation against GFW, and AIS used only afterwards to flag dark-vessel candidates. It also requires an honest caveat. A learned detector trained on CFAR∩GFW labels inherits some of CFAR's blind spots, small ships especially, and its scores are bounded by the quality of the reference.

---

## 5. How to use this in Chapter 2 and link it to Chapter 3

Suggested order:
1. Crisp's four-stage pipeline [E1].
2. CFAR variants and the multi-target problem [C2]–[C4].
3. Clutter models [C5], [C6].
4. Polarimetric and dual-pol detectors [C7]–[C9].
5. Operational Sentinel-1 systems [C11], [C12], [C13], [C14].
6. Challenges, using [C10], [C12], [C13], [O1].
7. Close with Section 4 above, which leads into Chapter 3. There the CFAR baseline stands for the classical family, and the learned point detector is set against the specific gaps listed.

---

## 6. Verification tally and flags

- **VERIFIED:** 19 entries. That is the 14 core references, the 2 optional ones, the 2 already in the thesis ([E1] Crisp, [E2] Torres) and the 1 cross-category pointer (Paolo 2024).
- **UNVERIFIED:** 0 entries.
- **Details flagged as not confirmed** (do not state them as fact without checking the full text):
  - [C1]: the names of the taxonomy's groups.
  - [C2]: the size of OS-CFAR's loss in uniform clutter.
  - [C3]: the specific GO/SO/OS conclusions; only the scope was confirmed from the abstract.
  - [C4] and [C6]: the sensor and numbers of the test images.
  - [C7]: the publication month.
  - [O2]: the method description, taken from a secondary source.
  - [O1]: no volume or issue assigned yet.
  - Paolo 2024: how its detection stage works internally.
