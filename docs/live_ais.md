# Fresh AIS and experimental short-gap estimates

The application can receive free Gulf observations through Open Waters' native
server-sent events feed. Coverage and availability vary. This is separate from
the stored Sentinel-1 detections and historical GFW fusion; none of those radar
positions is advanced to the present.

## Run locally on Windows

From the repository root, start the AIS worker in its own terminal:

```powershell
$env:OPENWATERS_ENABLED = "1"
.venv-eval/Scripts/python -m services.ais_ingest.worker
```

Start the API in another terminal, then the web client:

```powershell
.venv-eval/Scripts/python -m uvicorn services.api.main:app --host 127.0.0.1 --port 8000
```

```powershell
cd web
npm run dev
```

Open http://127.0.0.1:5173. The API alone does not collect AIS. The worker must
remain running for fresh observations. `OPENWATERS_ENABLED` is opt-in and does
not require an account or key. Existing configured feeds remain supported.
Avoid concurrent duplicate workers: the anonymous provider limits connections
per network address. The normal worker reconnects with backoff; HTTP 429 adds
a longer wait. Stop the worker with Ctrl+C when finished.

The current local evaluation environment was repaired with hash-verified mirror
packages. No package download occurs when starting these commands.

## What the map means

- The ordinary AIS symbol remains at its last observed position.
- The opt-in «نمایش موقعیت تخمینی» control adds a purple estimated point and
  dashed connection. It starts off because empirical performance is mixed.
- The estimate uses reported speed over ground and course over ground. It
  assumes constant motion and stops once the observation is over 300 seconds
  old. This is **not** a five-minute prediction beyond the present: existing
  provider latency consumes part of that limit.
- Unknown/fallback timestamps, simulated records, suspect positions and missing
  motion fields produce no estimate. Heading does not substitute for course.
- Historical radar playback disables estimated markers. Its AIS layer remains
  explicitly independent of the historical radar clock.
- Provider event time may be a source timestamp or the provider's receipt time;
  it is not independently verified GNSS fix time. The interface states this.

`GET /api/vessels` retains observed `lon`, `lat` and `ts` and adds provenance
and an `estimate` object. `?estimates=false` disables that object for historical
views. `GET /api/health` reports actual fresh received observations rather than
interpreting configured credentials as proof of live coverage.

Provider-reencoded source observations are distinct from this project's
synthetic demonstration traffic. Static messages may enrich identity but never
create or freshen a position. Original evidence and measured radar results stay
unchanged. Provider/source attribution is shown in the application.

## Reproduce the descriptive validation

```powershell
.venv-eval/Scripts/python scripts/evaluate_ais_motion.py --input data/qa/ais_motion_20261029/events.jsonl --out data/qa/ais_motion_20261029/evaluation.json
```

The evaluator compares one-, three- and five-minute nominal receipt-time
horizons with later provider positions and a hold-last-position baseline.
Only data available at each origin is used. It reports actual sample counts,
distinct vessels, median/P90 errors, latency, exclusions and movement strata.
There may be no admissible five-minute pairs because of the 300-second cap.
Provider positions are a comparison reference, not independent ground truth.
The small pilot is descriptive, with no fitted settings or calibrated confidence.

No data, credentials, model weights or report build outputs belong in Git.

## Measured pilot on 29 October 2026

The approved 20-minute collection contained 439 events: 436 position reports
from 38 vessels and three static messages. Ninety positions lacked usable
motion; 346 were eligible prediction origins. Median provider-to-local receipt
lag was 80.6 seconds (P90 90.4).

| Nominal horizon after receipt | Matched pairs / vessels | Motion P90 error | Hold-last P90 error |
| --- | --- | --- | --- |
| 1 minute | 114 / 20 | 26.5 m | 7.2 m |
| 3 minutes | 69 / 20 | 59.6 m | 32.8 m |
| 5 minutes | 0 / 0 | Not measured | Not measured |

Most matched cases were stationary. In the moving-only subset, median errors
were 145 m versus 472 m at one minute (12 pairs/four vessels), and 498 m versus
514 m at three minutes (10 pairs/four vessels). These tiny, correlated subsets
do not establish general predictive accuracy. The hold-last baseline was
better in the overall stationary-heavy comparison. Apparent sub-metre medians
from repeated provider positions are not evidence of sub-metre real accuracy.

All five-minute predictions were suppressed because the provider delay plus
the forecast horizon exceeded the fixed 300-second observation-age cap. No
setting was fitted to these results. Errors compare with later positions from
the same provider, not independent truth.

Collection was intermittent: the initial five-second read timeout caused
reconnects, and overlapping short QA connections triggered HTTP 429. The
collector resumed with a longer timeout; the runtime client uses a 45-second
idle timeout and respects rate-limit waits. These checks do not establish
continuous availability or complete Gulf coverage. A separate 60-second
end-to-end check stored 36 real source reports and returned 27 fresh vessels
from the application API; no radar records were changed.
