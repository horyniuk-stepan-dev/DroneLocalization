# `topnew` reference-map repair: straight-leg anchor model

Date: 2026-09-25. Source video: `D:/My Projects/FlightSimulator/flight.mp4`.
Ground truth is **evaluation-only**; propagation receives the reference video,
its extracted features, and the existing 83 calibration anchors. The video
SHA-256 recorded in each evaluated database matches the source video.

## Measured failure

The GUI's adaptive database retained 468/1688 slots, including all 83 exact
anchors after the anchor-preservation fix. Graph propagation marked 336 slots
supported, but those slots had 32.14 m p95 and 85.30 m maximum ground error.
The dense diagnostic database retained all 1688 slots and built 1687 temporal
edges. It nevertheless marked every slot supported while reaching 19.28 m p95
and 47.60 m maximum. Adding frames alone does not correct VO drift or false
loop closures. A separate anchored-temporal-chain experiment was worse (32.12 m
p95 after endpoint correction), so it was not integrated.

The existing anchor-gap check identified eight dense gaps with 151–277 m
temporal-chain endpoint disagreement. Its warning claimed those gaps would be
anchor-interpolated, but the implementation preserved graph-supported nodes;
the rejected edge chain therefore still produced confirmed geography. The
other long gaps also contained errors above 10 m. Graph connectivity and low
matching RMSE are insufficient geographic-quality evidence on this mission.

## Opt-in repair

`--anchor-linear-fallback` uses a motion model only where **at least three
consecutive anchor intervals** are each at least 20 slots long and their
per-slot *two-dimensional velocity vectors* agree within 1%. Short intervals
break a run, so turn clusters retain the visual-graph solution. For accepted
intervals, the affine image-centre position is linearly interpolated between
the exact anchors; log scale and the shortest angular arc are interpolated in
the same 5-DoF representation. Model endpoints are pinned to exact anchor
affines, even when the graph uses soft anchors, to prevent a footprint jump at
the boundary. The database stores origin code 5 and a
`frame_anchor_linear_support` mask for those frames, plus the interval list.
The regular propagation path remains the default.

The anchor geometry in `topnew` contains 13 such runs, each spanning three
long intervals. The maximum velocity-vector deviation within a run is 0.14%.
The mode replaced 1549 interior slots; the turn clusters retained the graph.

| Reference DB | Stored features | Supported slots | Ground-centre p95 | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Active adaptive DB | 468 | 336/1688 | 32.14 m (supported only) | 85.30 m |
| Dense graph only | 1688 | 1688/1688 | 19.28 m | 47.60 m |
| Dense + anchor-linear mode | 1688 | 1688/1688 | **0.93 m** | **1.47 m** |

The repaired dense map has angle p95 0.17 degrees and scale p95 1.34% against
simulator ground truth. Its maximum error among the four image corners has
3.99 m p95 and 9.12 m maximum relative to the simulator's affine model; a
centre-only score is insufficient to describe the whole footprint. The
unrelated 43-slot curved reference selected zero
linear intervals and retained its prior 0.64 m p95 / 0.69 m maximum. Source
ground truth was used to score both tests, never to choose the intervals or
compute their transforms.

Reproduce from a database built with all sampled slots:

```powershell
& .venv\Scripts\python.exe scripts/run_calibration_propagation.py `
  --db 'D:\My Projects\TEST\topnew\sources\main\.linear-fallback-pinned-20260925\database.h5' `
  --calibration 'D:\My Projects\TEST\topnew\sources\main\calibration.json' `
  --anchor-linear-fallback

& .venv\Scripts\python.exe scripts/validate_vs_ground_truth.py `
  --db 'D:\My Projects\TEST\topnew\sources\main\.linear-fallback-pinned-20260925\database.h5' `
  --gt 'D:\My Projects\FlightSimulator\ground_truth.json' `
  --video 'D:\My Projects\FlightSimulator\flight.mp4' `
  --max-supported-p95 3 --max-supported-corner-p95 5
```

The repaired copy is at
`D:/My Projects/TEST/topnew/sources/main/.linear-fallback-pinned-20260925/database.h5`.
The active `topnew/sources/main/database.h5` has not been replaced.

## Scope and remaining risk

This opt-in mode depends on a straight, approximately constant-speed survey
leg observed by four or more reliable anchors. A hidden bend or speed change
*between* anchors can satisfy the endpoint test and produce wrong positions.
It is therefore unsuitable as a universal fallback for arbitrary curved
flights; that case still needs stronger visual constraints or additional
measurements. The 43-slot curved test demonstrates rejection of its visible
turns, not proof against a hidden S-curve. The mode repairs this simulator
mission's reference map; cross-layer query localization still needs a separate
end-to-end replay against this repaired map.

`frame_graph_component`, `frame_support_anchor_count`, and the existing
`frame_disagreement` remain graph diagnostics even where the anchor-linear
model supplies the affine. In particular, localization confidence currently
reads graph disagreement. Query replay must measure whether that conservative
but mismatched confidence input suppresses otherwise valid fixes; it is not
appropriate to zero it out without such evidence.

## Independent same-area query replay

The `FlightSimulator/output/topnew_combined_scale_tilt_20260925` query flight
was generated over the same terrain raster as `topnew` and decoded into 80
sampled slots. Its programmed camera altitude is 500–1000 m with up to 20°
pitch and 10° roll; terrain makes the actual camera AGL range 461–898 m.
The query ground truth and camera altitude were passed only to the evaluator,
not `Localizer.localize_frame`.

Against the repaired dense single layer, 74/80 slots produced confirmed fixes.
Among those 74, raw visual error had median 0.62 m, p95 1.38 m, maximum 2.05 m;
final output error had median 0.78 m, p95 2.89 m, maximum 3.57 m. The six
failures were slots 4, 74, and 76–79, all with no confirmed geometric observation
and AGL below 600 m. The longest failure run is four one-second slots at the end of
the flight. Excluding the first model-load sample, successful-frame processing
time had median 66 ms and p95 267 ms; across both successes and failures,
warm p95 was 485 ms. These are synchronous replay processing times, not a
measurement of camera-to-output age in the live UI.

An evaluator-only overlap check compared query and reference affine footprint
polygons from simulator GT; those GT transforms never entered localization.
The best single 1000 m reference frame covered approximately 63%, 76%, 80%,
84%, 87%, and 92% of query slots 74–79 respectively. Slot 4 had approximately
97% footprint coverage. Thus missing geographic coverage alone does not explain
the failures. At slot 74, matching found hundreds of RANSAC inliers against
reference frames near 143–144, but the query centre lay outside the inlier
convex hull beyond the configured extrapolation limit. This is a deliberate
safety rejection, not evidence that the matched patch supports the centre.
The later failures still require retrieval and feature-matching diagnosis.

Full [JSON report](D:/My%20Projects/TEST/topnew/sources/main/.linear-fallback-pinned-20260925/query_full.json)
and [per-slot CSV](D:/My%20Projects/TEST/topnew/sources/main/.linear-fallback-pinned-20260925/query_full.csv)
are retained beside the repaired database. P5 is therefore a measured partial
success on this larger same-area test; the six low-AGL failures and P6's
physical two-layer handoff remain open.
