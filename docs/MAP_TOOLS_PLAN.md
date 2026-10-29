# Map Tools Plan (point, area, ruler, ...)

Date: 2026-10-28. Status: proposal, nothing implemented.

## Current state (verified in code)
- `src/gui/widgets/map_widget.py` (208 lines): `MapWidget(QWebEngineView)` + `MapBridge` over QWebChannel.
  Python->JS is one-way signals (marker, trajectory, FOV, panorama, verification/object markers).
  JS->Python only: `mapClicked(lat, lon)` and `ready(mode)`.
- Two pages: `resources/maps/map_template.html` (Leaflet 1.9.4 from unpkg, satellite/OSM, GEOGRAPHIC)
  and `local_map.html` (LOCAL mode, units are arbitrary "умовні одиниці").
- The only consumer of clicks: `MainWindow._on_map_clicked` -> status bar message.
- No toolbar, no user layers, no persistence.

## Goal
A standard map toolset usable in both modes, with results available to Python
(for later use: AOI for DB build, reference points for calibration checks, geofence).

## Tool set
| Tool | Interaction | Output |
|---|---|---|
| Pan / select (default) | drag, click a feature to select | selected feature id |
| Point (marker) | click; drag to move; double-click to rename | `{id, type:"point", name, lat, lon}` |
| Polyline / route | click vertices, double-click/Enter to finish | coords + total length |
| Polygon / area | click vertices, close on first vertex | coords + area + perimeter |
| Rectangle | drag | bounds + area |
| Circle | center + drag radius | center, radius, area |
| Ruler | click A, B, C... live distance per segment + total; Esc clears; not saved | temporary |
| Edit / delete | vertex drag, Del key, right-click menu | update/delete events |
| Coordinates readout | cursor position in corner (lat/lon or X/Y), click-to-copy | — |
| Go to coordinates | input field lat,lon (or MGRS later) | pan + temp marker |
| Layers panel | show/hide user layer, drone layers, basemap | — |
| Import / export | GeoJSON save/load | file |

## Architecture
1. **Library:** Leaflet-Geoman (free, MIT) for draw/edit/drag, OR own minimal code.
   Recommendation: Geoman — covers point/line/polygon/rect/circle/edit/drag out of the box.
   Ruler + coord readout written by hand (small).
2. **Offline:** Leaflet already comes from unpkg; vendor both Leaflet and Geoman into
   `resources/maps/vendor/` so the map works without internet (field use). Separate small step.
3. **Shared JS:** move tool code into `resources/maps/map_tools.js` + `map_tools.css`,
   included by both HTML pages, so GEOGRAPHIC and LOCAL behave identically.
4. **Bridge (JS->Python), new slots in `MapBridge`:**
   - `featureCreated(json)`, `featureEdited(json)`, `featureDeleted(id)`
   - `toolChanged(name)`
   Python->JS: `setTool(name)`, `loadFeatures(geojson)`, `clearFeatures()`, `zoomToFeature(id)`.
   Keep the existing queue-until-ready mechanism (`_emit_update`).
5. **Python model:** `src/gui/map_features.py` — `MapFeatureStore` (dict id->feature, Qt signals
   `changed`), GeoJSON (de)serialization. Single source of truth on the Python side;
   JS is a view. Re-sync to JS after `set_coordinate_mode(reset=True)` reloads the page
   (today a reload drops everything).
6. **Geometry math in Python, not JS**, for anything reported as a number that matters:
   geodesic distance/area via `pyproj.Geod` (already a dependency) in GEOGRAPHIC;
   plain Euclidean in LOCAL. JS shows quick approximate live values while drawing,
   Python sends back the authoritative value on finish. Testable in sandbox
   (`tests/test_map_features.py`).
7. **UI:** vertical toolbar inside the map (Leaflet control) — keeps MainWindow untouched;
   hotkeys: Esc cancel, Del delete, M point, R ruler, A area.
8. **Persistence:** autosave features to the project/mission folder as `map_features.geojson`
   (exact location to agree on); explicit Import/Export too.

## LOCAL mode caveat
In LOCAL mode coordinates are arbitrary units (see `_on_map_clicked`), so ruler/area show
"units" / "units^2", not metres — unless a scale is known. Do not show metres there
until the scale layer provides a factor (see scale-layer work). Label units explicitly.

## Stages (each independently testable)
- **S0** Vendor Leaflet (+Geoman) locally; maps still load. Check: offline load on Windows.
- **S1** Coordinate readout + Go-to + Ruler (no persistence, no Python model). Check: ruler
  distance vs known pair of points (compare with pyproj value).
- **S2** `MapFeatureStore` + bridge events + GeoJSON; unit tests for geodesic length/area
  (e.g. 1 km x 1 km square at 50°N -> ~1.000 km^2).
- **S3** Draw/edit tools via Geoman (point, line, polygon, rect, circle) wired to the store;
  re-sync after page reload/mode switch.
- **S4** Layers panel, context menu, rename, colors, import/export, autosave.
- **S5 (optional, uses)** "Use polygon as AOI for database build", "point as reference for
  error check against drone estimate" (distance from current estimate to a point), geofence alert.

## Risks / open questions
- Click handler: existing `map.on('click')` -> `mapClicked` will fire during drawing; must be
  suppressed while a tool is active.
- Large redraws: drone trajectory updates must not be blocked by many user features (separate layer).
- Do you want features tied to a mission/database, or global? Decides where autosave goes.
- MGRS / UTM input in Go-to — needed?

## Implementation status (2026-10-29)
- S0-S4 implemented. Own tool code (no Geoman): `resources/maps/map_tools.js/.css` for the Leaflet map,
  inline tools in `local_map.html` (canvas). Leaflet 1.9.4 vendored in `resources/maps/vendor/leaflet/`.
- Python: `src/gui/map_features.py` (store, WGS84 measurements via pyproj, GeoJSON), bridge + autosave in
  `map_widget.py`. Autosave: `user_data_dir()/map_features.geojson` (global, not per mission).
- LOCAL map: point/line/polygon/rectangle/ruler/delete/rename; no vertex editing or circle there.
- Not done: S5 (AOI for DB build, reference-point error check, geofence), MGRS input.
- Verified: 12 unit tests; both pages driven in headless Chromium with a mock bridge. Not yet run in the real Qt app.
