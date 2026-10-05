# Local FPV simulator maps (no GPS)

This test mode builds a **relative planar map** from a reference video of a
fictional environment. It does not need control points, latitude/longitude,
camera altitude or a real-world DEM.

## GUI workflow

1. Create a project/layer with the simulator reference video and build its
   feature database using the normal database action. Do not add GPS anchors.
2. Select that layer and click **Локальна карта без GPS (FPV)** in the calibration
   panel. This matches neighboring reference frames, adds verified loop closures
   and optimizes the existing pose graph. The first frame with features fixes the
   arbitrary coordinate origin, scale and orientation during optimization.
3. On completion the map switches to an offline X/Y grid. Numbered markers show
   reference frame centers. The connected frame footprints are uniformly scaled
   into a 0–100 square; the shorter dimension is centered without stretching.
   X points right and Y points up relative to the initial reference image.
4. Start video tracking or image localization against the selected layer. The
   query must show the same simulator environment. The map displays the matched
   view center, footprint and trajectory in local units.
5. Save tracking results as CSV. Columns are `x`, `y`, `coordinate_kind=local_planar`
   and `coordinate_units=arbitrary`. REST/WebSocket fixes use the same explicit
   local-coordinate fields. GeoJSON/KML are unavailable for these coordinates.

The map and its coordinate mode are stored in the HDF5 database and survive a
project restart. No geographic coordinates are written to `frame_gps`. The layer
calibration JSON records LOCAL mode with **zero surveyed anchors**; the internal
gauge is not a user-provided geographic anchor.

Rebuilding the feature database requires rebuilding its local map. Running the
local-map action again recomputes the normalization, so old exported tracks must
not be combined with results from a newly built map without alignment.

## Video panorama on the local map

After building the local map, select its layer and click **Згенерувати панораму з відео**.
In LOCAL mode this uses the reference
video recorded in the active database, paints its connected frames with their
existing map affines and automatically places the result beneath the markers.
No additional GPS calibration or neural re-localization is needed. If the video
was moved, select the original recording when prompted; its recorded SHA-256 is
checked when available. Another recording cannot reuse these per-frame affines.

Choose a PNG output path. The default is `panoramas/local_map.png` beside the
active layer's database. Keep the adjacent `local_map.png.json` file: it records
the map identity and image corners for repeat loading through **Накласти панораму на карту**.
A saved panorama from another map or an earlier geometry is rejected; rebuild
it after rebuilding the local map. The image is not automatically loaded on
project reopen; use Show Panorama to load it again.

The mosaic has a maximum side of 2048 pixels and is built one frame at a time.
Unmapped regions stay transparent, including disconnected reference frames;
valid black image pixels remain opaque. Feather blending softens overlaps.
The map's **Панорама** checkbox hides/shows the background without affecting
tracking, and **Показати все** fits the image and trajectory. Switching layers
or rebuilding the local map clears the old overlay.

Existing stitched images without this JSON can also be opened through Show
Panorama; they are registered by matching image crops against the selected
database. This can fail when too little recognizable content remains. The
reference-video mosaic is more direct and shares the map's geometry, including
any accumulated drift. Strong parallax or moving objects may cause ghosting;
the mosaic is not a 3D reconstruction or a surveyed orthophoto.

## Scope and limitations

- Values are arbitrary map units, **not meters**. Neither camera position nor
  altitude is estimated. This localizes the observed view using the existing
  image-to-reference geometry.
- Motion includes translation, rotation and scale estimated from feature matches;
  it is not restricted to summing pixel translations. The map remains a planar
  approximation, not full 3D SLAM. Strong parallax, nearby walls and acrobatics can
  cause failed matches or accumulated distortion/drift. Verified loop closures
  can reduce drift but do not establish real-world scale or guarantee accuracy.
- Frames without a verified graph path to the initial frame stay invalid. The
  system does not interpolate a fictional path across a scene cut. If too few
  frames connect, reduce `database.frame_step` and rebuild the feature database;
  adjacent selected frames need enough static scene overlap.
- Each independently built local layer has its own coordinate frame. Select one
  local layer at a time; automatic switching across independent local layers is
  disabled. Geographic multi-layer retrieval excludes local layers.
- Meter-based motion filters, geographic terrain priors and GPS exports are
  bypassed in this mode. Image matching and geometric verification still apply.
- Reference footprints fit into 0–100; subsequent queries may return coordinates
  outside that square. They are not clipped. Use **Показати все** to fit the view.
- A layer with existing geographic calibration is protected from conversion to
  local mode. Create a separate layer for the simulator video.

The Qt/localizer compatibility transport retains internal `(vertical, horizontal)`
fields named `lat/lon`; LOCAL interprets them as `(y, x)`. Public broker outputs
and CSV exports never publish these arbitrary values as GPS.
