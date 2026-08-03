# Planet Education & Research — Application Text

**Title:** Vision-based UAV geolocalization against multi-temporal satellite reference maps in GNSS-denied conditions

---

## Field 1 — Project description

My dissertation research addresses absolute geolocalization of unmanned aerial vehicles when
GNSS is unavailable, jammed, or spoofed. The system I am developing estimates a drone's
geographic position by matching its downward-looking camera frames against a pre-built
georeferenced satellite reference database, without any inertial or radio-navigation input. The
central difficulty is domain gap: the aerial frame and the satellite tile differ in viewing
geometry, ground sample distance, illumination, shadow direction, and — most severely — season.
A reference map built from summer imagery degrades sharply against a winter or early-spring
flight, and this seasonal failure mode is the main open problem in my work.

The questions I hope to answer with Planet data are: (1) How does absolute localization accuracy
degrade as the temporal gap between the reference imagery and the flight widens, across full
seasonal cycles? (2) Does a multi-temporal reference database — several acquisitions of the same
area under different seasonal and illumination conditions — recover the accuracy lost to a single
static basemap, and at what storage and retrieval cost? (3) Which land-cover classes (fragmented
cropland, deciduous forest, dense historic urban fabric) drive the residual error, and can they be
predicted before flight?

**Methods.** Reference tiles over my study area are converted into a topometric database storing
global descriptors from the DINOv3 vision foundation model for coarse retrieval, plus local
keypoints and descriptors (ALIKED / RDD with LightGlue matching) for fine registration. Query
frames are matched to retrieved candidates, pose is estimated with RANSAC, and the resulting
positions are fused over time using a Kalman filter and a 5-DoF pose-graph optimization with
Levenberg-Marquardt, anchored on surveyed GPS control points. Because my study area has
significant terrain relief, the planar-homography approximation used by most published work is
not valid there; the pipeline therefore also runs monocular depth estimation (Depth-Anything-V2)
to recover scale-aware, non-planar pose. Dynamic objects are masked with YOLOv11-Seg and contrast
is normalized with CLAHE before feature extraction. Accuracy is evaluated as absolute geodesic
error against GPS ground truth from instrumented flights and against a photorealistic flight
simulator that produces synthetic imagery with exact pose labels.

**Why Planet.** PlanetScope's near-daily revisit at roughly 3 m resolution is the only source that
lets me construct genuinely multi-temporal reference stacks — the same footprint imaged repeatedly
across a season — rather than the single-date, multi-year-old mosaics that public basemaps
provide. That temporal density is what makes question (2) answerable at all. Results will form
part of my dissertation and the evaluation code will be released openly.

---

## Field 2 — Geography

The study area is a single contiguous polygon of approximately 30 x 30 km (~900 km2) centred on
Chernivtsi, Chernivtsi Oblast, south-western Ukraine (approx. 48.29 N, 25.94 E). The footprint is
deliberately small and fixed: the research question concerns temporal variation over an identical
area, not spatial extent, so the same polygon is re-acquired across a full annual cycle rather than
expanded geographically. At ~900 km2 this fits within the monthly quota while allowing roughly
three acquisitions per month. The design depends on paired data — a Planet reference stack and
matched UAV imagery over the same ground in each season — so the site sits within routine
field-campaign range of Chernivtsi National University, where I am based.

The site lies in the Bukovinian foothills of the Eastern Carpathians and spans the Prut river
valley. Terrain rises from roughly 150 m in the valley floor to over 400 m on the surrounding
ridges within the polygon. This relief is scientifically central rather than incidental: it
invalidates the locally-planar ground assumption that most published UAV-to-satellite matching
relies on, and makes the area a genuine test of depth-aware pose estimation.

Land cover is heterogeneous at fine scale, which is what makes it informative. Fragmented
small-parcel agriculture and orchards occupy the terraces; beech, hornbeam and oak forest covers
the slopes and undergoes complete leaf-on / leaf-off radiometric turnover between seasons; dense
linear village settlement follows the valleys; and the historic core of Chernivtsi provides a
compact patch of stable, high-contrast urban structure that should act as the strongest anchor
class. Comparing error across these classes is question (3).

Two regional characteristics make PlanetScope's revisit necessary rather than convenient. First,
foothill terrain generates persistent orographic cloud, so a lower-cadence sensor would not
reliably deliver a usable acquisition in every target month. Second, at 48.3 N latitude the winter
solar elevation at PlanetScope's mid-morning overpass falls to roughly 13 degrees, casting long
terrain shadows that fall across slopes in a direction never seen in summer imagery. That
combination of seasonal canopy turnover and seasonally rotating terrain shadow is the strongest
version of the domain gap this project is designed to measure.
