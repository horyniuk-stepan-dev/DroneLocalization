import html


def create_box(id, value, x, y, width, height, fill_color, border_color):
    return f"""
    <mxCell id="{id}" value="{html.escape(value)}" style="rounded=1;whiteSpace=wrap;html=1;fillColor={fill_color};strokeColor={border_color};strokeWidth=2;fontColor=#ffffff;fontSize=12;verticalAlign=middle;align=center;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry" />
    </mxCell>
    """


def create_edge(id, source, target, color="#aaaaaa", style="orthogonalEdgeStyle"):
    # Using entry and exit ports to make lines predictable and prevent crossing in the center
    # rounded=1 makes corners nice
    return f"""
    <mxCell id="{id}" style="edgeStyle={style};rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};strokeWidth=2;endArrow=block;endFill=1;entryX=0.5;entryY=0;entryDx=0;entryDy=0;exitX=0.5;exitY=1;exitDx=0;exitDy=0;" edge="1" parent="1" source="{source}" target="{target}">
      <mxGeometry relative="1" as="geometry" />
    </mxCell>
    """


def create_edge_custom(
    id,
    source,
    target,
    color="#aaaaaa",
    exitX=0.5,
    exitY=1,
    entryX=0.5,
    entryY=0,
    style="orthogonalEdgeStyle",
):
    return f"""
    <mxCell id="{id}" style="edgeStyle={style};rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};strokeWidth=2;endArrow=block;endFill=1;entryX={entryX};entryY={entryY};entryDx=0;entryDy=0;exitX={exitX};exitY={exitY};exitDx=0;exitDy=0;" edge="1" parent="1" source="{source}" target="{target}">
      <mxGeometry relative="1" as="geometry" />
    </mxCell>
    """


def main():
    xml = """<mxfile host="Electron" modified="2023-11-01T00:00:00.000Z" agent="Mozilla/5.0" version="22.0.4" type="device">
  <diagram id="components" name="Page-1">
    <mxGraphModel dx="1434" dy="844" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" background="#1e1e1e" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
"""

    # Colors
    c_gui = "#34495e"  # Dark Blue-Gray
    c_worker = "#2c3e50"  # Darker Blue
    c_model = "#d35400"  # Orange
    c_loc = "#2980b9"  # Blue
    c_db = "#27ae60"  # Green
    c_calib = "#16a085"  # Teal
    c_trk = "#2E8B57"  # Sea Green
    c_vid = "#8e44ad"  # Purple
    c_base = "#7f8c8d"  # Gray

    border = "#000000"

    w, h = 230, 80
    gap_x = 40
    gap_y = 70

    # Columns
    col0 = 50
    col1 = col0 + w + gap_x
    col2 = col1 + w + gap_x
    col3 = col2 + w + gap_x

    # Rows
    row0 = 50  # Title
    row1 = 120  # GUI / Network
    row2 = 250  # Workers / Models
    row3 = 380  # DB / Loc / Calib
    row4 = 510  # Video / Preproc / Tracking / Geometry
    row5 = 640  # Depth / config / main

    # Title
    xml += f"""
    <mxCell id="title" value="Архітектура програмних компонентів системи локалізації БПЛА" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor=#ffffff;fontSize=16;fontStyle=1;" vertex="1" parent="1">
      <mxGeometry x="{col1}" y="{row0}" width="{w*2 + gap_x}" height="40" as="geometry" />
    </mxCell>
    """

    # Nodes
    nodes = [
        # Row 1
        (
            "gui",
            "<b>src/gui</b><br>Графічний інтерфейс PyQt6<br>MainWindow, Dock-панелі",
            col1,
            row1,
            c_gui,
        ),
        (
            "net",
            "<b>src/network</b><br>REST API & WebSocket<br>Трансляція координат",
            col2,
            row1,
            c_gui,
        ),
        # Row 2
        (
            "wrk",
            "<b>src/workers</b><br>QThread Workers<br>Оркестрація фонових задач",
            col1,
            row2,
            c_worker,
        ),
        (
            "mdl",
            "<b>src/models</b><br>ModelManager<br>DINOv2, ALIKED, YOLOv11",
            col2,
            row2,
            c_model,
        ),
        # Row 3
        (
            "db",
            "<b>src/database</b><br>HDF5DatabaseManager<br>Зберігання ознак та масок",
            col0,
            row3,
            c_db,
        ),
        (
            "loc",
            "<b>src/localization</b><br>Localizer & GeoAwareRetriever<br>Ядро зіставлення ознак",
            col1 + w / 2 + gap_x / 2,
            row3,
            c_loc,
        ),  # Center between col1 and col2
        (
            "cal",
            "<b>src/calibration</b><br>AnchorManager<br>Пропагація GPS координат",
            col3,
            row3,
            c_calib,
        ),
        # Row 4
        (
            "vid",
            "<b>src/video</b><br>VideoSourceManager<br>Декодування кадрів (RTSP/MP4)",
            col0,
            row4,
            c_vid,
        ),
        (
            "prep",
            "<b>src/preprocessing</b><br>ImagePreprocessor<br>CLAHE, YOLO masking",
            col1,
            row4,
            c_vid,
        ),
        (
            "trk",
            "<b>src/tracking</b><br>TrajectoryFilter (Kalman)<br>OutlierDetector (Z-score)",
            col2,
            row4,
            c_trk,
        ),
        (
            "geo",
            "<b>src/geometry</b><br>HomographyEstimator<br>CoordinateConverter",
            col3,
            row4,
            c_calib,
        ),
        # Row 5
        ("dep", "<b>src/depth</b><br>DepthEstimator<br>Depth Anything V2", col2, row5, c_vid),
        ("conf", "<b>config/config.py</b><br>Глобальні налаштування", col0, row5, c_base),
        ("main", "<b>main.py</b><br>Точка входу, ініціалізація", col1, row5, c_base),
        ("util", "<b>src/utils</b><br>LoggingUtils (Loguru)", col3, row5, c_base),
    ]

    for id, text, x, y, color in nodes:
        xml += create_box(id, text, x, y, w, h, color, border)

    # Edges - using carefully selected ports to avoid crosses
    edges = [
        # GUI -> Workers (straight down)
        ("e1", "gui", "wrk", "#ffffff", 0.5, 1, 0.5, 0),
        # GUI -> Models (crosses right)
        ("e2", "gui", "mdl", "#ffffff", 0.75, 1, 0.25, 0),
        # Network -> Workers (crosses left)
        ("e3", "net", "wrk", "#ffffff", 0.25, 1, 0.75, 0),
        # Workers -> DB
        ("e4", "wrk", "db", "#aaaaaa", 0.25, 1, 0.5, 0),
        # Workers -> Loc
        ("e5", "wrk", "loc", "#aaaaaa", 0.75, 1, 0.25, 0),
        # Workers -> Calibration (routes over Loc)
        ("e6", "wrk", "cal", "#aaaaaa", 0.9, 1, 0.5, 0),
        # Models -> Loc
        ("e7", "mdl", "loc", "#f39c12", 0.5, 1, 0.75, 0),
        # Loc -> Prep
        ("e8", "loc", "prep", "#3498db", 0.25, 1, 0.5, 0),
        # Loc -> Trk
        ("e9", "loc", "trk", "#3498db", 0.75, 1, 0.5, 0),
        # Cal -> Geo
        ("e10", "cal", "geo", "#1abc9c", 0.5, 1, 0.5, 0),
        # Prep -> Vid
        ("e11", "prep", "vid", "#9b59b6", 0, 0.5, 1, 0.5),
        # Prep -> Depth
        ("e12", "prep", "dep", "#9b59b6", 0.75, 1, 0, 0.5),
        # DB <-> Loc (horizontal)
        ("e13", "loc", "db", "#aaaaaa", 0, 0.5, 1, 0.5),
    ]

    for eid, src, trg, color, ex, ey, nx, ny in edges:
        xml += create_edge_custom(eid, src, trg, color, ex, ey, nx, ny)

    xml += """
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""
    with open(
        "C:/Users/horyn/.gemini/antigravity/brain/a6c235ef-3f39-4712-922b-29b0ba7dea73/diagrams/fig6_component_diagram.drawio",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(xml)
    print("Component diagram Draw.io file created successfully.")


if __name__ == "__main__":
    main()
