import html
import os

out_dir = r"C:\Users\horyn\.gemini\antigravity\brain\a6c235ef-3f39-4712-922b-29b0ba7dea73\diagrams\presentation_visuals"

drawio_template = """<mxfile host="Electron" modified="2026-05-28T00:00:00.000Z" agent="Mozilla/5.0" version="21.6.8" type="device">
  <diagram id="diag_1" name="Visual">
    <mxGraphModel dx="1280" dy="720" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="500" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{content}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""


def box(id, text, x, y, w, h, fill="#ffffff", stroke="#333333", font_size=16, rounded=1):
    # У XML атрибутах < і > повинні бути екрановані як &lt; та &gt;
    safe_text = html.escape(text).replace("\n", "&lt;br&gt;")
    return f"""<mxCell id="{id}" value="{safe_text}" style="rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontSize={font_size};fontColor=#333333;verticalAlign=middle;align=center;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
    </mxCell>"""


def arrow(id, src, dst, label="", color="#666666"):
    safe_lbl = html.escape(label).replace("\n", "&lt;br&gt;")
    lbl = f' value="{safe_lbl}"' if label else ""
    return f"""<mxCell id="{id}"{lbl} style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=3;strokeColor={color};fontSize=12;fontColor=#333;endArrow=block;endFill=1;" edge="1" parent="1" source="{src}" target="{dst}">
      <mxGeometry relative="1" as="geometry" />
    </mxCell>"""


def arrow_coords(id, sx, sy, tx, ty, color="#666666", width=3, dashed=False):
    style = f"endArrow=block;html=1;strokeWidth={width};strokeColor={color};endFill=1;"
    if dashed:
        style += "dashed=1;"
    return f"""<mxCell id="{id}" value="" style="{style}" edge="1" parent="1">
      <mxGeometry width="50" height="50" relative="1" as="geometry">
        <mxPoint x="{sx}" y="{sy}" as="sourcePoint" />
        <mxPoint x="{tx}" y="{ty}" as="targetPoint" />
      </mxGeometry>
    </mxCell>"""


def text(id, text, x, y, w, h, font_size=18, color="#333333", bold=False):
    safe_text = html.escape(text).replace("\n", "&lt;br&gt;")
    style = f"text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={font_size};fontColor={color};"
    if bold:
        style += "fontStyle=1;"
    return f"""<mxCell id="{id}" value="{safe_text}" style="{style}" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
    </mxCell>"""


def circle(id, x, y, r, fill="#3498db"):
    return f"""<mxCell id="{id}" value="" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={fill};strokeColor=none;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{r}" height="{r}" as="geometry" />
    </mxCell>"""


def generate_file(filename, content):
    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(drawio_template.format(content="\n".join(content)))
    print(f"Created: {filename}")


# ---------------------------------------------------------
# 1. Slide 7: Global & Local Pipeline (Feature Engineering)
# ---------------------------------------------------------
c1 = []
c1.append(box("bg1", "", 10, 10, 420, 400, fill="#f8f9fa", stroke="#ced4da"))
c1.append(text("t1", "1. Глобальний пошук", 10, 20, 420, 30, bold=True))
c1.append(box("in_cam", "Кадр з дрона", 40, 80, 150, 50, fill="#e3f2fd", stroke="#2196f3"))
c1.append(box("dino", "DINOv2 (ViT-L)", 220, 80, 150, 50, fill="#fff9c4", stroke="#fbc02d"))
c1.append(arrow("a1", "in_cam", "dino"))
c1.append(
    box(
        "vec",
        "Глобальний\nДескриптор\n[1024-d]",
        220,
        180,
        150,
        60,
        fill="#e8f5e9",
        stroke="#4caf50",
    )
)
c1.append(arrow("a2", "dino", "vec"))
c1.append(
    box(
        "lancedb",
        "LanceDB / FAISS\nВекторний індекс",
        220,
        290,
        150,
        60,
        fill="#ffe0b2",
        stroke="#f57c00",
    )
)
c1.append(arrow("a3", "vec", "lancedb"))
c1.append(box("db_frames", "Еталонна HDF5\nБД", 40, 290, 150, 60, fill="#e1bee7", stroke="#9c27b0"))
c1.append(arrow("a4", "db_frames", "lancedb"))

c1.append(box("bg2", "", 450, 10, 430, 400, fill="#f8f9fa", stroke="#ced4da"))
c1.append(text("t2", "2. Локальне зіставлення", 450, 20, 430, 30, bold=True))
c1.append(
    box("top1", "Топ-кандидат\n(з LanceDB)", 500, 80, 120, 50, fill="#ffe0b2", stroke="#f57c00")
)
c1.append(box("cam2", "Кадр з дрона", 710, 80, 120, 50, fill="#e3f2fd", stroke="#2196f3"))
c1.append(
    box("aliked", "ALIKED\n(Ключові точки)", 570, 180, 190, 50, fill="#ffccbc", stroke="#e64a19")
)
c1.append(arrow("a5", "top1", "aliked"))
c1.append(arrow("a6", "cam2", "aliked"))
c1.append(
    box(
        "lightglue",
        "LightGlue\n(Матчинг точок)",
        570,
        260,
        190,
        50,
        fill="#d1c4e9",
        stroke="#5e35b1",
    )
)
c1.append(arrow("a7", "aliked", "lightglue"))
c1.append(
    box(
        "ransac",
        "RANSAC\n(Гомографія + Inliers)",
        570,
        340,
        190,
        50,
        fill="#c8e6c9",
        stroke="#388e3c",
    )
)
c1.append(arrow("a8", "lightglue", "ransac"))

generate_file("slide_07_global_local_pipeline.drawio", c1)

# ---------------------------------------------------------
# 2. Slide 8: Tracking (Kalman + Z-Score)
# ---------------------------------------------------------
c2 = []
c2.append(box("bg_t", "", 10, 10, 860, 460, fill="#ffffff", stroke="#bdc3c7"))
c2.append(
    text("title_trk", "Обробка аномалій та прогнозування", 10, 20, 860, 40, bold=True, font_size=22)
)

c2.append(
    """<mxCell id="curve1" value="" style="curved=1;endArrow=none;html=1;strokeWidth=6;strokeColor=#bdc3c7;dashed=1;" edge="1" parent="1">
  <mxGeometry width="50" height="50" relative="1" as="geometry">
    <mxPoint x="100" y="350" as="sourcePoint" />
    <mxPoint x="800" y="200" as="targetPoint" />
    <Array as="points">
      <mxPoint x="300" y="380" />
      <mxPoint x="550" y="300" />
    </Array>
  </mxGeometry>
</mxCell>"""
)

c2.append(
    text(
        "kalman_lbl",
        "Kalman Filter: Прогнозована траєкторія",
        550,
        170,
        300,
        30,
        color="#7f8c8d",
        bold=True,
    )
)

c2.append(circle("p1", 200, 360, 20, fill="#2ecc71"))
c2.append(circle("p2", 350, 360, 20, fill="#2ecc71"))
c2.append(circle("p3", 480, 315, 20, fill="#2ecc71"))
c2.append(text("ok_lbl", "Валідна координата\n(Inliers > 10)", 300, 400, 150, 40, color="#27ae60"))

c2.append(circle("bad_p", 550, 120, 24, fill="#e74c3c"))
c2.append(arrow_coords("arr_bad", 490, 320, 545, 140, color="#e74c3c", width=3, dashed=True))
c2.append(
    text(
        "bad_txt",
        "Аномальний стрибок CV!\nШвидкість > 150 км/год",
        580,
        100,
        220,
        40,
        color="#c0392b",
        bold=True,
    )
)

c2.append(
    box(
        "zscore",
        "Z-Score Детектор\nБлокує координату",
        560,
        40,
        200,
        50,
        fill="#fab1a0",
        stroke="#e17055",
    )
)
c2.append(arrow_coords("arr_z", 650, 95, 570, 120, color="#e17055"))

c2.append(
    box(
        "kalman",
        "Фільтр Калмана\nВидає координату по інерції\n[x, y, vx, vy]",
        600,
        320,
        240,
        70,
        fill="#74b9ff",
        stroke="#0984e3",
    )
)
c2.append(arrow_coords("arr_k", 600, 350, 560, 310, color="#0984e3", width=3))

generate_file("slide_08_tracking_zscore_kalman.drawio", c2)

# ---------------------------------------------------------
# 3. Slide 9: TensorRT Optimization (Benchmark)
# ---------------------------------------------------------
c3 = []
c3.append(box("bg3", "", 10, 10, 860, 460, fill="#ffffff", stroke="#bdc3c7"))
c3.append(text("t3", "Оптимізація інференсу (TensorRT)", 10, 20, 860, 40, bold=True, font_size=22))

c3.append(box("pt", "Нативний PyTorch\n(FP32)", 80, 150, 200, 80, fill="#f8d7da", stroke="#dc3545"))
c3.append(text("pt_lat", "Затримка: ~300 мс", 80, 240, 200, 30, color="#dc3545", bold=True))

c3.append(
    box(
        "export",
        "Експорт в ONNX\n+ TensorRT Build",
        340,
        165,
        180,
        50,
        fill="#fff3cd",
        stroke="#ffc107",
    )
)
c3.append(arrow("a_pt_ex", "pt", "export"))

c3.append(
    box(
        "trt",
        "TensorRT Engine\n(Квантування FP16)",
        580,
        150,
        220,
        80,
        fill="#d4edda",
        stroke="#28a745",
    )
)
c3.append(
    text(
        "trt_lat", "Затримка: < 100 мс\n(Real-time)", 580, 240, 220, 50, color="#28a745", bold=True
    )
)
c3.append(arrow("a_ex_trt", "export", "trt"))

c3.append(box("vram", "Оптимізація VRAM", 330, 320, 200, 50, fill="#cce5ff", stroke="#007bff"))
c3.append(
    text(
        "vram_txt",
        "Пам'ять звільняється через\nLazy Loading (ModelManager)\n< 4 GB VRAM usage",
        300,
        380,
        260,
        60,
        color="#0056b3",
    )
)

generate_file("slide_09_tensorrt_optimization.drawio", c3)

# ---------------------------------------------------------
# 4. Slide 10: Explainability (LightGlue Matching)
# ---------------------------------------------------------
c4 = []
c4.append(box("bg4", "", 10, 10, 860, 460, fill="#ffffff", stroke="#bdc3c7"))
c4.append(
    text(
        "t4",
        "Візуальна пояснюваність (Inliers vs Outliers)",
        10,
        20,
        860,
        40,
        bold=True,
        font_size=22,
    )
)

c4.append(box("img_cam", "Кадр з дрона", 100, 120, 250, 180, fill="#ecf0f1", stroke="#95a5a6"))
c4.append(
    box(
        "img_map",
        "Еталонний супутниковий кадр",
        550,
        120,
        250,
        180,
        fill="#ecf0f1",
        stroke="#95a5a6",
    )
)

for y in [150, 180, 250, 270]:
    c4.append(arrow_coords(f"inl_{y}", 350, y, 550, y + 10, color="#2ecc71", width=2))
c4.append(
    text("inl_lbl", "Inliers (Правильні збіги)", 350, 150, 200, 30, color="#27ae60", bold=True)
)

c4.append(arrow_coords("out1", 350, 200, 550, 290, color="#e74c3c", width=2, dashed=True))
c4.append(arrow_coords("out2", 350, 280, 550, 140, color="#e74c3c", width=2, dashed=True))
c4.append(text("out_lbl", "Outliers\n(Відхилені RANSAC)", 350, 280, 200, 40, color="#c0392b"))

c4.append(
    box(
        "conf",
        "Confidence Score (Впевненість)",
        250,
        360,
        400,
        80,
        fill="#e8f5e9",
        stroke="#4caf50",
    )
)
c4.append(
    text(
        "conf_txt",
        "Якщо Inliers > 10 → Координата прийнята\nІнакше → Використовується Kalman Filter",
        260,
        380,
        380,
        50,
        color="#2e7d32",
    )
)

generate_file("slide_10_explainability_lightglue.drawio", c4)
