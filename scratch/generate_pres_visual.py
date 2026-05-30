import html

# --- ШАБЛОНИ ---
drawio_template = """<mxfile host="Electron" type="device">
{slides}
</mxfile>"""

slide_template = """  <diagram id="{id}" name="{name}">
    <mxGraphModel dx="1280" dy="720" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1280" pageHeight="720" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="{id}_bg" value="" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FAFAFA;strokeColor=none;" vertex="1" parent="1">
          <mxGeometry width="1280" height="720" as="geometry" />
        </mxCell>
        <mxCell id="{id}_title" value="{title}" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=36;fontStyle=1;fontColor=#0050ef;" vertex="1" parent="1">
          <mxGeometry x="60" y="40" width="1160" height="60" as="geometry" />
        </mxCell>
        <mxCell id="{id}_line" value="" style="line;strokeWidth=4;html=1;strokeColor=#0050ef;" vertex="1" parent="1">
          <mxGeometry x="60" y="100" width="1160" height="10" as="geometry" />
        </mxCell>
{content}
      </root>
    </mxGraphModel>
  </diagram>
"""


# Хелпери для малювання
def box(id, text, x, y, w, h, fill="#ffffff", stroke="#333333", font_size=18, rounded=1):
    return f"""<mxCell id="{id}" value="{html.escape(text)}" style="rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontSize={font_size};fontColor=#333333;verticalAlign=middle;align=center;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
    </mxCell>"""


def text(id, text, x, y, w, h, font_size=20, align="left", color="#333333"):
    return f"""<mxCell id="{id}" value="{html.escape(text)}" style="text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=top;whiteSpace=wrap;rounded=0;fontSize={font_size};fontColor={color};" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
    </mxCell>"""


def arrow(id, src, dst, label=""):
    lbl_str = ""
    if label:
        lbl_str = f' value="{html.escape(label)}"'
    return f"""<mxCell id="{id}"{lbl_str} style="edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=3;strokeColor=#666666;fontSize=14;fontColor=#333;" edge="1" parent="1" source="{src}" target="{dst}">
      <mxGeometry relative="1" as="geometry" />
    </mxCell>"""


def arrow_coords(id, sx, sy, tx, ty, color="#666666", width=3, dashed=False):
    style = f"endArrow=block;html=1;strokeWidth={width};strokeColor={color};"
    if dashed:
        style += "dashed=1;"
    return f"""<mxCell id="{id}" value="" style="{style}" edge="1" parent="1">
      <mxGeometry width="50" height="50" relative="1" as="geometry">
        <mxPoint x="{sx}" y="{sy}" as="sourcePoint" />
        <mxPoint x="{tx}" y="{ty}" as="targetPoint" />
      </mxGeometry>
    </mxCell>"""


def circle(id, x, y, r, fill="#3498db"):
    return f"""<mxCell id="{id}" value="" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={fill};strokeColor=none;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{r}" height="{r}" as="geometry" />
    </mxCell>"""


slides_data = []

# ==========================================
# Slide 1: ML Pipeline (DINOv2 + ALIKED)
# ==========================================
s1_content = []
# Вхід
s1_content.append(box("ml_in1", "Поточний кадр з дрона", 100, 250, 200, 80, fill="#ecf0f1"))
s1_content.append(box("ml_in2", "Еталонний кадр (база)", 100, 450, 200, 80, fill="#ecf0f1"))

# DINOv2
s1_content.append(
    box(
        "ml_dino",
        "DINOv2 (ViT-Small)\\nГлобальний екстрактор",
        400,
        250,
        250,
        100,
        fill="#ffeaa7",
        stroke="#fdcb6e",
    )
)
s1_content.append(arrow("arr_d1", "ml_in1", "ml_dino"))
s1_content.append(
    box(
        "ml_faiss",
        "FAISS\\nПошук найсхожішого",
        400,
        450,
        250,
        80,
        fill="#81ecec",
        stroke="#00cec9",
    )
)
s1_content.append(arrow("arr_d2", "ml_dino", "ml_faiss", "Вектор (1024 dim)"))

# ALIKED
s1_content.append(
    box(
        "ml_aliked",
        "ALIKED\\nЛокальний екстрактор",
        800,
        350,
        250,
        100,
        fill="#fab1a0",
        stroke="#e17055",
    )
)
s1_content.append(arrow("arr_a1", "ml_in1", "ml_aliked"))
s1_content.append(arrow("arr_a2", "ml_in2", "ml_aliked"))

s1_content.append(
    box(
        "ml_out",
        "N ключових точок (x,y)\\n+ дескриптори",
        1100,
        360,
        150,
        80,
        fill="#55efc4",
        stroke="#00b894",
        font_size=14,
    )
)
s1_content.append(arrow("arr_a3", "ml_aliked", "ml_out"))

slides_data.append(
    slide_template.format(
        id="slide7",
        name="7. Конвеєр ML",
        title="Конвеєр машинного навчання (Витягування ознак)",
        content="\\n".join(s1_content),
    )
)

# ==========================================
# Slide 2: LightGlue + RANSAC
# ==========================================
s2_content = []
s2_content.append(text("t1", "1. Ключові точки (Дрон)", 100, 200, 250, 40))
s2_content.append(text("t2", "2. Точки (Еталон)", 400, 200, 200, 40))

# Малюємо точки (Дрон)
for i, y in enumerate([280, 340, 400, 460, 520]):
    s2_content.append(circle(f"d_pt{i}", 180, y, 20, fill="#e74c3c"))

# Малюємо точки (Еталон)
for i, y in enumerate([260, 320, 420, 480, 550]):
    s2_content.append(circle(f"r_pt{i}", 450, y, 20, fill="#2ecc71"))

# Зв'язки (Inliers - паралельні, Outliers - перехрещені)
# Inliers
s2_content.append(arrow_coords("m1", 200, 290, 450, 270, color="#3498db"))
s2_content.append(arrow_coords("m2", 200, 350, 450, 330, color="#3498db"))
s2_content.append(arrow_coords("m3", 200, 410, 450, 490, color="#e74c3c", dashed=True))  # Outlier
s2_content.append(arrow_coords("m4", 200, 470, 450, 430, color="#3498db"))
s2_content.append(arrow_coords("m5", 200, 530, 450, 300, color="#e74c3c", dashed=True))  # Outlier

s2_content.append(
    box(
        "lg_box",
        "LightGlue\\nГрафова нейромережа\\n(Знаходить зв'язки)",
        230,
        600,
        200,
        80,
        fill="#a29bfe",
        stroke="#6c5ce7",
    )
)

# RANSAC
s2_content.append(
    box(
        "ransac_box",
        "Фільтр RANSAC\\n(Відсіює червоні хибні зв'язки\\nі рахує Гомографію)",
        650,
        350,
        250,
        100,
        fill="#fdcb6e",
        stroke="#e17055",
    )
)

s2_content.append(arrow_coords("arr_r1", 500, 400, 650, 400))

# Result
s2_content.append(
    box(
        "res_box",
        "Фінальні координати (X, Y)\\nПолігон кадру",
        1000,
        360,
        200,
        80,
        fill="#55efc4",
        stroke="#00b894",
    )
)
s2_content.append(arrow("arr_r2", "ransac_box", "res_box"))

slides_data.append(
    slide_template.format(
        id="slide8",
        name="8. Зіставлення",
        title="Геометричне зіставлення (LightGlue + RANSAC)",
        content="\\n".join(s2_content),
    )
)

# ==========================================
# Slide 3: Kalman + Z-Score
# ==========================================
s3_content = []
s3_content.append(
    text(
        "k_t", "Проблема: Одиничні хибні розпізнавання (аномалії)", 100, 150, 800, 40, font_size=24
    )
)

# Крива (шлях дрона)
s3_content.append(
    """<mxCell id="curve1" value="" style="curved=1;endArrow=none;html=1;strokeWidth=4;strokeColor=#34495e;" edge="1" parent="1">
  <mxGeometry width="50" height="50" relative="1" as="geometry">
    <mxPoint x="150" y="500" as="sourcePoint" />
    <mxPoint x="800" y="300" as="targetPoint" />
    <Array as="points">
      <mxPoint x="300" y="450" />
      <mxPoint x="500" y="480" />
      <mxPoint x="650" y="350" />
    </Array>
  </mxGeometry>
</mxCell>"""
)

# Точки (Good)
s3_content.append(circle("pt1", 300, 445, 15, fill="#2ecc71"))
s3_content.append(circle("pt2", 500, 475, 15, fill="#2ecc71"))
s3_content.append(circle("pt3", 650, 345, 15, fill="#2ecc71"))

# Точка (Outlier)
s3_content.append(circle("pt_bad", 550, 200, 20, fill="#e74c3c"))
s3_content.append(
    text("bad_lbl", "Аномальний стрибок\\n(Помилка CV)", 580, 180, 200, 40, color="#e74c3c")
)

# Z-score box
s3_content.append(
    box(
        "z_box",
        "Z-Score Детектор\\nВідхилення > 2.5σ (швидкість)",
        900,
        150,
        250,
        80,
        fill="#fab1a0",
        stroke="#e17055",
    )
)
s3_content.append(arrow_coords("arr_z", 560, 210, 900, 190, dashed=True))

# Kalman box
s3_content.append(
    box(
        "kal_box",
        "Фільтр Калмана\\nЗгладжує траєкторію та\\nрухає дрон по інерції",
        900,
        400,
        250,
        80,
        fill="#74b9ff",
        stroke="#0984e3",
    )
)
s3_content.append(arrow_coords("arr_k", 800, 300, 900, 440, dashed=True))

slides_data.append(
    slide_template.format(
        id="slide9",
        name="9. Трекінг",
        title="Фільтрація та трекінг (Kalman + Z-Score)",
        content="\\n".join(s3_content),
    )
)

# Зберігаємо
output_xml = drawio_template.format(slides="\\n".join(slides_data))
out_path = "C:/Users/horyn/.gemini/antigravity/brain/a6c235ef-3f39-4712-922b-29b0ba7dea73/diagrams/presentation_visuals.drawio"

with open(out_path, "w", encoding="utf-8") as f:
    f.write(output_xml)

print(f"Visual presentation saved to {out_path}")
