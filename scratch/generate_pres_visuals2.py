import html
import os

out_dir = r"C:\Users\horyn\.gemini\antigravity\brain\a6c235ef-3f39-4712-922b-29b0ba7dea73\diagrams\presentation_visuals"

drawio_template = """<mxfile host="Electron" modified="2026-05-28T00:00:00.000Z" agent="Mozilla/5.0" version="21.6.8" type="device">
  <diagram id="diag_1" name="Visual">
    <mxGraphModel dx="1280" dy="720" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="300" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{content}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""


def box(id, text, x, y, w, h, fill="#ffffff", stroke="#333333", font_size=16, rounded=1):
    safe_text = html.escape(text).replace("\n", "&lt;br&gt;")
    return f"""<mxCell id="{id}" value="{safe_text}" style="rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontSize={font_size};fontColor=#333333;verticalAlign=middle;align=center;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
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


def text(id, text, x, y, w, h, font_size=18, color="#333333", bold=False, align="center"):
    safe_text = html.escape(text).replace("\n", "&lt;br&gt;")
    style = f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={font_size};fontColor={color};"
    if bold:
        style += "fontStyle=1;"
    return f"""<mxCell id="{id}" value="{safe_text}" style="{style}" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
    </mxCell>"""


def circle(id, x, y, r, fill="#3498db"):
    return f"""<mxCell id="{id}" value="" style="ellipse;whiteSpace=wrap;html=1;aspect=fixed;fillColor={fill};strokeColor=none;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{r}" height="{r}" as="geometry" />
    </mxCell>"""


c2 = []
# Прибираємо сірий фон та рамку. Ширина 860, висота 280.
# Траєкторія (зліва направо, плавна дуга)
c2.append(
    """<mxCell id="curve1" value="" style="curved=1;endArrow=none;html=1;strokeWidth=5;strokeColor=#95a5a6;dashed=1;" edge="1" parent="1">
  <mxGeometry width="50" height="50" relative="1" as="geometry">
    <mxPoint x="50" y="220" as="sourcePoint" />
    <mxPoint x="820" y="150" as="targetPoint" />
    <Array as="points">
      <mxPoint x="200" y="240" />
      <mxPoint x="450" y="220" />
      <mxPoint x="650" y="140" />
    </Array>
  </mxGeometry>
</mxCell>"""
)

c2.append(
    text(
        "kalman_lbl",
        "Kalman Filter:\nЗгладжена траєкторія",
        100,
        160,
        200,
        40,
        color="#7f8c8d",
        bold=True,
        align="left",
    )
)

# Валідні точки (зелені, зліва і посередині)
c2.append(circle("p1", 100, 227, 18, fill="#2ecc71"))
c2.append(circle("p2", 250, 235, 18, fill="#2ecc71"))
c2.append(circle("p3", 400, 218, 18, fill="#2ecc71"))
c2.append(
    text(
        "ok_lbl",
        "Валідні координати\n(Inliers > 10)",
        220,
        260,
        150,
        30,
        color="#27ae60",
        font_size=14,
    )
)

# Аномальна точка (червона, справа, різкий стрибок вгору)
c2.append(circle("bad_p", 580, 50, 22, fill="#e74c3c"))
c2.append(arrow_coords("arr_bad", 420, 215, 570, 70, color="#e74c3c", width=3, dashed=True))
c2.append(
    text(
        "bad_txt",
        "Аномальний стрибок CV!\nШвидкість > 150 км/год",
        400,
        40,
        170,
        40,
        color="#c0392b",
        bold=True,
        font_size=14,
    )
)

# Z-Score Детектор вказує на червону точку
c2.append(
    box(
        "zscore",
        "Z-Score Детектор\nБлокує цю координату",
        630,
        10,
        180,
        40,
        fill="#fab1a0",
        stroke="#e17055",
        font_size=14,
    )
)
c2.append(arrow_coords("arr_z", 620, 30, 605, 50, color="#e17055", width=2))

# Фільтр Калмана продовжує шлях
c2.append(circle("p_pred", 600, 155, 18, fill="#3498db"))  # Прогнозована точка на сірій лінії
c2.append(
    box(
        "kalman",
        "Фільтр Калмана\nІгнорує стрибок і видає\nкоординату по інерції",
        580,
        200,
        200,
        50,
        fill="#74b9ff",
        stroke="#0984e3",
        font_size=14,
    )
)
c2.append(arrow_coords("arr_k", 650, 190, 615, 175, color="#0984e3", width=2))

path = os.path.join(out_dir, "slide_08_tracking_zscore_kalman.drawio")
with open(path, "w", encoding="utf-8") as f:
    f.write(drawio_template.format(content="\n".join(c2)))
print(f"Updated: {path}")
