import html
import os

out_dir = r"C:\Users\horyn\.gemini\antigravity\brain\a6c235ef-3f39-4712-922b-29b0ba7dea73\diagrams\presentation_visuals"
os.makedirs(out_dir, exist_ok=True)

drawio_template = """<mxfile host="Electron" modified="2026-05-28T00:00:00.000Z" agent="Mozilla/5.0" version="21.6.8" type="device">
  <diagram id="diag_eval" name="Evaluation">
    <mxGraphModel dx="1280" dy="720" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="900" pageHeight="450" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
{content}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>"""


def box(
    id,
    text,
    x,
    y,
    w,
    h,
    fill="#ffffff",
    stroke="#333333",
    font_size=16,
    rounded=1,
    font_color="#333333",
    bold=False,
):
    safe_text = html.escape(text).replace("\n", "&lt;br&gt;")
    style = f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};strokeWidth=2;fontSize={font_size};fontColor={font_color};verticalAlign=middle;align=center;"
    if bold:
        style += "fontStyle=1;"
    return f"""<mxCell id="{id}" value="{safe_text}" style="{style}" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry" />
    </mxCell>"""


def text(id, text, x, y, w, h, font_size=18, color="#333333", bold=False, align="center"):
    safe_text = html.escape(text).replace("\n", "&lt;br&gt;")
    style = f"text;html=1;strokeColor=none;fillColor=none;align={align};verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize={font_size};fontColor={color};"
    if bold:
        style += "fontStyle=1;"
    return f"""<mxCell id="{id}" value="{safe_text}" style="{style}" vertex="1" parent="1">
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


c = []

# --- ЛІВА ЧАСТИНА: Швидкість (Inference) ---
c.append(box("bg_left", "", 10, 10, 420, 400, fill="#f8f9fa", stroke="#ced4da", rounded=1))
c.append(
    text(
        "t_left",
        "Швидкість (Inference TensorRT)",
        10,
        20,
        420,
        40,
        bold=True,
        font_size=20,
        color="#1e3799",
    )
)

# Графік порівняння швидкості (горизонтальні бари)
c.append(
    text(
        "pt_lbl",
        "Нативний PyTorch (FP32)",
        30,
        80,
        380,
        30,
        align="left",
        font_size=16,
        color="#e74c3c",
        bold=True,
    )
)
c.append(
    box(
        "pt_bar",
        "~250 мс",
        30,
        110,
        350,
        40,
        fill="#fad390",
        stroke="#e55039",
        font_size=16,
        font_color="#e55039",
        bold=True,
    )
)

c.append(
    text(
        "trt_lbl",
        "TensorRT Engine (FP16)",
        30,
        180,
        380,
        30,
        align="left",
        font_size=16,
        color="#27ae60",
        bold=True,
    )
)
c.append(
    box(
        "trt_bar",
        "~85 мс",
        30,
        210,
        150,
        40,
        fill="#b8e994",
        stroke="#78e08f",
        font_size=18,
        font_color="#006266",
        bold=True,
    )
)

c.append(arrow_coords("speed_arr", 380, 130, 180, 230, color="#e58e26", width=3, dashed=True))
c.append(text("speed_txt", "Прискорення\nу ~3 рази", 220, 150, 150, 40, color="#e58e26", bold=True))

c.append(
    box(
        "vram_box",
        "Ефективне використання пам'яті\nVRAM < 4 ГБ",
        30,
        300,
        380,
        60,
        fill="#82ccdd",
        stroke="#60a3bc",
        font_size=16,
        font_color="#0a3d62",
        bold=True,
    )
)


# --- ПРАВА ЧАСТИНА: Якість та Точність ---
c.append(box("bg_right", "", 450, 10, 420, 400, fill="#f8f9fa", stroke="#ced4da", rounded=1))
c.append(
    text(
        "t_right",
        "Якість та Точність (Метрики)",
        450,
        20,
        420,
        40,
        bold=True,
        font_size=20,
        color="#1e3799",
    )
)

# MAE (Mean Absolute Error)
c.append(box("mae_bg", "", 480, 80, 360, 130, fill="#e1b12c", stroke="none", rounded=1))
c.append(box("mae_inner", "", 485, 85, 350, 120, fill="#f5f6fa", stroke="none", rounded=1))
c.append(
    text(
        "mae_title",
        "MAE (Середня похибка позиції)",
        490,
        95,
        340,
        30,
        font_size=18,
        color="#718093",
    )
)
c.append(text("mae_val", "~ 4.2 м", 490, 120, 340, 70, font_size=56, color="#0097e6", bold=True))

# Recall@1
c.append(box("rec_bg", "", 480, 240, 360, 130, fill="#44bd32", stroke="none", rounded=1))
c.append(box("rec_inner", "", 485, 245, 350, 120, fill="#f5f6fa", stroke="none", rounded=1))
c.append(
    text(
        "rec_title",
        "Recall @ 1 (Точність розпізнавання)",
        490,
        255,
        340,
        30,
        font_size=18,
        color="#718093",
    )
)
c.append(text("rec_val", "~ 88 %", 490, 280, 340, 70, font_size=56, color="#4cd137", bold=True))


# Збереження
path = os.path.join(out_dir, "slide_12_evaluation_improved.drawio")
with open(path, "w", encoding="utf-8") as f:
    f.write(drawio_template.format(content="\n".join(c)))
print(f"Created: {path}")
