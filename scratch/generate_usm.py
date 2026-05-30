import html


def create_card(
    id, value, x, y, width, height, border_color, fill_color="#ffffff", font_size=12, rx=10
):
    return f"""
    <mxCell id="{id}" value="{html.escape(value)}" style="rounded=1;whiteSpace=wrap;html=1;strokeColor={border_color};strokeWidth=2;fillColor={fill_color};fontSize={font_size};align=left;spacingLeft=10;verticalAlign=top;spacingTop=10;fontColor=#333333;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry" />
    </mxCell>
    """


def create_story_card(id, value, author, x, y, width, height, border_color):
    # Added the author badge at the bottom of the card
    inner_html = f"""<div style='display:flex;flex-direction:column;justify-content:space-between;height:100%;'>
        <div>{html.escape(value)}</div>
        <div style='margin-top:10px;font-size:10px;color:#666;'><span style='background:#6a5acd;color:white;border-radius:50%;padding:2px 5px;'>S</span> {html.escape(author)}</div>
    </div>"""
    return f"""
    <mxCell id="{id}" value="{html.escape(inner_html)}" style="rounded=1;whiteSpace=wrap;html=1;strokeColor={border_color};strokeWidth=2;fillColor=#ffffff;fontSize=12;align=left;spacingLeft=10;verticalAlign=top;spacingTop=10;fontColor=#333333;html=1;whiteSpace=wrap;" vertex="1" parent="1">
      <mxGeometry x="{x}" y="{y}" width="{width}" height="{height}" as="geometry" />
    </mxCell>
    """


def main():
    xml = """<mxfile host="Electron" modified="2023-11-01T00:00:00.000Z" agent="Mozilla/5.0" version="22.0.4" type="device">
  <diagram id="usm-diagram" name="Page-1">
    <mxGraphModel dx="1434" dy="844" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="1169" pageHeight="827" background="#f5f5f5" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />

        <!-- Header -->
        <mxCell id="header" value="User Story Map для оператора системи локалізації БПЛА" style="rounded=1;whiteSpace=wrap;html=1;fillColor=#2d3436;fontColor=#ffffff;fontSize=16;fontStyle=1;strokeColor=none;arcSize=20;" vertex="1" parent="1">
          <mxGeometry x="40" y="20" width="1000" height="60" as="geometry" />
        </mxCell>

        <!-- MVP separator -->
        <mxCell id="mvp_line" value="" style="endArrow=none;html=1;strokeColor=#cccccc;strokeWidth=1;" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="40" y="270" as="sourcePoint" />
            <mxPoint x="1040" y="270" as="targetPoint" />
          </mxGeometry>
        </mxCell>
        <mxCell id="mvp_label" value="MVP | 9" style="text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontColor=#7f8c8d;fontSize=11;" vertex="1" parent="1">
          <mxGeometry x="40" y="240" width="60" height="30" as="geometry" />
        </mxCell>
"""
    col_w = 230
    gap = 20
    x_start = 50

    border_act = "#ff7675"
    border_step = "#a29bfe"
    border_story = "#55efc4"
    author = "Stepan Horyniuk"

    # Columns definition
    cols = [
        {
            "activity": "Підготовка еталонних даних",
            "steps": [
                {
                    "name": "Завантаження відео",
                    "stories_mvp": [
                        "Відкрити відеофайл .mp4 з дрона",
                        "Переглянути метадані відео",
                    ],
                    "stories_future": [],
                },
                {
                    "name": "Генерація HDF5-бази",
                    "stories_mvp": [
                        "Запуск створення бази в 1 клік",
                        "Відображення прогресу обробки",
                    ],
                    "stories_future": ["Підтримка кількох місій"],
                },
            ],
        },
        {
            "activity": "Калібрування системи",
            "steps": [
                {
                    "name": "Розстановка якорів",
                    "stories_mvp": [
                        "Встановлення GPS-координат для опорних кадрів",
                        "Автоматичний розрахунок масштабу (GSD)",
                    ],
                    "stories_future": ["Збереження якорів у JSON"],
                }
            ],
        },
        {
            "activity": "Локалізація в реальному часі",
            "steps": [
                {
                    "name": "Підключення відеопотоку",
                    "stories_mvp": [
                        "Налаштування джерела RTSP/Webcam",
                        "Увімкнення YOLO-маскування об'єктів",
                    ],
                    "stories_future": ["Підтримка тепловізора"],
                },
                {
                    "name": "Трекінг на карті",
                    "stories_mvp": [
                        "Відображення маркера на карті Leaflet",
                        "Відображення полігону поля зору камери",
                    ],
                    "stories_future": ["Анімація маршруту польоту"],
                },
            ],
        },
        {
            "activity": "Аналітика та інтеграція",
            "steps": [
                {
                    "name": "Отримання координат (API)",
                    "stories_mvp": ["Логування подій (Loguru)"],
                    "stories_future": [
                        "Налаштування REST/WebSocket сервера",
                        "Трансляція JSON координат в QGroundControl",
                    ],
                }
            ],
        },
    ]

    current_x = x_start
    id_counter = 100

    for col in cols:
        act = col["activity"]
        # Top Activity Card
        xml += create_card(f"act_{id_counter}", act, current_x, 100, col_w, 60, border_act)
        id_counter += 1

        # Steps
        step_x = current_x
        for step in col["steps"]:
            # Step Card
            xml += create_card(
                f"step_{id_counter}", step["name"], step_x, 180, col_w, 60, border_step
            )
            id_counter += 1

            # Stories MVP
            story_y = 290
            for story in step["stories_mvp"]:
                xml += create_story_card(
                    f"story_{id_counter}", story, author, step_x, story_y, col_w, 80, border_story
                )
                story_y += 90
                id_counter += 1

            # Future Separator
            if step["stories_future"]:
                xml += f"""
                <mxCell id="future_line_{id_counter}" value="" style="endArrow=none;html=1;strokeColor=#cccccc;strokeWidth=1;dashed=1;" edge="1" parent="1">
                  <mxGeometry width="50" height="50" relative="1" as="geometry">
                    <mxPoint x="{step_x}" y="500" as="sourcePoint" />
                    <mxPoint x="{step_x + col_w}" y="500" as="targetPoint" />
                  </mxGeometry>
                </mxCell>
                """

            # Stories Future
            story_y = 520
            for story in step["stories_future"]:
                xml += create_story_card(
                    f"story_{id_counter}", story, author, step_x, story_y, col_w, 80, border_story
                )
                story_y += 90
                id_counter += 1

            step_x += col_w + gap

        # Next column X is based on how many steps were in this column
        current_x += max(1, len(col["steps"])) * (col_w + gap)

    xml += """
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
"""

    with open(
        "C:/Users/horyn/.gemini/antigravity/brain/a6c235ef-3f39-4712-922b-29b0ba7dea73/diagrams/fig9_user_story_map.drawio",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(xml)
    print("User Story Map Draw.io file created successfully.")


if __name__ == "__main__":
    main()
