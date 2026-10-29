"""Georeference a local (no-GPS) map from control points placed on it.

The user drops 📍 points on the local X/Y map (the regular point tool), then
types the true lat/lon of each one here. The dialog fits local -> projected
metres live and shows per-point residuals; the caller applies the result.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)
from src.geometry.coordinates import CoordinateConverter

from src.geometry.georef_fit import AFFINE, MIN_POINTS, SIMILARITY, GeorefFit, fit_georef
from src.geometry.georef_graph import (
    GRAPH,
    MIN_GRAPH_POINTS,
    GraphGeorefFit,
    LocalGraph,
    fit_graph_georef,
)

_NUM = r"[-+]?\d+(?:[.,]\d+)?"
_LATLON = re.compile(rf"^\s*({_NUM})\s*[,; ]\s*({_NUM})\s*$")
_COLS = ["✓", "Назва", "X", "Y", "Координати (lat, lon)", "Похибка, м", "Перевірка, м"]


def parse_latlon(text: str) -> tuple[float, float] | None:
    """'48.123, 37.456' / '48.123 37.456' / '48,123; 37,456' -> (lat, lon)."""
    m = _LATLON.match(text)
    if not m:
        return None
    lat, lon = (float(g.replace(",", ".")) for g in m.groups())
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return lat, lon


@dataclass
class ControlPoint:
    id: str
    name: str
    x: float  # local map units
    y: float
    latlon_text: str = ""
    enabled: bool = True

    def to_dict(self):
        return self.__dict__.copy()


def load_points(sidecar: Path, local_features) -> list[ControlPoint]:
    """Saved GCPs merged with the LOCAL point features currently on the map."""
    saved: dict[str, ControlPoint] = {}
    try:
        for d in json.loads(sidecar.read_text(encoding="utf-8")).get("points", []):
            cp = ControlPoint(**d)
            saved[cp.id] = cp
    except (OSError, ValueError, TypeError):
        pass
    for f in local_features:
        if f.type != "point":
            continue
        y, x = f.coords[0]
        cp = saved.get(f.id)
        if cp is None:
            saved[f.id] = ControlPoint(f.id, f.name or f"Точка {len(saved) + 1}", x, y)
        else:  # the point may have been dragged or renamed
            cp.x, cp.y, cp.name = x, y, f.name or cp.name
    return list(saved.values())


def save_points(sidecar: Path, points: list[ControlPoint]) -> None:
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    data.update(version=1, points=[p.to_dict() for p in points])
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


class GeorefDialog(QDialog):
    def __init__(
        self,
        points: list[ControlPoint],
        converter: CoordinateConverter,
        rotation_sign: int,
        *,
        can_reset: bool = False,
        graph: LocalGraph | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Прив’язати локальну карту до GPS")
        self.resize(820, 460)
        self.points = points
        self.converter = converter
        self.rotation_sign = rotation_sign
        self.graph = graph
        self.fit: GeorefFit | GraphGeorefFit | None = None
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(350)
        self._timer.timeout.connect(self._refit_now)
        self.used: list[ControlPoint] = []
        self.reset_requested = False
        self.deleted_ids: list[str] = []
        self._filling = False

        lay = QVBoxLayout(self)
        hint = QLabel(
            "Поставте 📍 точки на локальній карті на впізнавані об’єкти (ріг будівлі, перехрестя), "
            "потім впишіть їхні справжні координати, напр. <code>48.512345, 37.981234</code>.<br>"
            "Мінімум 2 точки; 3+ дають перевірку. Ставте їх по краях карти, не в лінію."
        )
        hint.setWordWrap(True)
        lay.addWidget(hint)

        self.table = QTableWidget(0, len(_COLS))
        self.table.setHorizontalHeaderLabels(_COLS)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        hh.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.table)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)

        row_btns = QHBoxLayout()
        btn_delete = QPushButton("🗑 Видалити вибрані точки")
        btn_delete.setToolTip("Видаляє точку з цього списку і з карти (клавіша Delete)")
        btn_delete.clicked.connect(self._delete_selected)
        row_btns.addWidget(btn_delete)
        row_btns.addStretch()
        lay.addLayout(row_btns)

        opts = QHBoxLayout()
        self.chk_graph = QCheckBox("Нежорстка по графу кадрів (≥3 точки, виправляє дрейф масштабу)")
        self.chk_graph.setToolTip(
            "Кожен кадр отримує власну поправку масштабу/повороту/зсуву, плавну вздовж "
            "графа кадрів. 2 точки або узгоджені точки дають те саме, що й загальна модель; "
            "з 3+ точками карта «підтягується» до кожної з них. За межами крайніх точок "
            "поправка не екстраполюється."
        )
        self.chk_graph.setEnabled(graph is not None)
        self.chk_graph.setChecked(graph is not None)
        self.chk_graph.toggled.connect(self._on_model_toggled)
        opts.addWidget(self.chk_graph)
        self.chk_affine = QCheckBox(
            "Афінна модель (≥4 точки, дозволяє перекіс — лише для перевірки)"
        )
        self.chk_affine.toggled.connect(self._on_model_toggled)
        opts.addWidget(self.chk_affine)
        opts.addStretch()
        self.chk_features = QCheckBox("Перенести об’єкти локальної карти в GPS")
        self.chk_features.setChecked(True)
        opts.addWidget(self.chk_features)
        lay.addLayout(opts)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.summary)

        buttons = QDialogButtonBox()
        self.btn_apply = buttons.addButton("Застосувати", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("Закрити", QDialogButtonBox.ButtonRole.RejectRole)
        if can_reset:
            btn_reset = QPushButton("Скинути прив’язку (назад до X/Y)")
            buttons.addButton(btn_reset, QDialogButtonBox.ButtonRole.DestructiveRole)
            btn_reset.clicked.connect(self._on_reset)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay.addWidget(buttons)

        self._fill()
        self._refit_now()

    # ── table ────────────────────────────────────────────────────────────────

    def _fill(self):
        self._filling = True
        self.table.setRowCount(len(self.points))
        for r, p in enumerate(self.points):
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked if p.enabled else Qt.CheckState.Unchecked)
            self.table.setItem(r, 0, chk)
            for c, v in ((1, p.name), (2, f"{p.x:.3f}"), (3, f"{p.y:.3f}")):
                it = QTableWidgetItem(v)
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, it)
            self.table.setItem(r, 4, QTableWidgetItem(p.latlon_text))
            for c in (5, 6):
                it = QTableWidgetItem("")
                it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, c, it)
        self._filling = False

    def _on_item_changed(self, item):
        if self._filling:
            return
        p = self.points[item.row()]
        if item.column() == 0:
            p.enabled = item.checkState() == Qt.CheckState.Checked
        elif item.column() == 4:
            p.latlon_text = item.text().strip()
        self._refit()

    def _set_cell(self, r, c, text, color=None):
        self._filling = True
        it = self.table.item(r, c)
        it.setText(text)
        if color:
            it.setForeground(color)
        self._filling = False

    # ── fitting ──────────────────────────────────────────────────────────────

    def _on_model_toggled(self, on):
        sender = self.sender()
        if on and sender is self.chk_graph:
            self.chk_affine.setChecked(False)
        elif on and sender is self.chk_affine:
            self.chk_graph.setChecked(False)
        self._refit()

    def _kind(self) -> str:
        if self.chk_affine.isChecked():
            return AFFINE
        if self.chk_graph.isChecked() and self.graph is not None:
            return GRAPH
        return SIMILARITY

    def _refit(self, *_):
        # The graph fit re-solves the whole layer; do not run it per keystroke.
        if self._kind() == GRAPH:
            self.btn_apply.setEnabled(False)
            self._timer.start()
        else:
            self._refit_now()

    def _refit_now(self, *_):
        from PyQt6.QtGui import QColor

        self.fit, self.used = None, []
        rows, src, dst, lats = [], [], [], []
        for r, p in enumerate(self.points):
            self._set_cell(r, 5, "")
            self._set_cell(r, 6, "")
            ll = parse_latlon(p.latlon_text) if p.latlon_text else None
            if p.latlon_text and ll is None:
                self._set_cell(r, 5, "формат?", QColor("#e74c3c"))
            if not (p.enabled and ll):
                continue
            rows.append(r)
            src.append([p.x, p.y])
            dst.append(self.converter.gps_to_metric(*ll))
            lats.append(ll[0])
            self.used.append(p)

        kind = self._kind()
        if kind == GRAPH and len(src) < MIN_GRAPH_POINTS:
            kind = SIMILARITY  # 2 points: the graph model is exactly the global one
        need = MIN_POINTS[kind] if kind != GRAPH else MIN_GRAPH_POINTS
        if len(src) < need:
            self._summary(
                f"Потрібно щонайменше {need} точки з координатами (є {len(src)}).", ok=False
            )
            return
        try:
            if kind == GRAPH:
                fit = fit_graph_georef(self.graph, src, dst, rotation_sign=self.rotation_sign)
            else:
                fit = fit_georef(src, dst, kind=kind, rotation_sign=self.rotation_sign)
        except (ValueError, RuntimeError) as e:
            self._summary(f"Не вдалося обчислити: {e}", ok=False)
            return
        # Web-Mercator metres are stretched by 1/cos(lat); report real ground metres.
        k = self.converter.ground_scale_factor(float(np.mean(lats)))
        self.ground_k = k
        for i, r in enumerate(rows):
            res = fit.residuals[i] * k
            self._set_cell(r, 5, f"{res:.2f}", QColor("#2ecc71" if res < 5 else "#e67e22"))
            if fit.loo_residuals is not None and np.isfinite(fit.loo_residuals[i]):
                loo = fit.loo_residuals[i] * k
                self._set_cell(r, 6, f"{loo:.2f}", QColor("#2ecc71" if loo < 10 else "#e74c3c"))
        self.fit = fit
        lines = [
            f"<b>RMSE: {fit.rmse * k:.2f} м</b> · масштаб {fit.scale * k:.3f} м/од. · "
            f"поворот {fit.rotation_deg:.1f}° · {len(src)} точок · модель: "
            + {SIMILARITY: "подібність", AFFINE: "афінна", GRAPH: "нежорстка по графу"}[kind]
        ]
        if kind == GRAPH:
            lo, hi = fit.scale_range
            lines.append(
                f"Найбільша поправка відносно загальної моделі: {fit.max_correction * k:.1f} м · "
                f"масштаб уздовж карти {lo * 100:.0f}–{hi * 100:.0f}% від загального · "
                f"жорсткість {fit.stiffness:g}"
            )
            if fit.loo_residuals is not None and fit.global_loo_residuals is not None:
                g_loo = float(np.sqrt(np.nanmean(fit.global_loo_residuals**2))) * k
                n_loo = float(np.sqrt(np.nanmean(fit.loo_residuals**2))) * k
                better = n_loo < g_loo
                lines.append(
                    f"<b>Перевірка (RMSE): загальна {g_loo:.2f} м → по графу {n_loo:.2f} м</b>"
                    + ("" if better else
                       " <span style='color:#e67e22'>— граф не кращий, краще вимкнути</span>")
                )
        if fit.loo_residuals is not None:
            lines.append(
                "«Перевірка» = похибка точки, коли її не використовують для розрахунку. "
                "Велике значення в одній точці — помилка в її координатах."
            )
        lines += [f"<span style='color:#e67e22'>⚠ {w}</span>" for w in fit.warnings]
        self._summary("<br>".join(lines), ok=True)

    def _summary(self, html, ok):
        self.summary.setText(html)
        self.btn_apply.setEnabled(ok)

    def _delete_selected(self):
        rows = sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True)
        if not rows:
            return
        names = ", ".join(self.points[r].name for r in sorted(rows))
        answer = QMessageBox.question(self, "Видалити точки", f"Видалити: {names}?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        for r in rows:
            self.deleted_ids.append(self.points.pop(r).id)
        self._fill()
        self._refit()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete and self.table.hasFocus():
            self._delete_selected()
            return
        super().keyPressEvent(event)

    def _on_reset(self):
        self.reset_requested = True
        self.accept()

    @property
    def convert_features(self) -> bool:
        return self.chk_features.isChecked()

    def gcp_records(self) -> list[dict]:
        out = []
        k = getattr(self, "ground_k", 1.0)
        for i, p in enumerate(self.used):
            lat, lon = parse_latlon(p.latlon_text)
            rec = {"name": p.name, "x": p.x, "y": p.y, "lat": lat, "lon": lon}
            if self.fit is not None:
                rec["residual_m"] = float(self.fit.residuals[i] * k)
            out.append(rec)
        return out
