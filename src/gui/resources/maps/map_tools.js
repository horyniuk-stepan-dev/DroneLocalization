/*
 * Map tools for the Leaflet (GEOGRAPHIC) map: point, line, polygon, rectangle,
 * circle, ruler, select/edit, delete, coordinate readout, go-to, import/export.
 *
 * The page is only a view. Every create/edit/delete goes to Python
 * (MapBridge.featureUpsert / featureDelete); Python stores the feature,
 * computes the geodesic measurement and sends the full list back through
 * loadFeaturesSignal, which re-renders the user layer.
 */
(function () {
    "use strict";

    var TOOLS = [
        { id: "select",    icon: "✋", title: "Вибір / переміщення карти (Esc)" },
        { id: "point",     icon: "📍", title: "Точка (M)" },
        { id: "line",      icon: "〰", title: "Лінія / маршрут (L) — подвійний клік або Enter завершує" },
        { id: "polygon",   icon: "⬠", title: "Площа / полігон (A) — подвійний клік, Enter або клік у першу вершину" },
        { id: "rectangle", icon: "▭", title: "Прямокутник (P) — два кліки по кутах" },
        { id: "circle",    icon: "◯", title: "Коло (C) — центр, потім точка на краю" },
        { id: "ruler",     icon: "📏", title: "Лінійка (R) — не зберігається" },
        { id: "delete",    icon: "🗑", title: "Видалити об'єкт кліком (D)" }
    ];
    var COLORS = ["#ffb000", "#ff4d4d", "#4da6ff", "#3ddc84", "#c77dff", "#ffffff"];
    var HOTKEYS = { m: "point", l: "line", a: "polygon", p: "rectangle", c: "circle", r: "ruler", d: "delete" };

    function esc(s) {
        return String(s).replace(/[&<>"']/g, function (c) {
            return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
        });
    }

    function MapTools(map, bridge) {
        this.map = map;
        this.bridge = bridge;
        this.tool = "select";
        this.draft = [];          // vertices of the shape being drawn
        this.draftLayer = L.layerGroup().addTo(map);
        this.rulerLayer = L.layerGroup().addTo(map);
        this.ruler = [];
        this.rulerDone = false;
        this.userLayer = L.layerGroup().addTo(map);
        this.features = [];
        this.editing = null;      // feature id with vertex handles shown
        this.counter = 0;
        this._buildToolbar();
        this._buildReadout();
        this._buildGoto();
        this._bindMap();
        this._buildContextMenu();
    }

    MapTools.prototype.isBusy = function () { return this.tool !== "select"; };

    // ── UI ─────────────────────────────────────────────────────────────────
    MapTools.prototype._buildToolbar = function () {
        var self = this;
        var Bar = L.Control.extend({
            options: { position: "topleft" },
            onAdd: function () {
                var box = L.DomUtil.create("div", "leaflet-bar ut-toolbar");
                self.buttons = {};
                TOOLS.forEach(function (t) {
                    var a = L.DomUtil.create("a", "ut-btn", box);
                    a.href = "#"; a.title = t.title; a.innerHTML = t.icon;
                    L.DomEvent.on(a, "click", function (e) { L.DomEvent.stop(e); self.setTool(t.id); });
                    self.buttons[t.id] = a;
                });
                L.DomUtil.create("div", "ut-sep", box);
                [["⇩", "Експорт у GeoJSON", function () { self.bridge.requestExport(); }],
                 ["⇧", "Імпорт GeoJSON", function () { self.bridge.requestImport(); }],
                 ["✖", "Видалити всі об'єкти на цій карті", function () { self.bridge.clearFeatures(); }]]
                    .forEach(function (b) {
                        var a = L.DomUtil.create("a", "ut-btn", box);
                        a.href = "#"; a.title = b[1]; a.innerHTML = b[0];
                        L.DomEvent.on(a, "click", function (e) { L.DomEvent.stop(e); b[2](); });
                    });
                L.DomEvent.disableClickPropagation(box);
                return box;
            }
        });
        new Bar().addTo(this.map);
        this._syncButtons();
    };

    MapTools.prototype._buildReadout = function () {
        var self = this;
        var Readout = L.Control.extend({
            options: { position: "bottomleft" },
            onAdd: function () {
                var d = L.DomUtil.create("div", "ut-panel ut-readout");
                d.title = "Клік — скопіювати координати";
                d.textContent = "—";
                L.DomEvent.on(d, "click", function (e) {
                    L.DomEvent.stop(e);
                    if (self.lastCursor) self.copy(self.lastCursor);
                    d.classList.add("ut-flash");
                    setTimeout(function () { d.classList.remove("ut-flash"); }, 300);
                });
                L.DomEvent.disableClickPropagation(d);
                self.readout = d;
                return d;
            }
        });
        new Readout().addTo(this.map);
    };

    MapTools.prototype._buildGoto = function () {
        var self = this;
        var Goto = L.Control.extend({
            options: { position: "topright" },
            onAdd: function () {
                var d = L.DomUtil.create("div", "ut-panel ut-goto");
                var input = L.DomUtil.create("input", "", d);
                input.placeholder = "lat, lon → Enter";
                L.DomEvent.on(input, "keydown", function (e) {
                    e.stopPropagation();
                    if (e.key !== "Enter") return;
                    var m = input.value.replace(/;/g, ",").match(/(-?\d+(?:\.\d+)?)[\s,]+(-?\d+(?:\.\d+)?)/);
                    if (!m) { input.classList.add("ut-bad"); return; }
                    var lat = parseFloat(m[1]), lon = parseFloat(m[2]);
                    if (Math.abs(lat) > 90 || Math.abs(lon) > 180) { input.classList.add("ut-bad"); return; }
                    input.classList.remove("ut-bad");
                    self.gotoPoint(lat, lon);
                });
                L.DomEvent.disableClickPropagation(d);
                L.DomEvent.disableScrollPropagation(d);
                return d;
            }
        });
        new Goto().addTo(this.map);
    };

    MapTools.prototype.gotoPoint = function (lat, lon) {
        var self = this;
        if (this.gotoMarker) this.map.removeLayer(this.gotoMarker);
        this.gotoMarker = L.circleMarker([lat, lon], { radius: 8, color: "#ff00ff", weight: 3, fill: false })
            .addTo(this.map)
            .bindTooltip(lat.toFixed(6) + ", " + lon.toFixed(6) + " (клік — зберегти як точку)", { direction: "top" });
        this.gotoMarker.on("click", function () {
            self._emitFeature({ type: "point", coords: [[lat, lon]] });
            self.map.removeLayer(self.gotoMarker); self.gotoMarker = null;
        });
        this.map.setView([lat, lon], Math.max(this.map.getZoom(), 16));
    };

    // ── Tool state ───────────────────────────────────────────────────────────
    MapTools.prototype.setTool = function (id) {
        this._cancelDraft();
        if (id !== "ruler") this._clearRuler();
        this.tool = id;
        var drawing = ["line", "polygon", "ruler"].indexOf(id) >= 0;
        if (drawing) this.map.doubleClickZoom.disable(); else this.map.doubleClickZoom.enable();
        this.map.getContainer().style.cursor = id === "select" ? "" : (id === "delete" ? "not-allowed" : "crosshair");
        this._syncButtons();
    };

    MapTools.prototype._syncButtons = function () {
        var self = this;
        Object.keys(this.buttons || {}).forEach(function (k) {
            self.buttons[k].classList.toggle("ut-active", k === self.tool);
        });
    };

    MapTools.prototype._cancelDraft = function () {
        this.draft = [];
        this.draftLayer.clearLayers();
    };

    MapTools.prototype._clearRuler = function () {
        this.ruler = []; this.rulerDone = false;
        this.rulerLayer.clearLayers();
    };

    // ── Map events ───────────────────────────────────────────────────────────
    MapTools.prototype._bindMap = function () {
        var self = this, map = this.map;
        map.on("click", function (e) { self._onClick(e.latlng); });
        map.on("dblclick", function (e) {
            if (self.tool === "line" || self.tool === "polygon") self._finishPath();
            else if (self.tool === "ruler") self.rulerDone = true;
        });
        map.on("mousemove", function (e) {
            self.lastCursor = e.latlng.lat.toFixed(6) + ", " + e.latlng.lng.toFixed(6);
            if (self.readout) self.readout.textContent = "Lat " + e.latlng.lat.toFixed(6) + "  Lon " + e.latlng.lng.toFixed(6);
            self._preview(e.latlng);
        });
        document.addEventListener("keydown", function (e) {
            if (e.target && e.target.tagName === "INPUT") return;
            if (e.key === "Escape") { self._hideMenu(); self.setTool("select"); return; }
            if (e.key === "Enter" && (self.tool === "line" || self.tool === "polygon")) { self._finishPath(); return; }
            if (e.key === "Backspace" && self.draft.length) {
                e.preventDefault(); self.draft.pop(); self._redrawDraft(null); return;
            }
            var t = HOTKEYS[e.key.toLowerCase()];
            if (t && !e.ctrlKey && !e.altKey) self.setTool(t);
        });
    };

    function samePlace(map, a, b) {
        return map.latLngToContainerPoint(a).distanceTo(map.latLngToContainerPoint(b)) < 4;
    }

    MapTools.prototype._onClick = function (ll) {
        var t = this.tool, map = this.map;
        if (t === "point") {
            this._emitFeature({ type: "point", coords: [[ll.lat, ll.lng]] });
        } else if (t === "line" || t === "polygon") {
            var d = this.draft;
            if (t === "polygon" && d.length >= 3 && samePlace(map, d[0], ll)) { this._finishPath(); return; }
            if (d.length && samePlace(map, d[d.length - 1], ll)) return; // second click of a dblclick
            d.push(ll); this._redrawDraft(null);
        } else if (t === "rectangle" || t === "circle") {
            if (!this.draft.length) { this.draft.push(ll); this._redrawDraft(ll); return; }
            var a = this.draft[0];
            if (samePlace(map, a, ll)) return;
            if (t === "rectangle") {
                this._emitFeature({ type: "rectangle", coords: [[a.lat, a.lng], [ll.lat, ll.lng]] });
            } else {
                // Python recomputes the radius geodesically from `edge`.
                this._emitFeature({ type: "circle", coords: [[a.lat, a.lng]], edge: [ll.lat, ll.lng] });
            }
            this._cancelDraft();
        } else if (t === "ruler") {
            if (this.rulerDone) this._clearRuler();
            if (this.ruler.length && samePlace(map, this.ruler[this.ruler.length - 1], ll)) return;
            this.ruler.push(ll);
            this._redrawRuler(null);
            if (this.ruler.length >= 2) {
                this.bridge.measureRuler(JSON.stringify(this.ruler.map(function (p) { return [p.lat, p.lng]; })));
            }
        }
    };

    MapTools.prototype._finishPath = function () {
        var d = this.draft, t = this.tool;
        // a dblclick adds a duplicate last vertex on some builds — drop it
        if (d.length >= 2 && samePlace(this.map, d[d.length - 1], d[d.length - 2])) d.pop();
        var need = t === "polygon" ? 3 : 2;
        if (d.length >= need) {
            this._emitFeature({ type: t, coords: d.map(function (p) { return [p.lat, p.lng]; }) });
        }
        this._cancelDraft();
    };

    MapTools.prototype._preview = function (ll) {
        if (this.draft.length) this._redrawDraft(ll);
        if (this.tool === "ruler" && this.ruler.length && !this.rulerDone) this._redrawRuler(ll);
    };

    MapTools.prototype._redrawDraft = function (cursor) {
        var t = this.tool, d = this.draft, g = this.draftLayer;
        g.clearLayers();
        if (!d.length) return;
        var style = { color: "#ffb000", weight: 2, dashArray: "6 4", interactive: false };
        if (t === "rectangle" && cursor) {
            L.rectangle(L.latLngBounds(d[0], cursor), style).addTo(g);
        } else if (t === "circle" && cursor) {
            L.circle(d[0], { radius: this.map.distance(d[0], cursor), color: style.color, weight: 2,
                dashArray: style.dashArray, interactive: false }).addTo(g);
            L.circleMarker(cursor, { radius: 1, opacity: 0, fillOpacity: 0, interactive: false }).addTo(g)
                .bindTooltip("r ≈ " + fmtLen(this.map.distance(d[0], cursor)),
                    { permanent: true, direction: "right", className: "ut-label" }).openTooltip();
        } else {
            var pts = cursor ? d.concat([cursor]) : d;
            if (t === "polygon" && pts.length >= 3) L.polygon(pts, style).addTo(g);
            else L.polyline(pts, style).addTo(g);
        }
        d.forEach(function (p) { L.circleMarker(p, { radius: 4, color: "#ffb000", interactive: false }).addTo(g); });
    };

    MapTools.prototype._redrawRuler = function (cursor) {
        var g = this.rulerLayer, pts = this.ruler;
        g.clearLayers();
        if (!pts.length) return;
        var all = cursor ? pts.concat([cursor]) : pts;
        L.polyline(all, { color: "#00e5ff", weight: 3, dashArray: "8 6", interactive: false }).addTo(g);
        pts.forEach(function (p) { L.circleMarker(p, { radius: 4, color: "#00e5ff", interactive: false }).addTo(g); });
        var text = this.rulerText || "";
        if (cursor && pts.length) {
            // Live preview is spherical (Leaflet); the clicked value comes from Python (WGS84).
            var total = 0;
            for (var i = 1; i < all.length; i++) total += this.map.distance(all[i - 1], all[i]);
            text = "≈ " + fmtLen(total);
        }
        if (text) {
            L.circleMarker(all[all.length - 1], { radius: 1, opacity: 0, fillOpacity: 0, interactive: false }).addTo(g)
                .bindTooltip(esc(text), { permanent: true, direction: "right", className: "ut-label ut-ruler" })
                .openTooltip();
        }
    };

    MapTools.prototype.setRulerText = function (text) {
        this.rulerText = text;
        this._redrawRuler(null);
    };

    // ── Features ─────────────────────────────────────────────────────────────
    MapTools.prototype._emitFeature = function (f) {
        var names = { point: "Точка", line: "Лінія", polygon: "Площа", rectangle: "Прямокутник", circle: "Коло" };
        this.counter += 1;
        f.name = f.name || (names[f.type] + " " + (this.features.length + 1));
        f.color = f.color || COLORS[0];
        f.mode = "GEOGRAPHIC";
        this.bridge.featureUpsert(JSON.stringify(f));
    };

    MapTools.prototype.loadFeatures = function (json) {
        var self = this;
        this.features = JSON.parse(json);
        this.userLayer.clearLayers();
        this.features.forEach(function (f) { self._renderFeature(f); });
    };

    MapTools.prototype._renderFeature = function (f) {
        var self = this, layer;
        var ll = f.coords.map(function (p) { return L.latLng(p[0], p[1]); });
        var style = { color: f.color, weight: 3, fillOpacity: 0.15, bubblingMouseEvents: false };
        if (f.type === "point") {
            layer = L.marker(ll[0], {
                draggable: true, bubblingMouseEvents: false,
                icon: L.divIcon({ className: "", html: '<div class="ut-pin" style="background:' + esc(f.color) + '"></div>',
                    iconSize: [16, 16], iconAnchor: [8, 16] })
            });
            layer.on("dragend", function () {
                var p = layer.getLatLng();
                self._update(f, { coords: [[p.lat, p.lng]] });
            });
        } else if (f.type === "line") {
            layer = L.polyline(ll, style);
        } else if (f.type === "polygon") {
            layer = L.polygon(ll, style);
        } else if (f.type === "rectangle") {
            layer = L.rectangle(L.latLngBounds(ll[0], ll[1]), style);
        } else if (f.type === "circle") {
            layer = L.circle(ll[0], L.extend({ radius: f.radius }, style));
        }
        if (!layer) return;
        var label = esc(f.name) + (f.measure ? "<br><small>" + esc(f.measure) + "</small>" : "");
        layer.bindTooltip(label, { permanent: true, direction: f.type === "point" ? "top" : "center",
            className: "ut-label", offset: f.type === "point" ? [0, -14] : [0, 0] });
        layer.on("contextmenu", function (e) { self._showMenu(e); });
        layer.on("click", function (e) {
            if (self.tool === "delete") { self.bridge.featureDelete(f.id); return; }
            if (self.tool === "select") { self._openPopup(f, e.latlng); return; }
            self._onClick(e.latlng); // drawing tools: treat as a map click
        });
        layer.addTo(this.userLayer);
        if (this.editing === f.id && f.type !== "point") this._addHandles(f);
    };

    MapTools.prototype._update = function (f, patch) {
        var g = {};
        ["id", "type", "coords", "name", "color", "radius", "mode"].forEach(function (k) {
            if (f[k] !== undefined) g[k] = f[k];
        });
        L.extend(g, patch);
        this.bridge.featureUpsert(JSON.stringify(g));
    };

    MapTools.prototype._addHandles = function (f) {
        var self = this, coords = f.coords.map(function (p) { return p.slice(); });
        coords.forEach(function (p, i) {
            var h = L.marker(p, { draggable: true, bubblingMouseEvents: false,
                icon: L.divIcon({ className: "", html: '<div class="ut-handle"></div>', iconSize: [12, 12], iconAnchor: [6, 6] }) });
            h.on("dragend", function () {
                var q = h.getLatLng(); coords[i] = [q.lat, q.lng];
                self._update(f, { coords: coords });
            });
            h.on("contextmenu", function () {
                var min = f.type === "polygon" ? 3 : (f.type === "line" ? 2 : 99);
                if (coords.length > min) { coords.splice(i, 1); self._update(f, { coords: coords }); }
            });
            h.addTo(self.userLayer);
        });
    };

    MapTools.prototype._openPopup = function (f, at) {
        var self = this;
        var colors = COLORS.map(function (c) {
            return '<span class="ut-swatch' + (c === f.color ? " ut-on" : "") + '" data-c="' + c + '" style="background:' + c + '"></span>';
        }).join("");
        var radius = f.type === "circle"
            ? '<label>Радіус, м <input class="ut-r" type="number" min="0.1" step="any" value="' + (+f.radius).toFixed(1) + '"></label>' : "";
        var html =
            '<div class="ut-popup">' +
            '<input class="ut-name" value="' + esc(f.name) + '">' +
            '<div class="ut-colors">' + colors + "</div>" + radius +
            '<div class="ut-measure">' + esc(f.measure || coordText(f)) + "</div>" +
            '<div class="ut-actions">' +
            '<button data-a="save">Зберегти</button>' +
            (f.type !== "point" ? '<button data-a="edit">' + (self.editing === f.id ? "Готово" : "Вершини") + "</button>" : "") +
            '<button data-a="copy">Копіювати</button>' +
            '<button data-a="del" class="ut-danger">Видалити</button>' +
            "</div></div>";
        var popup = L.popup({ minWidth: 220 }).setLatLng(at).setContent(html).openOn(this.map);
        var el = popup.getElement();
        var color = f.color;
        el.querySelectorAll(".ut-swatch").forEach(function (s) {
            s.onclick = function () {
                color = s.getAttribute("data-c");
                el.querySelectorAll(".ut-swatch").forEach(function (o) { o.classList.toggle("ut-on", o === s); });
            };
        });
        el.querySelectorAll("button").forEach(function (b) {
            b.onclick = function () {
                var a = b.getAttribute("data-a");
                if (a === "save") {
                    var patch = { name: el.querySelector(".ut-name").value.trim() || f.name, color: color };
                    var r = el.querySelector(".ut-r");
                    if (r && parseFloat(r.value) > 0) patch.radius = parseFloat(r.value);
                    self._update(f, patch);
                } else if (a === "edit") {
                    self.editing = self.editing === f.id ? null : f.id;
                    self.loadFeatures(JSON.stringify(self.features));
                } else if (a === "copy") {
                    self.copy(coordText(f));
                } else if (a === "del") {
                    self.bridge.featureDelete(f.id);
                }
                self.map.closePopup();
            };
        });
        el.querySelector(".ut-name").addEventListener("keydown", function (e) {
            e.stopPropagation();
            if (e.key === "Enter") el.querySelector('[data-a="save"]').click();
        });
    };

    // ── Right-click menu ─────────────────────────────────────────────────────
    MapTools.prototype._buildContextMenu = function () {
        var self = this, map = this.map;
        var menu = this.menu = L.DomUtil.create("div", "ut-menu", map.getContainer());
        menu.style.display = "none";
        var item = L.DomUtil.create("div", "ut-menu-item", menu);
        item.textContent = "📋 Копіювати координати";
        L.DomEvent.disableClickPropagation(menu);
        L.DomEvent.on(menu, "contextmenu", L.DomEvent.stop);
        L.DomEvent.on(item, "click", function (e) {
            L.DomEvent.stop(e);
            if (self.menuLatLng) {
                var t = self.menuLatLng.lat.toFixed(6) + ", " + self.menuLatLng.lng.toFixed(6);
                self.copy(t);
                self._toast("Скопійовано: " + t);
            }
            self._hideMenu();
        });
        map.on("contextmenu", function (e) { self._showMenu(e); });
        map.on("click movestart zoomstart", function () { self._hideMenu(); });
    };

    MapTools.prototype._showMenu = function (e) {
        if (e.originalEvent) L.DomEvent.preventDefault(e.originalEvent);
        this.menuLatLng = e.latlng;
        var size = this.map.getSize(), m = this.menu;
        m.style.display = "block";
        var x = Math.min(e.containerPoint.x, size.x - m.offsetWidth - 4);
        var y = Math.min(e.containerPoint.y, size.y - m.offsetHeight - 4);
        m.style.left = Math.max(0, x) + "px"; m.style.top = Math.max(0, y) + "px";
    };

    MapTools.prototype._hideMenu = function () {
        if (this.menu) this.menu.style.display = "none";
    };

    MapTools.prototype._toast = function (text) {
        var t = this.toastEl;
        if (!t) t = this.toastEl = L.DomUtil.create("div", "ut-toast", this.map.getContainer());
        t.textContent = text; t.style.display = "block";
        clearTimeout(this.toastTimer);
        this.toastTimer = setTimeout(function () { t.style.display = "none"; }, 1500);
    };

    // Python clipboard (QApplication) is reliable inside QWebEngine; execCommand is the fallback.
    MapTools.prototype.copy = function (text) {
        if (this.bridge && this.bridge.copyToClipboard) this.bridge.copyToClipboard(text);
        else copyText(text);
    };

    // ── helpers ──────────────────────────────────────────────────────────────
    function coordText(f) {
        return f.coords.map(function (p) { return p[0].toFixed(6) + ", " + p[1].toFixed(6); }).join("\n");
    }

    function fmtLen(m) { return m < 1000 ? m.toFixed(1) + " м" : (m / 1000).toFixed(3) + " км"; }

    function copyText(text) {
        var ta = document.createElement("textarea");
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand("copy"); } catch (e) { /* ignore */ }
        document.body.removeChild(ta);
    }

    window.MapTools = MapTools;
})();
