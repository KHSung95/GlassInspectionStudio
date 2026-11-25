"""High-resolution image viewer (QGraphicsView). Scene coordinates == original image pixel coordinates
(pixel (i, j) covers [i, i+1) x [j, j+1)). Display transforms never touch the original array."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QGraphicsEllipseItem, QGraphicsItem, QGraphicsPathItem, QGraphicsPixmapItem,
                               QGraphicsRectItem, QGraphicsScene, QGraphicsView)

STATE_STYLE = {  # label_state -> (pen style, fill alpha)
    "positive": (Qt.SolidLine, 0),
    "negative": (Qt.SolidLine, 40),
    "unknown": (Qt.DotLine, 0),
    "ignore": (Qt.DashDotLine, 60),
}


def gray_qimage(arr8: np.ndarray) -> QImage:
    arr8 = np.ascontiguousarray(arr8)
    h, w = arr8.shape
    img = QImage(arr8.data, w, h, arr8.strides[0], QImage.Format_Grayscale8)
    return img.copy()  # detach from numpy buffer


def colormap_qimage(score: np.ndarray, vmin: float, vmax: float, alpha: float = 0.6) -> QImage:
    """Score map -> RGBA 'inferno-like' overlay (transparent below vmin)."""
    s = np.clip((score - vmin) / max(vmax - vmin, 1e-12), 0, 1).astype(np.float32)
    r = np.clip(1.5 * s, 0, 1); g = np.clip(1.5 * s - 0.5, 0, 1); b = np.clip(0.6 - np.abs(s - 0.3) * 2, 0, 1)
    a = np.where(score > vmin, alpha * (0.35 + 0.65 * s), 0)
    rgba = (np.stack([r, g, b, a], -1) * 255).astype(np.uint8)
    h, w = s.shape
    return QImage(np.ascontiguousarray(rgba).data, w, h, w * 4, QImage.Format_RGBA8888).copy()


class ImageViewer(QGraphicsView):
    mouseMoved = Signal(float, float)                  # image coords
    shapeCreated = Signal(str, dict, object)           # tool, geometry, mask(np.ndarray|None)
    itemClicked = Signal(str, int)                     # 'annotation'|'candidate', id
    viewChanged = Signal()
    roiMeasured = Signal(dict)

    TOOLS = ("select", "pan", "point", "bbox", "polygon", "brush", "normal_roi", "ignore_roi", "measure")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setBackgroundBrush(QColor(24, 24, 24))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)  # nearest-neighbour: see real pixels
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.pix = QGraphicsPixmapItem(); self.pix.setTransformationMode(Qt.FastTransformation)
        self.pix.setZValue(0)
        self.scene().addItem(self.pix)
        self.overlay = QGraphicsPixmapItem(); self.overlay.setZValue(1)
        self.overlay.setTransformationMode(Qt.FastTransformation)
        self.scene().addItem(self.overlay)
        self.tool = "select"
        self.raw: np.ndarray | None = None
        self.ann_items: list[QGraphicsItem] = []
        self.cand_items: list[QGraphicsItem] = []
        self._drag_start: QPointF | None = None
        self._temp: QGraphicsItem | None = None
        self._poly: list[QPointF] = []
        self._brush_mask: np.ndarray | None = None
        self.brush_radius = 3
        self._panning = None
        self.show_grid = True
        self.draw_color = QColor("#ff3030")

    # ------------------------------------------------------------------ image
    def set_image(self, raw: np.ndarray, disp8: np.ndarray, keep_view: bool = True) -> None:
        first = self.raw is None
        same_size = self.raw is not None and self.raw.shape == raw.shape
        self.raw = raw
        self.pix.setPixmap(QPixmap.fromImage(gray_qimage(disp8)))
        self.scene().setSceneRect(QRectF(-50, -50, raw.shape[1] + 100, raw.shape[0] + 100))
        if first or not (keep_view and same_size):
            self.fit()

    def update_display(self, disp8: np.ndarray) -> None:
        self.pix.setPixmap(QPixmap.fromImage(gray_qimage(disp8)))

    def set_overlay(self, rgba: QImage | None, opacity: float = 1.0) -> None:
        if rgba is None:
            self.overlay.setPixmap(QPixmap())
        else:
            self.overlay.setPixmap(QPixmap.fromImage(rgba))
            self.overlay.setOpacity(opacity)

    def fit(self) -> None:
        if self.raw is not None:
            self.fitInView(QRectF(0, 0, self.raw.shape[1], self.raw.shape[0]), Qt.KeepAspectRatio)
            self.viewChanged.emit()

    def one_to_one(self) -> None:
        c = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(c)
        self.viewChanged.emit()

    def zoom(self) -> float:
        return self.transform().m11()

    def zoom_by(self, f: float) -> None:
        z = self.zoom()
        f = max(0.02 / z, min(f, 96.0 / z))
        self.scale(f, f)
        self.viewChanged.emit()

    def center_on_image(self, x: float, y: float, zoom: float | None = None) -> None:
        if zoom is not None:
            self.resetTransform(); self.scale(zoom, zoom)
        self.centerOn(QPointF(x, y))
        self.viewChanged.emit()

    def view_state(self):
        return self.transform(), self.mapToScene(self.viewport().rect().center())

    def set_view_state(self, st) -> None:
        tr, c = st
        self.setTransform(tr); self.centerOn(c)

    # ------------------------------------------------------------------ overlays
    def clear_items(self, which: str) -> None:
        lst = self.ann_items if which == "ann" else self.cand_items
        for it in lst:
            self.scene().removeItem(it)
        lst.clear()

    def _shape_item(self, gtype: str, g: dict, color: QColor, style, fill_alpha: int, mask=None, width=1.5):
        pen = QPen(color, width); pen.setCosmetic(True); pen.setStyle(style)
        brush = QBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha)) if fill_alpha else QBrush(Qt.NoBrush)
        if gtype == "point":
            r = max(float(g.get("r", 2.0)), 1.0) + 2.0
            it = QGraphicsEllipseItem(g["x"] - r, g["y"] - r, 2 * r, 2 * r)
            # centre cross (cosmetic, always visible)
            path = QPainterPath()
            path.addEllipse(QPointF(g["x"], g["y"]), r, r)
            path.moveTo(g["x"] - r * 1.8, g["y"]); path.lineTo(g["x"] - r * 0.6, g["y"])
            path.moveTo(g["x"] + r * 0.6, g["y"]); path.lineTo(g["x"] + r * 1.8, g["y"])
            path.moveTo(g["x"], g["y"] - r * 1.8); path.lineTo(g["x"], g["y"] - r * 0.6)
            path.moveTo(g["x"], g["y"] + r * 0.6); path.lineTo(g["x"], g["y"] + r * 1.8)
            it = QGraphicsPathItem(path)
        elif gtype == "bbox":
            it = QGraphicsRectItem(QRectF(QPointF(g["x0"], g["y0"]), QPointF(g["x1"], g["y1"])).normalized())
        elif gtype == "polygon":
            poly = QPolygonF([QPointF(x, y) for x, y in g["points"]])
            path = QPainterPath(); path.addPolygon(poly); path.closeSubpath()
            it = QGraphicsPathItem(path)
        elif gtype == "mask" and mask is not None:
            rgba = np.zeros(mask.shape + (4,), np.uint8)
            rgba[mask > 0] = (color.red(), color.green(), color.blue(), 110)
            img = QImage(np.ascontiguousarray(rgba).data, mask.shape[1], mask.shape[0], mask.shape[1] * 4,
                         QImage.Format_RGBA8888).copy()
            it = QGraphicsPixmapItem(QPixmap.fromImage(img)); it.setPos(g["x0"], g["y0"])
            return it
        else:
            it = QGraphicsRectItem(QRectF(g.get("x0", 0), g.get("y0", 0), g.get("w", 1), g.get("h", 1)))
        it.setPen(pen)
        if hasattr(it, "setBrush"):
            it.setBrush(brush)
        return it

    def show_annotations(self, anns: list[dict], colors: dict[str, str], masks: dict[int, np.ndarray],
                         selected: set[int] | None = None) -> None:
        self.clear_items("ann")
        for a in anns:
            col = QColor(colors.get(a["class_name"], "#ffffff"))
            style, fa = STATE_STYLE.get(a["label_state"], (Qt.SolidLine, 0))
            if not a.get("confirmed", 1):
                style = Qt.DashLine
            w = 3.0 if selected and a["id"] in selected else 1.5
            it = self._shape_item(a["geometry_type"], a["geometry"], col, style, fa, masks.get(a["id"]), w)
            it.setZValue(5)
            it.setData(0, ("annotation", a["id"]))
            it.setToolTip(f"#{a['id']} {a['class_name']} [{a['label_state']}] vis={a.get('visibility')} "
                          f"src={a.get('source')} confirmed={a.get('confirmed')}")
            self.scene().addItem(it); self.ann_items.append(it)

    def show_candidates(self, cands: list[dict], selected: int | None = None) -> None:
        """AI candidates: dashed yellow/cyan, visually distinct from confirmed labels."""
        self.clear_items("cand")
        for c in cands:
            st = c.get("review_status", "pending")
            col = QColor("#ffe14d") if st == "pending" else QColor("#4dd2ff") if st == "accepted" else QColor("#888888")
            it = self._shape_item(c["geometry_type"], c["geometry"], col, Qt.DashLine, 0, None,
                                  3.0 if selected == c["id"] else 1.2)
            it.setZValue(4)
            it.setData(0, ("candidate", c["id"]))
            it.setToolTip(f"candidate #{c['id']} {c['class_name']} score={c['score']:.4g} ({c['source_algorithm']}) "
                          f"[{st}]")
            self.scene().addItem(it); self.cand_items.append(it)

    # ------------------------------------------------------------------ grid / readout
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:
        z = self.zoom()
        if self.raw is None or not self.show_grid or z < 12:
            return
        pen = QPen(QColor(90, 90, 90, 160), 0); painter.setPen(pen)
        x0, x1 = max(0, int(math.floor(rect.left()))), min(self.raw.shape[1], int(math.ceil(rect.right())))
        y0, y1 = max(0, int(math.floor(rect.top()))), min(self.raw.shape[0], int(math.ceil(rect.bottom())))
        for x in range(x0, x1 + 1):
            painter.drawLine(QPointF(x, y0), QPointF(x, y1))
        for y in range(y0, y1 + 1):
            painter.drawLine(QPointF(x0, y), QPointF(x1, y))
        if z >= 40 and (x1 - x0) * (y1 - y0) < 1600:
            # raw pixel values, drawn in viewport coordinates with a normal font size
            painter.save()
            painter.resetTransform()
            f = painter.font(); f.setPointSize(8); painter.setFont(f)
            painter.setPen(QColor(255, 220, 0))
            for y in range(y0, y1):
                for x in range(x0, x1):
                    tl = self.mapFromScene(QPointF(x, y)); br = self.mapFromScene(QPointF(x + 1, y + 1))
                    r = QRectF(tl, br); t = str(int(self.raw[y, x]))
                    painter.setPen(QColor(0, 0, 0)); painter.drawText(r.translated(1, 1), Qt.AlignCenter, t)
                    painter.setPen(QColor(255, 220, 0)); painter.drawText(r, Qt.AlignCenter, t)
            painter.restore()

    # ------------------------------------------------------------------ events
    def wheelEvent(self, e) -> None:
        self.zoom_by(1.25 if e.angleDelta().y() > 0 else 0.8)

    def set_tool(self, t: str) -> None:
        if t not in self.TOOLS:
            raise ValueError(t)
        self._cancel_temp()
        self.tool = t
        self.setCursor(Qt.OpenHandCursor if t == "pan" else (Qt.ArrowCursor if t == "select" else Qt.CrossCursor))

    def _cancel_temp(self) -> None:
        if self._temp is not None:
            self.scene().removeItem(self._temp); self._temp = None
        self._poly = []; self._drag_start = None; self._brush_mask = None

    def mousePressEvent(self, e) -> None:
        p = self.mapToScene(e.position().toPoint())
        if e.button() == Qt.MiddleButton or (e.button() == Qt.LeftButton and self.tool == "pan"):
            self._panning = e.position(); self.setCursor(Qt.ClosedHandCursor); return
        if e.button() != Qt.LeftButton or self.raw is None:
            return super().mousePressEvent(e)
        if self.tool == "select":
            for it in self.items(e.position().toPoint()):
                d = it.data(0)
                if d:
                    self.itemClicked.emit(d[0], int(d[1])); return
            return
        if self.tool == "point":
            self.shapeCreated.emit("point", {"x": p.x(), "y": p.y(), "r": 2.0}, None); return
        if self.tool in ("bbox", "normal_roi", "ignore_roi", "measure"):
            self._drag_start = p
            self._temp = QGraphicsRectItem(QRectF(p, p))
            pen = QPen(self.draw_color if self.tool == "bbox" else QColor("#00ff88") if self.tool == "normal_roi"
                       else QColor("#aaaaaa") if self.tool == "ignore_roi" else QColor("#00ccff"), 1.5)
            pen.setCosmetic(True); self._temp.setPen(pen); self._temp.setZValue(10)
            self.scene().addItem(self._temp); return
        if self.tool == "polygon":
            self._poly.append(p)
            path = QPainterPath(); path.addPolygon(QPolygonF(self._poly))
            if self._temp is None:
                self._temp = QGraphicsPathItem(); pen = QPen(self.draw_color, 1.5); pen.setCosmetic(True)
                self._temp.setPen(pen); self._temp.setZValue(10); self.scene().addItem(self._temp)
            self._temp.setPath(path); return
        if self.tool == "brush":
            self._brush_mask = np.zeros(self.raw.shape, np.uint8)
            self._paint(p); return

    def _paint(self, p: QPointF) -> None:
        import cv2
        cv2.circle(self._brush_mask, (int(p.x()), int(p.y())), int(self.brush_radius), 1, -1)
        ys, xs = np.nonzero(self._brush_mask)
        if self._temp is not None:
            self.scene().removeItem(self._temp)
        x0, y0 = xs.min(), ys.min()
        sub = self._brush_mask[y0:ys.max() + 1, x0:xs.max() + 1]
        rgba = np.zeros(sub.shape + (4,), np.uint8); rgba[sub > 0] = (255, 60, 60, 120)
        img = QImage(np.ascontiguousarray(rgba).data, sub.shape[1], sub.shape[0], sub.shape[1] * 4,
                     QImage.Format_RGBA8888).copy()
        self._temp = QGraphicsPixmapItem(QPixmap.fromImage(img)); self._temp.setPos(x0, y0); self._temp.setZValue(10)
        self.scene().addItem(self._temp)

    def mouseMoveEvent(self, e) -> None:
        p = self.mapToScene(e.position().toPoint())
        self.mouseMoved.emit(p.x(), p.y())
        if self._panning is not None:
            d = e.position() - self._panning; self._panning = e.position()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(d.x()))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(d.y()))
            self.viewChanged.emit(); return
        if self._drag_start is not None and isinstance(self._temp, QGraphicsRectItem):
            self._temp.setRect(QRectF(self._drag_start, p).normalized()); return
        if self.tool == "brush" and self._brush_mask is not None and e.buttons() & Qt.LeftButton:
            self._paint(p); return
        super().mouseMoveEvent(e)

    def mouseReleaseEvent(self, e) -> None:
        if self._panning is not None:
            self._panning = None
            self.setCursor(Qt.OpenHandCursor if self.tool == "pan" else Qt.CrossCursor); return
        if self._drag_start is not None and isinstance(self._temp, QGraphicsRectItem):
            r = self._temp.rect()
            self._cancel_temp()
            if r.width() >= 1 and r.height() >= 1:
                g = {"x0": r.left(), "y0": r.top(), "x1": r.right(), "y1": r.bottom()}
                if self.tool == "measure":
                    self.roiMeasured.emit(g)
                else:
                    self.shapeCreated.emit(self.tool, g, None)
            return
        if self.tool == "brush" and self._brush_mask is not None:
            ys, xs = np.nonzero(self._brush_mask)
            x0, y0 = int(xs.min()), int(ys.min())
            sub = self._brush_mask[y0:ys.max() + 1, x0:xs.max() + 1].copy()
            self._cancel_temp()
            self.shapeCreated.emit("brush", {"x0": x0, "y0": y0, "w": sub.shape[1], "h": sub.shape[0]}, sub)
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:
        if self.tool == "polygon":
            self._finish_polygon()
        else:
            super().mouseDoubleClickEvent(e)

    def _finish_polygon(self) -> None:
        pts = [(p.x(), p.y()) for p in self._poly]
        # a double click adds a duplicate last point
        if len(pts) >= 2 and pts[-1] == pts[-2]:
            pts.pop()
        self._cancel_temp()
        if len(pts) >= 3:
            self.shapeCreated.emit("polygon", {"points": pts}, None)

    def keyPressEvent(self, e) -> None:
        if e.key() in (Qt.Key_Return, Qt.Key_Enter) and self.tool == "polygon":
            self._finish_polygon(); return
        if e.key() == Qt.Key_Escape:
            self._cancel_temp(); return
        super().keyPressEvent(e)

    def scrollContentsBy(self, dx, dy) -> None:
        super().scrollContentsBy(dx, dy)
        self.viewChanged.emit()
