# -*- coding: utf-8 -*-
import sys, os, json, math, pathlib
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget,
    QPushButton, QHBoxLayout, QMessageBox, QComboBox, QToolBar, QStatusBar, QSpinBox,
    QSlider, QLineEdit
)
from PySide6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QAction
from PySide6.QtCore import Qt, QPointF

from offside_core import compute_homography, offside_verdict, build_offside_lines, Calibration, project_points, line_image

def cv_to_qpix(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    qimg = QImage(img_rgb.data, w, h, ch * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)

class ImageCanvas(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.base = None
        self.overlay = []
        self.click_callback = None
        # zoom/pan
        self.scale = 1.0
        self.tx = 0.0
        self.ty = 0.0
        self._panning = False
        self._last_pos = None
        self.setMouseTracking(True)

    def set_image(self, img_bgr):
        if img_bgr is None:
            self.base = None
        else:
            self.base = img_bgr.copy()
            self.fit_to_widget()
        self.repaint()

    def fit_to_widget(self):
        if self.base is None: return
        h, w, _ = self.base.shape
        lbl_w = max(self.width(), 1); lbl_h = max(self.height(), 1)
        s = min(lbl_w / w, lbl_h / h)
        self.scale = s
        self.tx = (lbl_w - w*s) * 0.5
        self.ty = (lbl_h - h*s) * 0.5

    def wheelEvent(self, ev):
        if self.base is None: return
        delta = ev.angleDelta().y()
        if delta == 0: return
        factor = 1.25 if delta > 0 else 0.8
        old_scale = self.scale
        new_scale = max(0.05, min(20.0, old_scale * factor))
        pos = ev.position(); x = pos.x(); y = pos.y()
        ix = (x - self.tx) / old_scale; iy = (y - self.ty) / old_scale
        self.scale = new_scale
        self.tx = x - ix * new_scale
        self.ty = y - iy * new_scale
        self.repaint()

    def mousePressEvent(self, ev):
        if self.base is None: return
        if ev.button() == Qt.RightButton:
            self._panning = True; self._last_pos = ev.position()
        elif ev.button() == Qt.LeftButton and self.click_callback is not None:
            pt = self.widget_to_image(ev.position())
            if pt is not None: self.click_callback(pt)

    def mouseMoveEvent(self, ev):
        if self._panning and self._last_pos is not None:
            dp = ev.position() - self._last_pos
            self.tx += dp.x(); self.ty += dp.y()
            self._last_pos = ev.position()
            self.repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self._panning = False; self._last_pos = None

    def widget_to_image(self, qpf):
        if self.base is None: return None
        x = (qpf.x() - self.tx) / self.scale
        y = (qpf.y() - self.ty) / self.scale
        h, w, _ = self.base.shape
        if 0 <= x < w and 0 <= y < h:
            return (float(x), float(y))
        return None

    def paintEvent(self, ev):
        painter = QPainter(self)
        painter.fillRect(self.rect(), Qt.black)
        if self.base is None: return
        pix = cv_to_qpix(self.base)
        painter.save()
        painter.translate(self.tx, self.ty)
        painter.scale(self.scale, self.scale)
        painter.drawPixmap(0, 0, pix)
        for item in self.overlay:
            kind = item["kind"]; color = item.get("color", (0,255,0)); thick = item.get("thick", 3)
            painter.setPen(QPen(QColor(*color), thick))
            if kind == "points":
                for i,p in enumerate(item["pts"]):
                    qp = QPointF(p[0], p[1])
                    painter.drawEllipse(qp, 4, 4)
                    painter.drawText(qp + QPointF(6,-6), str(i+1))
            elif kind == "polyline":
                pts = item["pts"]
                for i in range(len(pts)-1):
                    p1 = QPointF(pts[i][0], pts[i][1]); p2 = QPointF(pts[i+1][0], pts[i+1][1])
                    painter.drawLine(p1, p2)
                if item.get("closed", False) and len(pts) >= 2:
                    p1 = QPointF(pts[-1][0], pts[-1][1]); p2 = QPointF(pts[0][0], pts[0][1])
                    painter.drawLine(p1, p2)
            elif kind == "line":
                p1 = item["p1"]; p2 = item["p2"]
                painter.drawLine(QPointF(p1[0], p1[1]), QPointF(p2[0], p2[1]))
            elif kind == "rect":
                x,y,w,h = item["rect"]
                painter.drawRect(x, y, w, h)
            elif kind == "text":
                pos = item["pos"]
                painter.drawText(QPointF(pos[0], pos[1]), item["text"])
        painter.restore()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Offside Line (VAR-like) – Prototipo foto única (V3c8)")
        self.canvas = ImageCanvas()
        self.status = QStatusBar(); self.setStatusBar(self.status)

        # Controls (create first)
        self.combo_template = QComboBox()
        self.combo_template.addItems([
            "Área de penal (40.32 x 16.5 m)",
            "Área chica (18.32 x 5.5 m)",
            "Rectángulo genérico (sin escala)",
            "Rectángulo con escala (profundidad conocida)"
        ])
        self.btn_start_cal = QPushButton("Empezar calibración")
        self.btn_undo_pt = QPushButton("Deshacer punto")
        self.btn_reset_cal = QPushButton("Reiniciar")

        self.btn_mark_def = QPushButton("Marcar 2.º último defensor")
        self.btn_mark_att = QPushButton("Marcar atacante")
        self.btn_mark_ball = QPushButton("Marcar pelota (opcional)")
        self.btn_compute = QPushButton("Calcular offside")

        # Guide widgets
        self.btn_add_guide = QPushButton("Añadir guía")
        self.btn_clear_guides = QPushButton("Borrar guías")
        self.combo_guide_axis = QComboBox()
        self.combo_guide_axis.addItems(["⊥ Offside (Auto)", "Eje X", "Eje Y", "Eje Z"])
        self.btn_define_z = QPushButton("Definir Z")

        # Zoom & thickness
        self.btn_zoom_in = QPushButton("Zoom +")
        self.btn_zoom_out = QPushButton("Zoom −")
        self.btn_fit = QPushButton("Ajustar")
        self.btn_1_1 = QPushButton("1:1")
        self.spin_thick = QSpinBox(); self.spin_thick.setRange(1, 20); self.spin_thick.setValue(4); self.spin_thick.setPrefix("Grosor: ")

        # Tolerance slider
        self.slider_tol = QSlider(Qt.Horizontal); self.slider_tol.setRange(0, 30); self.slider_tol.setValue(5)
        self.lbl_tol = QLabel("Tolerancia: 5 cm")

        # Axis override for offside
        self.combo_axis = QComboBox(); self.combo_axis.addItems(["Auto", "Forzar eje Y", "Forzar eje X"])

        # Scale input for "generic_scaled"
        self.edit_depth = QLineEdit(); self.edit_depth.setPlaceholderText("profundidad en metros (ej: 16.5)")

        # Hough & Auto-detect players
        self.btn_hough = QPushButton("Detectar líneas (Hough)")
        self.btn_auto_players = QPushButton("Autodetectar jugadores")

        # Layouts
        top1 = QHBoxLayout()
        top1.addWidget(self.combo_template); top1.addWidget(self.btn_start_cal)
        top1.addWidget(self.btn_undo_pt); top1.addWidget(self.btn_reset_cal)

        top2 = QHBoxLayout()
        top2.addWidget(self.btn_mark_def); top2.addWidget(self.btn_mark_att)
        top2.addWidget(self.btn_mark_ball); top2.addWidget(self.btn_compute)
        top2.addWidget(self.btn_add_guide); top2.addWidget(self.btn_clear_guides)
        top2.addWidget(self.combo_guide_axis); top2.addWidget(self.btn_define_z)

        top3 = QHBoxLayout()
        top3.addWidget(self.btn_zoom_in); top3.addWidget(self.btn_zoom_out)
        top3.addWidget(self.btn_fit); top3.addWidget(self.btn_1_1)
        top3.addWidget(self.spin_thick)

        top4 = QHBoxLayout()
        top4.addWidget(self.slider_tol); top4.addWidget(self.lbl_tol)
        top4.addWidget(QLabel("Eje:")); top4.addWidget(self.combo_axis)
        top4.addWidget(QLabel("Profundidad (m):")); top4.addWidget(self.edit_depth)
        top4.addWidget(self.btn_hough); top4.addWidget(self.btn_auto_players)

        v = QVBoxLayout()
        v.addLayout(top1); v.addLayout(top2); v.addLayout(top3); v.addLayout(top4); v.addWidget(self.canvas)

        center = QWidget(); center.setLayout(v); self.setCentralWidget(center)

        # State
        self.image = None
        self.calib_points = []
        self.calib: Calibration | None = None
        self.defender2 = None
        self.attacker = None
        self.ball = None
        self.tolerance_m = 0.05
        self.guides = []
        self.vp_z = None
        self._vpz_temp_pts = []
        self._vpz_capturing = False

        # Connections
        self.btn_start_cal.clicked.connect(self.start_calibration)
        self.btn_undo_pt.clicked.connect(self.undo_point)
        self.btn_reset_cal.clicked.connect(self.reset_all)
        self.btn_mark_def.clicked.connect(self.mark_defender2)
        self.btn_mark_att.clicked.connect(self.mark_attacker)
        self.btn_mark_ball.clicked.connect(self.mark_ball)
        self.btn_compute.clicked.connect(self.compute_offside)
        self.btn_zoom_in.clicked.connect(lambda: self.zoom_step(1.25))
        self.btn_zoom_out.clicked.connect(lambda: self.zoom_step(0.8))
        self.btn_fit.clicked.connect(self.fit_view)
        self.btn_1_1.clicked.connect(self.view_1_1)
        self.slider_tol.valueChanged.connect(self.on_tol_changed)
        self.btn_hough.clicked.connect(self.hough_detect_lines)
        self.btn_auto_players.clicked.connect(self.autodetect_players)
        self.btn_add_guide.clicked.connect(self.add_guide_click)
        self.btn_clear_guides.clicked.connect(self.clear_guides)
        self.btn_define_z.clicked.connect(self.define_z)

        # Menu
        tb = QToolBar()
        act_open = QAction("Abrir imagen", self); act_open.triggered.connect(self.open_image)
        act_export = QAction("Exportar anotación", self); act_export.triggered.connect(self.export_annotation)
        tb.addAction(act_open); tb.addAction(act_export); self.addToolBar(tb)

        self.update_overlay()

    # ---------- Utility UI methods ----------
    def on_tol_changed(self, val):
        self.tolerance_m = val / 100.0
        self.lbl_tol.setText(f"Tolerancia: {val} cm")

    def zoom_step(self, factor):
        if self.image is None: return
        w = self.canvas.width(); h = self.canvas.height()
        center = QPointF(w*0.5, h*0.5)
        old_s = self.canvas.scale
        new_s = max(0.05, min(20.0, old_s*factor))
        ix = (center.x() - self.canvas.tx) / old_s; iy = (center.y() - self.canvas.ty) / old_s
        self.canvas.scale = new_s
        self.canvas.tx = center.x() - ix * new_s
        self.canvas.ty = center.y() - iy * new_s
        self.canvas.repaint()

    def fit_view(self):
        self.canvas.fit_to_widget(); self.canvas.repaint()

    def view_1_1(self):
        if self.image is None: return
        h, w, _ = self.image.shape
        self.canvas.scale = 1.0
        self.canvas.tx = (self.canvas.width() - w) * 0.5
        self.canvas.ty = (self.canvas.height() - h) * 0.5
        self.canvas.repaint()

    # ---------- File operations ----------
    def open_image(self):
        file, _ = QFileDialog.getOpenFileName(self, "Abrir imagen", "", "Imágenes (*.png *.jpg *.jpeg *.bmp)")
        if not file: return
        img = cv2.imread(file, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Error", "No se pudo abrir la imagen.")
            return
        self.image = img
        self.canvas.set_image(self.image)
        self.status.showMessage("Imagen cargada. Elige plantilla y calibra.", 5000)
        self.reset_annotations(keep_image=True)

    def export_annotation(self):
        if self.image is None:
            QMessageBox.information(self, "Info", "No hay imagen para exportar.")
            return
        file, _ = QFileDialog.getSaveFileName(self, "Exportar anotación (PNG)", "offside_annot.png", "PNG (*.png)")
        if not file: return
        out = self.image.copy()
        thick = max(1, self.spin_thick.value())
        verdict=None; lines=None

        axis_override = self.get_axis_override()

        if self.calib is not None and self.defender2 is not None and self.attacker is not None:
            ball_pt = self.ball if self.ball is not None else None
            verdict = offside_verdict(self.calib, self.attacker, self.defender2, ball_pt,
                                      tolerance_m=self.tolerance_m, axis_override=axis_override)
            lines = build_offside_lines(self.calib, verdict, axis_override=axis_override)
            # draw lines (clipped)
            def iline(L, color):
                p1_raw = (float(L[0][0]), float(L[0][1])); p2_raw = (float(L[1][0]), float(L[1][1]))
                clipped = self._clip_line_to_image(p1_raw, p2_raw)
                if clipped is None: return
                p1, p2 = clipped
                p1i = (int(round(p1[0])), int(round(p1[1]))); p2i = (int(round(p2[0])), int(round(p2[1])))
                cv2.line(out, p1i, p2i, color, thick, cv2.LINE_AA)
            iline(lines["defender_line"], (0,0,255))
            iline(lines["attacker_line"], (0,255,255))
            cv2.putText(out, f"{verdict['verdict']}  Δ={verdict['delta']:.03f} (tol {verdict['tolerance']:.02f}) eje={verdict['axis']}",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2, cv2.LINE_AA)

        # export guides
        if self.guides:
            for (g1,g2) in self.guides:
                p1i = (int(round(g1[0])), int(round(g1[1]))); p2i = (int(round(g2[0])), int(round(g2[1])))
                cv2.line(out, p1i, p2i, (0,255,255), max(1, thick-1), cv2.LINE_AA)

        # Draw points
        def draw_circle(img, p, color, r=5, t=-1):
            cv2.circle(img, (int(round(p[0])), int(round(p[1]))), r, color, t, lineType=cv2.LINE_AA)
        def draw_text(img, p, text, color):
            cv2.putText(img, text, (int(p[0])+6, int(p[1])-6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        if self.calib_points:
            for i,p in enumerate(self.calib_points):
                draw_circle(out, p, (0,255,0), r=4, t=-1)
                draw_text(out, p, str(i+1), (0,200,0))
        if self.defender2 is not None:
            draw_circle(out, self.defender2, (255,180,0), r=6, t=-1)
            draw_text(out, self.defender2, "Defensa 2", (255,180,0))
        if self.attacker is not None:
            draw_circle(out, self.attacker, (0,220,255), r=6, t=-1)
            draw_text(out, self.attacker, "Atacante", (0,220,255))
        if self.ball is not None:
            draw_circle(out, self.ball, (255,255,255), r=6, t=-1)
            draw_text(out, self.ball, "Pelota", (255,255,255))

        import json
        cv2.imwrite(file, out)
        meta = {
            "calibration_points": self.calib_points,
            "defender2": self.defender2,
            "attacker": self.attacker,
            "ball": self.ball,
            "template": ["penal","goal","generic","generic_scaled"][self.combo_template.currentIndex()],
            "tolerance_m": self.tolerance_m,
            "axis_mode": None if self.calib is None else self.calib.axis_mode,
            "axis_override": axis_override,
            "line_thickness": thick,
            "depth_override_m": self.get_depth_override()
        }
        json_path = pathlib.Path(file).with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        QMessageBox.information(self, "Exportado", f"Se guardaron:\n{file}\n{json_path}")

    # ---------- State helpers ----------
    def reset_annotations(self, keep_image=False):
        self.calib_points = []; self.calib = None
        self.defender2 = None; self.attacker = None; self.ball = None
        self.guides = []
        self.update_overlay()

    def reset_all(self):
        self.image = None; self.canvas.set_image(None); self.reset_annotations()

    def get_depth_override(self):
        txt = self.edit_depth.text().strip()
        if not txt: return None
        try:
            val = float(txt)
            if val > 0: return val
        except: pass
        return None

    def get_axis_override(self):
        idx = self.combo_axis.currentIndex()
        if idx == 1: return "y"
        elif idx == 2: return "x"
        return None

    # ---------- Overlay ----------
    def _clip_line_to_image(self, p1, p2):
        if self.image is None: return None
        h, w, _ = self.image.shape
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        if abs(x2-x1) < 1e-6 and abs(y2-y1) < 1e-6: return None
        eps = 1e-6; W = w - 1; H = h - 1
        inter = []
        if abs(x2-x1) > eps:
            t = (0 - x1) / (x2 - x1); y = y1 + t*(y2 - y1)
            if -5 <= y <= H + 5: inter.append((0.0, float(y)))
            t = (W - x1) / (x2 - x1); y = y1 + t*(y2 - y1)
            if -5 <= y <= H + 5: inter.append((float(W), float(y)))
        if abs(y2-y1) > eps:
            t = (0 - y1) / (y2 - y1); x = x1 + t*(x2 - x1)
            if -5 <= x <= W + 5: inter.append((float(x), 0.0))
            t = (H - y1) / (y2 - y1); x = x1 + t*(x2 - x1)
            if -5 <= x <= W + 5: inter.append((float(x), float(H)))
        uniq = []
        for pt in inter:
            if not any((abs(pt[0]-q[0])<1e-3 and abs(pt[1]-q[1])<1e-3) for q in uniq):
                uniq.append(pt)
        if len(uniq) < 2: return None
        uniq = [(min(max(p[0], 0.0), W), min(max(p[1], 0.0), H)) for p in uniq]
        best = (uniq[0], uniq[1]); bestd = -1
        for i in range(len(uniq)):
            for j in range(i+1, len(uniq)):
                dx = uniq[i][0]-uniq[j][0]; dy = uniq[i][1]-uniq[j][1]
                d = dx*dx + dy*dy
                if d > bestd: bestd = d; best = (uniq[i], uniq[j])
        return best

    def update_overlay(self, verdict_dict=None, lines=None, extra_items=None):
        self.canvas.overlay = []
        if self.image is None:
            self.canvas.repaint(); return
        thick = self.spin_thick.value()

        # VP_Z marker (if defined)
        if self.vp_z is not None:
            vx, vy = self.vp_z
            cross = [((vx-8, vy), (vx+8, vy)), ((vx, vy-8), (vx, vy+8))]
            for (p1,p2) in cross:
                self.canvas.overlay.append({"kind":"line","p1":p1,"p2":p2,"color":(200,200,255),"thick":max(1,thick-1)})
            self.canvas.overlay.append({"kind":"text","pos":(vx+10, vy-10),"text":"VP_Z","color":(200,200,255),"thick":max(1,thick-1)})

        # Temp Z-capture: draw numbered points and extended lines (clipped)
        if self._vpz_temp_pts:
            pts = self._vpz_temp_pts
            # points + numbers
            self.canvas.overlay.append({"kind":"points","pts":pts,"color":(180,180,255),"thick":max(2,thick)})
            for i,p in enumerate(pts):
                self.canvas.overlay.append({"kind":"text","pos":(p[0]+6,p[1]-6),"text":str(i+1),"color":(180,180,255),"thick":max(1,thick-1)})
            # helper for extended lines
            def draw_ext(pa, pb):
                c = self._clip_line_to_image(pa, pb)
                if c is not None:
                    self.canvas.overlay.append({"kind":"line","p1":c[0],"p2":c[1],"color":(120,120,255),"thick":max(2,thick)})
            if len(pts) >= 2:
                draw_ext(pts[0], pts[1])
            if len(pts) >= 4:
                draw_ext(pts[2], pts[3])

        # Guides first (cyan)
        if self.guides:
            for (g1, g2) in self.guides:
                self.canvas.overlay.append({"kind":"line","p1":g1,"p2":g2,"color":(0,255,255),"thick":max(1,thick-1)})

        # Calib points/poly
        if self.calib_points:
            self.canvas.overlay.append({"kind":"points","pts":self.calib_points,"color":(0,255,0),"thick":thick})
            if len(self.calib_points)>=2:
                self.canvas.overlay.append({"kind":"polyline","pts":self.calib_points,"color":(0,200,0),"thick":max(1,thick-1),"closed":len(self.calib_points)==4})

        # Marks
        if self.defender2 is not None:
            self.canvas.overlay.append({"kind":"points","pts":[self.defender2],"color":(0,180,255),"thick":thick})
            self.canvas.overlay.append({"kind":"text","pos":self.defender2,"text":"Defensa 2","color":(0,180,255),"thick":thick})
        if self.attacker is not None:
            self.canvas.overlay.append({"kind":"points","pts":[self.attacker],"color":(255,220,0),"thick":thick})
            self.canvas.overlay.append({"kind":"text","pos":self.attacker,"text":"Atacante","color":(255,220,0),"thick":thick})
        if self.ball is not None:
            self.canvas.overlay.append({"kind":"points","pts":[self.ball],"color":(255,255,255),"thick":thick})
            self.canvas.overlay.append({"kind":"text","pos":self.ball,"text":"Pelota","color":(255,255,255),"thick":thick})

        # Lines and verdict
        if verdict_dict is not None and self.calib is not None and lines is not None:
            p1_raw = tuple(map(float, lines["defender_line"][0])); p2_raw = tuple(map(float, lines["defender_line"][1]))
            c = self._clip_line_to_image(p1_raw, p2_raw)
            if c is not None:
                p1, p2 = c
                self.canvas.overlay.append({"kind":"line","p1":p1,"p2":p2,"color":(255,0,0),"thick":thick})
            p1a_raw = tuple(map(float, lines["attacker_line"][0])); p2a_raw = tuple(map(float, lines["attacker_line"][1]))
            ca = self._clip_line_to_image(p1a_raw, p2a_raw)
            if ca is not None:
                p1a, p2a = ca
                self.canvas.overlay.append({"kind":"line","p1":p1a,"p2":p2a,"color":(255,220,0),"thick":thick})
            msg = f"{verdict_dict['verdict']} | Δ={verdict_dict['delta']:.03f} (tol {verdict_dict['tolerance']:.02f}) eje={verdict_dict['axis']}"
            self.canvas.overlay.append({"kind":"text","pos":(20.0,20.0),"text":msg,"color":(255,0,0),"thick":thick})
            self.status.showMessage(msg, 10000)

        if extra_items:
            self.canvas.overlay.extend(extra_items)

        self.canvas.repaint()

    # ---------- Calibration ----------
    def start_calibration(self):
        if self.image is None:
            QMessageBox.information(self, "Info", "Primero abre una imagen."); return
        self.calib_points = []; self.calib = None; self.guides = []

        def on_click(pt):
            self.calib_points.append(pt)
            self.update_overlay()
            if len(self.calib_points) == 4:
                template_idx = self.combo_template.currentIndex()
                if template_idx == 0: template = "penal"
                elif template_idx == 1: template = "goal"
                elif template_idx == 2: template = "generic"
                else: template = "generic_scaled"
                depth = self.get_depth_override() if template=="generic_scaled" else None
                axis_override = self.get_axis_override()
                try:
                    pts = np.array(self.calib_points, dtype=np.float32)
                    self.calib = compute_homography(pts, template, depth_override=depth, axis_override=axis_override)
                    self.status.showMessage(f"Calibración OK. Eje: {self.calib.axis_mode}. Ahora marca jugadores o usa autodetección.", 6000)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Falló la calibración: {e}")
                self.canvas.click_callback = None

        self.canvas.click_callback = on_click
        QMessageBox.information(self, "Calibración",
            "Haz 4 clics sobre el rectángulo de referencia **en este orden**:\n"
            "1) esquina lejos-izquierda\n2) lejos-derecha\n3) cerca-derecha\n4) cerca-izquierda\n"
            "(La arista 3→4 se toma como línea de gol). Para 'Rectángulo con escala', asegúrate de que esas aristas sean paralelas a la línea de gol y coloca la profundidad real en metros.")
        self.update_overlay()

    def undo_point(self):
        if self.canvas.click_callback is None:
            if self.ball is not None: self.ball = None
            elif self.attacker is not None: self.attacker = None
            elif self.defender2 is not None: self.defender2 = None
            elif self.calib_points: self.calib_points.pop()
            self.update_overlay(); return
        if self.calib_points:
            self.calib_points.pop(); self.update_overlay()

    # ---------- Mark points ----------
    def mark_defender2(self):
        if self.image is None or self.calib is None:
            QMessageBox.information(self, "Info", "Necesitas calibrar primero."); return
        self.canvas.click_callback = lambda pt: self._set_point("def", pt)

    def mark_attacker(self):
        if self.image is None or self.calib is None:
            QMessageBox.information(self, "Info", "Necesitas calibrar primero."); return
        self.canvas.click_callback = lambda pt: self._set_point("att", pt)

    def mark_ball(self):
        if self.image is None or self.calib is None:
            QMessageBox.information(self, "Info", "Necesitas calibrar primero."); return
        self.canvas.click_callback = lambda pt: self._set_point("ball", pt)

    def _set_point(self, which, pt):
        if which == "def": self.defender2 = pt
        elif which == "att": self.attacker = pt
        else: self.ball = pt
        self.canvas.click_callback = None; self.update_overlay()

    # ---------- Compute ----------
    def compute_offside(self):
        if self.image is None or self.calib is None or self.defender2 is None or self.attacker is None:
            QMessageBox.information(self, "Info", "Faltan datos: calibración + defensor + atacante."); return
        ball_pt = self.ball if self.ball is not None else None
        axis_override = self.get_axis_override()
        verdict = offside_verdict(self.calib, self.attacker, self.defender2, ball_pt,
                                  tolerance_m=self.tolerance_m, axis_override=axis_override)
        lines = build_offside_lines(self.calib, verdict, axis_override=axis_override)
        self.update_overlay(verdict_dict=verdict, lines=lines)

    # ---------- Hough line detection ----------
    def hough_detect_lines(self):
        if self.image is None:
            QMessageBox.information(self, "Info", "Abrí una imagen primero."); return
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3, L2gradient=True)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=120, minLineLength=80, maxLineGap=20)
        items = []
        if lines is None:
            QMessageBox.information(self, "Hough", "No se detectaron líneas fuertes. Ajusta la imagen o calibra manualmente."); 
            self.update_overlay(extra_items=items); return

        angles = []
        segs = []
        for l in lines[:,0,:]:
            x1,y1,x2,y2 = l
            dx = x2-x1; dy = y2-y1
            ang = math.degrees(math.atan2(dy, dx))
            angles.append(ang); segs.append((x1,y1,x2,y2))
        angles = np.array(angles)
        famA = []; famB = []
        median = np.median(angles)
        for (x1,y1,x2,y2),ang in zip(segs, angles):
            if abs(ang - median) < 20:
                famA.append((x1,y1,x2,y2))
            else:
                famB.append((x1,y1,x2,y2))

        def draw_family(fam, color):
            for (x1,y1,x2,y2) in fam[:30]:
                items.append({"kind":"line","p1":(x1,y1),"p2":(x2,y2),"color":color,"thick":2})

        draw_family(famA, (0,255,0))
        draw_family(famB, (0,200,255))

        import itertools
        rect_pts = None
        def top2_longest(fam):
            fam_sorted = sorted(fam, key=lambda s: (s[2]-s[0])**2+(s[3]-s[1])**2, reverse=True)
            return fam_sorted[:2]

        if len(famA)>=2 and len(famB)>=2:
            for a1,a2 in itertools.combinations(top2_longest(famA), 2):
                for b1,b2 in itertools.combinations(top2_longest(famB), 2):
                    def intersect(s1, s2):
                        x1,y1,x2,y2 = s1; x3,y3,x4,y4 = s2
                        den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
                        if den == 0: return None
                        px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
                        py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
                        return (float(px), float(py))
                    Pll = intersect(a1, b1); Plr = intersect(a1, b2); Prl = intersect(a2, b1); Prr = intersect(a2, b2)
                    if None in (Pll,Plr,Prl,Prr): continue
                    pts = np.array([Pll, Plr, Prr, Prl], dtype=np.float32)
                    idx = np.argsort(pts[:,1])
                    top = pts[idx[:2]][np.argsort(pts[idx[:2],0])]
                    bot = pts[idx[2:]][np.argsort(pts[idx[2:],0])]
                    ordered = np.array([top[0], top[1], bot[1], bot[0]], dtype=np.float32)
                    rect_pts = ordered; break
                if rect_pts is not None: break

        if rect_pts is not None:
            self.calib_points = [tuple(map(float,p)) for p in rect_pts]
            items.append({"kind":"points","pts":self.calib_points,"color":(255,0,0),"thick":4})
            self.status.showMessage("Hough sugirió un rectángulo. Revisá las esquinas y, si sirven, presioná 'Empezar calibración' para fijarlo.", 8000)
        else:
            self.status.showMessage("Hough detectó líneas pero no armó un rectángulo robusto. Usá la sugerencia visual y calibrá manualmente.", 8000)

        self.update_overlay(extra_items=items)

    # ---------- Autodetect players ----------
    def autodetect_players(self):
        if self.image is None:
            QMessageBox.information(self, "Info", "Abrí una imagen primero."); return
        if self.calib is None:
            QMessageBox.information(self, "Info", "Calibrá antes de autodetectar jugadores."); return
        hog = cv2.HOGDescriptor(); hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        img = self.image
        scale_factor = 1.0
        h, w, _ = img.shape
        if max(h,w) > 1600:
            scale_factor = 1600.0 / max(h,w)
            img_small = cv2.resize(img, (int(w*scale_factor), int(h*scale_factor)))
        else:
            img_small = img.copy()
        rects, weights = hog.detectMultiScale(img_small, winStride=(8,8), padding=(8,8), scale=1.05)
        rects = [(int(x/scale_factor), int(y/scale_factor), int(wd/scale_factor), int(ht/scale_factor)) for (x,y,wd,ht) in rects]

        if len(rects) == 0:
            QMessageBox.information(self, "Jugadores", "No se detectaron personas de forma fiable. Ajustá la imagen o marcá manualmente."); 
            return

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        feats = []; centers_img = []; axis_vals = []
        axis = self.get_axis_override() or (self.calib.axis_mode if self.calib is not None else "y")
        for (x,y,wid,hei) in rects:
            x1,y1,x2,y2 = max(0,x),max(0,y),min(w-1,x+wid),min(h-1,y+hei)
            cx = (x1+x2)/2.0; cy = (y1+y2)/2.0
            centers_img.append((cx, cy))
            patch = hsv[int(y1+0.2*hei):int(y1+0.6*hei), int(x1+0.2*wid):int(x1+0.8*wid)]
            mean = (0,0,0) if patch.size==0 else tuple(np.mean(patch.reshape(-1,3), axis=0).tolist())
            feats.append(mean)
            wpt = project_points(self.calib.H, np.array([(cx,cy)], dtype=np.float32))[0]
            axis_vals.append(float(wpt[1] if axis=="y" else wpt[0]))

        feats = np.array(feats, dtype=np.float32)
        axis_vals = np.array(axis_vals, dtype=np.float32)
        centers_img = np.array(centers_img, dtype=np.float32)

        if len(feats) >= 2:
            Z = feats.copy()
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
            ret, labels, centers = cv2.kmeans(Z, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            labels = labels.flatten()
        else:
            labels = np.zeros((len(feats),), dtype=np.int32)

        def cluster_axis_stats(k):
            vals = axis_vals[labels==k]
            if len(vals)==0: return 1e9
            return np.percentile(vals, 25)
        def_k_def = 0 if cluster_axis_stats(0) < cluster_axis_stats(1) else 1

        idx_def = np.where(labels==def_k_def)[0]
        if len(idx_def) >= 2:
            order = idx_def[np.argsort(axis_vals[idx_def])]
            d2_idx = order[1]
            self.defender2 = tuple(map(float, centers_img[d2_idx]))
        else:
            d2_idx = int(np.argmin(axis_vals))
            self.defender2 = tuple(map(float, centers_img[d2_idx]))

        k_att = 1 - def_k_def
        idx_att = np.where(labels==k_att)[0]
        if len(idx_att) >= 1:
            a_idx = idx_att[np.argmin(axis_vals[idx_att])]
            self.attacker = tuple(map(float, centers_img[a_idx]))
        else:
            a_idx = int(np.argmax(axis_vals))
            self.attacker = tuple(map(float, centers_img[a_idx]))

        items = []
        for i,(x,y,wid,hei) in enumerate(rects):
            color = (0,180,255) if labels[i]==def_k_def else (255,220,0)
            items.append({"kind":"rect","rect":(x,y,wid,hei),"color":color,"thick":2})
        items.append({"kind":"points","pts":[self.defender2], "color":(0,180,255), "thick":5})
        items.append({"kind":"text","pos":self.defender2, "text":"Defensa 2 (auto)", "color":(0,180,255), "thick":2})
        items.append({"kind":"points","pts":[self.attacker], "color":(255,220,0), "thick":5})
        items.append({"kind":"text","pos":self.attacker, "text":"Atacante (auto)", "color":(255,220,0), "thick":2})

        self.update_overlay(extra_items=items)
        self.status.showMessage("Autodetección lista. Revisá las cajas y corrige manualmente si hace falta.", 7000)

    # ---------- Guides & Z axis ----------
    def add_guide_click(self):
        if self.image is None or self.calib is None:
            QMessageBox.information(self, "Guía", "Calibrá primero."); return
        self.status.showMessage("Click en el hombro/punto para trazar guía.", 5000)
        self.canvas.click_callback = lambda pt: self._add_guide_from_point(pt)

    def clear_guides(self):
        self.guides = []
        self.update_overlay()

    def define_z(self):
        """Toggle captura de VP_Z con 4 clics: (1,2) vertical A, (3,4) vertical B."""
        if self.image is None:
            QMessageBox.information(self, "Eje Z", "Abrí una imagen primero."); return
        if self._vpz_capturing:
            self._vpz_capturing = False
            self._vpz_temp_pts = []
            self.canvas.click_callback = None
            self.status.showMessage("Captura de Z cancelada.", 4000)
            self.update_overlay()
            return
        # Start capture
        self._vpz_temp_pts = []
        self._vpz_capturing = True
        self.canvas.click_callback = self._on_click_define_z
        QMessageBox.information(self, "Definir Z",
            "Hacé 4 clics:\n"
            "  1 y 2: sobre una misma vertical real (palo del arco, columna, etc.)\n"
            "  3 y 4: sobre otra vertical real distinta.\n\n"
            "Se calculará el punto de fuga vertical (VP_Z).\n"
            "Tip: podés volver a apretar 'Definir Z' para cancelar.")
        self.status.showMessage("Definir Z: clic 1–2 en vertical A, 3–4 en vertical B.", 8000)
        self.update_overlay()

    def _on_click_define_z(self, pt):
        self._vpz_temp_pts.append(pt)
        n = len(self._vpz_temp_pts)
        if n == 1:
            self.status.showMessage("Punto 1 registrado. Marcá el punto 2 sobre la MISMA vertical.", 4000)
        elif n == 2:
            self.status.showMessage("Vertical A lista. Ahora puntos 3 y 4 sobre OTRA vertical.", 5000)
        elif n == 3:
            self.status.showMessage("Punto 3 registrado. Falta el punto 4.", 4000)

        if n >= 4:
            (x1,y1),(x2,y2),(x3,y3),(x4,y4) = self._vpz_temp_pts[:4]
            den = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
            if abs(den) < 1e-6:
                self.vp_z = None
                QMessageBox.information(self, "Eje Z", "Las rectas parecen paralelas en la imagen. Usaré vertical de imagen como aproximación.")
            else:
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / den
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / den
                self.vp_z = (float(px), float(py))
                QMessageBox.information(self, "Eje Z", f"VP_Z definido en ({self.vp_z[0]:.1f}, {self.vp_z[1]:.1f}).")
            self.canvas.click_callback = None
            self._vpz_temp_pts = []
            self._vpz_capturing = False
        self.update_overlay()

    def _add_guide_from_point(self, pt_img):
        axis_opt = self.combo_guide_axis.currentIndex() if hasattr(self, "combo_guide_axis") else 0
        if axis_opt == 1:
            guide_axis = "x"
        elif axis_opt == 2:
            guide_axis = "y"
        elif axis_opt == 3:
            guide_axis = "z"
        else:
            offside_axis = self.get_axis_override() or (self.calib.axis_mode if self.calib is not None else "y")
            guide_axis = "x" if offside_axis == "y" else "y"

        if guide_axis in ("x","y"):
            wpt = project_points(self.calib.H, np.array([pt_img], dtype=np.float32))[0]
            wx, wy = float(wpt[0]), float(wpt[1])
            ref = wx if guide_axis=="x" else wy
            L = line_image(self.calib.Hinv, self.calib.width_m, self.calib.depth_m, ref, guide_axis)
            p1_raw = (float(L[0][0]), float(L[0][1]))
            p2_raw = (float(L[1][0]), float(L[1][1]))
        else:
            if self.vp_z is not None:
                p1_raw = (float(pt_img[0]), float(pt_img[1]))
                p2_raw = (float(self.vp_z[0]), float(self.vp_z[1]))
            else:
                p1_raw = (float(pt_img[0]), -1e6)
                p2_raw = (float(pt_img[0]),  1e6)

        clipped = self._clip_line_to_image(p1_raw, p2_raw)
        self.canvas.click_callback = None
        if clipped is not None:
            self.guides.append(clipped)
            self.update_overlay()
        else:
            QMessageBox.information(self, "Guía", "La guía quedó fuera del frame. Probá definir Z o recalibrar.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow(); w.resize(1400, 900); w.show()
    sys.exit(app.exec())
