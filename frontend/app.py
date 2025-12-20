from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import requests
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QHBoxLayout,
    QVBoxLayout,
    QMessageBox,
)

BACKEND_URL = "http://127.0.0.1:8000/api/simpsonify"



def frame_to_pixmap(frame_bgr) -> QPixmap:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class Main(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simpsonify - Camera → Result")
        self.resize(1200, 650)

        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)

        self.last_frame = None
        self.out_dir = Path("outputs_gui")
        self.out_dir.mkdir(exist_ok=True)

        # UI
        root = QHBoxLayout(self)

        # Left panel
        left = QVBoxLayout()
        root.addLayout(left, 1)

        self.lbl_cam = QLabel("Camera preview")
        self.lbl_cam.setAlignment(Qt.AlignCenter)
        self.lbl_cam.setMinimumSize(520, 420)
        self.lbl_cam.setStyleSheet("border: 1px solid #999;")
        left.addWidget(self.lbl_cam)

        self.btn_start = QPushButton("Start Camera")
        self.btn_capture = QPushButton("Capture + Convert")
        self.btn_stop = QPushButton("Stop Camera")
        self.btn_capture.setEnabled(False)
        self.btn_stop.setEnabled(False)

        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_capture.clicked.connect(self.capture_and_convert)

        left.addWidget(self.btn_start)
        left.addWidget(self.btn_capture)
        left.addWidget(self.btn_stop)

        self.lbl_status = QLabel("Status: Ready.")
        left.addWidget(self.lbl_status)
        left.addStretch(1)

        # Right panel
        right = QVBoxLayout()
        root.addLayout(right, 1)

        self.lbl_result = QLabel("Result will appear here")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        self.lbl_result.setMinimumSize(520, 520)
        self.lbl_result.setStyleSheet("border: 1px solid #999;")
        right.addWidget(self.lbl_result)

        self.lbl_hint = QLabel("Backend must be running on http://127.0.0.1:8000")
        right.addWidget(self.lbl_hint)
        right.addStretch(1)

    def start_camera(self):
        if self.cap is not None:
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.cap = None
            QMessageBox.critical(self, "Camera Error", "Could not open camera (index 0).")
            return

        self.timer.start(30)
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_capture.setEnabled(True)
        self.lbl_status.setText("Status: Camera running.")

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.last_frame = None
        self.lbl_cam.setText("Camera preview")

        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_capture.setEnabled(False)
        self.lbl_status.setText("Status: Camera stopped.")

    def _tick(self):
        if self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok:
            return
        self.last_frame = frame

        pix = frame_to_pixmap(frame).scaled(
            self.lbl_cam.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.lbl_cam.setPixmap(pix)

    def capture_and_convert(self):
        if self.last_frame is None:
            return

        self.lbl_status.setText("Status: Capturing…")
        ts = int(time.time())
        input_path = self.out_dir / f"capture_{ts}.png"
        cv2.imwrite(str(input_path), self.last_frame)

        self.lbl_status.setText("Status: Sending to backend… (this can take a while)")
        try:
            with open(input_path, "rb") as f:
                files = {"image": ("capture.png", f, "image/png")}
                data = {
                    "prompt": "simpsons character, thick black outline, very simple face, flat solid colors, yellow skin, 2D cel animation, cartoon TV show style, no shading, simple shapes",
                    "strength": "0.75",
                    "guidance": "6.0",
                    "steps": "25",
                    "seed": "0",
                }
                r = requests.post(BACKEND_URL, files=files, data=data, timeout=1800)

            if r.status_code != 200:
                raise RuntimeError(f"Backend error {r.status_code}: {r.text}")

            out_path = self.out_dir / f"result_{ts}.png"
            out_path.write_bytes(r.content)

            pix = QPixmap(str(out_path)).scaled(
                self.lbl_result.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_result.setPixmap(pix)
            self.lbl_status.setText(f"Status: Done. Saved {out_path.name}")

        except Exception as e:
            QMessageBox.critical(self, "Convert Error", str(e))
            self.lbl_status.setText("Status: Error.")

    def closeEvent(self, event):
        try:
            self.stop_camera()
        finally:
            super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec())
