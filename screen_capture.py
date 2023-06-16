import numpy as np
import cv2
from mss import mss
from PIL import Image


from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import sys

width = 500
height = 500


class MainApp(QWidget):
    def __init__(self):
        QWidget.__init__(self, None, Qt.WindowStaysOnTopHint)
        self.video_size = QSize(width, height)
        self.setup_ui()
        self.setup_capture()
        self.t_dino = cv2.cvtColor(cv2.imread("dino.png"), cv2.COLOR_BGR2GRAY)
        self.t_cactus_1 = cv2.cvtColor(cv2.imread("cactus_1.png"), cv2.COLOR_BGR2GRAY)
        self.bird_templates = [
            cv2.cvtColor(cv2.imread(f"bird_{i}.png"), cv2.COLOR_BGR2GRAY)
            for i in range(3)
        ]

    def handle_x_slider_value_change(self, x):
        self.bounding_box["left"] = int(1920 * x / 100)

    def handle_y_slider_value_change(self, y):
        self.bounding_box["top"] = int(1080 * y / 100)

    def setup_ui(self):
        """Initialize widgets."""
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)

        self.x_pos_slider = QSlider(Qt.Horizontal, self)
        self.x_pos_slider.valueChanged.connect(self.handle_x_slider_value_change)
        self.y_pos_slider = QSlider(Qt.Horizontal, self)
        self.y_pos_slider.valueChanged.connect(self.handle_y_slider_value_change)

        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.image_label)
        self.main_layout.addWidget(self.quit_button)
        self.main_layout.addWidget(self.x_pos_slider)
        self.main_layout.addWidget(self.y_pos_slider)

        self.setLayout(self.main_layout)

    def setup_capture(self):
        """Initialize camera."""
        self.bounding_box = {"top": 300, "left": 600, "width": 500, "height": height}
        self.sct = mss()
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def get_frame(self):
        self.current_frame = np.array(self.sct.grab(self.bounding_box))
        self.current_grey_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)

        return self.current_frame

    def detect_dino(self):
        dinos = self.detect_object(self.t_dino)
        w, h = dinos["w"], dinos["h"]
        for pt in dinos["points"]:
            cv2.rectangle(
                self.current_frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2
            )

    def detect_bird(self):
        detected_birds = map(
            lambda bird_template: self.detect_object(bird_template), self.bird_templates
        )
        for bird in detected_birds:
            w, h = bird["w"], bird["h"]
            for pt in bird["points"]:
                cv2.rectangle(
                    self.current_frame, pt, (pt[0] + w, pt[1] + h), (255, 0, 0), 2
                )

    def detect_cactii(self):
        cactus = self.detect_object(self.t_cactus_1)
        w, h = cactus["w"], cactus["h"]
        for pt in cactus["points"]:
            cv2.rectangle(
                self.current_frame, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2
            )

    def detect_object(self, template, threshold=0.8):
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(self.current_grey_frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        return {"points": zip(*loc[::-1]), "w": w, "h": h}

    def display_video_stream(self):
        self.get_frame()
        self.detect_dino()
        self.detect_bird()
        self.detect_cactii()
        image = QImage(
            self.current_frame,
            self.current_frame.shape[1],
            self.current_frame.shape[0],
            self.current_frame.strides[0],
            QImage.Format_RGB32,
        )
        self.image_label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())
