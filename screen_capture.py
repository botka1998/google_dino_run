import numpy as np
import cv2
from mss import mss
from PIL import Image, ImageDraw


from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import sys

top = 100
left = 400
width = 400
height = 300


class MainApp(QWidget):
    def __init__(self):
        QWidget.__init__(self, None, Qt.WindowStaysOnTopHint)
        self.draw = True
        self.video_size = QSize(width, height)
        self.bounding_box = {"top": top, "left": left, "width": 500, "height": height}
        self.setup_ui()
        self.setup_capture()
        self.t_dino = cv2.cvtColor(cv2.imread("dino.png"), cv2.COLOR_BGR2GRAY)
        self.bird_templates = [
            cv2.cvtColor(cv2.imread(f"bird_{i}.png"), cv2.COLOR_BGR2GRAY)
            for i in range(4)
        ]
        self.cactii_templates = [
            cv2.cvtColor(cv2.imread(f"cactus_{i}.png"), cv2.COLOR_BGR2GRAY)
            for i in range(7)
        ]

    def handle_x_slider_value_change(self, x):
        self.bounding_box["left"] = int(1920 * x / 100)

    def handle_y_slider_value_change(self, y):
        self.bounding_box["top"] = int(1080 * y / 100)

    def setup_ui(self):
        """Initialize widgets."""
        self.main_layout = QVBoxLayout()
        self.image_label = QLabel()
        self.image_label.setFixedSize(self.video_size)
        self.main_layout.addWidget(self.image_label)
        self.add_sliders()
        self.quit_button = QPushButton("Quit")
        self.quit_button.clicked.connect(self.close)

        self.main_layout.addWidget(self.quit_button)
        self.b_draw_checkbox = QCheckBox("Should draw:", self)
        self.b_draw_checkbox.stateChanged.connect(self.set_draw)
        self.setLayout(self.main_layout)

    def set_draw(self, value):
        self.draw = value

    def add_sliders(self):
        self.x_pos_slider = QSlider(Qt.Horizontal, self)
        self.x_pos_slider.setValue(int(left * 100 / 1920))
        self.x_pos_slider.valueChanged.connect(self.handle_x_slider_value_change)
        self.y_pos_slider = QSlider(Qt.Horizontal, self)
        self.y_pos_slider.valueChanged.connect(self.handle_y_slider_value_change)
        self.y_pos_slider.setValue(int(top * 100 / 1080))
        self.slider_label = QLabel("Adjust captrure window position", self)

        self.main_layout.addWidget(self.slider_label)
        self.main_layout.addWidget(self.x_pos_slider)
        self.main_layout.addWidget(self.y_pos_slider)

    def setup_capture(self):
        """Initialize camera."""
        self.sct = mss()
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        delay = 10 if self.draw else 10
        self.timer.start(delay)

    def get_frame(self):
        self.current_frame = np.array(self.sct.grab(self.bounding_box))
        self.current_grey_frame = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2GRAY)
        self.current_pil_frame = Image.fromarray(self.current_frame)
        self.drawer = ImageDraw.Draw(self.current_pil_frame)

        return self.current_frame

    def detect_dino(self):
        dino = self.detect_object(self.t_dino)
        if dino is None:
            return
        if self.draw:
            self.draw_bounding_box(
                dino,
            )

    def detect_bird(self):
        detected_birds = map(self.detect_object, self.bird_templates)
        if self.draw:
            for bird in detected_birds:
                if bird is None:
                    continue
                self.draw_bounding_box(bird, color=(0, 0, 255))

    def detect_cactii(self):
        detected_cactii = map(
            self.detect_object,
            self.cactii_templates,
        )

        if self.draw:
            for cactus in detected_cactii:
                if cactus is None:
                    continue
                self.draw_bounding_box(cactus, color=(0, 255, 0), thicc=1)

    def draw_bounding_box(self, detected_object, color=(255, 255, 255), thicc=2):
        w, h = detected_object["w"], detected_object["h"]
        x, y = detected_object["point"]

        shape = [(x, y), (w + x, h + y)]

        self.drawer.rectangle(shape, fill=None, outline="red")

    def detect_object(self, template, threshold=0.85):
        w, h = template.shape[::-1]
        res = cv2.matchTemplate(self.current_grey_frame, template, cv2.TM_CCOEFF_NORMED)
        loc = np.where(res >= threshold)
        if loc[0].size == 0:
            return None
        return {"point": list(zip(*loc[::-1]))[0], "w": w, "h": h}

    def display_video_stream(self):
        self.get_frame()
        self.detect_dino()
        self.detect_bird()
        self.detect_cactii()
        self.current_frame = np.array(self.current_pil_frame)
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
    win.setGeometry(1400, 200, win.width(), win.height())
    sys.exit(app.exec_())
