import numpy as np
import cv2
from mss import mss
from PIL import Image


from PySide6.QtCore import *
from PySide6.QtGui import *
from PySide6.QtWidgets import *

import sys


class MainApp(QWidget):
    def __init__(self):
        QWidget.__init__(self)
        self.video_size = QSize(600, 600)
        self.setup_ui()
        self.setup_capture()

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
        self.bounding_box = {"top": 300, "left": 600, "width": 600, "height": 600}
        self.sct = mss()
        self.timer = QTimer()
        self.timer.timeout.connect(self.display_video_stream)
        self.timer.start(30)

    def display_video_stream(self):
        sct_img = np.array(self.sct.grab(self.bounding_box))
        grey_screen = cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)
        template = cv2.imread("dino.png")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        w, h = template.shape[::-1]

        res = cv2.matchTemplate(grey_screen, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        for pt in zip(*loc[::-1]):  # Switch columns and rows
            # print(pt)
            cv2.rectangle(sct_img, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

        image = QImage(
            sct_img,
            sct_img.shape[1],
            sct_img.shape[0],
            sct_img.strides[0],
            QImage.Format_RGB32,
        )
        self.image_label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainApp()
    win.show()
    sys.exit(app.exec_())


# bounding_box = {"top": 300, "left": 600, "width": 600, "height": 400}

# sct = mss()

# while True:
#     sct_img = sct.grab(bounding_box)
#     cv2.imshow("screen", np.array(sct_img))

#     if (cv2.waitKey(1) & 0xFF) == ord("q"):
#         cv2.destroyAllWindows()
#         break
