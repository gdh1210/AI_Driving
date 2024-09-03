from _2_myjoystickapp import MyJoystickApp
import threading
import cv2
import time
from PyQt5 import QtGui
import sys

class MyJoystickCamApp(MyJoystickApp):
    def __init__(self, cbJoyPos=None):
        super().__init__(cbJoyPos)

        self.camThread = threading.Thread(target=self.camMain)
        self.camThread.daemon = True
        self.camThread.start()

        self.app.aboutToQuit.connect(lambda: sys.exit(0))

        self.t_prev = time.time()
        self.cnt_frame = 0
        self.total_frame = 0
        self.cnt_time = 0

    def camMain(self):
        cap = cv2.VideoCapture(0)
        width, height = 640, 480
        self.video.resize(width, height)

        while True:
            # 영상 받기
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            # 영상 출력
            frame2 = cv2.resize(frame, (640, 480))

            h, w, c = frame2.shape
            qImg = QtGui.QImage(frame2.data, w, h, w * c, QtGui.QImage.Format_RGB888)
            pixmap = QtGui.QPixmap.fromImage(qImg.rgbSwapped())
            self.video.setPixmap(pixmap)

            self.collectData(frame)
            self.checkFrameRate()

    def collectData(self, frame):
        # 데이터 수집 로직을 여기에 구현
        pass

    def checkFrameRate(self):
        self.cnt_frame += 1
        t_now = time.time()
        if t_now - self.t_prev >= 1.0:
            self.t_prev = t_now
            self.total_frame += self.cnt_frame
            self.cnt_time += 1
            print("FPS: %d, avg: %.2f" % (self.cnt_frame, self.total_frame / self.cnt_time))
            self.cnt_frame = 0
