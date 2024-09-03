#myjoystickapp.py
from PyQt5.QtWidgets import *
from PyQt5.Qt import Qt
from _1_myjoystick import MyJoystick

class MyJoystickApp:
    def __init__(self, cbJoyPos=None):
        self.app = QApplication([])
        self.mw = QMainWindow()
        self.mw.setWindowTitle('RC Car Joystick')
        self.mw.setGeometry(100, 100, 300, 200)
        
        cw = QWidget()
        cw.setStyleSheet("background-color:gray;")
        self.mw.setCentralWidget(cw)
        
        ml = QGridLayout()
        cw.setLayout(ml)
        
        self.video = QLabel('Video here~')
        ml.addWidget(self.video, 0, 0)
        
        self.joystick = MyJoystick(cbJoyPos=cbJoyPos, app=self)
        ml.addWidget(self.joystick, 1, 0)
        
        self.speed = 33
        speedbar = QSlider(Qt.Horizontal)
        speedbar.setRange(0, 100)
        speedbar.setTickInterval(10)
        speedbar.setTickPosition(QSlider.TicksBelow)
        speedbar.setValue(self.speed)
        speedbar.valueChanged.connect(self.setSpeed)
        ml.addWidget(speedbar, 2, 0)
        
        self.app.aboutToQuit.connect(self.app.deleteLater)
        
        self.mw.show()

    def setSpeed(self, speed):
        self.speed = speed

    def getSpeed(self):
        return self.speed

    def run(self):
        self.app.exec_()
