from _3_myjoystickcamapp import MyJoystickCamApp
import time
import os
import cv2
import keyboard

class MyDataCollectionApp(MyJoystickCamApp):
    def __init__(self, cbJoyPos=None):
        super().__init__(cbJoyPos)
        
        self.rl = 0
        self.cnt_frame_total = 0
        
        self.datadir = 'data_%f' % (time.time())
        os.mkdir(self.datadir)
        os.mkdir(os.path.join(self.datadir, '_0_forward'))
        os.mkdir(os.path.join(self.datadir, '_1_right'))
        os.mkdir(os.path.join(self.datadir, '_2_left'))
        os.mkdir(os.path.join(self.datadir, '_3_stop'))
        
        self.names = ['_0_forward', '_1_right', '_2_left', '_3_stop']

    def setRL(self, rl):
        self.rl = rl

    def collectData(self, frame):
        rl = self.rl
        collect_data = (rl & 4) >> 2
        if collect_data == 1:
            rl = rl & 3
            frame = cv2.resize(frame, (160, 120))
            road_file = '%f.png' % (time.time())
            cv2.imwrite(os.path.join(os.path.join(self.datadir, self.names[rl]), road_file), frame)
            self.cnt_frame_total += 1

    def checkFrameRate(self):
        self.cnt_frame += 1
        t_now = time.time()
        if t_now - self.t_prev >= 1.0:
            self.t_prev = t_now
            print("frame count : %d" % self.cnt_frame, "total frame : %d" % self.cnt_frame_total)
            self.cnt_frame = 0

if __name__ == '__main__':
    import serial
    
    mot_serial = serial.Serial('COM9', 9600)
    
    def cbJoyPos(joystickPosition, app=None):
        posX, posY = joystickPosition
        
        speed = 0
        if app is not None:
            speed = app.getSpeed()
        
        command = 'x'
        collect_data = 1
        
        if posY < -0.5:
            command = 's'  # backward
            collect_data = 0
        elif posY > 0.15:
            if -0.15 <= posX <= 0.15:
                command = 'w'  # forward
            elif posX < -0.15:
                command = 'a'  # left
            elif posX > 0.15:
                command = 'd'  # right
            else:
                command = 'x'  # stop driving
                collect_data = 0
        
        if keyboard.is_pressed('w'):
            command = 'w'
        elif keyboard.is_pressed('a'):
            command = 'a'
        elif keyboard.is_pressed('d'):
            command = 'd'
        elif keyboard.is_pressed('s'):
            command = 's'
            
        if command == 'w':
            right, left = 0, 0  # forward
        elif command == 'a':
            right, left = 1, 0  # left
        elif command == 'd':
            right, left = 0, 1  # right
        elif command == 'x':
            right, left = 1, 1  # stop
        else:
            right, left = 1, 1  # stop
        
        
        rl = collect_data << 2 | right << 1 | left
        myDataCollectionApp.setRL(rl)
        
        mot_serial.write(command.encode())
    
    myDataCollectionApp = MyDataCollectionApp(cbJoyPos=cbJoyPos)
    myDataCollectionApp.run()

