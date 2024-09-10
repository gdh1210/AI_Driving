<div align="center">
<img src="https://capsule-render.vercel.app/api?type=Rounded&color=0:649173,100:DBD5A4&height=250&&section=header&text=AI%20Driving%20project&fontColor=FFFFFF&stroke=202020&fontSize=90">
</div>

<div align="center">
<img src="https://img.shields.io/badge/python-3776AB?style=for-the-badge&logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-Arduino-00979D?style=for-the-badge&logo=Arduino&logoColor=white">
</div>

# 자율 주행 프로젝트

<p align="center">
<img src="https://github.com/user-attachments/assets/09507add-d095-4659-83d7-c5c6b459c12e">
</p>

## 목차
* [06-19(수) 아두이노 차량 조립 및 모터 핀제어 코딩](#0619수)
* [06-20(목) 차량제어를 위한 조이스틱 UI 코딩](#0620목)
* [06-21(금) 데이터 전달을 위한 카메라 연결](#0621금)
* [06-24(월) 모의주행을 통한 데이터 수집 및 학습 데이터 변환](#0624월)
* [06-25(화) 학습된 모델을 이용한 자율주행](#0625화)

---
### 06.19(수)


사용된 아두이노 차량 사진 프로젝트를 시행하기에 앞서 차량 조립을 진행하였다.

<div align="center">
<img src="https://github.com/user-attachments/assets/25e8f66b-24ce-4167-ad53-f2cd2da8d6be">
</div>


조립이후 arduino_running_test.zip 안에있는 아두이노 소스 자료를 활용하여 연결된 각종 센서와 모터의 구동이 정상적으로 작동하는지 확인해 보고.

<div align="center">
<img src="https://github.com/user-attachments/assets/cd6e7f48-aad4-4f04-ae67-29131436b4f7" width="500" height="200">
</div>

확인 결과 결함이 있거나 동작하지 않는 센서는 없었다.<br>
이후 모터 동작을 설정하기 위해 아두이노코딩을 진행했다.

# AI_Driving_arduino.ino

```c++
#include "DCmotor.h"
#include "rccar.h"
//초음파 지정자
const int trigPin = 8;
const int echoPin = 13;
long duration;
int distance;
//모터 지정자
DCmotor dcF_L(2, 3);
DCmotor dcF_R(4, 5);
DCmotor dcB_L(7, 6);
DCmotor dcB_R(19, 10);
RCCar car(dcF_L, dcF_R, dcB_L, dcB_R);
void setup() {
  pinMode(trigPin, OUTPUT); 
  pinMode(echoPin, INPUT); 
  Serial.begin(9600);}
void loop() {
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  duration = pulseIn(echoPin, HIGH);
  distance= duration*0.034/2;
  Serial.println(distance);
  if (distance<20) {
    car.backward();
    delay(2000);
    car.stop(); }
  if(Serial.available()>0) {
    int cmd=Serial.read();
    switch(cmd) {
      case 'w': car.forward(); break;
      case 's': car.backward(); break;
      case 'a': car.left(); break;
      case 'd': car.right(); break;
      case 'x': car.stop(); break;
      default: break; } } }
```

초음파 감지 센서로 거리를 측정하고 20 이하의 거리가 감지되면 뒤로 2초간 이동후 정지 하는 코드이다 또한 핸드폰과 블루투스 페어링을 통해 w 전진, s 후진, a 좌회전, d 우회전, x 정지를 시행 가능하다.<br>
우선은 이렇게 짜여진 코드를 이용해 모의주행을 실시하여 학습시킬 데이터를 수집할 것이다.

### 06.20(목)
# _1_myjoystick.py

```py
class MyJoystick(QWidget):
    def __init__(self, parent=None, cbJoyPos=None, app=None):
        super(MyJoystick, self).__init__(parent)
        self.setMinimumSize(200, 200)
        self.movingOffset = QPointF(0, 0)  # 조이스틱 위치
        self.grabCenter = False  # 조이스틱을 잡았느냐?
        self.__maxDistance = 50  # 조이스틱 범위
        self.cbJoyPos = cbJoyPos  # 사용자 정의 콜백 함수
        self.app = app  # 앱 객체가 넘어옴, 뒤에서 정의함
        
        self.timer = QTimer(self)
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.timeout)
        self.timer.start()

    def paintEvent(self, event):
        painter = QPainter(self)
        bounds = QRectF(
            -self.__maxDistance, 
            -self.__maxDistance, 
            self.__maxDistance * 2, 
            self.__maxDistance * 2
        ).translated(self._center())
        painter.drawEllipse(bounds)
        painter.setBrush(Qt.black)
        painter.drawEllipse(self._centerEllipse())

    def _center(self):
        return QPointF(self.width() / 2, self.height() / 2)

    def _centerEllipse(self):
        if self.grabCenter:
            return QRectF(-20, -20, 40, 40).translated(self.movingOffset)
        return QRectF(-20, -20, 40, 40).translated(self._center())

    def mousePressEvent(self, event):
        self.grabCenter = self._centerEllipse().contains(event.pos())
        self.movingOffset = self._boundJoystick(event.pos())
        self.update()
        return super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.grabCenter:
            self.movingOffset = self._boundJoystick(event.pos())
            self.update()
        if self.cbJoyPos is not None:
            self.cbJoyPos(self._joystickPosition(), self.app)

    def _boundJoystick(self, point):
        limitLine = QLineF(self._center(), point)
        if limitLine.length() > self.__maxDistance:
            limitLine.setLength(self.__maxDistance)
        return limitLine.p2()

    def mouseReleaseEvent(self, event):
        self.grabCenter = False
        self.movingOffset = QPointF(0, 0)
        self.update()
        if self.cbJoyPos is not None:
            self.cbJoyPos(self._joystickPosition(), self.app)

    def timeout(self):
        sender = self.sender()
        if id(sender) == id(self.timer):
            if self.cbJoyPos is not None:
                self.cbJoyPos(self._joystickPosition(), self.app)

    def _joystickPosition(self):
        if not self.grabCenter:
            return (0, 0)
        normVector = QLineF(self._center(), self.movingOffset)
        currentDistance = normVector.length()
        angle = normVector.angle()
        distance = min(currentDistance / self.__maxDistance, 1.0)
        posX = math.cos(angle * math.pi / 180) * distance
        posY = math.sin(angle * math.pi / 180) * distance
        return (posX, posY)
```
우선 조이스틱 부분을 작성하고 실행해 보았다. 가동범위가 원안으로 제한되어있는지 앞,뒤 양옆 이동간에 좌표값을 제대로 송신 받는지 확인 했다.

<div align="center">
<img src="https://github.com/user-attachments/assets/881a3daf-52d2-4ad3-ba88-a9041c2b16b2" width="300" height="300">
<img src="https://github.com/user-attachments/assets/2e98304c-1585-4adb-a140-d089b1d66caf" width="300" height="300">
<img src="https://github.com/user-attachments/assets/9a9a4f91-7630-4793-b04f-4489dc6561df" width="100" height="150">
</div>

# _2_myjoystickapp.py
```py
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
```

배경색이 너무 진한거 같아 변경하였고 창의 크기 또한 좀더 크게 늘려주었다.<br>
위쪽에 위젯을 넣어 카메라를 연결하려고 블루투스 무선 통신을 위해 외부 프로그램인 Camo Studio를 사용했다.

<div align="center">
<img src="https://github.com/user-attachments/assets/8aa03a29-f86e-4933-af04-787b4d349715" width="600" height="600">
</div>

프로그램 설치랑 핸드폰과 연결만 확인하고 나머지는 내일 작업을 진행하겠다.

---

### 06.21(금)

# _3_MyJoystickCamApp.py
```py
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
```

카메라 연결을 cv2 모델의 camMain 함수에서 처리하고 이전의 조이스틱 파일을 불러와 결합시켜 동작을 한다.

<div align="center">
<img src="https://github.com/user-attachments/assets/f43de05c-ce71-439d-840e-f1f20c599252" width="600" height="600">
<img src="https://github.com/user-attachments/assets/ad3635fe-156b-48a9-8faa-bcd5e31e02ba" width="300" height="300">
</div>

10분 정도 동작결과 카메라 프레임이 30대에서 안정화 되어있고 영상을 받고 출력하는데에 딜레이 발생이 없었다.<br>
이제 연결된 카메라를 통해 받은 영상을 사진데이터로 수집하여 학습용으로 만드는 과정을 진행할 것이다.

# _4_MyDataCollectionApp.py
```py
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
```

시범주행 과정에서 마우스로 조이스틱을 움직이는데 조금 불편해서 키보드를 입력받아 움직이도록 코드를 수정했다.

# _4-1_MyDataCollectionApp.py

커맨드를 입력받는 부분을 keyboard.is_pressed 를 통해 입력을 감지하고 명령을 시행한다.

```py
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
```
데이터 수집을위한 모의 주행이후 폴더가 생성된 것을 확인할 수 있었고

<div align="center">
<img src="https://github.com/user-attachments/assets/070e0b4b-bb5f-466c-b77b-6de55a56540a" width="600" height="100">
</div>

_0_forward 폴더(좌)내부와 _2_left 폴더(우)내부의 수집된 데이터의 모습 또한 확인이 가능하다.

<div align="center">
<img src="https://github.com/user-attachments/assets/f73a92e5-14b5-446d-b3c1-7136c723f519" width="300" height="300">
<img src="https://github.com/user-attachments/assets/26ed32cc-81c5-49e8-aa58-3246ddc2d111" width="300" height="300">
</div>

# _5_data_labelling.py
```py
dataDir = 'data'  # 데이터 저장 디렉터리

print(os.getcwd())  # 현재 디렉터리 어딘지 확인
os.chdir(dataDir)  # 디렉터리 이동
roadDirs = os.listdir()  # 현재 디렉터리 확인
print(roadDirs)

f_csv = open('0_road_labels.csv', 'w', newline='')
wr = csv.writer(f_csv)
wr.writerow(["file", "label", "labelNames"])

roadDirs = [road for road in roadDirs if os.path.isdir(road)]
print(roadDirs)

for num, roadDir in enumerate(roadDirs):
    roadFiles = os.listdir(roadDir)
    for roadFile in roadFiles:
        wr.writerow([os.path.join(roadDir, roadFile), num, roadDir])

f_csv.flush()
f_csv.close()
```
학습데이터로 사용하기 위해 각 폴더의 사진에 라벨링 작업을 해주었다.

<div align="center">
<img src="https://github.com/user-attachments/assets/7728c1a1-6ea6-4613-a5db-5ef9338ecf58" width="600" height="100">
</div>

# _6_cnn_training.py
```py
dirname = "data"

def image_to_tensor(img_path):
    img = keras_image.load_img(
        os.path.join(dirname, img_path),
        target_size=(120, 160)
    )
    x = keras_image.img_to_array(img)
    return np.expand_dims(x, axis=0)

def data_to_tensor(img_paths):
    list_of_tensors = [
        image_to_tensor(img_path) for img_path in tqdm(img_paths)
    ]
    return np.vstack(list_of_tensors)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Load the data
data = pd.read_csv(os.path.join(dirname, "0_road_labels.csv"))
data = data.sample(frac=1)

files = data['file']
targets = data['label'].values

tensors = data_to_tensor(files)

print(data.tail())
print(tensors.shape)
print(targets.shape)
```
이미지 데이터를 처리하여 학습에 사용할 수 있는 형태로 변환하고, 데이터를 랜덤하게 섞어서 모델이 편향되지 않도록 만들었다.

<div align="center">
<img src="https://github.com/user-attachments/assets/8a1c2eb2-f46c-476e-aa5e-ab62d404312d" width="1000" height="300">
</div>

---
### 06.24(월)

# _6-1_cnn_reading.py

데이터를 로드한 후 출력하는 작업을 진행하여 검수작업을 진행한다.

```py
# Name list
names = ['_0_forward', '_1_right', '_2_left', '_3_stop']

def display_images(img_path, ax):
    img = cv2.imread(os.path.join(dirname, img_path))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for i, ax in enumerate(axes.flat):
    if i < len(files):
        ax.set_title(names[targets[i]], color='blue')
        display_images(files[i], ax)
    else:
        ax.axis('off')  # 이미지가 없으면 서브플롯을 비활성화

plt.tight_layout()
plt.show()
```

데이터의 랜덤화 결과 9개의 사진을 테스트 해보았을때 우측 중앙과 하단이 전진을 해야하는 지 우회전을 해야하는지 애매하고<br>
모의 주행시 수집한 데이터의 대부분이 전진이다 보니 데이터의 랜덤화를 진행했음에도 전진이 대부분인건 어쩔 수 없는것 같다.
> 전진 데이터 2,085 장, 우회전 데이터 685 장, 좌회전 데이터 492장

<div align="center">
<img src="https://github.com/user-attachments/assets/28829a5a-a73d-4a43-a721-1839a4949531" width="600" height="600">
</div> 

# _7_tensorflow_training.py
정규화된 데이터를 바탕으로 학습을 진행 하였다.

```py
from cnn_training import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt

tensors = tensors.astype('float32') / 255
targets = to_categorical(targets, 4)

x_train, x_test, y_train, y_test = train_test_split(
    tensors,
    targets,
    test_size=0.2,
    random_state=1
)

n = int(len(x_test) / 2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_valid.shape, y_valid.shape)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), padding="same",
                           activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50,
                    validation_data=(x_valid, y_valid))

loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("model.h5")

```
다층 신경망 구조는 ANN 구조를 참고해 만들었고 처음 학습을 진행할때 빠른 학습을 위해 Flatten 을 기준으로 위아래 층을 하나씩 빼고 학습 횟수도 5번정도로 진행을 했었다 <br>
대략 적중률이 70% ~ 80% 정도를 보이고 높아보이지만 사실 자율주행을 하면 30프레임(0.5초)에 한번 씩 판단을 내려 주행을 해야하는데<br>
30초만 주행해도 60번의 판단을 내려야 하며 75%의 정답률로 계산시 15번이 오판된 명령이 내려진다 이는 경로에서 이탈할 가능성이 크다.

<div align="center">
<img src="https://github.com/user-attachments/assets/de480a43-d259-4a39-b73c-09e6ce4d8b73" width="600" height="300">
</div>

학습횟수를 50회로 늘리고 학습간의 신경층을 조금더 늘렸다 결과적으로 적중률을 95% 이상으로 끌어올리는데 성공했다.

<div align="center">
<img src="https://github.com/user-attachments/assets/7d2e8f1f-1de3-4195-aa39-be3f48bd855c" width="600" height="300">
</div>


# _7-1_tensorflow_reading.py

학습이 끝난 모델을 검수 하기위해 그림을 하나하나 살펴보며 검수를 하기에는 너무 많은 작업이 필요하므로 대체 시각자료인 그래프로 변경하였다.  

```py
from tensorflow_training import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model1 = load_model('model.h5')

# Model predictions for the testing dataset
y_test_predict = model1.predict(x_test)
print(y_test_predict.shape, y_test_predict[0])
y_test_predict = np.argmax(y_test_predict, axis=1)
print(y_test_predict.shape, y_test_predict[0])

# Name list
names = ['_0_forward', '_1_right', '_2_left', '_3_stop']

# Display true labels and predictions
fig = plt.figure(figsize=(18, 18))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = y_test_predict[idx]
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(names[pred_idx], names[true_idx]),
                 color=("#4876ff" if pred_idx == true_idx else "darkred"))
plt.show()

```

그래프 확인결과 학습효과가 눈에 보이는 수준으로 좋다가 점점 0에 수렴하는 수평이 되는데 더 학습량을 늘리거나 신경망을 두텁게 한다해도 유의미한 정확도의 증가는 없을듯 하다.

<div align="center">
<img src="https://github.com/user-attachments/assets/2d816a96-d80f-4003-a011-08c7fb62a3d8" width="400" height="400">
</div>

---
### 06.25(화)

# _8_AI_driving.py

학습이 끝난 model.h5를 가지고 자율 주행을 하기위한 코드작성을 진행하였다.

```py
# OpenCV로부터 영상 받기
cap = cv2.VideoCapture(0)

# 시리얼 통신 설정
mot_serial = serial.Serial('COM9', 9600)

# 모델 로드
model = load_model('model.h5')

t_now = time.time()
t_prev = time.time()
cnt_frame = 0

# 클래스/라벨 이름 설정
names = ['_0_forward', '_1_right', '_2_left', '_3_stop']

# 메시지 큐 설정
HOW_MANY_MESSAGES = 10
mq = queue.Queue(HOW_MANY_MESSAGES)

# CNN 메인 함수 정의
def cnn_main(args):
    while True:
        frame = mq.get()
        frame_thrown = 0
        while not mq.empty():
            frame = mq.get()
            frame_thrown += 1
            
        print(f'{frame_thrown} frame thrown')

        # 전처리: 이미지 스케일링
        image = frame
        image = frame / 255.0

        # 이미지 텐서로 변환
        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)

        # 예측
        y_predict = model.predict(image_tensor)
        y_predict = np.argmax(y_predict, axis=1)
        print(names[y_predict[0]], y_predict[0])

        # 예측 결과에 따른 명령 전송
        cmd = y_predict[0].item()
        if cmd == 0:
            command = 'w'
        elif cmd == 1:
            command = 'd'
        elif cmd == 2:
            command = 'a'
        else:
            command = 'x'
        mot_serial.write(command.encode())

# CNN 메인 쓰레드 실행
cnnThread = threading.Thread(target=cnn_main, args=(0,))
cnnThread.daemon = True
cnnThread.start()

# 메인 루프: 영상 받기 및 처리
while True:
    # 영상 받기
    ret, frame = cap.read()

    # 영상 출력
    frame = cv2.resize(frame, (640, 480))
    cv2.imshow('frame', frame)

    # 프레임 크기 축소 후 메시지 큐에 추가
    frame = cv2.resize(frame, (160, 120))
    mq.put(frame)

    # 키 입력 확인 (ESC 키 입력 시 종료)
    key = cv2.waitKey(1)
    if key == 27:
        break

    # 프레임 수 카운트
    cnt_frame += 1
    t_now = time.time()

    # 1초마다 프레임 수 출력
    if t_now - t_prev >= 1.0:
        t_prev = t_now
        print("frame count : %f" % cnt_frame)
        cnt_frame = 0

# 종료 처리
mot_serial.write('s'.encode())
mot_serial.close()
cap.release()
cv2.destroyAllWindows()
sys.exit(0)
```

주행결과 자율주행이 되는 모습을 확인할 수 있다.

<p align="center">
<img src="https://github.com/user-attachments/assets/18be9e08-471c-433c-91ca-a31047a138ab">
</p>

라인을 두 줄로 변경하고 재학습 하여 진행 결과 잘 가다가 경로를 이탈해 버리는 것을 확인하였다.

<p align="center">
<img src="https://github.com/user-attachments/assets/e8a619d6-7eaf-49e3-89d1-45f7730c738f">
</p>

시행착오 및 문제점

:x:문제점
* 자율주행 결과 새로 학습을 진행했음에도 2줄로 되어있는 라인은 잘 따라가지 못헀다. 아마 한줄로 진행할경우 화면 중앙으로 라인이 따라가도록 움직이고 라인이 2개일경우 2라인 사이로 움직이려고 해서 서로 기준이 다르다 보니 학습시 혼동이 오는것 같다.
* 아두이노 보드의 처리량 한계로 추축되는 문제 인해 앞에 장애물이 끼어 들거나 무언가가 막고 있다면 초음파 센서로 거리를 감지하고 가까우면 2초동안 후진하는 코드를 넣었는데 자율주행과 동시에 처리하면서 센서까지 활용할 경우 후진이후 자율주행이 동작하지 않았다.
* 여러번 돌려보니 항상 같은 경로에서는 잘 움직이지만 다른 환경적인 요인에의한 변수에 대해서는 취약하여 주변환경이 조금만 바뀌어도 잘 못 동작하는 경우가 많았다.

:o:해결법
* 이전 학습데이터를 삭제하고 기준이 명확한 모의주행 데이터만 남겨서 재학습 하면 정상적으로 주행이 가능 할 것이다. 아니면 학습모델을 분리하여 2개의 라인과 1개의 라인 운행시 학습데이터를 바꾸는 식으로 해도 가능하다.
* 아무래도 데이터 학습시 후진에대한건 따로 지정해주지 않았기 때문에 후진이후 정지하거나 후진이후 경로를 이탈해버려서 정지하는 걸로 판단이 되어서 후진대신 2초간 정지로 바꾸었다.
* 학습 데이터의 부족, ANN의 구조적 한계, 주변 환경에 대한 부정확한 판단이 원인이기 때문에 보완하기 위한 행동으로는 신경망의 개조 혹은 엔비디아에서 내놓은 오픈소스를 활용하여 변경을 하던가 아두이노 대신 처리량이 커서 자체적으로 신경망을 탑재할 수 있는 보드를 구해서 차량쪽에서 바로바로 학습을 진행하면 환경변화 요인에 대응할 수 있을것 같다.
 
# 마치며
* 5일간 진행한 프로젝트에서 아두이노 핀제어 부분과 조이스틱 제작 부분에서는 교수님이 주신 소스와 인터넷에 있는 소스를 활용해 제작을 진행했기에 서로 코드를 연결할때 약간의 수정만 진행하면 큰 문제없이 작동했기 때문에 크게 어려울게 없었지만 모의주행 데이터를 수집하는 과정에서 정규화된 데이터를 얻어야 하는데 사람이 직접 하다보니 직선구간에서 차량이 틀어져서 똑바로 가기위해 방향을 살짝 조정해주면 직선구간에 좌,우 회전이 학습되어 오류데이터가 넘어가거나 학습신경망의 학습과정에서 약2시간 정도의 시간이 소요되는데 이때 툴은 학습중이라 컴퓨터의 많은 자원을 끌어 쓰다보니 다른 작업이 진행이 어려웠다.
* 자율주행을 조건을 바꿔서 2줄로 진행할때 한줄로 학습한 모델은 조금 삐걱거리는 모습을 보여도 잘 가는데 2줄의 라인으로 학습된 데이터는 기준이 문제인건지 직선코스는 잘가는데 곡선코스에서 이탈하는 경우가 자주 보였다. 정확한 요인은 파악하지 못했지만 환경적인 요소가 변화하면 학습데이터가 고정되어있기 때문에 잘 인식을 못하는 것 같았다 이를 해결하기 위해 여러 자료들을 찾아 봤는데 차량의 내부에 학습을 진행할 수 있게 신경망을 탑재하면 환경 변화요소 또한 학습하여 유연한 대처가 가능하다고 하는데 차량의 보드가 아두이노이다 보니 처리량에 한계가 있어서 해결하지는 못했다.

### 참조
아두이노 핀제어 코드 활용 - https://carrotweb.tistory.com/132
학습 신경망에 대한 간단한 지식 및 비교 - https://ebbnflow.tistory.com/119
NVIDIA 자율주행 관련 자료 - https://www.nvidia.com/en-us/self-driving-cars/
NVIDIA 신경망 관련 자료 - https://developer.nvidia.com/blog/deep-learning-self-driving-cars/
트렌스포머 신경망 구조 이해(적용X) - https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04
파이썬으로 JSON 외부데이터 읽고 쓰기(코드 가져와서 CSV로 대체) - https://rfriend.tistory.com/474
외부프로그램 camo studio - https://reincubate.com/ko/camo/



