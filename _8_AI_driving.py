import serial
import time
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
import threading
import queue
import sys

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
