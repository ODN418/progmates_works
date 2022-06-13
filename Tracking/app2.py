import cv2
import sys, os
from keras.models import load_model
import numpy as np

cap = cv2.VideoCapture(0) #読み込む動画のパス
fps = cap.get(cv2.CAP_PROP_FPS)


keras_param = "C:/Users/cs18017/Documents/progmates/progmates_works/Tracking/cnn.h5"
model = load_model(keras_param)
# https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ja


while True:
    # 1フレームずつ取得する。
    success, frame = cap.read()
    if not success:
            break
    # ↓任意の処理をここに書く↓ 
    frame = cv2.resize(frame, (64, 64))
    prd = model.predict(np.array([frame]))
    prelabel = np.argmax(prd, axis=1)
    if prelabel == 0:
        print(">>> 犬")
    elif prelabel == 1:
        print(">>> 猫")
    # ↑任意の処理をここに書く↑

    cv2.imshow("img", frame)

    key = cv2.waitKey(30)
    if key == 27:
        break

# 後始末。しなくても終わる。
cap.release()
cv2.destroyAllWindows()