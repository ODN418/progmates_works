import cv2
from keras.models import load_model
import numpy as np
cap = cv2.VideoCapture(0) #読み込む動画のパス
fps = cap.get(cv2.CAP_PROP_FPS)

# keras_param = "./cnn.h5"
# model = load_model(keras_param)
model = load_model("cnn.h5")
# https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ja
#avg = None

while True:
    # 1フレームずつ取得する。
    success, img = cap.read()

    # ↓任意の処理をここに書く↓ 
    # 白黒画像に
    # フィルターをセット
    # ↑任意の処理をここに書く↑

    cv2.imshow("Image", img)

    key = cv2.waitKey(30)
    if key == 27:
        break

# 後始末。しなくても終わる。
cap.release()
cv2.destroyAllWindows()