import cv2
from keras.models import load_model
import numpy as np
cap = cv2.VideoCapture(0)

keras_param = "./cnn.h5"
model = load_model(keras_param)

while True:
    # ret, frame = cap.read()

    # prd = model.predict(np.array([frame]))
    # prelabel = np.argmax(prd, axis=1)
    # if prelabel == 0:
    #     text = 'p'
    #     font = cv2.FONT_HERSHEY_PLAIN
                
    # 表示
    cv2.imshow("Show FLAME Image", frame) 
    # qを押したら終了。
    k = cv2.waitKey(1)
    if k == ord('q'):
        break
 
# 後始末。しなくても終わる。
cap.release()
cv2.destroyAllWindows()