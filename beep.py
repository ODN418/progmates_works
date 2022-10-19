import numpy as np
import cv2
import winsound
#顔認識カスケード
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
#Videoキャプチャ
cap = cv2.VideoCapture(0)

#ビープ音1000Hz 吹鳴時間100ms
freq = 2000
delay = 100

#繰り返し
while(True):
    #最新画像を取得
    ret, img = cap.read()
    #グレースケール化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #顔を認識
    faces = face_cascade.detectMultiScale(gray)
    #認識箇所を青枠で囲む
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #顔が検出されたらビープ音
        winsound.Beep(freq, delay)
        
    #最新画像を表示
    cv2.imshow('img', img)
    #imshowの絵を描画するコマンドがwaitkeyに入っているらしく、waitkeyによる1msキー待ち
    #実際には1ms後に抜けるのでこのままで随時画面更新になる。
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()