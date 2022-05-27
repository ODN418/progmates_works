import cv2
import mediapipe as mp
 
# カメラ解像度の設定
wCam, hCam = 640, 480
 
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
 
# カメラが1台のみ接続されている場合は0を指定。
# 2台以上接続されている場合は、カメラIDを指定。
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)
pTime = 0
 
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
    while cap.isOpened():
        success, image = cap.read()
 
        if success==False:
            continue
 
        # パオ―マンス向上のため、オプションで参照渡しの画像を書き込み不可にする。
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
 
        # 顔検出の時間を計測開始
        results = face_detection.process(image)
 
        # 顔検出アノテーションをイメージ上に描画する。
        # Draw the face detection annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if results.detections:
            for detection in results.detections:
                mp_drawing.draw_detection(image, detection)
 
        cv2.imshow("Img",image)
 
        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break
 
# 後始末。しなくても終わる。
cap.release()
cv2.destroyAllWindows()