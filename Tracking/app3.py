import cv2
import numpy as np

cap = cv2.VideoCapture(0) #読み込む動画のパス
fps = cap.get(cv2.CAP_PROP_FPS)


# https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ja


while True:
    # 1フレームずつ取得する。
    success, frame = cap.read()
      # 画像を読込
    src = frame
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    # 2値化
    retval, bw = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # 輪郭を抽出
    #   contours : [領域][Point No][0][x=0, y=1]
    #   cv2.CHAIN_APPROX_NONE: 中間点も保持する
    #   cv2.CHAIN_APPROX_SIMPLE: 中間点は保持しない
    contours, hierarchy = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # 矩形検出された数（デフォルトで0を指定）
    detect_count = 0

    # 各輪郭に対する処理
    for i in range(0, len(contours)):

        # 輪郭の領域を計算
        area = cv2.contourArea(contours[i])

        # ノイズ（小さすぎる領域）と全体の輪郭（大きすぎる領域）を除外
        if area < 1e2 or 1e5 < area:
            continue

        # 外接矩形
        if len(contours[i]) > 0:
            rect = contours[i]
            x, y, w, h = cv2.boundingRect(rect)
            cv2.rectangle(src, (x, y), (x + w, y + h), (0, 255, 0), 2)

            detect_count = detect_count + 1

    cv2.imshow('output', src)

    key = cv2.waitKey(30)
    if key == 27:
        break

# 後始末。しなくても終わる。
cap.release()
cv2.destroyAllWindows()