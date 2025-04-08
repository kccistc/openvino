import cv2
import numpy as np

CAM_W = 1920
CAM_H = 1080

WINDOW_W = 1920
WINDOW_H = 1080

WINDOW_POS_X = 500
WINDOW_POS_Y = 0

WINDOW_TITLE = "Result"

cap = cv2.VideoCapture(0)

# 웹캠이 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("웹캠을 열 수 없습니다.")
    exit

# Create Window
cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
#cv2.setWindowProperty("WINDOW_TITLE", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Resize Window Size
cv2.resizeWindow(WINDOW_TITLE, WINDOW_W, WINDOW_H)

# Move Window
cv2.moveWindow(WINDOW_TITLE, WINDOW_POS_X, WINDOW_POS_Y)

# 카메라 인풋 포멧을 MJPG 로 설정
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
if not cap.set(cv2.CAP_PROP_FOURCC, fourcc):
    print("Failed to set FOURCC to MJPG.")

# 카메라 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
print(f"Resolution = {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)},FPS = {cap.get(cv2.CAP_PROP_FPS)}")

while True:
    ret, frame = cap.read()  # 웹캠으로부터 프레임을 읽어옴
    if not ret:
        print("이미지 캡쳐에 실패했습니다.")
        break

    # 결과를 화면에 표시
    cv2.imshow(WINDOW_TITLE, frame)
    if cv2.waitKey(1) == 27:  # ESC 키를 누르면 종료
        break

cap.release()
cv2.destroyAllWindows()
