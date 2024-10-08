# OpenCV examples

## 00. Overview
 - OpenCV site: https://opencv.org/
 - Docs: https://docs.opencv.org/4.10.0/
 - Tutorials: https://docs.opencv.org/master/d9/df8/tutorial_root.html

## 01. Image
 - 01_image.py: 이미지 파일 Read / Preview / Save
 - 02_image.py: 색공간 제어
 - 03_image.py: Resize & Crop
 - `Bonus Challenges:`
   1. 오른쪽으로 90도 회전시키기 - 참고: cv2.rotate
   2. 좌우 반전, 상하 반전 시키기 - 참고: cv2.flip

## 02. Video
 - 01_video.py: 비디오 파일 Real / Play
 - `Bonus Challenges:`
   1. 비디오 파일이 끝까지 재생 되면 반복 재생시키기
   2. 비디오 파일 재생속도 키 입력으로 실시간 변경 적용하기 ('1': 느리게, '2': 빠르게) 

## 03. Camera
 - 01_camera.py: webcam 으로부터 input 을 받아 출력
 - 02_camera.py: Line / Rectangle / Text 출력
 - 03_camera.py: trackbar
 - `Bonus Challenges:`
   1. 'c' 키 입력 시 현재 webcam frame 을 capture 해서 이미지 파일로 저장하기
   2. 저장된 이미지 파일들이 overwrite 되지 않도록 날짜/시간을 저장되는 파일 이름으로 사용하기
   3. 2개의 webcam 을 사용해 각각 preview 를 보여주도록 구현하기

## Homework #1
 - Base code: 02_camera.py
 - Text 문구 / Font / 색상 / 크기 / 굵기 / 출력위치 등 모든 값을 변경해 보자.
 - 동그라미 그리는 OpenCV 함수를 찾아서 적용해 보자.
 - 마우스 왼쪽 버튼을 click 하면 해당 위치에 동그라미가 그려지도록 코드를 추가해 보자. (Reference: cv2.EVENT_LBUTTONDOWN)
 - 결과물 예시: homework_opencv_1.mp4

## Homework #2 (Optional)
 - Base code: 03_camera.py
 - Trackbar 를 control 해서 TEXT 의 굵기가 변하는 것을 확인해 보자.
 - Trackbar 를 추가해서 font size 를 변경하는 것을 구현해 보자.
 - R/G/B Trackbar 를 추가해서 글자의 font color 를 runtime 에 변경해 보자.
 - 결과물 예시: homework_opencv_2.mp4
