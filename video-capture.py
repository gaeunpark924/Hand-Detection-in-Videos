import cv2
import numpy as np
from math import acos, pi, sqrt

video_file = 'test.mov'       
cap = cv2.VideoCapture(video_file)   #동영상 파일 읽기

count = 1
while(cap.isOpened()):
    ret, frame = cap.read() #ret 객체 초기화를 확인하는 변수, 잘 읽으면 true 반환 #frame 각 프레임에 해당하는 이미지 변수 
    if ret:
        cv2.imwrite("./hand_gesture/%d.jpg" % count, frame)
        if cv2.waitKey(100) & 0xFF == ord('q'): #waitKey 100ms 만큼 기다림
            break
        count += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()
