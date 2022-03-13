import cv2
import numpy as np
from math import acos, pi, sqrt

def hand_color_model(H,S,I):
    #피부색이 있는 부분만을 마킹할 배열을 새로 만든다
    height, width =H.shape[0], src.shape[1]
    dst = np.zeros((height, width), dtype=np.uint8)
    
    #피부색이 있는 픽셀 값은 1로 하기 위해 모든 값이 1인 배열 생성
    dst_1 = dst+1
    
    for i in range(height):
        for j in range(width):
            if H[i][j] >=0 and H[i][j] <=0.65:         #Hue값 사용해서 손 detection
                dst[i][j] = dst_1[i][j]#src[i][j]       #1만 있는 배열을 복사
            elif S[i][j] >= 0.1 and S[i][j] <=0.4:     #Saturation 사용해서 손 detection
                dst[i][j] = dst_1[i][j]#src[i][j]       #1만 있는 배열을 복사
    return dst

def four_connect(src):
    h, w = src.shape[0], src.shape[1]

    pad = np.zeros((h+2,w+2), dtype=np.uint8)
    pad[1:-1, 1:-1]=src.copy()
    
    h1, w1 = h+2, w+2

    eqiv_table = []
    mmax = 0
    for i in range(1,h1-1):                                      #Connected Components Labeling #row by row로 계산
        for j in range(1,w1-1):
            top = pad[i-1,j]
            left = pad[i, j-1]
            if pad[i][j]!=0:
                if top >= 1 or left >= 1:                        #top, left 둘 중에 1개 이상 1보다 클 때
                    if top >=1 and left >= 1:                    #둘 다 1보다 클때 #아니면 하나가 0이므로 큰 값으로 라벨링
                        if top == left:                           #같으면 그 값으로 라벨링
                            pad[i][j] = top
                        else:
                            minn = min(top,left)                   #두 개 중에 작은 값
                            maxx = max(top,left)                   #두 개 중에 큰 값
                            pad[i][j] == minn                      #작은 값을 넣음
                            if len(eqiv_table)-1>=minn:           #eqiv_table 업데이트 #작은값이 리스트의 길이보다 작거나 같으면
                                eqiv_table[minn] = [minn, maxx]   #리스트의 값만 수정하고
                            elif len(eqiv_table)-1<minn:          #최소값이 리스트의 길이보다 길면
                                eqiv_table.append([minn, maxx])   #리스트의 마지막에 한 칸 추가    #maxx -> minn
                    else:                                         #top, left 둘 중에 하나만 1보다 클 때
                        pad[i][j] = max(top, left)                #큰 값을 넣음
                else:                                            #top, left 둘 다 0일 때
                    if(mmax==0):
                        pad[i][j] = 1
                        mmax = 1
                    else:
                        pad[i][j] = mmax+1
                        mmax += 1
    for i in range(len(eqiv_table)):
        li = eqiv_table[-(i+1)]                                  # 리스트의 뒤에서부터 li = [[0,1]]
        pad[pad==li[1]]=li[0]

    return pad

#B,G,R
def compute_Hue(B, G, R):
    angle = 0
    if B != G != R:
        angle = 0.5*((R - G) + (R - B)) / sqrt((R-G)*(R-G) + (R-B)*(G-B))
    return acos(angle) if B <= G else (2*pi - acos(angle))

#이미지 읽기
FRAME = "60"

src = cv2.imread("./hand_gesture/"+FRAME+".jpg", cv2.IMREAD_COLOR)#./hand_gesture/100.jpg ./hand_video2/30.jpg

#이미지 크기 조절
height, width = src.shape[0], src.shape[1]
if(height>=400):
    src= cv2.resize(src, dsize=(0, 0), fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    height, width = src.shape[0], src.shape[1]

I = np.zeros((height, width))
S = np.zeros((height, width))
H = np.zeros((height, width))

#stock man 정의식을 사용해서 RGB값을 HSI로 변환
for i in range(height):
    for j in range(width):
        B, G, R = src[i][j][0]/255., src[i][j][1]/255., src[i][j][2]/255. #정규화       
        #수식
        I[i][j] = (B*G*R)/3.
        if B*G*R != 0:
            S[i][j] = 1 - 3*np.min([B,G,R])/(B+G+R)
        H[i][j] = compute_Hue(B, G, R)
        
I=I*255

#피부색이 있는 부분만을 마킹해서 0,1 binary 이미지로 저장
dst = hand_color_model(H,S,I) #dst값 RGB

#4-connectivity
dst = four_connect(dst)
#print(dst.shape)
for i in range(height):
    for j in range(width):
        if(dst[i+1][j+1]==0): #dst는 zero-padding 이미지여서 인덱스에 +1해야함
            src[i][j][0]=255
            src[i][j][1]=255
            src[i][j][2]=255

#시각화를 위해 0~255로 맞춰줌
dst = dst*255

cv2.imshow("frame"+FRAME, src)
cv2.waitKey(0)
cv2.destroyAllWindows()
