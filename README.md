- 디지털 영상처리 과제
## Hand Detection in Videos
**1. 영상 데이터 프레임 단위로 나누기**
```Python
video_file = 'test.mov'
cap = cv2.VideoCapture(video_file)
```
```Python
count = 1
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        cv2.imwrite("./hand_gesture/%d.jpg" % count, frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        count += 1
    else:
        break
```

<p>
  <img src="https://user-images.githubusercontent.com/51811995/158065611-a52f256f-5b78-4a12-89f5-084ef7d73cc7.jpg" width=400>
  <img src="https://user-images.githubusercontent.com/51811995/158065716-7709cb57-85ff-4098-8cb7-4deb43522c48.jpg" width=400>
</p>

**2. Skin-Color 모델**
- HSI 값을 이용하여 skin color 모델 구축
```Python
def hand_color_model(H,S,I):
    height, width =H.shape[0], src.shape[1]
    dst = np.zeros((height, width), dtype=np.uint8)
    dst_1 = dst+1
    for i in range(height):
        for j in range(width):
            if H[i][j] >=0 and H[i][j] <=0.65:         
                dst[i][j] = dst_1[i][j]       
            elif S[i][j] >= 0.1 and S[i][j] <=0.4:     
                dst[i][j] = dst_1[i][j]     
    return dst
```

**3. Binary 이미지 생성**
- Skin-color model을 이용하여 thresholding 하여 foreground와 background 구별
- 피부색이 있는 부분은 1로 없는 부분은 0으로 마킹

<p>
  <img src="https://user-images.githubusercontent.com/51811995/158066343-2dc4ae6b-ec92-49e9-8de3-62842f5fa1cc.png" width=400>
  <img src="https://user-images.githubusercontent.com/51811995/158066368-be41665c-6c0b-44ae-a08c-2f33c8a65443.png" width=400>
</p>

**4. Connected Component Labeling**
- 4-connectivity 사용해서 Conected Component Labeling
```Python
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
```

<p>
  <img src="https://user-images.githubusercontent.com/51811995/158065662-14b73f8b-f581-4f12-a84c-4b6b211d359c.png" width=400>
  <img src="https://user-images.githubusercontent.com/51811995/158065764-8e2e4ce2-c74e-49ff-9222-d1181e325c36.png" width=400>
</p>

