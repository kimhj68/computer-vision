# OpenCV 기초 및 실습 도구 프로젝트

이 프로젝트는 OpenCV를 활용하여 이미지 처리의 기초를 익히고, 실무에서 활용 가능한 간단한 도구(브러쉬, 크롭 툴)를 구현한 예제 모음입니다.

---

## 1. 이미지 수평 결합 (`01.py`)

### 문제 설명
채널 수가 다른 두 이미지(3채널 컬러 이미지와 1채널 명암 이미지)를 가로로 나란히 붙여서 출력하고자 할 때 발생하는 차원(Dimension) 불일치 문제를 해결하고, 이미지를 결합하는 방법을 학습합니다.

### 전체 코드
```python
import cv2 as cv
import numpy as np
import sys


img = cv.imread('soccer.jpg') # 폴더 내 이미지 로드
img = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5) # 원본 이미지를 작게 변환

if img is None: # 파일 존재 여부 확인 
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 이미지를 명암 이미지로 변환 

gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 1 채널 이미지를 3 채널 이미지로 변환

imgs = np.hstack((img,gray_bgr)) # 이미지들을 수평으로 이어붙임

cv.imshow('collected images', imgs) # 이미지 표시

cv.waitKey()
cv.destroyAllWindows()

print(type(imgs))
print(imgs.shape)
```

### 핵심 코드 및 설명
- **`cv.cvtColor(gray, cv.COLOR_GRAY2BGR)`**: 흑백(Grayscale) 이미지는 1개의 채널을 가지지만, 원본 컬러 이미지는 3개의 채널(BGR)을 가집니다. `np.hstack`으로 붙이려면 채널 수가 같아야 하므로 흑백 이미지를 다시 3채널 형식으로 변환합니다.
- **`np.hstack((img, gray_bgr))`**: Numpy의 수평 결합 함수를 사용하여 두 이미지 행렬을 가로 방향으로 이어붙입니다.

### 결과 화면
<img width="1795" height="631" alt="image" src="https://github.com/user-attachments/assets/d1fe5e5e-854f-4337-b662-4ecb155ab407" />


---

## 2. 대화형 브러쉬 도구 (`02.py`)

### 문제 설명
마우스 이벤트를 제어하여 이미지 위에 사용자가 자유롭게 그림을 그릴 수 있는 브러쉬 기능을 구현합니다. 또한 마우스 드래그를 감지하고 키보드 입력을 통해 브러쉬의 크기를 실시간으로 조절하는 기능을 포함합니다.

### 전체 코드
```python
import cv2 as cv
import numpy as np
import sys

img = cv.imread('soccer.jpg') # 폴더 내 이미지 로드
img = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5) # 원본 이미지를 작게 변환

if img is None: # 파일 존재 여부 확인 
    sys.exit('파일이 존재하지 않습니다.')

draw_size = 5 # 붓 크기 5 지정

def draw(event, x, y, flags, param): # 마우스 이벤트 처리 함수

    if event == cv.EVENT_LBUTTONDOWN: # 왼쪽 마우스 버튼 클릭 시
        cv.circle(img, (x, y), draw_size, (255, 0, 0), -1) # 파란색 원 그리기
    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 마우스 버튼 클릭 시
        cv.circle(img, (x, y), draw_size, (0, 0, 255), -1) # 빨간색 원 그리기
        
    elif event == cv.EVENT_MOUSEMOVE: # 마우스 이동 시
        if flags & cv.EVENT_FLAG_LBUTTON: # 왼쪽 마우스 버튼 클릭 상태에서 이동 시
            cv.circle(img, (x, y), draw_size, (255, 0, 0), -1) # 파란색 원 그리기
        elif flags & cv.EVENT_FLAG_RBUTTON: # 오른쪽 마우스 버튼 클릭 상태에서 이동 시
            cv.circle(img, (x, y), draw_size, (0, 0, 255), -1) # 빨간색 원 그리기
    

cv.namedWindow('mouse') # 마우스 이벤트를 처리할 창 생성
cv.setMouseCallback('mouse', draw) # 마우스 이벤트 처리 함수 등록

while True:
    cv.imshow('mouse', img) # 창에 이미지 표시
    key = cv.waitKey(1) # 키보드 입력 대기
    
    if key == ord('q'): # 'q' 키 입력 시
        cv.destroyAllWindows() # 모든 창 닫기
        break
    elif key == ord('+') or key == ord('='): # '+' 또는 '=' 키 입력 시
        if draw_size < 15: # 붓 크기가 15보다 작으면
            draw_size += 1 # 붓 크기 증가
            print(f'현재 붓 크기: {draw_size}') # 붓 크기 출력
    elif key == ord('-'): # '-' 키 입력 시
        if draw_size > 1: # 붓 크기가 1보다 크면
            draw_size -= 1 # 붓 크기 감소
        print(f'현재 붓 크기: {draw_size}') # 붓 크기 출력
```

### 핵심 코드 및 설명
- **`flags & cv.EVENT_FLAG_LBUTTON`**: 마우스가 이동할 때(`EVENT_MOUSEMOVE`) 왼쪽 버튼이 눌려 있는 상태인지를 비트 연산으로 확인합니다. 이를 통해 클릭한 상태로 움직일 때만 그림이 그려지는 "드래그" 기능을 구현합니다.
- **`cv.setMouseCallback('mouse', draw)`**: 지정한 윈도우에서 발생하는 모든 마우스 이벤트를 사용자가 정의한 `draw` 함수로 전달합니다.
- **`key = cv.waitKey(1)`**: 메인 루프에서 1ms 간격으로 키 입력을 확인하여 `+`, `-` 키에 따라 `draw_size` 변수를 동적으로 변경합니다.

### 결과 화면
<img width="898" height="631" alt="image" src="https://github.com/user-attachments/assets/dc3f7848-b6c7-481e-b734-54d551e258e8" />


---

## 3. 스마트 이미지 크롭 도구 (`03.py`)

### 문제 설명
사용자가 드래그하여 이미지의 특정 영역을 선택하고, 해당 영역을 추출(Crop)하는 기능을 구현합니다. 선택 영역의 시각적 피드백 유지, 리셋 기능, 그리고 파일 저장 기능을 포함한 종합적인 도구 제작을 목표로 합니다.

### 전체 코드
```python
import cv2 as cv
import numpy as np
import sys

img = cv.imread('girl_laughing.jpg') # 폴더 내 이미지 로드
img = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5) # 원본 이미지를 작게 변환

if img is None: # 파일 존재 여부 확인 
    sys.exit('파일이 존재하지 않습니다.')

ix, iy = -1, -1
cx, cy = -1, -1
drawing = False
show_rect = False # 사각형 표시 여부 제어
roi = None # 잘라낸 영역을 저장할 전역 변수

def draw(event, x, y, flags, param):
    global ix, iy, cx, cy, drawing, show_rect, roi

    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN:
        ix, iy = x, y
        cx, cy = x, y
        drawing = True
        show_rect = True # 그리기 시작할 때 사각형 표시 활성화
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing: # 드래그 중일 때만 현재 좌표 갱신
            cx, cy = x, y
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        if drawing: # 드래그 중에 버튼을 놓았을 때만 작동
            drawing = False
            # 사각형 표시(show_rect)는 True로 유지하여 화면에 남도록 함
            x_start, x_end = min(ix, x), max(ix, x)
            y_start, y_end = min(iy, y), max(iy, y)
            
            roi = img[y_start:y_end, x_start:x_end]
            
            if roi.size > 0:
                cv.imshow('Cropped Image', roi)

cv.namedWindow('Selection')
cv.setMouseCallback('Selection', draw)

print("'r': 리셋(사각형 제거 및 창 닫기), 's': 현재 영역 저장, 'q': 종료")

while True:
    if show_rect: # 사각형 표시가 활성화된 경우에만 그리기
        img_copy = img.copy()
        cv.rectangle(img_copy, (ix, iy), (cx, cy), (0, 0, 255), 2)
        cv.imshow('Selection', img_copy)
    else:
        cv.imshow('Selection', img)
        
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'): # 리셋: 사각형 숨기기, 창 닫기, 변수 초기화
        show_rect = False
        try:
            cv.destroyWindow('Cropped Image')
        except:
            pass
        roi = None
        print("영역 선택이 리셋되었습니다.")
    elif key == ord('s'):
        if roi is not None:
            cv.imwrite('cropped_result.jpg', roi)
            print("선택한 영역이 'cropped_result.jpg'로 저장되었습니다.")
        else:
            print("저장할 영역이 없습니다. 먼저 영역을 선택해 주세요.")

cv.destroyAllWindows()
```

### 핵심 코드 및 설명
- **영역 유지 처리**: `img.copy()`와 `cv.rectangle`을 `while True` 루프 안에서 처리함으로써, 마우스 버튼을 놓거나 이동을 멈추어도 빨간색 선택 사각형이 화면에 지속적으로 표시되도록 구현했습니다.
- **`roi = img[y_start:y_end, x_start:x_end]`**: 파이썬의 리스트/행렬 슬라이싱 기능을 사용하여 선택한 좌표 내의 이미지 데이터만을 추출합니다.
- **`cv.imwrite('cropped_result.jpg', roi)`**: 's' 키를 누르면 추출된 영역(ROI)을 실제 이미지 파일로 저장합니다.
- **범용 클릭 처리**: `LBUTTON`과 `RBUTTON` 이벤트를 동시에 체크하여 마우스 어느 쪽 버튼으로든 영역 선택이 가능하도록 편의성을 높였습니다.

### 결과 화면
<img width="901" height="640" alt="image" src="https://github.com/user-attachments/assets/f6c4b2a3-90b3-4bf5-b3db-7d7e4a535092" />


