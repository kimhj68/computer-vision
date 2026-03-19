# Edge and Region 실습 프로젝트

이 프로젝트는 이미지 처리 기술인 소벨(Sobel) 에지 검출, 캐니(Canny) 에지 검출 및 허프(Hough) 선 변환을 통한 직선 검출, 그리고 GrabCut 알고리즘을 활용한 대화식 객체 영역 분할 등을 다룹니다.

---

## 1. 소벨 에지 검출 및 시각화 (`01.py`)

### 문제 설명
주어진 이미지를 그레이스케일로 변환한 후, 소벨(Sobel) 필터를 사용하여 x축과 y축 방향의 에지를 검출합니다. 이후 두 방향의 에지를 합쳐 전체 에지 강도(Magnitude)를 계산하고, 원본 이미지와 검출된 에지 맵을 나란히 시각화하여 확인합니다.

### 전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# 이미지 파일의 절대 경로 설정
image_path = os.path.join("images", "edgeDetectionImage.jpg")

# 이미지 불러오기
img = cv.imread(image_path)
if img is None:
    print(f"이미지 불러오기 실패: {image_path}")
    exit(1)

# Matplotlib에서 원본 이미지를 올바르게 표시하기 위해 BGR을 RGB로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# x축 및 y축 방향으로 소벨(Sobel) 에지 계산
sobel_x = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

# 에지 강도(Magnitude) 계산
magnitude = cv.magnitude(sobel_x, sobel_y)

# 0~255 범위의 uint8 타입으로 변환
magnitude_uint8 = cv.convertScaleAbs(magnitude)

# matplotlib를 사용하여 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image') # 원본 이미지
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Sobel Edge Magnitude') # 소벨 에지 강도
plt.imshow(magnitude_uint8, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 핵심 코드 및 설명
- **`cv.Sobel()`**: 원본 이미지에 소벨 마스크를 적용해 지정한 축(x축 또는 y축) 방향 픽셀들의 밝기 변화량(미분값)을 계산하여 방향성 있는 에지를 찾습니다.
- **`cv.magnitude()`**: 계산된 x축 미분값과 y축 미분값에 피타고라스 정리를 적용하여 두 방향의 에지 성분을 합친 절대적인 에지 세기(피쳐의 강도)를 구합니다.
- **`cv.convertScaleAbs()`**: 에지 강도는 내부적으로 소수점(float) 데이터 타입으로 계산되는데, 이를 이미지로 띄우기 위해 픽셀 색상 범위인 양수 0~255 사이의 정수(uint8) 구조로 스케일링하여 변환합니다.

### 결과 화면
<img width="1000" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/336379e0-b145-47a0-b8c3-cc5493d4a684" />

---

## 2. 허프 변환을 이용한 직선 검출 (`02.py`)

### 문제 설명
캐니(Canny) 에지 검출을 사용하여 이미지 내의 얇고 확실한 윤곽선 맵을 생성합니다. 생성된 에지 맵을 기반으로 확률적 허프 선 변환(Probabilistic Hough Transform) 알고리즘을 수행하여 실질적인 직선 성분을 찾아내고, 원본 이미지 위에 검출된 선을 빨간색으로 그려 시각화합니다.

### 전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# 이미지 파일의 절대 경로 설정
image_path = os.path.join("images", "dabo.jpg")

# 이미지 불러오기
img = cv.imread(image_path)
if img is None:
    print(f"이미지 불러오기 실패: {image_path}")
    exit(1)

# Matplotlib에서 원본 이미지를 올바르게 표시하기 위해 BGR을 RGB로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_hough = img_rgb.copy()

# 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 캐니(Canny) 에지 검출 적용
edges = cv.Canny(gray, threshold1=100, threshold2=200)

# 허프 선 변환(Hough Line Transform) 적용
lines = cv.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=110, minLineLength=60, maxLineGap=5)

# 이미지 위에 검출된 선 그리기
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # img_hough가 RGB 형식이므로 빨간색 선은 (255, 0, 0)으로 지정
        # (요구사항의 (0, 0, 255)는 OpenCV의 기본 BGR 기준이므로 RGB에서는 (255, 0, 0)이 됨)
        cv.line(img_hough, (x1, y1), (x2, y2), (255, 0, 0), 2)

# matplotlib를 사용하여 시각화
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image') # 원본 이미지
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Hough Lines') # 허프 선 변환 결과
plt.imshow(img_hough)
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 핵심 코드 및 설명
- **`cv.Canny()`**: 잡음과 그라데이션을 세밀하게 제거하고 선의 두께를 얇게 압축하여 가장 강한 뼈대 에지만 남깁니다. 허프 변환이 직선을 인식하기 전, 군더더기 곡선들을 걸러내는 필수 전처리 과정입니다.
- **`cv.HoughLinesP()`**: 에지 점들을 파라미터($\rho, \theta$) 공간에 투표하여 직선을 골라내는 확률 기반 알고리즘입니다. 직선으로 확정지을 교차점 개수(`threshold`)나 검출할 선의 최소 길이(`minLineLength`) 등을 튜닝함으로써 다양한 검출 결과를 이끌어냅니다.
- **`cv.line()`**: 허프 변환 함수가 찾아준 직선의 (시작점, 끝점) 픽셀 좌표 두 개를 이어서 영상 위에 두께를 지정하여 그래픽 선으로 그립니다.

### 결과 화면
<img width="1000" height="500" alt="Figure_2" src="https://github.com/user-attachments/assets/45ff5012-f775-4915-87ac-6c7593addc70" />

---

## 3. GrabCut을 이용한 대화식 영역 분할 및 객체 추출 (`03.py`)

### 문제 설명
사용자가 지정한 대략적인 사각형 박스(Bounding Box)를 바탕으로 GrabCut 알고리즘을 사용해 배경과 전경(객체)을 분리해 냅니다. 결과로 얻어진 분할 마스크를 사용하여 이미지에서 객체 주변의 배경을 완전히 삭제하고 순수한 커피잔 객체만 추출합니다.

### 전체 코드
```python
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# 이미지 파일의 절대 경로 설정
image_path = os.path.join("images", "coffee cup.JPG")

# 이미지 불러오기
img = cv.imread(image_path)
if img is None:
    print(f"이미지 불러오기 실패: {image_path}")
    exit(1)

# Matplotlib에서 원본 이미지를 올바르게 표시하기 위해 BGR을 RGB로 변환
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# GrabCut을 위한 초기 사각형 영역 정의 (x, y, width, height)
# 객체가 주로 중앙에 위치하므로 일반적인 중앙 사각형 설정 사용
h, w = img.shape[:2]
rect = (50, 50, w - 100, h - 100)

# 마스크(mask), 배경 모델(bgdModel), 전경 모델(fgdModel) 초기화
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# GrabCut 알고리즘 적용
cv.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount=5, mode=cv.GC_INIT_WITH_RECT)

# 마스크 값을 0과 1로 변경
# 배경 (GC_BGD, GC_PR_BGD) -> 0
# 전경 (GC_FGD, GC_PR_FGD) -> 1
mask2 = np.where((mask == cv.GC_BGD) | (mask == cv.GC_PR_BGD), 0, 1).astype('uint8')

# 원본 이미지에 마스크를 곱하여 배경이 제거된 이미지 생성
result_img = img_rgb * mask2[:, :, np.newaxis]

# matplotlib를 사용하여 시각화
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.title('Original Image') # 원본 이미지
plt.imshow(img_rgb)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Mask') # 마스크 (0은 검은색, 1은 흰색으로 시각화)
plt.imshow(mask2 * 255, cmap='gray')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Background Removed') # 배경 제거 결과
plt.imshow(result_img)
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 핵심 코드 및 설명
- **`cv.grabCut()`**: 주어진 초기 사각형 외부를 확실한 배경으로, 내부를 물체 후보로 두고 가우시안 모델 기반으로 지속적인 학습을 통해 객체와 배경을 분할합니다. 자동으로 테두리를 찾아주는 형태의 영역 분리 알고리즘입니다.
- **`np.where()`**: GrabCut이 찾아준 상세 마스크 배열(4가지 속성값) 중, 배경이 될 확률이 높은 곳들에 0(검은색)을 할당하고 전경 물체일 확률이 높은 픽셀엔 1(흰색)을 할당해 새로운 흑백 이진 배열(`mask2`)로 정리합니다.
- **`img_rgb * mask2[:, :, np.newaxis]`**: 배열 연산을 응용해 원본 영상 픽셀에 각 위치의 마스크값(0 또는 1)을 곱합니다. 배경 위치의 화소는 $\times 0$이 되어 검정색으로 변하고(제거됨), 전경 화소는 $\times 1$이 되어 물체 색상 정보만 깨끗하게 남게 됩니다.

### 결과 화면
<img width="1500" height="500" alt="Figure_3" src="https://github.com/user-attachments/assets/126cdabd-799c-4329-ae4c-d5f43c7ba6c5" />

