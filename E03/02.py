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
