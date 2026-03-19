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
