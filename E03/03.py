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
