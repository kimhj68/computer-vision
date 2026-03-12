import cv2
import numpy as np
from pathlib import Path

# 출력 폴더 생성
output_dir = Path("../outputs")
output_dir.mkdir(parents=True, exist_ok=True)

# 이미지 불러오기
img = cv2.imread("../images/rose.png")
if img is None:
    raise FileNotFoundError("이미지를 찾지 못했습니다.")

rows, cols = img.shape[:2]

# 1. 이미지의 중심 기준으로 +30도 회전 및 0.8배 크기 조절
# cv2.getRotationMatrix2D(center, angle, scale)
center = (cols / 2, rows / 2)
angle = 30
scale = 0.8
M = cv2.getRotationMatrix2D(center, angle, scale)

# 2. x축 방향으로 +80px, y축 방향으로 -40px만큼 평행이동
# 아핀 행렬 M의 마지막 열이 [tx, ty]를 담당함
M[0, 2] += 80
M[1, 2] -= 40

# 3. 변환 적용
# cv2.warpAffine(src, M, dsize)
dst = cv2.warpAffine(img, M, (cols, rows))

# 결과 시각화
cv2.imshow('Original', img)
cv2.imshow('Rotated + Scaled + Translated', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 결과 저장
cv2.imwrite("../outputs/02_transformation_result.jpg", dst)
print("Result saved to ../outputs/02_transformation_result.jpg")
