import cv2 as cv
import matplotlib.pyplot as plt
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 이미지 로드
img_path = 'images/mot_color70.jpg'
if not os.path.exists(img_path):
    print(f"오류: {img_path} 파일을 찾을 수 없습니다.")
    exit()

img = cv.imread(img_path)
img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# 2. SIFT 객체 생성
# 매개변수를 변경하여 테스트 가능 (예: cv.SIFT_create(nfeatures=500))
sift = cv.SIFT_create()

# 3. 특징점 검출
keypoints, descriptors = sift.detectAndCompute(img_gray, None)

# 4. 특징점 그리기
img_with_keypoints = cv.drawKeypoints(
    img_gray, 
    keypoints, 
    img_rgb.copy(), 
    flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# 5. matplotlib을 이용한 결과 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.imshow(img_rgb)
ax1.set_title('원본 이미지 (Original Image)')
ax1.axis('off')

ax2.imshow(img_with_keypoints)
ax2.set_title(f'SIFT 특징점 검출 결과 (n={len(keypoints)})')
ax2.axis('off')

plt.tight_layout()
plt.show()
