import cv2 as cv
import matplotlib.pyplot as plt
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 이미지 로드
img1_path = 'images/mot_color70.jpg'
img2_path = 'images/mot_color83.jpg' # PDF는 80을 명시하지만, 제공된 83 이미지를 사용

if not os.path.exists(img1_path) or not os.path.exists(img2_path):
    print("오류: 이미지를 찾을 수 없습니다.")
    exit()

img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
img1_rgb = cv.cvtColor(img1, cv.COLOR_BGR2RGB)
img2_rgb = cv.cvtColor(img2, cv.COLOR_BGR2RGB)

# 2. SIFT 특징점 추출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 3. BFMatcher와 KNN 매칭을 이용한 특징점 매칭
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# 비율 테스트 (Lowe's ratio test) 적용하여 오매칭 제거
good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)
    if len(good_matches) >= 100:
        break

# 4. 매칭 결과 그리기
img_matches = cv.drawMatches(
    img1_rgb, kp1, 
    img2_rgb, kp2, 
    good_matches, 
    None, 
    flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    matchColor=(0, 255, 0) # 매칭선 색상을 초록색으로 설정
)

# 5. 결과 시각화
plt.figure(figsize=(15, 6))
plt.imshow(img_matches)
plt.title(f'SIFT 특징점 매칭 결과 (Good Matches: {len(good_matches)})')
plt.axis('off')
plt.show()
