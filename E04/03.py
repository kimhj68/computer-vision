import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 1. 대상 이미지 로드 (img1.jpg, img2.jpg, img3.jpg 중 선택)
img1_path = 'images/img1.jpg'
img2_path = 'images/img2.jpg'

if not os.path.exists(img1_path) or not os.path.exists(img2_path):
    print("오류: 이미지를 찾을 수 없습니다.")
    exit()

img1 = cv.imread(img1_path)
img2 = cv.imread(img2_path)
img1_gray = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
img2_gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

# 2. SIFT 특징 추출
sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1_gray, None)
kp2, des2 = sift.detectAndCompute(img2_gray, None)

# 3. 특징점 매칭 및 비율 테스트 (오매칭 제거)
bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good_matches.append(m)

MIN_MATCH_COUNT = 10

if len(good_matches) > MIN_MATCH_COUNT:
    # 4. 좋은 매칭점들의 픽셀 좌표 추출 
    # (img1을 캔버스 기준으로 삼기 위해 src와 dst를 바꿔서 변환: img2 -> img1)
    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 5. RANSAC 알고리즘을 사용하여 호모그래피 행렬 계산
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # 6. 원근 변환(Warping) 수행 (img2 이미지를 img1 시점으로 변환)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # 두 이미지가 나란히 들어갈 수 있도록 충분히 큰 출력 공간 파노라마 생성
    out_width = w1 + w2
    out_height = max(h1, h2)
    
    warped_img2 = cv.warpPerspective(img2, M, (out_width, out_height))
    panorama = warped_img2.copy()
    
    # 변환된 넓은 공간 왼쪽에 기준 이미지(img1)를 덮어쓰기
    panorama[0:h1, 0:w1] = img1

    # 특징점 매칭 결과(선의 형태)를 그리기 위한 파라미터 셋업
    draw_params = dict(matchColor=(0, 255, 0),    # 매칭선은 초록색
                       singlePointColor=None,
                       matchesMask=matchesMask, # Inlier(정상) 매칭점들에만 선을 그림
                       flags=2)
    img_matches = cv.drawMatches(img1, kp1, img2, kp2, good_matches, None, **draw_params)

    # 시각화(matplotlib)를 위해 BGR에서 RGB 배열로 변환
    img_matches_rgb = cv.cvtColor(img_matches, cv.COLOR_BGR2RGB)
    panorama_rgb = cv.cvtColor(panorama, cv.COLOR_BGR2RGB)

    # 7. 최종 처리 결과 시각화
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.imshow(img_matches_rgb)
    ax1.set_title('특징점 매칭 결과 - 1차 (Inliers)')
    ax1.axis('off')

    ax2.imshow(panorama_rgb)
    ax2.set_title('호모그래피 원근 변환 이미지 (파노라마 합성이미지)')
    ax2.axis('off')

    plt.tight_layout(pad=2.0, h_pad=4.0)
    plt.subplots_adjust(top=0.93)
    plt.show()

else:
    print(f"매칭점이 충분하지 않습니다 - {len(good_matches)}/{MIN_MATCH_COUNT}")
