# Local Feature 실습 프로젝트 (E04)

이 프로젝트는 SIFT(Scale-Invariant Feature Transform) 알고리즘을 활용하여 이미지의 특징점을 검출하고 매칭하며, 이를 통해 기하학적 변환(Homography) 정보를 계산하여 겹친 사진들을 하나의 파노라마 형태로 정합(Image Alignment)해 보는 컴퓨터 비전 실습 예제입니다.

---

## 1. SIFT를 이용한 특징점 검출 및 시각화 (`01.py`)

### 문제 설명
주어진 원본 이미지(`mot_color70.jpg`)에서 크기 변환이나 회전에도 크게 변하지 않는 강건한 SIFT 특징점을 찾습니다. 검출된 각각의 특징점을 위치, 스케일(영향 범위), 중심 방향 정보와 함께 시각화하여 알고리즘의 동작을 직관적으로 확인하는 과정을 학습합니다.

### 전체 코드
```python
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
```

### 핵심 코드 및 설명
- **`cv.SIFT_create()`**: SIFT 특징점 객체를 초기화합니다. `nfeatures` 등의 인자를 지정하여 추출할 특징점의 최대 개수를 제한하거나 조절할 수 있습니다.
- **`sift.detectAndCompute(img_gray, None)`**: 회색조 이미지에서 SIFT 알고리즘을 수행하여 Keypoint(특징점 위치 및 스케일 정보 객체)와 Descriptor(주변 그래디언트 정보 벡터 특징량)를 동시에 반환합니다.
- **`cv.drawKeypoints(..., flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)`**: OpenCV에서 제공하는 드로잉 함수로, 특징점들의 픽셀 단위 위치 뿐만 아니라 계산된 정보의 영향 스케일을 원의 크기로, 그라디언트 기준 방향을 선분 형태로 상세하게 시각화합니다.

### 결과 화면
<img width="1200" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/1e73f3e1-4d3d-4e82-848b-25d459e71e05" />


---

## 2. SIFT를 이용한 두 영상 간 특징점 매칭 (`02.py`)

### 문제 설명
두 개의 촬영 각도가 다른 사진(`mot_color70.jpg`, `mot_color83.jpg`)에서 각각 SIFT 특징량을 구하고, 두 이미지 간 서로 대응하는 유사 특징점끼리 이어주는 특징점 매칭(Feature Matching)을 수행합니다. 오매칭을 줄이고 높은 신뢰도를 얻기 위해 KNN(K-Nearest Neighbors) 방식의 매칭과 Ratio Test를 도입합니다.

### 전체 코드
```python
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
```

### 핵심 코드 및 설명
- **`bf = cv.BFMatcher()` / `bf.knnMatch(des1, des2, k=2)`**: Brute-Force 방식의 매처를 사용하여 첫 번째 이미지의 각 특징점에서 대상 이미지의 가장 가까운 최근접 특징점(이웃)을 2개(`k=2`)씩 추출합니다. 
- **Ratio Test (`m.distance < 0.7 * n.distance`)**: 이른바 'Lowe의 거리 비율(Lowe's ratio test)' 검증입니다. 1번 매칭의 최단 거리가 2순위 후보 매칭 거리의 70% 이하일 만큼 압도적으로 가까울 때에만 올바른 매칭('good_match')으로 판별하여, 잘못된 특징점 연결선 생성을 차단합니다.

### 결과 화면
<img width="1514" height="486" alt="image" src="https://github.com/user-attachments/assets/57fdc146-41bd-45f5-a0c2-ae9be9e13b77" />



---

## 3. 호모그래피를 이용한 이미지 정합 (`03.py`)

### 문제 설명
비율 검증을 거쳐 걸러진 좋은 매칭쌍(Good Matches) 지점들의 픽셀 좌표값을 바탕으로 두 이미지 면 간의 투시 및 공간을 일치시키는 호모그래피(Homography) 변환 행렬을 찾고 한 이미지를 회전, 이동 및 왜곡시키면서 나머지 이미지의 시점과 일치하도록 공간 정합(Image Alignment)해 파노라마 형상을 만드는 과정입니다.

### 전체 코드
```python
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
```

### 핵심 코드 및 설명
- **`cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)`**: 호모그래피(공간 변환) 행렬 `M`을 추정합니다. 이때 `RANSAC` 파라미터 옵션을 활성화하여 잘못 매칭된 예외값(Outliers)들을 제외하고, 허용 오차거리 내의 올바른 `Inlier`만 바탕으로 원근 행렬을 도출하도록 합니다.
- **좌표계 설계 (`src_pts=(img2)`, `dst_pts=(img1)`)**: 사진의 구도상 `img1`이 왼쪽이고 `img2`가 오른쪽이기 때문에 `img1`을 대상(기준 좌표계 0, 0)으로 두고 `img2`의 시야를 `img1`의 관점으로 공간 변환(Warping)하도록 설계했습니다. 기준 화면의 왼편(음의 좌표 구역)으로 넘어가는 사진이 잘려나가는 것을 완벽하게 예방하기 위함입니다.
- **`cv.warpPerspective(img2, M, (out_width, out_height))`**: 구해진 호모그래피 행렬 `M`을 사용하여 원본 사진인 `img2`의 시계 공간 자체를 비틀어 변환(Warping)합니다. 여유있게 넓어진 캔버스에 `img2`의 변환 이미지가 그려지고, 그 위에 원본 `img1`을 덮어 씌움으로써 자연스러운 형태의 파노라마 정합을 시연할 수 있었습니다.

### 결과 화면
<img width="1200" height="808" alt="Figure_3" src="https://github.com/user-attachments/assets/4e6fa738-e053-4278-8317-c9b34427858d" />

