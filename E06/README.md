# Dynamic Vision 실습 프로젝트 (E06)

이 프로젝트는 컴퓨터 비전의 핵심 분야 중 하나인 **동적 비전(Dynamic Vision)**의 응용 기술을 학습하는 과정입니다. YOLOv3와 SORT 알고리즘을 결합한 실시간 다중 객체 추적(Multi-Object Tracking) 방식을 구현해 보고, Google의 Mediapipe FaceMesh를 활용하여 실시간 웹캠 영상에서 얼굴의 세밀한 468개 랜드마크를 빠르게 추출하고 시각화하는 과정을 실습합니다.

---

## 1. SORT 알고리즘을 활용한 다중 객체 추적기 구현 (`01.py`)

### 문제 설명
사전 학습된 YOLOv3 객체 검출 모델(`yolov3.weights`, `yolov3.cfg`)을 사용하여 실시간 영상(`slow_traffic_small.mp4`) 프레임 내의 차량 등을 검출해 냅니다. 이후 **SORT (Simple Online and Realtime Tracking)** 알고리즘에 추출한 바운딩 박스(Bounding Box) 데이터를 전달하여, 프레임이 넘어가며 객체가 이동하더라도 동일한 객체임을 추적하고 식별하여 고유 ID를 부여하는 다중 객체 추적기 구조를 완성합니다.

### 전체 코드
```python
import cv2
import numpy as np
from sort import Sort

# YOLO 모델 로드
# yolov3.weights와 yolov3.cfg 파일이 같은 폴더에 있어야 합니다.
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except IndexError:
    # opencv 버전에 따른 예외 처리
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# SORT 추적기 초기화
tracker = Sort()

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture("slow_traffic_small.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width, channels = frame.shape

    # 1. 객체 검출 (YOLOv3)
    # 이미지를 YOLO 입력 형식(blob)으로 변환
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []
    
    # 검출된 객체 정보 추출
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # 신뢰도가 0.5 이상인 경우만 (차량 등 모든 객체 포함)
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # NMS(Non-Maximum Suppression)를 적용하여 중복된 bounding box 제거
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
    # SORT 추적기 입력 형식에 맞게 리스트 구성: [x1, y1, x2, y2, score]
    dets = []
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            dets.append([x, y, x+w, y+h, confidences[i]])
            
    dets = np.array(dets)
    if len(dets) == 0:
        dets = np.empty((0, 5))

    # 2. 객체 추적 (SORT)
    # SORT 추적기 업데이트 (현재 프레임의 검출 결과를 바탕으로 추적 갱신)
    tracks = tracker.update(dets)
    
    # 3. 결과 시각화
    for track in tracks:
        x1, y1, x2, y2, track_id = [int(v) for v in track]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Tracking View", frame)
    
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

### 핵심 코드 및 설명
- **`cv2.dnn.NMSBoxes(...)`**: 검출된 여러 개의 오브젝트 바운딩 박스 중에서 동일한 대상 객체를 가리키고 겹쳐 있는 중복 박스들을 비최대 억제(Non-Maximum Suppression) 기법을 사용해 제거하고, 신뢰성 높은 대표 박스 하나만 필터링합니다.
- **`tracker.update(dets)`**: 현재 프레임에서 도출한 객체들의 좌표를 SORT 추적 클래스(`Sort()`)에 주입합니다. SORT 알고리즘은 내부적으로 칼만 필터(Kalman Filter)와 헝가리안 매칭 알고리즘을 활용하여, 이전 프레임의 위치값들과 비교해 연속적인 동일성을 연산하고 현재 추적 중인 객체의 상태와 `track_id`를 반환합니다.
- **`cv2.putText(...)`**: SORT 알고리즘을 거쳐 부여된 고유 식별 번호(`track_id`)를 바운딩 박스 상단에 텍스트로 실시간 렌더링함으로써, 복잡하게 이동하는 다수의 객체가 동일 화면에 출현해도 지속해서 개별 추적이 유지되는 프로세스를 관찰합니다.

### 결과 화면
<img width="801" height="490" alt="image" src="https://github.com/user-attachments/assets/733778a7-8140-424a-856e-572731dae8fd" />

---

## 2. Mediapipe를 활용한 얼굴 랜드마크 추출 및 시각화 (`02.py`)

### 문제 설명
모바일 및 데스크톱 환경상에서 복잡하고 무거운 AI 인퍼런스를 빠르고 정확하게 수행해 내는 Google의 파이프라인 프레임워크인 **Mediapipe**를 활용합니다. 내장된 FaceMesh 모듈을 호출하여 PC 웹캠을 통한 실시간 디스플레이 내부에서 사용자의 얼굴을 분석합니다. 이를 통해 눈, 코, 입, 윤곽 등을 구성하는 468개의 정교한 타겟 랜드마크를 추출하고 좌표에 원형 점을 찍어 시각적인 얼굴 데이터 그물망(Mesh) 구조를 구현합니다.

### 전체 코드
```python
import cv2
import mediapipe as mp

# Mediapipe 그리기 유틸리티 및 Face Mesh 모듈 초기화
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Face Mesh 객체 생성
# max_num_faces=1: 최대 1명의 얼굴 검출
# refine_landmarks=True: 눈동자, 입술 등의 랜드마크를 더 정밀하게 추출 (468개 이상)
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 열기 (0번 카메라)
cap = cv2.VideoCapture(0)

print("영상 창을 클릭하고 ESC를 누르면 종료됩니다.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("프레임을 읽어올 수 없습니다.")
        break

    # 이미지 처리 성능 향상을 위해 writeable 플래그를 False로 설정하여 참조로 전달
    image.flags.writeable = False
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 추출
    results = face_mesh.process(image_rgb)

    image.flags.writeable = True

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, c = image.shape
            
            # 각각의 랜드마크(468여개) 좌표 순회
            for landmark in face_landmarks.landmark:
                # 랜드마크 좌표는 0.0 ~ 1.0 사이로 정규화(normalized) 되어 있음
                # 실제 영상 해상도 픽셀 좌표로 변환 반영
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # 좌우 반전하여 거울 모드로 셀피(selfie) 구성 후 화면 출력
    cv2.imshow('Mediapipe Face Mesh', cv2.flip(image, 1))
    
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

### 핵심 코드 및 설명
- **`mp_face_mesh.FaceMesh(refine_landmarks=True)`**: 얼굴 메시에 매칭할 핵심 인공신경망 모듈을 초기화합니다. `refine_landmarks=True` 플래그를 활성화하면 홍채, 입술 주변의 눈/입 등 곡률의 중요 부위와 관련된 추가 랜드마크를 통합하여 한 단계 더 정밀화된 468개(개선판에선 눈동자 포함 478개)의 3D 윤곽 벡터 구조를 얻어냅니다.
- **`int(landmark.x * w), int(landmark.y * h)`**: Mediapipe 시스템 내에서 얻어진 `.x`, `.y` 렌더링 좌표값 등은 사용된 화면 규격(해상도)에 의존하지 않는 0.0 ~ 1.0 비율 구조의 정규화(Normalized) 값으로 반환됩니다. 이를 OpenCV 그리기 연산을 위한 실제 비트맵 픽셀 인덱스로 사용하려면 현재 적용 중인 영상 프레임의 실제 가로 세로 길이를 곱한 형변환 산술식이 적용되어야 합니다.
- **`cv2.flip(image, 1)`**: Mediapipe로 수집 및 렌더링 처리된 데이터 화면의 좌우를 직접 반전(`flip`)시킵니다. 사용자가 마치 디지털 거울을 응시하듯 이질감을 덜어내는 방향인 셀피 뷰(Selfie-View) 모드 경험을 구성하며, 사용자 친화적인 인터페이스의 기반이 됩니다.

### 결과 화면
<img width="535" height="426" alt="image" src="https://github.com/user-attachments/assets/397512bf-189e-4fd2-83c7-cdadaf70946b" />
