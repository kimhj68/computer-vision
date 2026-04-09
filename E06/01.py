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
                # 객체의 bounding box 중심 좌표 및 너비, 높이 계산
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # bounding box의 좌상단 좌표 계산
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
    # 검출된 객체가 없을 경우 빈 배열 생성
    if len(dets) == 0:
        dets = np.empty((0, 5))

    # 2. 객체 추적 (SORT)
    # SORT 추적기 업데이트 (현재 프레임의 검출 결과를 바탕으로 추적 갱신)
    tracks = tracker.update(dets)
    
    # 3. 결과 시각화
    for track in tracks:
        x1, y1, x2, y2, track_id = [int(v) for v in track]
        
        # bounding box 그리기
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 객체 ID 텍스트 표시
        cv2.putText(frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 결과 프레임 출력
    cv2.imshow("Tracking View", frame)
    
    # ESC 키(27)를 누르면 종료
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
