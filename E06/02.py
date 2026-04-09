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

    # 이미지 처리 성능 향상을 위해 writeable 플래그를 False로 설정하여
    # 참조로 전달되게 합니다.
    image.flags.writeable = False
    # OpenCV는 BGR을 사용하고 Mediapipe는 RGB를 사용하므로 색상 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 얼굴 랜드마크 추출
    results = face_mesh.process(image_rgb)

    # 이미지 상에 랜드마크를 그리기 위해 다시 writeable 플래그를 True로 설정
    image.flags.writeable = True

    # 결과가 있을 경우 (얼굴이 검출된 경우)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, c = image.shape
            
            # 각각의 랜드마크(468여개) 좌표 순회
            for landmark in face_landmarks.landmark:
                # 랜드마크 좌표는 0.0 ~ 1.0 사이로 정규화(normalized)되어 있으므로
                # 영상의 실제 가로(w), 세로(h) 크기를 곱하여 실제 픽셀 좌표로 변환합니다.
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                
                # 추출된 좌표에 OpenCV circle 함수로 초록색 점 표시 (반지름=1)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # 좌우 반전하여 거울 모드로 보여줌 (selfie-view)
    cv2.imshow('Mediapipe Face Mesh', cv2.flip(image, 1))
    
    # ESC 키(27)를 누르면 프로그램 종료
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
