import cv2 as cv
import numpy as np
import sys

img = cv.imread('girl_laughing.jpg') # 폴더 내 이미지 로드
img = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5) # 원본 이미지를 작게 변환

if img is None: # 파일 존재 여부 확인 
    sys.exit('파일이 존재하지 않습니다.')

ix, iy = -1, -1
cx, cy = -1, -1
drawing = False
show_rect = False # 사각형 표시 여부 제어
roi = None # 잘라낸 영역을 저장할 전역 변수

def draw(event, x, y, flags, param):
    global ix, iy, cx, cy, drawing, show_rect, roi

    if event == cv.EVENT_LBUTTONDOWN or event == cv.EVENT_RBUTTONDOWN:
        ix, iy = x, y
        cx, cy = x, y
        drawing = True
        show_rect = True # 그리기 시작할 때 사각형 표시 활성화
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing: # 드래그 중일 때만 현재 좌표 갱신
            cx, cy = x, y
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        if drawing: # 드래그 중에 버튼을 놓았을 때만 작동
            drawing = False
            # 사각형 표시(show_rect)는 True로 유지하여 화면에 남도록 함
            x_start, x_end = min(ix, x), max(ix, x)
            y_start, y_end = min(iy, y), max(iy, y)
            
            roi = img[y_start:y_end, x_start:x_end]
            
            if roi.size > 0:
                cv.imshow('Cropped Image', roi)

cv.namedWindow('Selection')
cv.setMouseCallback('Selection', draw)

print("'r': 리셋(사각형 제거 및 창 닫기), 's': 현재 영역 저장, 'q': 종료")

while True:
    if show_rect: # 사각형 표시가 활성화된 경우에만 그리기
        img_copy = img.copy()
        cv.rectangle(img_copy, (ix, iy), (cx, cy), (0, 0, 255), 2)
        cv.imshow('Selection', img_copy)
    else:
        cv.imshow('Selection', img)
        
    key = cv.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('r'): # 리셋: 사각형 숨기기, 창 닫기, 변수 초기화
        show_rect = False
        try:
            cv.destroyWindow('Cropped Image')
        except:
            pass
        roi = None
        print("영역 선택이 리셋되었습니다.")
    elif key == ord('s'):
        if roi is not None:
            cv.imwrite('cropped_result.jpg', roi)
            print("선택한 영역이 'cropped_result.jpg'로 저장되었습니다.")
        else:
            print("저장할 영역이 없습니다. 먼저 영역을 선택해 주세요.")

cv.destroyAllWindows()
