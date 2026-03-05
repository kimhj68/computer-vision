import cv2 as cv
import numpy as np
import sys

img = cv.imread('soccer.jpg') # 폴더 내 이미지 로드
img = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5) # 원본 이미지를 작게 변환

if img is None: # 파일 존재 여부 확인 
    sys.exit('파일이 존재하지 않습니다.')

draw_size = 5 # 붓 크기 5 지정

def draw(event, x, y, flags, param): # 마우스 이벤트 처리 함수

    if event == cv.EVENT_LBUTTONDOWN: # 왼쪽 마우스 버튼 클릭 시
        cv.circle(img, (x, y), draw_size, (255, 0, 0), -1) # 파란색 원 그리기
    elif event == cv.EVENT_RBUTTONDOWN: # 오른쪽 마우스 버튼 클릭 시
        cv.circle(img, (x, y), draw_size, (0, 0, 255), -1) # 빨간색 원 그리기
        
    elif event == cv.EVENT_MOUSEMOVE: # 마우스 이동 시
        if flags & cv.EVENT_FLAG_LBUTTON: # 왼쪽 마우스 버튼 클릭 상태에서 이동 시
            cv.circle(img, (x, y), draw_size, (255, 0, 0), -1) # 파란색 원 그리기
        elif flags & cv.EVENT_FLAG_RBUTTON: # 오른쪽 마우스 버튼 클릭 상태에서 이동 시
            cv.circle(img, (x, y), draw_size, (0, 0, 255), -1) # 빨간색 원 그리기
    

cv.namedWindow('mouse') # 마우스 이벤트를 처리할 창 생성
cv.setMouseCallback('mouse', draw) # 마우스 이벤트 처리 함수 등록

while True:
    cv.imshow('mouse', img) # 창에 이미지 표시
    key = cv.waitKey(1) # 키보드 입력 대기
    
    if key == ord('q'): # 'q' 키 입력 시
        cv.destroyAllWindows() # 모든 창 닫기
        break
    elif key == ord('+') or key == ord('='): # '+' 또는 '=' 키 입력 시
        if draw_size < 15: # 붓 크기가 15보다 작으면
            draw_size += 1 # 붓 크기 증가
            print(f'현재 붓 크기: {draw_size}') # 붓 크기 출력
    elif key == ord('-'): # '-' 키 입력 시
        if draw_size > 1: # 붓 크기가 1보다 크면
            draw_size -= 1 # 붓 크기 감소
        print(f'현재 붓 크기: {draw_size}') # 붓 크기 출력