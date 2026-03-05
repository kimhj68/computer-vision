import cv2 as cv
import numpy as np
import sys


img = cv.imread('soccer.jpg') # 폴더 내 이미지 로드
img = cv.resize(img, dsize=(0,0), fx=0.5, fy=0.5) # 원본 이미지를 작게 변환

if img is None: # 파일 존재 여부 확인 
    sys.exit('파일이 존재하지 않습니다.')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) # BGR 컬러 이미지를 명암 이미지로 변환 

gray_bgr = cv.cvtColor(gray, cv.COLOR_GRAY2BGR) # 1 채널 이미지를 3 채널 이미지로 변환

imgs = np.hstack((img,gray_bgr)) # 이미지들을 수평으로 이어붙임

cv.imshow('collected images', imgs) # 이미지 표시

cv.waitKey()
cv.destroyAllWindows()

print(type(imgs))
print(imgs.shape)