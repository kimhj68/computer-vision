import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 1. CIFAR-10 데이터셋 로드
print("CIFAR-10 데이터셋을 로드합니다.")
cifar10 = tf.keras.datasets.cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리 (픽셀 값을 0~1 범위로 정규화)
x_train, x_test = x_train / 255.0, x_test / 255.0

# CIFAR-10 클래스명 (참고용)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# 3. CNN 모델 설계
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 4. 모델 훈련
print("\n[모델 훈련 시작]")
# 학습 속도와 결과를 위해 epoch를 10번 진행합니다.
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 5. 성능 평가
print("\n[모델 평가]")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 세트 정확도: {test_acc:.4f}")

# 6. 테스트 이미지(dog.jpg)에 대한 예측 진행
image_path = 'images/dog.jpg'

# OpenCV로 테스트 이미지를 불러옵니다.
img = cv2.imread(image_path)
if img is not None:
    # BGR 포맷을 RGB로 변환
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 모델 입력 크기에 맞춰 스케일링 (32x32)
    img_resized = cv2.resize(img_rgb, (32, 32))
    
    # 정규화 및 차원 확장 ((32, 32, 3) -> (1, 32, 32, 3))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)
    
    # 모델을 이용해 예측 수행
    predictions = model.predict(img_input)
    predicted_class = np.argmax(predictions[0])
    
    print(f"\n[예측 결과]")
    print(f"예측된 클래스: {class_names[predicted_class]} (Confidence: {predictions[0][predicted_class]*100:.2f}%)")
    
    # 결과 시각화
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()
else:
    print(f"\n이미지를 찾을 수 없습니다: {image_path}")
