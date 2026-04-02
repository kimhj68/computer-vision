# Image Recognition 실습 프로젝트 (E05)

이 프로젝트는 딥러닝 기반의 주요 컴퓨터 비전 태스크 중 하나인 **이미지 인식(Image Recognition)**의 기초를 학습하는 과정입니다. 흑백 숫자 이미지인 MNIST 데이터셋을 활용한 기본 인공신경망의 구축부터, 복잡한 컬러 사물/동물 이미지인 CIFAR-10 데이터셋을 판별하는 전형적인 합성곱 신경망(CNN)의 설계 및 실제 이미지에 대한 예측까지의 과정을 실습합니다.

---

## 1. 간단한 이미지 분류기 구현 (MNIST 데이터셋) (`01.py`)

### 문제 설명
주어진 손글씨 숫자(0~9) 이미지(MNIST 데이터셋)를 활용해 가장 기초적인 형태의 이미지 분류기를 제작합니다. 신경망 도구(Keras)를 이용해 훈련/테스트 세트를 분할하고 이미지를 `0~1` 범위로 정규화한 후, 1차원으로 픽셀들을 펴주는 `Flatten` 층과 일반 노드 네트워크인 `Dense` 층을 결합한 얕은 신경망 구조를 구축하여 정확도를 도출해 냅니다.

### 전체 코드
```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# 1. MNIST 데이터셋 로드 및 분할
print("MNIST 데이터셋을 로드합니다.")
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 데이터 전처리 (픽셀 값을 0~1 범위로 정규화)
x_train, x_test = x_train / 255.0, x_test / 255.0

# 2. 간단한 신경망 모델 구축
model = Sequential([
    Flatten(input_shape=(28, 28)),       # 28x28의 2차원 배열을 1차원 배열로 펼침
    Dense(128, activation='relu'),       # 은닉층 (노드 128개)
    Dense(10, activation='softmax')      # 출력층 (10개의 클래스 분류를 위한 Softmax)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# 3. 모델 훈련
print("\n[모델 훈련 시작]")
model.fit(x_train, y_train, epochs=5)

# 4. 모델 평가
print("\n[모델 평가]")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 세트 정확도: {test_acc:.4f}")
```

### 핵심 코드 및 설명
- **`x_train / 255.0`**: 0 ~ 255로 구성되어 있는 픽셀의 값 범위를 0 ~ 1 수준의 값으로 스케일링하는 정규화를 통해 모델 학습 과정 중의 수렴 속도 및 성능을 증대시킵니다.
- **`Flatten(input_shape=(28, 28))`**: 2차원 공간인 28x28 픽셀 형태의 다차원 이미지를 단순 신경망(MLP)의 입력으로 넣기 위해 길이 '784'의 1차원 벡터 배열로 평탄화합니다.
- **`Dense(10, activation='softmax')`**: '0'부터 '9'까지 총 10가지의 클래스로 분류하는 다중 분류 작업이므로 마지막 출력층의 노드는 10개로 지정하며, 결과들의 총합이 1(100%)이 되는 확률 계산 함수 `Softmax`를 활성화 함수로 지정합니다.

### 결과 화면
```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ flatten (Flatten)                    │ (None, 784)                 │               0 │
│ dense (Dense)                        │ (None, 128)                 │         100,480 │
│ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 101,770 (397.54 KB)

[모델 훈련 시작]
Epoch 1/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 3s 1ms/step - accuracy: 0.9275 - loss: 0.2517
Epoch 2/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9687 - loss: 0.1075
...
Epoch 5/5
1875/1875 ━━━━━━━━━━━━━━━━━━━━ 2s 1ms/step - accuracy: 0.9872 - loss: 0.0425

[모델 평가]
313/313 - 0s - 1ms/step - accuracy: 0.9761 - loss: 0.0759
테스트 세트 정확도: 0.9761
```
> 파라미터가 10만 개인 인공 신경망 모델로도 단 몇 초 만의 반복 학습을 통해 처음 보는 글씨체 이미지 1만 건에서 무려 **97.6%** 수준의 훌륭한 진단 정답률을 보여주었습니다.

---

## 2. CIFAR-10 데이터셋을 활용한 CNN 모델 구축 (`02.py`)

### 문제 설명
조금 더 복잡한 컬러 채널(RGB)을 지니고 있는 사물과 동물 이미지(CIFAR-10 데이터셋)들을 구별해내기 위하여, 영상 인식 등에서 압도적으로 큰 효율을 보여주는 다층적인 특징 추출 모델, 합성곱 신경망(Convolutional Neural Network, CNN)을 설계합니다. 모델 훈련 직후 실존하는 새로운 강아지 이미지 파일(`dog.jpg`)을 모델에 주입해 학습된 딥러닝 망이 해당 대상 객체를 정확하게 인지하고 분류하는지 실전 검증합니다.

### 전체 코드
```python
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

# 4. 모델 훈련 시작 (검증 셋 추가)
print("\n[모델 훈련 시작]")
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# 5. 성능 평가
print("\n[모델 평가]")
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"테스트 세트 정확도: {test_acc:.4f}")

# 6. 테스트 이미지(dog.jpg) 실증 예측
image_path = 'images/dog.jpg'

img = cv2.imread(image_path)
if img is not None:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 모델 입력 크기에 맞춰 스케일링 (32x32)
    img_resized = cv2.resize(img_rgb, (32, 32))
    
    # 정규화 및 1장 전용 데이터 차원 확장 ((32, 32, 3) -> (1, 32, 32, 3))
    img_input = np.expand_dims(img_resized / 255.0, axis=0)
    
    predictions = model.predict(img_input)
    predicted_class = np.argmax(predictions[0])
    
    print(f"\n[예측 결과]")
    print(f"예측된 클래스: {class_names[predicted_class]} (Confidence: {predictions[0][predicted_class]*100:.2f}%)")
    
    plt.imshow(img_rgb)
    plt.title(f"Prediction: {class_names[predicted_class]}")
    plt.axis('off')
    plt.show()
else:
    print(f"\n이미지를 찾을 수 없습니다: {image_path}")
```

### 핵심 코드 및 설명
- **`Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))`**: 이미지에서 다양한 형태의 특징(예: 가로/세로 엣지 구조)을 잡아내는 3x3 픽셀 모양의 필터 돋보기를 32개 겹쳐 사용하는 '합성곱' 연산입니다. 컬러 사진이므로 입력 데이터에 3개의 색 채널(`input_shape=(..., 3)`)이 세팅됩니다.
- **`MaxPooling2D((2, 2))`**: 2x2 사이즈별 영역 내의 최댓값만 통과시켜 다음 층의 데이터 크기(공간적 차원, 가로/세로 해상도)를 각각 절반으로 크게 줄입니다. 가장 뚜렷하고 강한 특징들만을 부각시켜 파라미터를 최적화할 뿐만 아니라 계산 부하량도 대폭 줄여줍니다.
- **`np.expand_dims(...)`**: `model.predict()`는 '(배치 크기수, 가로, 세로, 채널)'의 구조인 4차원 배열 배치를 입력으로 기대합니다. 1장의 단일 이미지인 `(32, 32, 3)` 데이터의 첫 축을 확장해 학습 때처럼 `(1, 32, 32, 3)` 포맷으로 맞춰주는 중요한 처리 과정입니다.

### 결과 화면
```text
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 30, 30, 32)          │             896 │
│ max_pooling2d (MaxPooling2D)         │ (None, 15, 15, 32)          │               0 │
│ conv2d_1 (Conv2D)                    │ (None, 13, 13, 64)          │          18,496 │
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 6, 6, 64)            │               0 │
│ conv2d_2 (Conv2D)                    │ (None, 4, 4, 64)            │          36,928 │
│ flatten (Flatten)                    │ (None, 1024)                │               0 │
│ dense (Dense)                        │ (None, 64)                  │          65,600 │
│ dense_1 (Dense)                      │ (None, 10)                  │             650 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘

[모델 훈련 시작]
Epoch 1/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 11s 6ms/step - accuracy: 0.4516 - loss: 1.5036 - val_accuracy: 0.5367 - val_loss: 1.2909
...
Epoch 10/10
1563/1563 ━━━━━━━━━━━━━━━━━━━━ 12s 8ms/step - accuracy: 0.7867 - loss: 0.6134 - val_accuracy: 0.7005 - val_loss: 0.9200

[모델 평가]
313/313 - 1s - 3ms/step - accuracy: 0.7005 - loss: 0.9200
테스트 세트 정확도: 0.7005

[예측 결과]
예측된 클래스: dog (Confidence: 87.37%)
```
> 합성곱으로 이미지를 파악하는 모델이 처음 보는 테스트 데이터들을 상대로 약 70%의 정확도를 기록했으며, 특히 실제 `images/dog.jpg` 강아지 사진을 투입해 예측을 수행하였을 때 무려 **87%의 높은 확률(Confidence)로 '개(dog)' 임을 정확히 인식**하는 결과를 시현했습니다.

<img width="640" height="480" alt="Figure_1" src="https://github.com/user-attachments/assets/aac15b28-1c9f-48de-9d6f-40cb14ceddab" />

