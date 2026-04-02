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
