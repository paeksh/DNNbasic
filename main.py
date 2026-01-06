import tensorflow as tf
from tensorflow.keras import layers, models

# CNN 모델 생성
model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # convolutional layer
        layers.MaxPooling2D((2, 2)), # pooling layer
        layers.Conv2D(64, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation = 'relu'),
        layers.Flatten(), # flattening to 1D
        layers.Dense(64, activation ='relu'), # fully-connected layer
        layers.Dense(10, activation = 'softmax') # output layer (0~9 숫자 분류)
])

# 모델 요약 출력
model.summary()