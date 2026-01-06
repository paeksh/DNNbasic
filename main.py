import tensorflow as tf
from tensorflow.keras import layers, models, Sequential

##### 1. CNN 모델 생성 #####

CNN_model = models.Sequential([
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
CNN_model.summary()

##### 2. RNN 모델 생성 #####
RNN_model = Sequential([
    layers.SimpleRNN(50, activation = 'tanh', input_shape = (10, 1)), # input sequence = 10, 특성 : 1개
    layers.Dense(1) # output layer (회귀 예측)
])

# 모델 요약 출력
RNN_model.summary()
