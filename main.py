import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM # jika runtime biasa
# from tensorflow.compat.v1.keras.layers import CuDNNLSTM # jika runtime GPU


mnist = tf.keras.datasets.mnist  # mnist adalah kumpulan data 28x28 gambar angka tulisan tangan dan labelnya
(x_train, y_train),(x_test, y_test) = mnist.load_data()  # membongkar gambar ke x_train/x_test dan memberi label ke y_train/y_test

x_train = x_train/255.0
x_test = x_test/255.0

print(x_train.shape)
print(x_train[0].shape)

model = Sequential()

model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# JIKA Anda menjalankan dengan GPU, cobalah jenis lapisan CuDNNLSTM sebagai gantinya (activation, tanh is required) dan ini akan jauh lebih cepat dan lebih efisien
"""
model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))
model.add(CuDNNLSTM(128))
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
"""

opt = tf.keras.optimizers.Adam(lr=0.001, decay=1e-6)

# Compile model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'],
)

model.fit(x_train,
          y_train,
          epochs=3,
          validation_data=(x_test, y_test))
