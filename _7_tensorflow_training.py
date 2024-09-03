from cnn_training import *
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

import tensorflow as tf
import matplotlib.pyplot as plt

tensors = tensors.astype('float32') / 255
targets = to_categorical(targets, 4)

x_train, x_test, y_train, y_test = train_test_split(
    tensors,
    targets,
    test_size=0.2,
    random_state=1
)

n = int(len(x_test) / 2)
x_valid, y_valid = x_test[:n], y_test[:n]
x_test, y_test = x_test[n:], y_test[n:]

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)
print(x_valid.shape, y_valid.shape)
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(24, (5, 5), strides=(2, 2), padding="same",
                           activation='relu', input_shape=x_train.shape[1:]),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(32, (5, 5), strides=(2, 2), padding="same", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation='softmax')
])
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=50,
                    validation_data=(x_valid, y_valid))

loss = history.history['loss']
epochs = range(1, len(loss) + 1)

plt.plot(epochs, loss, 'g', label='Training loss')
plt.title('Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save("model.h5")