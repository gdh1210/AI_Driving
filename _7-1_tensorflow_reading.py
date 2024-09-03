from tensorflow_training import *
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

model1 = load_model('model.h5')

# Model predictions for the testing dataset
y_test_predict = model1.predict(x_test)
print(y_test_predict.shape, y_test_predict[0])
y_test_predict = np.argmax(y_test_predict, axis=1)
print(y_test_predict.shape, y_test_predict[0])

# Name list
names = ['_0_forward', '_1_right', '_2_left', '_3_stop']

# Display true labels and predictions
fig = plt.figure(figsize=(18, 18))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=16, replace=False)):
    ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = y_test_predict[idx]
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(names[pred_idx], names[true_idx]),
                 color=("#4876ff" if pred_idx == true_idx else "darkred"))
plt.show()