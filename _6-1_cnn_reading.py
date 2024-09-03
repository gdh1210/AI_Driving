from cnn_training import *

import cv2
import matplotlib.pyplot as plt

# Name list
names = ['_0_forward', '_1_right', '_2_left', '_3_stop']

def display_images(img_path, ax):
    img = cv2.imread(os.path.join(dirname, img_path))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])

fig, axes = plt.subplots(3, 3, figsize=(15, 10))

for i, ax in enumerate(axes.flat):
    if i < len(files):
        ax.set_title(names[targets[i]], color='blue')
        display_images(files[i], ax)
    else:
        ax.axis('off')  # 이미지가 없으면 서브플롯을 비활성화

plt.tight_layout()
plt.show()
