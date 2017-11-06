from darkflow.net.build import TFNet
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
options = {"model": "cfg/yolo.cfg","load": "cfg/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("images/timg.jpeg")
result = tfnet.return_predict(imgcv)
fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
ax.imshow(cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB))
color = ['r', 'b', 'g', 'y']
for _, box in enumerate(result):
    idx = random.randint(0,3)
    label = box['label']
    confidence = box['confidence']
    topleft = box['topleft']
    bottomright = box['bottomright']
    if confidence > 0.4 :
        rect = mpatches.Rectangle(
            (topleft['x'], topleft['y']), bottomright['x'] - topleft['x'], bottomright['y'] - topleft['y'],
            fill=False, edgecolor=color[idx], linewidth=3)
        ax.add_patch(rect)
        ax.text(topleft['x'], topleft['y'] - 5, label, family = 'monospace', fontsize = 10)
        ax.text(topleft['x'], topleft['y'] + 15, "%2.2f %%" % (confidence * 100), family='monospace', fontsize = 10)

plt.show()
