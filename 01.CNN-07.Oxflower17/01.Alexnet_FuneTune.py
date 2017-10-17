from zknet.zkNet import zkNet
import tensorflow as tf
import tensorlayer as tl
from PIL import Image
import tflearn
import numpy as np
IMAGE_SIZE = 0
IMAGE_CHANNEL = 0
def lossfunction(logits, label, *data):
    global IMAGE_SIZE
    IMAGE_SIZE = data[0]['image_size']
    global IMAGE_CHANNEL
    IMAGE_CHANNEL = data[0]['image_channel']
    print('call my loss...')
    ce = tl.cost.cross_entropy(logits, label, name='cost')
    L2 = 0
    for p in tl.layers.get_variables_with_name('relu/W', True, True):
        print(p.name)
        L2 += tf.contrib.layers.l2_regularizer(0.004)(p)

    return ce + L2

def accuracyFunction(logits, label, *data):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32))

def predFunction(outputs):
    return tf.argmax(tf.nn.softmax(outputs), 1)

def image_proposal(img_path):
    img = Image.open(img_path)
    img = tflearn.data_utils.resize_image(img, IMAGE_SIZE, IMAGE_SIZE)
    if img.mode == 'RGB':
        r, g, b = img.split()  # rgb通道分离
    else:
        r, g, b, _ = img.split()
    r_arr = np.array(r).reshape(IMAGE_SIZE * IMAGE_SIZE)
    g_arr = np.array(g).reshape(IMAGE_SIZE * IMAGE_SIZE)
    b_arr = np.array(b).reshape(IMAGE_SIZE * IMAGE_SIZE)
    # 行拼接，类似于接火车；最终结果：共n行，一行3072列，为一张图片的rgb值
    image_arr = np.concatenate((r_arr, g_arr, b_arr))

    return image_arr.reshape([IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])

if __name__ == '__main__':
    zknet = zkNet('alexnet_funetune.xml', LossFunc=lossfunction, AccFunc=accuracyFunction)
    zknet.train()
    image_predict = image_proposal("test.png")
    print(zknet.predict(image_predict, predFunc=predFunction))