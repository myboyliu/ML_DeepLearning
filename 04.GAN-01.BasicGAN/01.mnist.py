import tensorflow as tf
import numpy as np
from zkreader.Reader import MnistReader
import pickle
import tensorlayer as tl

IMAGE_SIZE = 784
IMAGE_CHANNEL = 1
BATCH_SIZE = 100
NOISE_SIZE = 100
G_UNITS = 128
D_UNITS = 128
ALPHA = 0.01
SMOOTH = 0.1
LEARNING_RATE = 0.001
EPOCHS = 50
N_SAMPLE = 25
meta = {'image_size' : 28, 'image_channel' : IMAGE_CHANNEL, 'batch_size' : BATCH_SIZE,
        'dataType' : 'mnist', 'dataPath' : '../Total_Data/mnist/'}
reader = MnistReader(meta)
(image, label), (_,_) = reader.readData()
image = tf.reshape(image, shape=[BATCH_SIZE, IMAGE_SIZE])
# 存储测试样例
samples = []
# 存储loss
losses = []

real_image = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='realImg')
noise_image = tf.placeholder(dtype=tf.float32, shape=[None, NOISE_SIZE], name='noiseImg')

with tf.variable_scope("generator", reuse=False):
    # hidden layer
    hidden1 = tf.layers.dense(noise_image, G_UNITS)
    # leaky ReLU
    hidden1 = tf.maximum(ALPHA * hidden1, hidden1)
    # dropout
    hidden1 = tf.layers.dropout(hidden1, rate=0.2)

    # logits & outputs
    G_logits = tf.layers.dense(hidden1, IMAGE_SIZE)
    G_outputs = tf.tanh(G_logits)

with tf.variable_scope("discriminator", reuse=False):
    # hidden layer
    hidden1 = tf.layers.dense(real_image, D_UNITS)
    hidden1 = tf.maximum(ALPHA * hidden1, hidden1)

    # logits & outputs
    D_logits_real = tf.layers.dense(hidden1, 1)
    D_outputs_real = tf.sigmoid(D_logits_real)

with tf.variable_scope("discriminator", reuse=True):
    # hidden layer
    hidden1 = tf.layers.dense(G_outputs, D_UNITS)
    hidden1 = tf.maximum(ALPHA * hidden1, hidden1)

    # logits & outputs
    D_logits_fake = tf.layers.dense(hidden1, 1)
    D_outputs_fake = tf.sigmoid(D_logits_fake)

d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_real,
                                                                     labels=tf.ones_like(D_outputs_real)) * (1 - SMOOTH))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                                     labels=tf.zeros_like(D_outputs_fake)))
d_loss = tf.add(d_loss_real, d_loss_fake)
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logits_fake,
                                                                labels=tf.ones_like(D_outputs_fake)) * (1 - SMOOTH))
train_vars = tf.trainable_variables()
g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
d_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars)
g_train_opt = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    tf.train.start_queue_runners()
    for e in range(EPOCHS):
        for batch_i in range(reader.trainRecordCount // BATCH_SIZE):
            batch, _ = sess.run([image, label])
            batch_images = batch
            # 对图像像素进行scale，这是因为tanh输出的结果介于(-1,1),real和fake图片共享discriminator的参数
            batch_images = batch_images*2 - 1

            # generator的输入噪声
            batch_noise = np.random.uniform(-1, 1, size=(BATCH_SIZE, NOISE_SIZE))

            # Run optimizers
            _ = sess.run(d_train_opt, feed_dict={real_image: batch_images, noise_image: batch_noise})
            _ = sess.run(g_train_opt, feed_dict={noise_image: batch_noise})

        # 每一轮结束计算loss
        train_loss_d = sess.run(d_loss,
                                feed_dict = {real_image: batch_images,
                                             noise_image: batch_noise})
        # real img loss
        train_loss_d_real = sess.run(d_loss_real,
                                     feed_dict = {real_image: batch_images,
                                                  noise_image: batch_noise})
        # fake img loss
        train_loss_d_fake = sess.run(d_loss_fake,
                                     feed_dict = {real_image: batch_images,
                                                  noise_image: batch_noise})
        # generator loss
        train_loss_g = sess.run(g_loss,
                                feed_dict = {noise_image: batch_noise})


        print("Epoch {}/{}...".format(e+1, EPOCHS),
              "Discriminator Loss: {:.4f}(Real: {:.4f} + Fake: {:.4f})...".format(train_loss_d, train_loss_d_real, train_loss_d_fake),
              "Generator Loss: {:.4f}".format(train_loss_g))
        # 记录各类loss值
        losses.append((train_loss_d, train_loss_d_real, train_loss_d_fake, train_loss_g))

        # 抽取样本后期进行观察
        sample_noise = np.random.uniform(-1, 1, size=(N_SAMPLE, NOISE_SIZE))

        with tf.variable_scope("generator", reuse=True):
            # hidden layer
            hidden1 = tf.layers.dense(noise_image, G_UNITS)
            # leaky ReLU
            hidden1 = tf.maximum(ALPHA * hidden1, hidden1)
            # dropout
            hidden1 = tf.layers.dropout(hidden1, rate=0.2)

            # logits & outputs
            t_logits = tf.layers.dense(hidden1, IMAGE_SIZE)
            t_outputs = tf.tanh(t_logits)

        gen_samples = sess.run([t_logits, t_outputs],
                               feed_dict={noise_image: sample_noise})
        samples.append(gen_samples)

        # 存储checkpoints
       # / saver.save(sess, 'generator.ckpt')

    # 将sample的生成数据记录下来
with open('train_samples.pkl', 'wb') as f:
    pickle.dump(samples, f)