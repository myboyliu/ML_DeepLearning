'''
使用mnist数据集，
输入层数据：data = 28 * 18 * 1
卷积层：conv2d = 5 * 5 * 1 - 输出输出：data = 24 * 24 * K(K就是卷积核的个数)
激活层ReLU- 输出数据：data = 24 * 24 * K
池化层：pool2d - MaxPool 2 * 2 步长S=2 输出数据为12 * 12 * K
全连接层-非线性 units = 1024 输出数据:1024(新增的层)
全连接层-线性 units = 10 输出数据(特征):logits = 1 * 1 * 10(最终需要分10类：0-9)
Softmax：计算属于每个分类的概率
'''
import os

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow_network.basic_net as bn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate_init = 0.001
training_epochs = 1
batch_size = 100
display_step = 10

conv1_kernel_num = 16 #卷积核数量
fc1_units_num = 100 #非线性全连接层神经元个数
keep_prob_init = 0.5
n_input = 784
n_classes = 10

def EvaluateModelOnDataset(sess, images, labels):
    n_samples = images.shape[0]
    per_batch_size = 100
    loss = 0
    acc = 0

    if (n_samples <= per_batch_size): #样本比较少，一次评估
        batch_count = 1
        loss, acc = sess.run([cross_entropy_loss, accuracy],
                             feed_dict={X_origin : images, Y_true : labels, learning_rate : learning_rate_init})
    else: #样本比较大，分批次评估
        batch_count = int(n_samples / per_batch_size)
        batch_start = 0
        for idx in range(batch_count):
            batch_loss, batch_acc = sess.run([cross_entropy_loss, accuracy],
                                             feed_dict={X_origin : images[batch_start:batch_start + per_batch_size, :],
                                                        Y_true: labels[batch_start:batch_start + per_batch_size, :],
                                                        learning_rate : learning_rate_init})
            batch_start += per_batch_size
            loss += batch_loss
            acc += batch_acc

    return loss / batch_count, acc / batch_count

if __name__ == '__main__':
    with tf.Graph().as_default():
        # 输入
        with tf.name_scope('Inputs'):
            X_origin = tf.placeholder(tf.float32, [None, n_input], name='X_origin')
            Y_true = tf.placeholder(tf.float32, [None, n_classes], name='Y_true')
            X_image = tf.reshape(X_origin, [-1,28,28,1])

        #前向推断
        with tf.name_scope('Inference'):
            with tf.name_scope('Conv2d'): # 卷积层
                weights = bn.WeightsVariable(shape=[5,5,1,conv1_kernel_num], name_str='weights')
                biases = bn.BiasesVariable(shape=[conv1_kernel_num], name_str='biases')
                conv_out = bn.Conv2d(X_image, weights, biases, stride=1, padding='VALID')

            with tf.name_scope('Activate'):# 非线性激活层
                activate_out = bn.Activation(conv_out, activation=tf.nn.relu,name='relu')

            with tf.name_scope('Pool2d'): #池化层
                pool_out = bn.Pool2d(activate_out, pool=tf.nn.max_pool, k=2, stride=2)

            with tf.name_scope('FeatsReshape'): #将二维特征图变为一维特征向量，得到的是conv1_kernel_num个特征图，每个特征图是12*12的
                features = tf.reshape(pool_out, [-1, 12 * 12 * conv1_kernel_num])

            with tf.name_scope('FC_ReLU'): #非线性全连接层

                weights = bn.WeightsVariable(shape=[12 * 12 * conv1_kernel_num, fc1_units_num], name_str='weights')
                biases = bn.BiasesVariable(shape=[fc1_units_num], name_str='biases')
                fc1_out = bn.FullyConnected(features, weights, biases,
                                        activate=tf.nn.relu, # 恒等映射，没有经过激活函数，所以没有任何改变
                                        act_name='ReLU')
                with tf.name_scope('L2_Loss'): # 一般不会给卷积层和线性全连接层添加L2_Loss
                    weights_loss = tf.nn.l2_loss(weights)

            with tf.name_scope('FC_Linear'): #全连接层
                weights = bn.WeightsVariable(shape=[fc1_units_num, n_classes], name_str='weights')
                biases = bn.BiasesVariable(shape=[n_classes], name_str='biases')
                Ypred_logits = bn.FullyConnected(fc1_out, weights, biases,
                                              activate=tf.identity, # 恒等映射，没有经过激活函数，所以没有任何改变
                                              act_name='identity')

        #定义损失层
        with tf.name_scope('Loss'):
            wd = 0.00001
            with tf.name_scope('XEntropyLoss'): #交叉熵损失
                cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    labels=Y_true, logits=Ypred_logits
                ))
            with tf.name_scope('TotalLoss'):
                weights_loss = wd * weights_loss
                total_loss = cross_entropy_loss + weights_loss # 经验损失 + 正则化损失

        #定义优化训练层
        with tf.name_scope('Train'):
            learning_rate = tf.placeholder(tf.float32)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
            # optimizer = tf.train.AdagradDAOptimizer(learning_rate=learning_rate, global_step=global_step)
            # optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.ProximalGradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.ProximalAdagradOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
            # optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
            '''
            两个步骤：
            1.反向传播计算梯度
            2.利用梯度下降算法优化权重与偏置
            '''
            #minimize更新参数后，这个global_step参数会自动加一,可以当做调用minimize的次数
            trainer = optimizer.minimize(total_loss, global_step=global_step)

        #定义模型评估层
        with tf.name_scope('Evaluate'):
            correct_pred = tf.equal(tf.argmax(Ypred_logits, 1), tf.argmax(Y_true, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        init = tf.global_variables_initializer()

        # summary_writer = tf.summary.FileWriter(logdir='../logs/05/', graph=tf.get_default_graph())
        # summary_writer.close()
        mnist = input_data.read_data_sets('../Total_Data/MNIST_data/', one_hot=True)
        results_list = list()
        results_list.append(['learning_rate', learning_rate_init,
                             'training_epochs', training_epochs,
                             'batch_size', batch_size,
                             'display_step', display_step,
                             'conv1_kernel_num', conv1_kernel_num])
        results_list.append(['train_step', 'train_loss', 'validation_loss',
                             'train_step', 'train_accuracy', 'validation_accuracy'])
        with tf.Session() as sess:
            sess.run(init)
            total_batches = int(mnist.train.num_examples / batch_size)
            print("Per batch Size: ", batch_size)
            print("Train sample Count: ", mnist.train.num_examples)
            print("Total batch Count: ", total_batches)

            training_step = 0
            for epoch in range(training_epochs):
                for batch_idx in range(total_batches):
                    batch_x, batch_y = mnist.train.next_batch(batch_size)
                    sess.run(trainer, feed_dict={X_origin : batch_x,
                                                 Y_true : batch_y,
                                                 learning_rate : learning_rate_init})
                    # training_step += 1 # global_step跟training_step是一个意思
                    training_step = sess.run(global_step) #跟上一句的效果是一样的
                    if training_step % display_step == 0:
                        start_idx = max(0, (batch_idx - display_step) * batch_size)
                        end_idx = batch_idx * batch_size
                        train_loss, train_acc =EvaluateModelOnDataset(sess,mnist.train.images[start_idx:end_idx, :],
                                                                      mnist.train.labels[start_idx:end_idx, :])
                        print("Training Step: " + str(training_step) +
                              ", Training Loss= " + "{:.6f}".format(train_loss) +
                              ", Training Accuracy= " + "{:.5f}".format(train_acc))

                        validation_loss, validation_acc = EvaluateModelOnDataset(sess, mnist.validation.images,mnist.validation.labels)
                        print("Training Step: " + str(training_step) +
                              ", Validation Loss= " + "{:.6f}".format(validation_loss) +
                              ", Validation Accuracy= " + "{:.5f}".format(validation_acc))

                        l2loss = sess.run(weights_loss) #计算正则化损失

                        results_list.append([training_step, train_loss, validation_loss,l2loss,
                                             training_step, train_acc, validation_acc])

            print("训练完毕")
            test_samples_count = mnist.test.num_examples
            test_loss, test_accuracy = EvaluateModelOnDataset(sess, mnist.test.images, mnist.test.labels)
            print("Testing Samples Count:", test_samples_count)
            print("Testing Loss:", test_loss)
            print("Testing Accuracy:", test_accuracy)
            results_list.append(['test step', 'loss', test_loss, 'accuracy', test_accuracy])