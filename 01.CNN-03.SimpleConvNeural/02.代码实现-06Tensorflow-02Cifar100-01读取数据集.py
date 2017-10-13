'''
Cifar10的读取是通过Cifar10_input读取的，但是它只能读取cifar10的数据，不能读取cifar100的数据，而且它读出来的数据
都是24*24大小的图片，我们希望找到一个办法可以读取cifar10与cifar100，并且图片大小为32*32的
cifar100有100个类，每个类有600个图像，其中包含训练集500以及测试集100.这100个类又被分组为20个大类
'''
import os

import numpy as np
import tensorflow as tf
import tensorflow_network.basic_net as bn

import cifarTools as ctf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

learning_rate_init = 0.001
training_epochs = 5
batch_size = 100
display_step = 10
conv1_kernel_num = 64
conv2_kernel_num = 64
fc1_units_num = 1024
fc2_units_num = 512

dataset_dir_cifar = '../Total_Data/cifar_tfrecords'
num_examples_per_epoch_for_train = ctf.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
num_examples_per_epoch_for_eval = ctf.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
image_size = ctf.IMAGE_SIZE
image_channel = ctf.IMAGE_DEPTH

cifar10or20or100 = 100
if cifar10or20or100 == 10:
    n_classes = ctf.NUM_CLASSES_CIFAR10
    dataset_dir = dataset_dir_cifar
elif cifar10or20or100 == 20:
    n_classes = ctf.NUM_CLASSES_CIFAR20
    dataset_dir = dataset_dir_cifar
else:
    n_classes = ctf.NUM_CLASSES_CIFAR100
    dataset_dir = dataset_dir_cifar

def get_distored_train_batch(data_dir, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')

    data_dir = os.path.join(data_dir, 'train_%i_package.tfrecords' % cifar10or20or100)
    images, labels = ctf.readFromTFRecords(
        data_dir, batch_size=batch_size,
        img_shape=[image_size,image_size,image_channel])
    return images, labels

def get_undistorted_eval_batch(data_dir, eval_data, batch_size):
    if not data_dir:
        raise ValueError('Please supply a data_dir')
    data_dir = os.path.join(data_dir, 'test_%i_package.tfrecords' % cifar10or20or100)
    images, labels = ctf.readFromTFRecords(
        data_dir, batch_size=batch_size,
        img_shape=[image_size,image_size,image_channel])
    return images, labels

def Inference(images_holder): # images_holder : [32*32*3]
    with tf.name_scope('Conv2d_1'):
        weights = bn.WeightsVariable(shape=[5,5, image_channel, conv1_kernel_num],
                                  name_str='weights', stddev=5e-2)
        biases = bn.BiasesVariable(shape=[conv1_kernel_num], name_str='biases',
                                init_value=0.0)
        conv1_out = bn.Conv2d(images_holder, weights, biases, stride=1, padding='SAME')
        bn.AddActivationSummary(conv1_out)

    with tf.name_scope('Pool2d_1'):
        pool1_out = bn.Pool2d(conv1_out, pool=tf.nn.max_pool,
                           k=3, stride=2, padding='SAME') # pool1_out : [batch_size, 16, 16, conv1_kernel_num]

    with tf.name_scope('Conv2d_2'):
        weights = bn.WeightsVariable(shape=[5,5, conv1_kernel_num, conv2_kernel_num], name_str='weights',
                                  stddev=5e-2)
        biases = bn.BiasesVariable(shape=[conv2_kernel_num], name_str='biases', init_value=0.1)
        conv2_out = bn.Conv2d(pool1_out, weights, biases, stride=1, padding='SAME') # [batch_size, 16, 16, conv2_kernel_num]
        bn.AddActivationSummary(conv2_out)

    with tf.name_scope('Pool2d_2'):
        pool2_out = bn.Pool2d(conv2_out, pool=tf.nn.max_pool, k=3, stride=2,
                           padding='SAME') # [batch_size, 8, 8, conv2_kernel_num]

    with tf.name_scope('FeatsReshape'):
        features = tf.reshape(pool2_out, [batch_size, -1])
        feats_dim = features.get_shape()[1].value # 8*8*conv2_kernel_num

    with tf.name_scope('FC1_nonlinear'):
        weights = bn.WeightsVariable(shape=[feats_dim, fc1_units_num], name_str='weights', stddev=0.04)
        biases = bn.BiasesVariable(shape=[fc1_units_num], name_str='biases', init_value=0.1)
        fc1_out = bn.FullyConnected(features, weights, biases, activate=tf.nn.relu, act_name='relu')
        bn.AddActivationSummary(fc1_out)

    with tf.name_scope('FC2_nonlinear'):
        weights = bn.WeightsVariable(shape=[fc1_units_num, fc2_units_num], name_str='weights', stddev=0.04)
        biases = bn.BiasesVariable(shape=[fc2_units_num], name_str='biases', init_value=0.1)
        fc2_out = bn.FullyConnected(fc1_out, weights, biases, activate=tf.nn.relu, act_name='relu')
        bn.AddActivationSummary(fc2_out)

    with tf.name_scope('FC3_linear'):
        fc3_units_num = n_classes
        weights = bn.WeightsVariable(shape=[fc2_units_num, fc3_units_num],
                                  name_str='weights', stddev=1.0/fc2_units_num)
        biases = bn.BiasesVariable(shape=[fc3_units_num], name_str='biases', init_value=0.0)
        logits = bn.FullyConnected(fc2_out, weights, biases, activate=tf.identity, act_name='linear')
        bn.AddActivationSummary(logits)
    return logits

with tf.Graph().as_default():
    with tf.name_scope('Inputs'):
        images_holder = tf.placeholder(tf.float32, [batch_size, image_size, image_size, image_channel], name='images')
        labels_holder = tf.placeholder(tf.int32, [batch_size], name='labels')

    with tf.name_scope('Inference'):
        logitis = Inference(images_holder)

    with tf.name_scope('Loss'):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_holder, logits=logitis)
        cross_entropy_loss = tf.reduce_mean(cross_entropy, name='xentropy_loss')
        total_loss = cross_entropy_loss
        average_losses = bn.AddLossesSummary([total_loss])

    with tf.name_scope('Train'):
        learning_rate = tf.placeholder(tf.float32)
        global_step = tf.Variable(0, name='global_step', trainable=False, dtype=tf.int64)
        optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(total_loss, global_step=global_step)

    with tf.name_scope('Evaluate'):
        top_K_op = tf.nn.in_top_k(predictions=logitis, targets=labels_holder, k=1)

    with tf.name_scope('GetTrainBatch'):
        images_train, labels_train = get_distored_train_batch(data_dir=dataset_dir,batch_size=batch_size)
        tf.summary.image('images', images_train, max_outputs=9)

    with tf.name_scope('GetTestBatch'):
        images_test, labels_test = get_undistorted_eval_batch(data_dir=dataset_dir, eval_data=True, batch_size=batch_size)
        tf.summary.image('images', images_test, max_outputs=9)

    merged_summaries = tf.summary.merge_all()
    init_op = tf.global_variables_initializer()

    # print('把计算图写入事件文件')
    # summary_writer = tf.summary.FileWriter(logdir='../logs')
    # summary_writer.add_graph(graph=tf.get_default_graph())
    # summary_writer.flush()

    results_list = list()
    results_list.append(['learning_rate', learning_rate_init,
                         'training_epochs', training_epochs,
                         'batch_size', batch_size,
                         'display_step', display_step,
                         'conv1_kernel_num', conv1_kernel_num,
                         'conv2_kernel_num', conv2_kernel_num,
                         'fc1_units_num', fc1_units_num,
                         'fc2_units_num', fc2_units_num])
    results_list.append(['train_step', 'train_loss', 'train_step', 'train_accuracy'])

    with tf.Session() as sess:
        sess.run(init_op)
        tf.train.start_queue_runners()
        print('==>>>>>>>>>>开始在训练集上训练模型<<<<<<<<<<==')
        num_batches_per_epoch = int(num_examples_per_epoch_for_train / batch_size) # 500
        print('Per batch Size: ', batch_size)
        print('Train sample Count Per Epoch: ', num_examples_per_epoch_for_train)
        print('Total batch Count Per Epoch: ', num_batches_per_epoch)

        training_step = 0

        for epoch in range(training_epochs):
            for batch_idx in range(num_batches_per_epoch):
                images_batch, labels_batch = sess.run([images_train, labels_train]) # 获取一个批次的数据
                _, loss_value = sess.run([train_op, total_loss],
                                                     feed_dict={images_holder: images_batch,
                                                                labels_holder: labels_batch,
                                                                learning_rate : learning_rate_init})
                training_step = sess.run(global_step)
                if training_step % display_step == 0:
                    predictions = sess.run([top_K_op], feed_dict={images_holder: images_batch,
                                                                  labels_holder: labels_batch})
                    batch_accuracy = np.sum(predictions) * 1.0 / batch_size
                    results_list.append([training_step, loss_value, training_step, batch_accuracy])
                    print("Training Epoch: " + str(epoch) +
                          ", Training Step: " + str(training_step) +
                          ", Training Loss= " + "{:.6f}".format(loss_value) +
                          ", Training Accuracy= " + "{:.5f}".format(batch_accuracy))
                    summaries_str = sess.run(merged_summaries, feed_dict={images_holder:images_batch,
                                                                          labels_holder:labels_batch})
                    # summary_writer.add_summary(summary=summaries_str, global_step=training_step)
                    # summary_writer.flush()

        # summary_writer.close()
        print('训练完毕')

        print('==>>>>>>>>>>==开始在测试集上评估模型==<<<<<<<<<<==')
        total_batches = int(num_examples_per_epoch_for_eval / batch_size)
        total_examples = total_batches * batch_size
        print('Per batch Size: ', batch_size)
        print('Test sample Count Per Epoch: ', total_examples)
        print('Total batch Count Per Epoch: ', total_batches)

        correct_predicted = 0
        for test_step in range(total_batches):
            images_batch, label_batch = sess.run([images_test, labels_test])
            predictions = sess.run([top_K_op], feed_dict={images_holder: images_batch,
                                                          labels_holder: label_batch})
            correct_predicted += np.sum(predictions)
        accuracy_score = correct_predicted / total_examples
        print('--------->Accuracy on Test Examples: ', accuracy_score)
        results_list.append(['Accuracy on Test Examples: ', accuracy_score])