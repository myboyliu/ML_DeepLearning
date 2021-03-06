from zknet.zkLayer import zkLayer, ConvLayer, LrnLayer, MaxPoolLayer, FlattenLayer, FullyConnectLayer, DropOutLayer
import zknet.zkUtils as cfg
import tensorlayer as tl
import tensorflow as tf
import numpy as np
from zkreader.ReaderUtil import reader_new
from zknet.timer import Timer
import sys

class zkNet(object):
    optimizerOpt = {
        "adam" : tf.train.AdamOptimizer,
        'gra' : tf.train.GradientDescentOptimizer,
        'mom' : tf.train.MomentumOptimizer,
        'ada' : tf.train.AdagradOptimizer,
        'rms' : tf.train.RMSPropOptimizer,
    }

    learningRateOpt = {
        "exp" : tf.train.exponential_decay,
        "pol" : tf.train.polynomial_decay,
        "nat" : tf.train.natural_exp_decay,
        "inv" : tf.train.inverse_time_decay
    }

    default_value = [['optimizer', 'adam'],
                 ['learning_rate', 0.001],
                 ['gpu_num', 0],
                 ['epoch', 1000],
                 ['dataType', 'cifar10'],
                 ['dataPath', '../Total_Data/cifar10/'],
                 ['isSaveNpzFile', 1],
                 ['NpzFileName', 'saver.npz'],
                 ['trainType', 0],
                 ['displayNum', 10],
                 ['IsOneHot', 0],
                 ['n_classes', 10],
                 ['IsRecordSummary', 1],
                 ['GraphDir', 'logs']
                 ]
    def __init__(self, model, LossFunc = None, AccFunc = None, UserDefinedLayer={}):
        self.meta, self.loss_meta, self.layers = cfg.create_network(model, UserDefinedLayer)
        self.init_meta()
        print('-------------META---------------')
        print(self.meta)
        print('--------------------------------')
        if LossFunc == None:
            self.lossFunc = self.LossFunction
        else:
            self.lossFunc = LossFunc
        if AccFunc == None:
            self.accFunc = self.AccFunction
        else:
            self.accFunc = AccFunc

        if 'accFunction' in self.meta and self.meta['accFunction'] == 'False':
            self.accFunc = None

        self.input = tf.placeholder(dtype=tf.float32, shape=[self.meta['batch_size'], self.meta['image_size'],
                                                             self.meta['image_size'], self.meta['image_channel']], name='images')
        if 'label_shape' in self.meta:
            shape = str(self.meta['batch_size']) + "," + self.meta['label_shape']
            self.label = tf.placeholder(dtype=tf.float32, shape=[int(i) for i in shape.split(',')], name='labels')
        else:
            self.label = tf.placeholder(dtype=tf.int64, shape=[self.meta['batch_size'], ], name='labels') if self.meta['IsOneHot'] == 0 else \
                tf.placeholder(dtype=tf.int64, shape=[self.meta['batch_size'], self.meta['n_classes']], name='labels')

        self.global_step = tf.Variable(initial_value=0, dtype=tf.int64, trainable=False, name='global_step')
        self.reader = reader_new(self.meta)

        if type(self.meta['learning_rate']) is dict:
            learningRateMeta = self.meta['learning_rate']
            learning_rate_init = float(learningRateMeta['init']) if 'init' in learningRateMeta else 0.1
            learning_rate_decay_rate = float(learningRateMeta['decay_rate']) if 'decay_rate' in learningRateMeta else 0.5
            num_examples_per_epoch_for_train = self.reader.trainRecordCount
            num_batches_per_epoch = int(num_examples_per_epoch_for_train / self.meta['batch_size'])
            num_epochs_per_decay = int(learningRateMeta['num_epochs_per_decay']) # 每次过多少个epoch，学习率就会降低
            learning_rate_decay_steps = int(num_batches_per_epoch * num_epochs_per_decay)

            if learningRateMeta["type"] != 'pol':
                self.learning_rate = self.learningRateOpt[learningRateMeta["type"]](
                    learning_rate = learning_rate_init,
                    global_step = self.global_step,
                    decay_steps = learning_rate_decay_steps,
                    decay_rate = learning_rate_decay_rate,
                    staircase = False
                )
            else:
                learning_rate_final = float(learningRateMeta['end_learning_rate']) if 'end_learning_rate' in learningRateMeta else 0.001
                self.learning_rate = self.learningRateOpt[learningRateMeta["type"]](
                    learning_rate = learning_rate_init,
                    global_step = self.global_step,
                    decay_steps = learning_rate_decay_steps,
                    decay_rate = learning_rate_decay_rate,
                    end_learning_rate = learning_rate_final,
                    power = 0.5,
                    cycle = False
                )
        else:
            self.learning_rate = float(self.meta['learning_rate'])

        self.optimizer = self.optimizerOpt[self.meta['optimizer']](learning_rate = self.learning_rate)

        self.build_forward()
    def init_meta(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.meta:
                self.meta[key] = defaultValue
        self.meta['gpu_num'] = int(self.meta['gpu_num'])
        self.meta['epoch'] = int(self.meta['epoch'])
        self.meta['n_classes'] = int(self.meta['n_classes'])
    def build_forward(self):
        tl.layers.set_name_reuse(True)
        out = self.input
        for layer in self.layers:
            out = layer.forward(out, self.meta)

        self.network = out
        self.logits = self.network.outputs

        self.cost = self.lossFunc(self.logits, self.label, self.meta)

        if self.accFunc == None:
            self.acc = None
        else:
            self.acc = self.accFunc(self.logits, self.label, self.meta)

    def LossFunction(self, logits, label, **data):
        ce = tl.cost.cross_entropy(logits, label, name='cost')
        L2 = 0
        for p in tl.layers.get_variables_with_name('relu/W', True, True):
            L2 += tf.contrib.layers.l2_regularizer(0.004)(p)

        return ce + L2
    def AccFunction(self, logits, label, **data):
        correct_prediction = tf.nn.in_top_k(logits, label, k=1)#tf.equal(tf.argmax(logits, 1), y)
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def multigpu(self):
        grdientList = []
        with tf.variable_scope(tf.get_variable_scope()): # 除了GradientDescentOptimizer之外，剩下的优化器都需要加上这一句
            for i in range(self.meta['gpu_num']):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                        tf.get_variable_scope().reuse_variables()
                        grads = self.optimizer.compute_gradients(self.cost)
                        grdientList.append(grads)
        avg_gradient = self.average_gradients(grdientList)
        avg_gradient_op = self.optimizer.apply_gradients(avg_gradient, global_step=self.global_step)
        return avg_gradient_op

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)

            grad = tf.concat(grads, 0)
            grad = tf.reduce_mean(grad, 0)

            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def to_categorical(self, y, nb_classes):
        y = np.asarray(y, dtype='int32')
        if not nb_classes:
            nb_classes = np.max(y)+1
        Y = np.zeros((len(y), nb_classes))
        Y[np.arange(len(y)),y] = 1.
        return Y

    def train(self):
        (images, labels), (images_test, labels_test) = self.reader.readData()
        if int(self.meta['gpu_num']) > 0:
            train_op = self.multigpu()
        else:
            train_op = self.optimizer.minimize(self.cost, global_step=self.global_step)

        if self.meta['IsRecordSummary'] == 1:
            tf.summary.scalar('Loss', self.cost)
            tf.summary.scalar('LearningRate', self.learning_rate)
            if self.acc != None:
                tf.summary.scalar('Accuracy', self.acc)
            tf.summary.image('images', images, max_outputs=9)
            tf.summary.image('images', images_test, max_outputs=9)
        merged_summary = tf.summary.merge_all()
        if self.meta['IsRecordSummary'] == 1:
            summary_writer = tf.summary.FileWriter(logdir=self.meta['GraphDir'])
            summary_writer.add_graph(graph=tf.get_default_graph())
            summary_writer.flush()
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            tl.layers.initialize_global_variables(sess)
            if self.meta['isSaveNpzFile'] == 2: # fune tune
                tl.files.load_and_assign_npz(sess=sess, name=self.meta['NpzFileName'], network=self.network)
                self.network.print_params()
                self.network.print_layers()
                tl.utils.test(sess, self.network, self.acc, images_test, labels_test, self.input, self.label, batch_size=self.meta['batch_size'], cost=self.cost)
            else:
                train_timer = Timer()

                tf.train.start_queue_runners()
                train_count = int(self.reader.trainRecordCount / self.meta['batch_size'])
                for epoch in range(int(self.meta['epoch'])):
                    train_timer.tic()
                    current_learning_rate = sess.run(self.learning_rate)
                    print("Epoch:%d/%d, Learning Rate : %s" % (epoch + 1, int(self.meta['epoch']), "{:.6f}".format(current_learning_rate)))

                    for idx in range(train_count):
                        batch_image, batch_labels = sess.run([images, labels])
                        if self.meta['IsOneHot'] == 1:
                            batch_labels_ex = self.to_categorical(batch_labels, self.meta['n_classes'])
                        else:
                            batch_labels_ex = batch_labels
                        _, loss_value = sess.run([train_op, self.cost], feed_dict={self.input : batch_image,
                                                                                   self.label : batch_labels_ex})

                        if self.meta['IsOneHot'] == 1:
                            batch_labels_ex = self.to_categorical(batch_labels, self.meta['n_classes'])
                        else:
                            batch_labels_ex = batch_labels

                        batch_accuracy = 0.0

                        if self.acc != None:
                            predictions = sess.run([self.acc], feed_dict={self.input:batch_image,
                                                                          self.label : batch_labels_ex})
                            batch_accuracy = np.sum(predictions) * 1.0

                        perCount = int(train_count / 100) # 7
                        percent = int(idx / perCount)

                        if train_count % perCount == 0:
                            dotcount = int(train_count / perCount) - 1
                        else:
                            dotcount = int(train_count / perCount)

                        if percent > 99 :
                            i = 0
                        s1 = "\r[%s%s] %d/%d,Loss=%s,Accuracy=%s"%(

                            "*"*(int(percent)),
                            " "*(dotcount-int(percent)),
                            (idx + 1),
                            (train_count),
                            "{:.3f}".format(loss_value),
                            "{:.3f}".format(batch_accuracy))

                        sys.stdout.write(s1)
                        sys.stdout.flush()

                        # if (idx != 0 and idx % perCount == 0 and percent <= 98):
                        #     s1 = "\r%d/%d[%s%s]%d%%,Loss=%s,Accuracy=%s"%((idx + 1),(train_count),"*"*(int(percent) + 1)," "*(100-int(percent) - 1),(int(percent) + 1),
                        #                                                      "{:.3f}".format(loss_value), "{:.3f}".format(batch_accuracy))
                        #     sys.stdout.write(s1)
                        #     sys.stdout.flush()
                        #     # time.sleep(0.3)
                        # elif (percent >= 99 and idx == train_count - 1):
                        #     percent = 99
                        #     s1 = "\r%d/%d[%s%s]%d%%,Loss=%s,Accuracy=%s"%((idx + 1),(train_count),"*"*(int(percent) + 1)," "*(100-int(percent) - 1),(int(percent) + 1),
                        #                                                   "{:.3f}".format(loss_value), "{:.3f}".format(batch_accuracy))
                        #     sys.stdout.write(s1)
                        #     sys.stdout.flush()

                        if idx % self.meta['displayNum'] == 0 :
                            if self.meta['IsRecordSummary'] == 1:
                                summaries_str = sess.run(merged_summary, feed_dict={self.input : batch_image,
                                                                                    self.label : batch_labels_ex})
                                summary_writer.add_summary(summary=summaries_str, global_step=idx)
                                summary_writer.flush()
                    print("\r")
                    train_timer.toc()
                print('Training Done. Time is : ' + "{:.3f}".format(train_timer.average_time * int(self.meta['epoch'])))
                if self.meta['isSaveNpzFile']:
                    tl.files.save_npz(self.network.all_params , name=self.meta['NpzFileName'])
                total_batches = int(self.reader.testRecordCount / self.meta['batch_size'])
                total_examples = total_batches * self.meta['batch_size']

                correct_predicted = 0
                for test_step in range(total_batches):
                    images_batch, label_batch = sess.run([images_test, labels_test])
                    if self.meta['IsOneHot'] == 1:
                        label_batch = self.to_categorical(label_batch, self.meta['n_classes'])
                    predictions = sess.run([self.acc], feed_dict={self.input:images_batch,
                                                                  self.label : label_batch})
                    correct_predicted += np.sum(predictions)
                accuracy_score = correct_predicted / total_batches
                print('--------->Accuracy on Test Examples: ', accuracy_score)

    def predict(self, image_predict, predFunc):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            pred = predFunc(self.network.outputs)

            imageList = []
            for idx in range(self.meta['batch_size']):
                imageList.append(image_predict)

            result = tl.utils.predict(sess, self.network, image_predict, self.input, pred)
            count = np.bincount(result)
            return np.argmax(count)
    def predict_dectection(self, image_predict):
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            results = []
            net_output = sess.run([self.network.outputs], feed_dict={self.input : image_predict})
            for i in range(net_output.shape[0]):
                results.append(self.interpret_output(net_output[i]))

        return results