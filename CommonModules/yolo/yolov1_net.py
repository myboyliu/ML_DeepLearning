import numpy as np
import tensorflow as tf
import yolo.yolo_cfg as cfg
import tensorlayer as tl
import tensorflow.contrib.slim as slim

class YOLONet(object):
    def __init__(self, is_training=True):
        self.classes = cfg.CLASSES # 20
        self.num_class = len(self.classes) # 20
        self.image_size = cfg.IMAGE_SIZE
        self.cell_size = cfg.CELL_SIZE # 7
        self.boxes_per_cell = cfg.BOXES_PER_CELL # 2 每个网格预测2个bounding box
        self.output_size = (self.cell_size * self.cell_size) * (self.num_class + self.boxes_per_cell * 5) # 7*7*(5 * 2 + 20)
        self.scale = 1.0 * self.image_size / self.cell_size
        self.boundary1 = self.cell_size * self.cell_size * self.num_class # 7 * 7 * 20
        self.boundary2 = self.boundary1 + self.cell_size * self.cell_size * self.boxes_per_cell # 7*7*20 + 7*7*2
        self.object_scale = cfg.OBJECT_SCALE # 1.0
        self.noobject_scale = cfg.NOOBJECT_SCALE # 1.0
        self.class_scale = cfg.CLASS_SCALE # 2.0
        self.coord_scale = cfg.COORD_SCALE # 5.0

        self.learning_rate = cfg.LEARNING_RATE
        self.batch_size = cfg.BATCH_SIZE
        self.alpha = cfg.ALPHA

        self.offset = np.transpose(np.reshape(np.array(
            [np.arange(self.cell_size)] * self.cell_size * self.boxes_per_cell),
            (self.boxes_per_cell, self.cell_size, self.cell_size)), (1, 2, 0))

        self.images = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 3], name='images')
        self.network, self.logits = self.build_network(self.images, num_outputs=self.output_size, alpha=self.alpha, is_training=is_training)

        # L2 = 0
        # for p in tl.layers.get_variables_with_name('relu/W', True, True):
        #     print(p.name)
        #     L2 += tf.contrib.layers.l2_regularizer(0.004)(p)

        if is_training:
            self.labels = tf.placeholder(tf.float32, [None, self.cell_size, self.cell_size, 5 + self.num_class])


            self.loss_layer(self.logits, self.labels)
            self.total_loss = tf.losses.get_total_loss()
            tf.summary.scalar('total_loss', self.total_loss)
    def build_network(self, images, num_outputs, alpha,keep_prob=0.5, is_training=True, scope='yolo'):
        tl.layers.set_name_reuse(True)
        with tf.variable_scope(scope):
            network = tl.layers.InputLayer(images, name='input')
            network = tl.layers.PadLayer(network, paddings=np.array([[0,0], [3,3], [3,3], [0,0]]), name='1_pad')
            network = tl.layers.Conv2d(network, n_filter=64, filter_size=(7,7), strides=(2,2), padding='VALID',name='1_conv',act=tf.nn.elu,
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.MaxPool2d(network, filter_size=(2,2), padding='SAME', name='1_pool')

            network = tl.layers.Conv2d(network, n_filter=192, filter_size=(3,3), name='2_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.MaxPool2d(network, filter_size=(2,2), padding='SAME', name='2_pool')

            network = tl.layers.Conv2d(network, n_filter=128, filter_size=(1,1), name='3_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(3,3), name='4_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(1,1), name='5_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3,3), name='6_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.MaxPool2d(network, filter_size=(2,2), padding='SAME', name='3_pool')

            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(1,1), name='7_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3,3), name='8_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(1,1), name='9_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3,3), name='10_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(1,1), name='11_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3,3), name='12_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=256, filter_size=(1,1), name='13_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(3,3), name='14_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(1,1), name='15_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=1024, filter_size=(3,3), name='16_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.MaxPool2d(network, filter_size=(2,2), padding='SAME', name='4_pool')

            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(1,1), name='17_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=1024, filter_size=(3,3), name='18_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=512, filter_size=(1,1), name='19_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=1024, filter_size=(3,3), name='20_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=1024, filter_size=(3,3), name='21_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.PadLayer(network, paddings=np.array([[0,0], [1,1], [1,1], [0,0]]), name='2_pad')

            network = tl.layers.Conv2d(network, n_filter=1024, filter_size=(3,3), strides=(2,2), padding='VALID', name='22_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=1024, filter_size=(3,3), name='23_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.Conv2d(network, n_filter=1024, filter_size=(3,3), name='24_conv',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                       W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.TransposeLayer(network, perm=[0,3,1,2], name='1_trans')
            network = tl.layers.FlattenLayer(network, name='1_flat')

            network = tl.layers.DenseLayer(network, n_units=512, name='1_fc',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                           W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.DenseLayer(network, n_units=4096, name='2_fc',act=lambda x : tl.act.leaky_relu(x, alpha=alpha),
                                           W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            network = tl.layers.DropoutLayer(network, keep=keep_prob, is_fix=True, is_train=is_training, name='1_dropout')
            network = tl.layers.DenseLayer(network, n_units=num_outputs, act=tl.act.identity, name='3_fc',
                                           W_init=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
        return network, network.outputs
    def loss_layer(self, predicts, labels, scope='loss_layer'):
        print('---------------------------------')
        # predicts = (?, 1470)
        # labels = (?,7,7,25)
        with tf.variable_scope(scope):
            # 将predicts[0:980]转为[?,7,7,20] 分类的计算
            predict_classes = tf.reshape(predicts[:, :self.boundary1], [self.batch_size, self.cell_size, self.cell_size, self.num_class])
            # 将predicts[980:1078]转为[?,7,7,2] confidence，信心量
            predict_scales = tf.reshape(predicts[:, self.boundary1:self.boundary2], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell])
            # 将predicts[1078:1470]转为[7,7,2,4] BoundingBox计算,2表示每个小区域生成2个BoundingBox，4代表每个小区域的尺寸
            predict_boxes = tf.reshape(predicts[:, self.boundary2:], [self.batch_size, self.cell_size, self.cell_size, self.boxes_per_cell, 4])

            response = tf.reshape(labels[:, :, :, 0], [self.batch_size, self.cell_size, self.cell_size, 1])
            boxes = tf.reshape(labels[:, :, :, 1:5], [self.batch_size, self.cell_size, self.cell_size, 1, 4])
            boxes = tf.tile(boxes, [1, 1, 1, self.boxes_per_cell, 1]) / self.image_size
            classes = labels[:, :, :, 5:]

            offset = tf.constant(self.offset, dtype=tf.float32)
            offset = tf.reshape(offset, [1, self.cell_size, self.cell_size, self.boxes_per_cell])
            offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
            predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / self.cell_size,
                                           (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / self.cell_size,
                                           tf.square(predict_boxes[:, :, :, :, 2]), # 2代表宽度w
                                           tf.square(predict_boxes[:, :, :, :, 3])]) # 3代表高度h
            predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])

            iou_predict_truth = self.calc_iou(predict_boxes_tran, boxes)

            # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
            object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * response

            # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
            noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

            boxes_tran = tf.stack([boxes[:, :, :, :, 0] * self.cell_size - offset,
                                   boxes[:, :, :, :, 1] * self.cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                                   tf.sqrt(boxes[:, :, :, :, 2]),
                                   tf.sqrt(boxes[:, :, :, :, 3])])
            boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

            # class_loss
            class_delta = response * (predict_classes - classes)
            class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * self.class_scale

            # object_loss
            object_delta = object_mask * (predict_scales - iou_predict_truth)
            object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * self.object_scale

            # noobject_loss
            noobject_delta = noobject_mask * predict_scales
            noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * self.noobject_scale

            # coord_loss
            coord_mask = tf.expand_dims(object_mask, 4)
            boxes_delta = coord_mask * (predict_boxes - boxes_tran)
            coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * self.coord_scale

            L2 = 0
            for p in tl.layers.get_variables_with_name('conv/W', True, True):
                L2 += tf.contrib.layers.l2_regularizer(0.005)(p)

            for p in tl.layers.get_variables_with_name('fc/W', True, True):
                L2 += tf.contrib.layers.l2_regularizer(0.005)(p)

            tf.losses.add_loss(class_loss)
            tf.losses.add_loss(object_loss)
            tf.losses.add_loss(noobject_loss)
            tf.losses.add_loss(coord_loss)
            tf.losses.add_loss(L2)

            tf.summary.scalar('class_loss', class_loss) # 类别预测
            tf.summary.scalar('object_loss', object_loss) #含object的box confidence预测
            tf.summary.scalar('noobject_loss', noobject_loss) #不含object的box confidence预测
            tf.summary.scalar('coord_loss', coord_loss) # 坐标预测
            tf.summary.scalar('L2_Loss', L2)

            tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
            tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
            tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
            tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
            tf.summary.histogram('iou', iou_predict_truth)
    def calc_iou(self, boxes1, boxes2, scope='iou'):
        """calculate ious
        Args:
          boxes1: 4-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4]  ====> (x_center, y_center, w, h)
          boxes2: 1-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL, 4] ===> (x_center, y_center, w, h)
        Return:
          iou: 3-D tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        """
        with tf.variable_scope(scope):
            boxes1 = tf.stack([boxes1[:, :, :, :, 0] - boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] - boxes1[:, :, :, :, 3] / 2.0,
                               boxes1[:, :, :, :, 0] + boxes1[:, :, :, :, 2] / 2.0,
                               boxes1[:, :, :, :, 1] + boxes1[:, :, :, :, 3] / 2.0])
            boxes1 = tf.transpose(boxes1, [1, 2, 3, 4, 0])

            boxes2 = tf.stack([boxes2[:, :, :, :, 0] - boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] - boxes2[:, :, :, :, 3] / 2.0,
                               boxes2[:, :, :, :, 0] + boxes2[:, :, :, :, 2] / 2.0,
                               boxes2[:, :, :, :, 1] + boxes2[:, :, :, :, 3] / 2.0])
            boxes2 = tf.transpose(boxes2, [1, 2, 3, 4, 0])

            # calculate the left up point & right down point
            lu = tf.maximum(boxes1[:, :, :, :, :2], boxes2[:, :, :, :, :2])
            rd = tf.minimum(boxes1[:, :, :, :, 2:], boxes2[:, :, :, :, 2:])

            # intersection
            intersection = tf.maximum(0.0, rd - lu)
            inter_square = intersection[:, :, :, :, 0] * intersection[:, :, :, :, 1]

            # calculate the boxs1 square and boxs2 square
            square1 = (boxes1[:, :, :, :, 2] - boxes1[:, :, :, :, 0]) * \
                      (boxes1[:, :, :, :, 3] - boxes1[:, :, :, :, 1])
            square2 = (boxes2[:, :, :, :, 2] - boxes2[:, :, :, :, 0]) * \
                      (boxes2[:, :, :, :, 3] - boxes2[:, :, :, :, 1])

            union_square = tf.maximum(square1 + square2 - inter_square, 1e-10)

        return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)