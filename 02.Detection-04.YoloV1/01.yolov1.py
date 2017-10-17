from zknet.zkNet import zkNet
import tensorflow as tf
import tensorlayer as tl
import numpy as np

def lossfunction(predicts, labels, *data):
    print('---------------------------------')
    # predicts = (?, 1470)
    # labels = (?,7,7,25)
    meta = data[0]
    cell_size = int(meta['cell_size']) if 'cell_size' in meta else 7
    num_class = meta['n_classes']
    boxes_per_cell = int(meta['boxes_per_cell']) if 'boxes_per_cell' in meta else 2
    batch_size = meta['batch_size']
    image_size = meta['image_size']
    class_scale = float(meta['class_scale'])
    object_scale = float(meta['object_scale'])
    noobject_scale = float(meta['noobject_scale'])
    coord_scale = float(meta['coord_scale'])
    boundary1 = cell_size * cell_size * num_class
    boundary2 = boundary1 + cell_size * cell_size * boxes_per_cell
    offset = np.transpose(np.reshape(np.array(
        [np.arange(cell_size)] * cell_size * boxes_per_cell),
        (boxes_per_cell, cell_size, cell_size)), (1, 2, 0))

    with tf.variable_scope('loss_layer'):
        # 将predicts[0:980]转为[?,7,7,20] 分类的计算
        predict_classes = tf.reshape(predicts[:, :boundary1], [batch_size, cell_size, cell_size, num_class])
        # 将predicts[980:1078]转为[?,7,7,2] confidence，信心量
        predict_scales = tf.reshape(predicts[:, boundary1:boundary2], [batch_size, cell_size, cell_size, boxes_per_cell])
        # 将predicts[1078:1470]转为[7,7,2,4] BoundingBox计算,2表示每个小区域生成2个BoundingBox，4代表每个小区域的尺寸
        predict_boxes = tf.reshape(predicts[:, boundary2:], [batch_size, cell_size, cell_size, boxes_per_cell, 4])

        response = tf.reshape(labels[:, :, :, 0], [batch_size, cell_size, cell_size, 1])
        boxes = tf.reshape(labels[:, :, :, 1:5], [batch_size, cell_size, cell_size, 1, 4])
        boxes = tf.cast(tf.tile(boxes, [1, 1, 1, boxes_per_cell, 1]) / image_size, dtype=tf.float32)
        classes = labels[:, :, :, 5:]

        offset = tf.constant(offset, dtype=tf.float32)
        offset = tf.reshape(offset, [1, cell_size, cell_size, boxes_per_cell])
        offset = tf.tile(offset, [batch_size, 1, 1, 1])
        predict_boxes_tran = tf.stack([(predict_boxes[:, :, :, :, 0] + offset) / cell_size,
                                       (predict_boxes[:, :, :, :, 1] + tf.transpose(offset, (0, 2, 1, 3))) / cell_size,
                                       tf.square(predict_boxes[:, :, :, :, 2]), # 2代表宽度w
                                       tf.square(predict_boxes[:, :, :, :, 3])]) # 3代表高度h
        predict_boxes_tran = tf.transpose(predict_boxes_tran, [1, 2, 3, 4, 0])



        iou_predict_truth = calc_iou(predict_boxes_tran, boxes)

        # calculate I tensor [BATCH_SIZE, CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        object_mask = tf.reduce_max(iou_predict_truth, 3, keep_dims=True)
        object_mask = tf.cast((iou_predict_truth >= object_mask), tf.float32) * tf.cast(response, tf.float32)

        object_mask = tf.cast(object_mask, tf.float32)

        # calculate no_I tensor [CELL_SIZE, CELL_SIZE, BOXES_PER_CELL]
        noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

        boxes_tran = tf.stack([boxes[:, :, :, :, 0] * cell_size - offset,
                               boxes[:, :, :, :, 1] * cell_size - tf.transpose(offset, (0, 2, 1, 3)),
                               tf.sqrt(boxes[:, :, :, :, 2]),
                               tf.sqrt(boxes[:, :, :, :, 3])])
        boxes_tran = tf.transpose(boxes_tran, [1, 2, 3, 4, 0])

        # class_loss
        class_delta = response * (predict_classes - classes)
        class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(class_delta), axis=[1, 2, 3]), name='class_loss') * class_scale

        # object_loss
        object_delta = object_mask * (predict_scales - iou_predict_truth)
        object_loss = tf.reduce_mean(tf.reduce_sum(tf.square(object_delta), axis=[1, 2, 3]), name='object_loss') * object_scale

        # noobject_loss
        noobject_delta = noobject_mask * predict_scales
        noobject_loss = tf.reduce_mean(tf.reduce_sum(tf.square(noobject_delta), axis=[1, 2, 3]), name='noobject_loss') * noobject_scale

        # coord_loss
        coord_mask = tf.expand_dims(object_mask, 4)
        boxes_delta = coord_mask * (predict_boxes - boxes_tran)
        coord_loss = tf.reduce_mean(tf.reduce_sum(tf.square(boxes_delta), axis=[1, 2, 3, 4]), name='coord_loss') * coord_scale

        tf.losses.add_loss(class_loss)
        tf.losses.add_loss(object_loss)
        tf.losses.add_loss(noobject_loss)
        tf.losses.add_loss(coord_loss)
        #
        tf.summary.scalar('class_loss', class_loss) # 类别预测
        tf.summary.scalar('object_loss', object_loss) #含object的box confidence预测
        tf.summary.scalar('noobject_loss', noobject_loss) #不含object的box confidence预测
        tf.summary.scalar('coord_loss', coord_loss) # 坐标预测
        #
        tf.summary.histogram('boxes_delta_x', boxes_delta[:, :, :, :, 0])
        tf.summary.histogram('boxes_delta_y', boxes_delta[:, :, :, :, 1])
        tf.summary.histogram('boxes_delta_w', boxes_delta[:, :, :, :, 2])
        tf.summary.histogram('boxes_delta_h', boxes_delta[:, :, :, :, 3])
        tf.summary.histogram('iou', iou_predict_truth)
    return tf.losses.get_total_loss()

def accuracyFunction(logits, label, *data):
    # correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    return 1 #tf.reduce_mean(tf.cast(correct_pred, tf.float32))
def calc_iou(boxes1, boxes2, scope='iou'):
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

if __name__ == '__main__':
    zknet = zkNet('yoloV1.xml', LossFunc=lossfunction, AccFunc=accuracyFunction)
    zknet.train()



