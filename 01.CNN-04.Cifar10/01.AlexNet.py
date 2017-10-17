from zknet.zkNet import zkNet
import tensorflow as tf
import tensorlayer as tl

# 训练集 - 87%
# 测试集 - 70.08%

def lossfunction(logits, label,*data):
    # cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label,
    #                                                                logits=logits)
    # total_loss_op = tf.reduce_mean(cross_entropy)
    # return total_loss_op

    print('call my loss...')
    ce = tl.cost.cross_entropy(logits, label, name='cost')
    L2 = 0
    for p in tl.layers.get_variables_with_name('relu/W', True, True):
        print(p.name)
        L2 += tf.contrib.layers.l2_regularizer(0.004)(p)

    return ce + L2

def accuracyFunction(logits, label,*data):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32))

if __name__ == '__main__':
    zknet = zkNet('alexnet.xml', LossFunc=lossfunction, AccFunc=accuracyFunction)
    zknet.train()



