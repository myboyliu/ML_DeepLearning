from zknet.zkNet import zkNet
import tensorflow as tf
import tensorlayer as tl

'''
01. alexnet.xml -> Training Accuracy: 98.4% Testing Accuracy: 81.3%  
02. vgg19.xml
03. resnet.xml  -> Training Accuracy: 100%  Testing Accuracy: 85%
'''
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

    # for one-hot version
    # print('call my loss...')
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label, name='cost')
    # ce = tf.reduce_mean(cross_entropy)
    # L2 = 0
    # for p in tl.layers.get_variables_with_name('relu/W', True, True):
    #     print(p.name)
    #     L2 += tf.contrib.layers.l2_regularizer(0.004)(p)
    #
    # return ce + L2

def accuracyFunction(logits, label,*data):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32))

    # for one-hot version
    # correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # return accuracy

if __name__ == '__main__':
    zknet = zkNet('01.alexnet.xml', LossFunc=lossfunction, AccFunc=accuracyFunction)
    zknet.train()



