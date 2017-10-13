from zknet.zkNet import zkNet
import tensorflow as tf
import tensorlayer as tl

def lossfunction(logits, label):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label)
    total_loss = tf.reduce_mean(cross_entropy)
    L2 = 0
    for p in tl.layers.get_variables_with_name('relu/W', True, True):
        print(p.name)
        L2 += tf.contrib.layers.l2_regularizer(0.004)(p)

    return total_loss + L2

def accuracyFunction(logits, label):
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))

if __name__ == '__main__':
    zknet = zkNet('vgg16.cfg', LossFunc=lossfunction, AccFunc=accuracyFunction)
    zknet.train()



