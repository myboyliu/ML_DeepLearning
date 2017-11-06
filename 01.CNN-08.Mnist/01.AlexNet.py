from zknet.zkNet import zkNet
import tensorflow as tf
import tensorlayer as tl
# 训练集 - 87%
# 测试集 - 70.08%

def lossfunction(logits, label,*data):
    cost = tl.cost.cross_entropy(logits, label, name='cost')
    return cost

def accuracyFunction(logits, label,*data):
    return tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=logits, targets=label, k=1), tf.float32))


if __name__ == '__main__':
    zknet = zkNet('alexnet.xml', LossFunc=lossfunction, AccFunc=accuracyFunction)
    zknet.train()



