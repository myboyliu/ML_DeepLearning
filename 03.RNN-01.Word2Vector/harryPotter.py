import collections
import math
import os
import random
import zipfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import re
vocabulary_size = 20000
data_index = 0

def read_file(filename):
    f=open(filename,'r')
    file_read=f.read()
    words_=re.sub("[^a-zA-Z]+", " ",file_read).lower() #正则匹配,只留下单词，且大写改小写
    words=list(words_.split())  #length of words:1121985
    return words

def read_data(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def build_dataset(words):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    #data 编码,词频高的在前，词频低的在后
    #count 词频
    #dictionary 词库
    #reverse_dictionary 反转词汇表
    return data, count, dictionary, reverse_dictionary

def generate_batch(batch_size, num_skips, skip_window):
    #num_skips : 每个单词生成多少样本, 必须小于等于skip_window的2倍，并且batch_size必须是它的整数倍
    #skip_window : 单词最远可以联系的距离，如果为1，那么表示只能跟紧邻的两个单词生成样本
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window
        targets_to_avoid = [ skip_window ]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

filepath = '../Total_Data/03Data/HarryPotter.txt'
# filepath = '../Total_Data/03Data/mytest.txt'
# words = read_data(filepath)
words = read_file(filepath)
print('Data size', len(words))
data, count, dictionary, reverse_dictionary = build_dataset(words)
del words

batch_size = 128
embedding_size = 128
skip_window = 1
num_skips = 2

# batch_inputs 是 batch_size个数字，顺序就是文本文字的顺序，内容就是单词的编码
# batch_labels 是 batch_size个数字，内容是对应位置上batch_inputs的内容在文本中紧邻的单词的编码。
# 比如Harry Potter and the Sorcerer's Stone.txt，这句话，通过编码，我们知道有
# Harry 7, Potter 129, and 2, the 1, Sorcerer 2885 .......，
# 那么batch_inputs就是129, 2, 1, 2885,
# 在文本中，Potter紧邻的单词是Harry和and，那么labels就会有两条记录7, 2,
# 也就形成了下面的形式
# 129 potter -> 7 harry
# 129 potter -> 2 and
# 2 and -> 1 the
# 2 and -> 129 potter
# 1 the -> 2 and
# 1 the -> 2885 sorcerer
# 2885 sorcerer -> 1 the
# 2885 sorcerer -> 15 s
# batch_inputs, batch_labels = generate_batch(batch_size=batch_size, skip_window=skip_window, num_skips=num_skips)
# for i in range(batch_size):
#     print(batch_inputs[i], reverse_dictionary[batch_inputs[i]], '->', batch_labels[i, 0], reverse_dictionary[batch_labels[i, 0]])
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, replace=False)
num_sampled = 64

graph = tf.Graph()
with graph.as_default():
    train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels =tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    with tf.device('/cpu:0'):
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,biases=nce_biases,labels=train_labels, inputs=embed,
                                         num_sampled=num_sampled, num_classes=vocabulary_size))
    optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)
    norm = tf.sqrt(tf.reduce_mean(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()
    num_steps = 100001
    with tf.Session(graph=graph) as session:
        session.run(init)
        average_loss = 0
        for step in range(num_steps):
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs : batch_inputs, train_labels : batch_labels}
            _, loss_val = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % 2000 == 0:
                if step > 0:
                    average_loss /= 2000
                print("Average loss at step ", step, ": ", average_loss)
                average_loss = 0

            if step % 10000 == 0:
                sim = similarity.eval()
                for i in range(valid_size):
                    valid_word = reverse_dictionary[valid_examples[i]]
                    top_k = 8
                    nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                    log_str = "Nearest to %s:" % valid_word
                    for k in range(top_k):
                        close_word = reverse_dictionary[nearest[k]]
                        log_str = "%s %s," % (log_str, close_word)
                    print(log_str)
        final_embeddings = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs,labels,filename):
    #low_dim_embs 降维到2维的单词的空间向量
    assert low_dim_embs.shape[0]>=len(labels),"more labels than embedding"
    plt.figure(figsize=(18,18))
    for i,label in enumerate(labels):
        x,y=low_dim_embs[i,:]
        plt.scatter(x, y)
        #展示单词本身
        plt.annotate(label,xy=(x,y),xytext=(5,2),textcoords='offset points',ha='right',va='bottom')
    plt.savefig(filename)

'''''
tsne实现降维，将原始的128维的嵌入向量降到2维
'''
print('123213123')
tsne=TSNE(perplexity=30,n_components=2,init='pca',n_iter=5000)
plot_number=100
low_dim_embs=tsne.fit_transform(final_embeddings[:plot_number,:])
labels=[reverse_dictionary[i] for i in range(plot_number)]
plot_with_labels(low_dim_embs, labels, './harryPotter.png')
print(final_embeddings)
print('1231231')

