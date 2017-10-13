import os
import tensorflow as tf

def readFromTFRecords(fileList, batch_size, img_shape, num_threads=2, min_after_dequeue=1000, IsLabelInt = True, LabelShape=(7,7,25)):
    def read_and_decode(filename_queue, img_shape):
        """Return a single example for queue"""
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label' if IsLabelInt == True else 'label_raw': tf.FixedLenFeature([], tf.int64 if IsLabelInt == True else tf.string),
            }
        )
        # some essential steps
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image = tf.reshape(image, img_shape)    # THIS IS IMPORTANT
        image.set_shape(img_shape)
        image = tf.cast(image, tf.float32) * (1 / 255.0)  # set to [0, 1]

        if IsLabelInt == True:
            sparse_label = tf.cast(features['label'], tf.int32)
        else:
            sparse_label = tf.decode_raw(features['label_raw'], tf.float32)
            sparse_label = tf.reshape(sparse_label, LabelShape)
        #
        return image, sparse_label

    filename_queue = tf.train.string_input_producer(fileList)

    image, sparse_label = read_and_decode(filename_queue, img_shape) # share filename_queue with multiple threads

    # tf.train.shuffle_batch internally uses a RandomShuffleQueue
    images, sparse_labels = tf.train.shuffle_batch(
        [image, sparse_label], batch_size=batch_size, num_threads=num_threads,
        min_after_dequeue=min_after_dequeue,
        capacity=min_after_dequeue + (num_threads + 1) * batch_size
    )

    return images, sparse_labels

class Reader(object):
    def __init__(self, meta):
        self.meta = meta
        self.datatype = meta['dataType']
        self.dataPath = meta['dataPath']
        self.trainFileList = []
        self.testFileList = []
        self.trainRecordCount = 0
        self.testRecordCount = 0
    def readData(self):
        pass

class CifarReader(Reader):
    def __init__(self, meta):
        super(CifarReader,self).__init__(meta)
        self.trainFileList = [os.path.join(self.dataPath, "train_%d.tfrecords" % i) for i in range(5)]
        self.testFileList = [os.path.join(self.dataPath, "test_%d.tfrecords" % i) for i in range(1)]
        self.trainRecordCount = 50000
        self.testRecordCount = 10000
    def readData(self):
        image_train, label_train = readFromTFRecords(self.trainFileList, batch_size=self.meta['batch_size'],
                          img_shape=[
                              self.meta['image_size'],
                              self.meta['image_size'],
                              self.meta['image_channel']
                          ])
        image_test, label_test = readFromTFRecords(self.testFileList, batch_size=self.meta['batch_size'],
                                                   img_shape=[
                                                       self.meta['image_size'],
                                                       self.meta['image_size'],
                                                       self.meta['image_channel']
                                                   ])
        return (image_train,label_train), (image_test, label_test)
class MnistReader(Reader):
    def __init__(self, meta):
        super(MnistReader,self).__init__()
        self.trainFileList = [os.path.join(self.dataPath, "train_%d.tfrecords" % i) for i in range(1)]
        self.testFileList = [os.path.join(self.dataPath, "test_%d.tfrecords" % i) for i in range(1)]
        self.trainRecordCount = 60000
        self.testRecordCount = 10000
    def readData(self):
        image_train, label_train = readFromTFRecords(self.trainFileList, batch_size=self.meta['batch_size'],
                                                     img_shape=[
                                                         self.meta['image_size'],
                                                         self.meta['image_size'],
                                                         self.meta['image_channel']
                                                     ])
        image_test, label_test = readFromTFRecords(self.testFileList, batch_size=self.meta['batch_size'],
                                                   img_shape=[
                                                       self.meta['image_size'],
                                                       self.meta['image_size'],
                                                       self.meta['image_channel']
                                                   ])
        return (image_train,label_train), (image_test, label_test)

class Oxflower17Reader(Reader):
    def __init__(self, meta):
        super(Oxflower17Reader,self).__init__(meta)
        self.trainFileList = [os.path.join(self.dataPath, "train_%d.tfrecords" % i) for i in range(1)]
        self.testFileList = [os.path.join(self.dataPath, "test_%d.tfrecords" % i) for i in range(1)]
        self.trainRecordCount = 1224
        self.testRecordCount = 136
    def readData(self):
        image_train, label_train = readFromTFRecords(self.trainFileList, batch_size=self.meta['batch_size'],
                                                     img_shape=[
                                                         self.meta['image_size'],
                                                         self.meta['image_size'],
                                                         self.meta['image_channel']
                                                     ])
        image_test, label_test = readFromTFRecords(self.testFileList, batch_size=self.meta['batch_size'],
                                                   img_shape=[
                                                       self.meta['image_size'],
                                                       self.meta['image_size'],
                                                       self.meta['image_channel']
                                                   ])
        return (image_train,label_train), (image_test, label_test)

class CatDogReader(Reader):
    def __init__(self, meta):
        super(CatDogReader,self).__init__(meta)
        self.trainFileList = [os.path.join(self.dataPath, "train_%d.tfrecords" % i) for i in range(10)]
        self.testFileList = [os.path.join(self.dataPath, "test_%d.tfrecords" % i) for i in range(1)]
        self.trainRecordCount = 25000
        self.testRecordCount = 1300
    def readData(self):
        image_train, label_train = readFromTFRecords(self.trainFileList, batch_size=self.meta['batch_size'],
                                                     img_shape=[
                                                         self.meta['image_size'],
                                                         self.meta['image_size'],
                                                         self.meta['image_channel']
                                                     ])
        image_test, label_test = readFromTFRecords(self.testFileList, batch_size=self.meta['batch_size'],
                                                   img_shape=[
                                                       self.meta['image_size'],
                                                       self.meta['image_size'],
                                                       self.meta['image_channel']
                                                   ])
        return (image_train,label_train), (image_test, label_test)