import pickle
import numpy as np
import os

NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000 #训练集
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 #测试集，评估集
IMAGE_SIZE = 32
IMAGE_DEPTH = 3

NUM_CLASSES_CIFAR10 = 10
NUM_CLASSES_CIFAR20 = 20
NUM_CLASSES_CIFAR100 = 100

def getCifarClassic(file_dir, classicNumber,cifar10or20or100=10):
    if cifar10or20or100 == 10:
        filename = os.path.join(file_dir, '10.txt')
    elif cifar10or20or100 == 20:
        filename = os.path.join(file_dir, '20.txt')
    else:
        filename = os.path.join(file_dir, '100.txt')
    with open(filename) as file:
        for number, line in enumerate(file.readlines()):
            if (classicNumber == number):
                return line
    return ""

def readDataFromPython(data_dir, train=True, cifar10or20or100=10):
    if cifar10or20or100 == 10:
        if train:
            batches = [pickle.load(open(os.path.join(data_dir, 'data_batch_%d' % i), 'rb'), encoding='bytes') for i in range(1, 6)]
        else:
            batches = [pickle.load(open(os.path.join(data_dir, 'test_batch'), 'rb'), encoding='bytes')]
    else:
        if train:
            batches = [pickle.load(open(os.path.join(data_dir, 'train'), 'rb'), encoding='bytes')]
        else:
            batches = [pickle.load(open(os.path.join(data_dir, 'test'), 'rb'), encoding='bytes')]

    images = np.zeros((NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype=np.uint8) if train else np.zeros((NUM_EXAMPLES_PER_EPOCH_FOR_EVAL, IMAGE_SIZE, IMAGE_SIZE, IMAGE_DEPTH), dtype=np.uint8)
    labels = np.zeros((NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN), dtype=np.int32) if train else np.zeros((NUM_EXAMPLES_PER_EPOCH_FOR_EVAL), dtype=np.int32)
    for i, b in enumerate(batches):
        if cifar10or20or100 == 10:
            for j, l in enumerate(b[b'labels']):
                images[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = b[b'data'][j].reshape([IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = l
        elif cifar10or20or100 == 20:
            for j, l in enumerate(b[b'coarse_labels']):
                images[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = b[b'data'][j].reshape([IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = l
        else:
            for j, l in enumerate(b[b'fine_labels']):
                images[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = b[b'data'][j].reshape([IMAGE_DEPTH, IMAGE_SIZE, IMAGE_SIZE]).transpose([2, 1, 0]).transpose(1, 0, 2)
                labels[i*NUM_EXAMPLES_PER_EPOCH_FOR_EVAL + j] = l
    return images, labels

if __name__ == '__main__':
    dataset_dir = '../Total_Data/cifar_tfrecords'

    # data, label = _convertImageToCifar('filelist', '', IMAGE_SIZE)
    # _convertToTFRecords(data, label, 4, dataset_dir, 'mytest.tfrecords')

    # print(getClassic(dataset_dir, 7, 10))

    # cifar10or20or100 = [[10,'../Total_Data/TempData/cifar-10-batches-py'],
    #                     [20,'../Total_Data/TempData/cifar-100-python'],
    #                     [100,'../Total_Data/TempData/cifar-100-python']]
    # train = [True, False]
    #
    # for i, cifar in enumerate(cifar10or20or100):
    #     for j, t in enumerate(train):
    #         if t:
    #             filename = 'train_%i_package.tfrecords' % cifar[0]
    #         else:
    #             filename = 'test_%i_package.tfrecords' % cifar[0]
    #
    #         if not os.path.exists(os.path.join(cifar[1], filename)):
    #             images, labels = _read_data_batches_py(cifar[1], t, cifar[0])
    #         _convertToTFRecords(images, labels, len(images), dataset_dir, filename)
    #         print(filename + ' done')