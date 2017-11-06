import numpy as np

l = [np.arange(7)] * 14
print(l)
l = np.array(l) # 14 * 7, 一共有14个元素，每个元素里面又有7个元素，所以是一个二维数组
print(l)

l = np.reshape(l, (2,7,7)) # 2 * 7 * 7， 一共有2个元素，每个元素里面有7个元素，每个元素里面又有7个元素，所以是三维数组
# print(l)
l = np.transpose(l, (1,2,0)) # 原有shape是(2,7,7),需要将它变为(7,7,2)，相当于shape[0]变到shape[2],shape[1] => shape[0], shape[2] => shape[1]
print(l)