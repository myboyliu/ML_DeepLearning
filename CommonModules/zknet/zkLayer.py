import tensorflow as tf
import tensorlayer as tl

def PrintLog(tensor, *data):
    print("[JJZHK] Input Tensor Shape : %s" % tensor.get_shape())
    pass
class zkLayer(object):
    def __init__(self, *args):

        self.data = args[1]
        self.name = self.data['name']
        self.setup()
    def setup(self):
        pass

    def forward(self, layerInput, *meta):
        pass

class InputLayer(zkLayer):
    def setup(self):
        size = int(self.data['size']) if 'size' in self.data  else 224
        batch_size = int(self.data['batch_size']) if 'batch_size' in self.data  else 100
        channel = int(self.data['channel']) if 'channel' in self.data  else 3

        self.size = size
        self.batch_size = batch_size
        self.channel = channel
    def forward(self, layerInput, *meta):
        PrintLog(layerInput)
        return tl.layers.InputLayer(layerInput, name=self.name)

class PadLayer(zkLayer):
    def setup(self):
        self.padding = self.data['padding'] if 'padding' in self.data else "0,0;0,0;0,0;0,0"

        self.padding = self.padding.split(';')
        self.padding = [[int(j) for j in i.split(',')] for i in self.padding]
    def forward(self, layerInput, *meta):
        # PrintLog(layerInput)
        return tl.layers.PadLayer(layerInput, paddings=self.padding, name=self.name)

class ConvLayer(zkLayer):
    default_value = [['n_filter', 96],
                     ['filter_size', "1,1"],
                     ['act', "relu"],
                     ['padding', "SAME"],
                     ['strides', "1,1"],
                     ['w_stddev', 0.02],
                     ['w_mean', 0.0],
                     ['b_value', '0.0'],
                     ['act_alpha', 0.1]]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['n_filter'] = int(self.data['n_filter'])
        self.data['filter_size'] = tuple(map(int, self.data['filter_size'].split(',')))

        self.data['strides'] = tuple(map(int, self.data['strides'].split(',')))
        self.data['w_stddev'] = float(self.data['w_stddev'])
        self.data['w_mean'] = float(self.data['w_mean'])
        self.data['b_value'] = float(self.data['b_value'])

        if 'act' in self.data and self.data['act'] == 'leaky_relu': # leaky_relu
            self.data['act_alpha'] = float(self.data['act_alpha'])
            self.data['act'] = lambda x : tl.act.leaky_relu(x, alpha=self.data['act_alpha'])
        else:
            self.data['act'] = getattr(tf.nn, self.data['act']) if self.data['act'] != "identity" else getattr(tf, "identity")

    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return tl.layers.Conv2d(layerInput, n_filter=self.data['n_filter'],
                                filter_size=self.data['filter_size'],
                                strides=self.data['strides'],
                                padding=self.data['padding'],
                                act=self.data['act'],
                                W_init=tf.truncated_normal_initializer(mean=self.data['w_mean'], stddev=self.data['w_stddev']),
                                b_init=tf.constant_initializer(value=self.data['b_value']),
                                name=self.name)
class LrnLayer(zkLayer):
    default_value = [['depth_radius', 4.0],
                     ['bias', 1.0],
                     ['alpha', 0.00001],
                     ['beta', 0.75]
                    ]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['depth_radius'] = float(self.data['depth_radius'])
        self.data['bias'] = float(self.data['bias'])
        self.data['alpha'] = float(eval(self.data['alpha']))
        self.data['beta'] = float(self.data['beta'])
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return tl.layers.LocalResponseNormLayer(layerInput, depth_radius=self.data['depth_radius'], bias=self.data['bias'],
                                                alpha=self.data['alpha'], beta=self.data['beta'], name=self.name)

class MaxPoolLayer(zkLayer):
    default_value = [['filter_size',"1,1"],
                     ['strides', "1,1"],
                     ['padding', "SAME"]
                     ]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['filter_size'] = tuple(map(int, self.data['filter_size'].split(',')))
        self.data['strides'] = tuple(map(int, self.data['strides'].split(',')))
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return tl.layers.MaxPool2d(layerInput, filter_size=self.data['filter_size'],
                                   strides=self.data['strides'], padding=self.data['padding'],
                                   name=self.name)
class FlattenLayer(zkLayer):
    def setup(self):
        pass
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return tl.layers.FlattenLayer(layerInput, name=self.name)

class FullyConnectLayer(zkLayer):
    default_value = [['n_units', 4096],
                     ['act', "relu"],
                     ['act_alpha', 0.1],
                     ['w_stddev', 0.02],
                     ['w_mean', 0.0],
                     ['b_value', '0.0'],
                     ['reshape', False]
                    ]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['n_units'] = int(self.data['n_units'])

        self.data['w_stddev'] = float(self.data['w_stddev'])
        self.data['w_mean'] = float(self.data['w_mean'])
        self.data['b_value'] = float(self.data['b_value'])
        self.data['reshape'] = bool(self.data['reshape'])

        if 'act' in self.data and self.data['act'] == 'leaky_relu': # leaky_relu
            self.data['act_alpha'] = float(self.data['act_alpha'])
            self.data['act'] = lambda x : tl.act.leaky_relu(x, alpha=self.data['act_alpha'])
        else:
            self.data['act'] = getattr(tf.nn, self.data['act']) if self.data['act'] != "identity" else getattr(tf, "identity")

    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        layer = tl.layers.DenseLayer(layerInput, n_units=self.data['n_units'], act=self.data['act'],
                                     W_init=tf.truncated_normal_initializer(mean=self.data['w_mean'], stddev=self.data['w_stddev']),
                                     b_init=tf.constant_initializer(value=self.data['b_value']),
                                     name=self.name)
        if self.data['reshape'] == True:
            recName = self.data['recName']
            recLayer = meta[0]['end_point'][recName]
            return tl.layers.LambdaLayer(layer, lambda x : recLayer.outputs * tf.expand_dims(tf.expand_dims(x, 1), 1), name=self.name)
        else:
            return layer
class DropOutLayer(zkLayer):
    default_value = [['keep', 0.5]
                     ]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['keep'] = float(self.data['keep'])
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return tl.layers.DropoutLayer(layerInput, keep=self.data['keep'],
                                      is_fix=True,
                                      name=self.name)

class AvergePoolLayer(zkLayer):
    default_value = [['filter_size',"1,1"],
                     ['strides', "1,1"],
                     ['padding', "SAME"]
                     ]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['filter_size'] = tuple(map(int, self.data['filter_size'].split(',')))
        self.data['strides'] = tuple(map(int, self.data['strides'].split(',')))
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return tl.layers.MeanPool2d(layerInput, filter_size=self.data['filter_size'],
                                   strides=self.data['strides'], padding=self.data['padding'],
                                   name=self.name)

class MergeLayer(zkLayer):
    default_value = [['combine',"concat"]
                     ]
    subLayers = []
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['combine'] = getattr(tf, self.data['combine'])

    def addSubLayers(self, layer):
        self.subLayers.append(layer)

    def _adjustSubLayers(self):
        layers = self.subLayers.copy()
        self.subLayers.clear()

        for layer in layers:
            if type(layer) != list:
                self.subLayers.append(layer)
            else:
                self.subLayers.append(self._addCommonLayer(layer))

    def _addCommonLayer(self, layers):
        tempLayer = list()
        for layer in layers:
            if type(layer) != list:
                tempLayer.append(layer)
            else:
                list1 = self._addCommonLayer(layer)
                for l in list1:
                    tempLayer.append(l)
        return tempLayer
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)

        self._adjustSubLayers()
        list_layer = []
        for layer in self.subLayers:
            if type(layer) != list:
                list_layer.append(layer.forward(layerInput))
            elif type(layer) == list and len(layer) == 1:
                list_layer.append(layer[0].forward(layerInput))
            else:
                subinput = layerInput
                for sublayer in layer:
                    subinput = sublayer.forward(subinput)

                list_layer.append(subinput)
        if self.data['combine'] == tf.concat:

            return tl.layers.ConcatLayer(layer=list_layer, concat_dim=3, name=self.name)
        else:
            return tl.layers.ElementwiseLayer(layer=list_layer, combine_fn=self.data['combine'], name=self.name)

class ResnetLayer(zkLayer):
    default_value = [['n_filter',16],
                     ['count', 1],
                     ['subsample_factor', 1]]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['n_filter'] = int(self.data['n_filter'])
        self.data['count'] = int(self.data['count'])
        self.data['subsample_factor'] = int(self.data['subsample_factor'])
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return self.Residual_layer(layerInput, self.data['count'], self.data['n_filter'],
                                   self.data['subsample_factor'])

    def Residual_layer(self, network, count, n_filter=16, subsample_factor=1):
        prev_nb_channels = network.outputs.get_shape().as_list()[3]

        if subsample_factor > 1:
            subsample = [1, subsample_factor, subsample_factor, 1]
            name_pool = 'pool_layer' + str(count)
            shortcut = tl.layers.PoolLayer(network, ksize=subsample, strides=subsample,
                                           padding='VALID', pool=tf.nn.avg_pool, name=self.name + "_" + name_pool)
        else:
            subsample = [1,1,1,1]
            shortcut = network

        if n_filter > prev_nb_channels:
            name_lambda = 'lambda_layer' + str(count)
            shortcut = tl.layers.LambdaLayer(shortcut, self.zero_pad_channels, fn_args={'pad': n_filter - prev_nb_channels},
                                             name=self.name + "_" + name_lambda)
        name_norm = 'norm' + str(count)
        y = tl.layers.BatchNormLayer(network, decay=0.999, epsilon=1e-05, is_train=True, name=self.name + "_" + name_norm)
        name_conv = 'conv_layer' + str(count)
        y = tl.layers.Conv2dLayer(y, act=tf.nn.relu, shape=[3,3,prev_nb_channels, n_filter],
                                  strides=subsample, padding='SAME', name=self.name + "_" + name_conv)
        name_norm_2 = 'norm_second' + str(count)
        y = tl.layers.BatchNormLayer(y,
                                     decay=0.999,
                                     epsilon=1e-05,
                                     is_train=True,
                                     name=self.name + "_" + name_norm_2)

        prev_input_channels = y.outputs.get_shape().as_list()[3]
        name_conv_2 = 'conv_layer_second' + str(count)
        y = tl.layers.Conv2dLayer(y,
                                  act=tf.nn.relu,
                                  shape=[3, 3, prev_input_channels, n_filter],
                                  strides=[1, 1, 1, 1],
                                  padding='SAME',
                                  name=self.name + "_" + name_conv_2)

        name_merge = 'merge' + str(count)
        out = tl.layers.ElementwiseLayer([y, shortcut],
                                         combine_fn=tf.add,
                                         name=self.name + "_" + name_merge)


        return out

    def zero_pad_channels(self, x, pad=0):
        pattern = [[0, 0], [0, 0], [0, 0], [pad - pad // 2, pad // 2]]
        return tf.pad(x, pattern)

class ResnetLayer2(zkLayer):
    default_value = [['shape',[16,16]],
                     ['increase_filter', False]]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['shape'] = list(map(int, self.data['shape'].split(',')))
        self.data['increase_filter'] = bool(self.data['increase_filter'])
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return self.Residual_layer(layerInput, self.data['shape'])

    def Residual_layer(self, x,shape,increase_filter=False):
        output_filter_num = shape[1]
        if increase_filter:
            first_stride = (2,2)
        else:
            first_stride = (1,1)

        pre_relu   = tl.layers.BatchNormLayer(x, act=tf.nn.relu, name=self.name + '_bat1')
        conv_1 = tl.layers.Conv2d(pre_relu, n_filter=output_filter_num, filter_size=(3,3), strides=first_stride, padding='SAME', name=self.name + '_con1')
        relu1   = tl.layers.BatchNormLayer(conv_1, act=tf.nn.relu, name=self.name+'_bat2')
        conv_2 =tl.layers.Conv2d(relu1, n_filter=output_filter_num, filter_size=(3,3), strides=first_stride, padding='SAME', name=self.name + "_con2")
        if increase_filter:
            projection = tl.layers.Conv2d(relu1, n_filter=output_filter_num, filter_size=(3,3), strides=first_stride, padding='SAME', name=self.name+"_con3")
            block = tl.layers.ConcatLayer(layer=[conv_2, projection], concat_dim=3, name=self.name + "_mer1")
        else:
            block = tl.layers.ConcatLayer(layer=[conv_2, x], concat_dim=3, name=self.name + "_mer2")
        return block

class BatchNormLayer(zkLayer):
    default_value = [['decay',0.9],
                     ['epsilon', 0.00001],
                     ['act', 'identity'],
                     ['act_alpha', 0.1]]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['decay'] = float(self.data['decay'])
        self.data['epsilon'] = float(self.data['epsilon'])
        if 'act' in self.data and self.data['act'] == 'leaky_relu': # leaky_relu
            self.data['act_alpha'] = float(self.data['act_alpha'])
            self.data['act'] = lambda x : tl.act.leaky_relu(x, alpha=self.data['act_alpha'])
        else:
            self.data['act'] = getattr(tf.nn, self.data['act']) if self.data['act'] != "identity" else getattr(tf, "identity")
    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return tl.layers.BatchNormLayer(layer=layerInput, decay=self.data['decay'],epsilon=self.data['epsilon'],
                                        act=self.data['act'], is_train=True, name=self.name)

class TransposeLayer(zkLayer):
    def setup(self):
        self.data['perm'] = [int(i) for i in self.data['perm'].split(',')]

    def forward(self, layerInput, *meta):
        PrintLog(layerInput.outputs)
        return tl.layers.TransposeLayer(layerInput, perm=self.data['perm'], name=self.name)

class RecordLayer(zkLayer):
    def setup(self):
        pass

    def forward(self, layerInput, *meta):
        sMeta = meta[0]
        if 'end_point' not in sMeta:
            sMeta['end_point'] = {}

        sMeta['end_point'][self.data['recName']] = layerInput
        print("[JJZHK] Record Layer - %s" % self.data['recName'])
        return layerInput

class SelfLayer(zkLayer):
    def setup(self):
        pass

    def forward(self, layerInput, *meta):

        return layerInput

class GlobalAvgLayer(zkLayer):
    def setup(self):
        pass

    def forward(self, layerInput, *meta):
        outputs = layerInput.outputs
        outputs = tf.reduce_mean(outputs, axis=[1,2])

        return tl.layers.InputLayer(outputs, name=self.name)

class GlobalMaxLayer(zkLayer):
    def setup(self):
        pass

    def forward(self, layerInput, *meta):
        outputs = layerInput.outputs
        outputs = tf.reduce_max(outputs, axis=[1,2])

        return tl.layers.InputLayer(outputs, name=self.name)