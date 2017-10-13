import tensorflow as tf
import tensorlayer as tl
class zkLayer(object):
    def __init__(self, *args):
        self.name = args[0]
        self.data = args[1]
        self.setup()
        # self.input = tl.layers.InputLayer()
        print(self.name)
    def setup(self):
        pass

    def forward(self, layerInput):
        pass

class InputLayer(zkLayer):
    def setup(self):
        size = int(self.data['size']) if 'size' in self.data  else 224
        batch_size = int(self.data['batch_size']) if 'batch_size' in self.data  else 100
        channel = int(self.data['channel']) if 'channel' in self.data  else 3

        self.size = size
        self.batch_size = batch_size
        self.channel = channel
    def forward(self, layerInput):
        return tl.layers.InputLayer(layerInput, name=self.name)

class ConvLayer(zkLayer):
    default_value = [['n_filter', 96],
                     ['filter_size', "1,1"],
                     ['act', "relu"],
                     ['padding', "SAME"],
                     ['strides', "1,1"]]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['n_filter'] = int(self.data['n_filter'])
        self.data['filter_size'] = tuple(map(int, self.data['filter_size'].split(',')))
        self.data['act'] = getattr(tf.nn, self.data['act']) if self.data['act'] != "identity" else getattr(tf, "identity")
        self.data['strides'] = tuple(map(int, self.data['strides'].split(',')))

    def forward(self, layerInput):
        return tl.layers.Conv2d(layerInput, n_filter=self.data['n_filter'],
                                filter_size=self.data['filter_size'],
                                strides=self.data['strides'],
                                padding=self.data['padding'],
                                act=self.data['act'],
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
    def forward(self, layerInput):
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
    def forward(self, layerInput):
        return tl.layers.MaxPool2d(layerInput, filter_size=self.data['filter_size'],
                                   strides=self.data['strides'], padding=self.data['padding'],
                                   name=self.name)
class FlattenLayer(zkLayer):
    def setup(self):
        pass
    def forward(self, layerInput):
        return tl.layers.FlattenLayer(layerInput, name=self.name)

class FullyConnectLayer(zkLayer):
    default_value = [['n_units', 4096],
                     ['act', "relu"]
                    ]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['n_units'] = int(self.data['n_units'])
        self.data['act'] = getattr(tf.nn, self.data['act']) if self.data['act'] != "identity" else getattr(tf, "identity")

    def forward(self, layerInput):
        return tl.layers.DenseLayer(layerInput, n_units=self.data['n_units'], act=self.data['act'], name=self.name)
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
    def forward(self, layerInput):
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
    def forward(self, layerInput):
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

    def forward(self, layerInput):
        list_layer = []
        for layer in self.subLayers:
            if len(layer) == 1:
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
    def forward(self, layerInput):
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

class BatchNormLayer(zkLayer):
    default_value = [['decay',0.9],
                     ['epsilon', 0.00001],
                     ['act', 'identity']]
    def setup(self):
        for v in self.default_value:
            key = v[0]
            defaultValue = v[1]
            if key not in self.data:
                self.data[key] = defaultValue
        self.data['decay'] = float(self.data['decay'])
        self.data['epsilon'] = float(self.data['epsilon'])
        self.data['act'] = getattr(tf.nn, self.data['act']) if self.data['act'] != "identity" else getattr(tf, "identity")
    def forward(self, layerInput):
        return tl.layers.BatchNormLayer(layer=layerInput, decay=self.data['decay'],epsilon=self.data['epsilon'],
                                        act=self.data['act'], is_train=True)