import json
from zknet.zkLayer import zkLayer
from zknet.zkLayer import InputLayer, ConvLayer, LrnLayer, MaxPoolLayer, FlattenLayer, \
    FullyConnectLayer, DropOutLayer, AvergePoolLayer, MergeLayer, BatchNormLayer, ResnetLayer
layerOpt = {
    "inp" : InputLayer,
    "con" : ConvLayer,
    "lrn" : LrnLayer,
    "max" : MaxPoolLayer,
    "fla" : FlattenLayer,
    "ful" : FullyConnectLayer,
    "drp" : DropOutLayer,
    "avg" : AvergePoolLayer,
    "ewl" : MergeLayer,
    "bnl" : BatchNormLayer,
    "res" : ResnetLayer
}

def my_obj_pairs_hook(lst):
    result={}
    count={}
    for key,val in lst:
        if key in count:
            count[key]=1+count[key]
        else:
            count[key]=1

        if key in result:
           result[key + str(count[key] - 1)]=val
        else:
            result[key]=val
    return result

def parseJson(data):
    for idx, vec in enumerate(data):
        d = data[vec]
        yield vec, d

def parseJsonFromFile(model):
    with open(model) as json_file:
        data = json.load(json_file, object_pairs_hook=my_obj_pairs_hook)
        vec, data = parseJson(data)

    return vec, data

def create_network(model):
    layers = list()
    meta = dict()
    loss_meta = dict()
    for vec, data in parseJsonFromFile(model):
        if vec == "TrainConfig":
            meta = data
        elif vec == "LossConfig":
            loss_meta = data
        else:
            for vec, data in parseJson(data):
                type_vec = vec[0:3]
                op_class = layerOpt.get(type_vec, zkLayer)
                layer = op_class(vec, data)
                if type_vec == 'ewl':
                    mylayer = parseEWL(vec, data['merge'])
                    layer.subLayers = mylayer
                    # index = 0
                    # for subdata in data['merge']:
                    #     mylayer = []
                    #     for sub_vec, sub_data in parseJson(subdata):
                    #         sub_type_vec = sub_vec[0:3]
                    #         sub_op_class = layerOpt.get(sub_type_vec, zkLayer)
                    #         sub_layer = sub_op_class(vec + "_" + str(index) + sub_vec, sub_data)
                    #         mylayer.append(sub_layer)
                    #     index += 1
                    #     layer.addSubLayers(mylayer)
                    # MergeLayer(layer).addSubLayers(layer)
                layers.append(layer)
                if type_vec == "inp":
                    meta['batch_size'] = layer.batch_size
                    meta['image_size'] = layer.size
                    meta['image_channel'] = layer.channel

    return meta, loss_meta, layers

def parseEWL(parentName, mergeNode):
    index = 0
    total_layer = []
    for data in mergeNode:
        mylayer = []
        for sub_vec, sub_data in parseJson(data):
            sub_type_vec = sub_vec[0:3]
            sub_op_class = layerOpt.get(sub_type_vec, zkLayer)
            name = parentName + "_" + str(index) + sub_vec
            sub_layer = sub_op_class(parentName + "_" + str(index) + sub_vec, sub_data)
            mylayer.append(sub_layer)
            if sub_type_vec == 'ewl':
                llLayer = parseEWL(name, sub_data['merge'])
                sub_layer.subLayers = llLayer

        total_layer.append(mylayer)
        index += 1

    return total_layer