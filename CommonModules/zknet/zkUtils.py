from xml.etree.ElementTree import parse
from zknet.zkLayer import zkLayer
from zknet.zkLayer import InputLayer, ConvLayer, LrnLayer, MaxPoolLayer, FlattenLayer, \
    FullyConnectLayer, DropOutLayer, AvergePoolLayer, MergeLayer, BatchNormLayer, ResnetLayer, \
    PadLayer, TransposeLayer, RecordLayer, SelfLayer
layerOpt = {
    "inp" : InputLayer,
    "pad" : PadLayer,
    "con" : ConvLayer,
    "lrn" : LrnLayer,
    "max" : MaxPoolLayer,
    "fla" : FlattenLayer,
    "ful" : FullyConnectLayer,
    "drp" : DropOutLayer,
    "avg" : AvergePoolLayer,
    "ewl" : MergeLayer,
    "bnl" : BatchNormLayer,
    "res" : ResnetLayer,
    "tra" : TransposeLayer,
    "rec" : RecordLayer,
    "slf" : SelfLayer
}
def print_node(node):
    '''''打印结点基本信息'''
    print("==============================================")
    print("node.attrib:%s" % node.attrib)
    if "age" in node.attrib:
        print("node.attrib['age']:%s" % node.attrib['age'])
    print("node.tag:%s" % node.tag)
    print("node.text:%s" % node.text)

def create_network(filePath, UserDefinedLayer={}):
    root = parse(filePath)
    trainConfig = root.find("TrainConfig")
    meta = dict()
    for child in trainConfig:
        if child.tag == 'learning_rate' and len(child) > 0:
            learning_meta = dict()
            for ll in child:
                learning_meta[ll.tag] = ll.text
            meta[child.tag] = learning_meta
        else:
            meta[child.tag] = child.text

    LayerNodeList = root.findall("NetConfig/Layer")
    count = {}
    layers = list()
    for LayerNode in LayerNodeList:
        if 'type' not in LayerNode.attrib:
            if 'Loop' not in LayerNode.attrib:
                loop = 1
            else:
                loop = LayerNode.attrib['Loop']

            for index in range(int(loop)):
                for child in LayerNode:
                    dealLayers(child, count, layers, meta, UserDefinedLayer)
        else:
            dealLayers(LayerNode, count, layers, meta, UserDefinedLayer)

    return meta, dict(), layers

def dealLayers(LayerNode, count, layers, meta, UserDefinedLayer ):
    type_vec = LayerNode.attrib['type']
    data = LayerNode.attrib.copy()
    # del data['type']

    if 'Loop' not in LayerNode.attrib:
        loop = 1
    else:
        loop = LayerNode.attrib['Loop']
    if type_vec == "inp":
        loop = 1

    for index in range(int(loop)):
        if type_vec in count :
            count[type_vec] += 1
        else:
            count[type_vec] = 1
        vec = type_vec + str(count[type_vec])
        data['name'] = vec
        if (type_vec == "usd"):
            op_class = UserDefinedLayer.get(LayerNode.attrib['class'], zkLayer)
        else:
            op_class = layerOpt.get(type_vec, zkLayer)

        layer = op_class(vec, data)
        if type_vec == 'ewl':
            mylayer = parseEWL(vec, LayerNode)
            layer.subLayers = mylayer
        layers.append(layer)
        if type_vec == "inp":
            meta['batch_size'] = layer.batch_size
            meta['image_size'] = layer.size
            meta['image_channel'] = layer.channel

def parseEWL(parentName, ewlNode):
    index = 0
    total_layer = []
    for child in ewlNode:
        if 'type' not in child.attrib:
            name = parentName + "_" + str(index)
            total_layer.append(parseEWL(name, child))
        else:
            sub_vec = child.attrib['type']
            sub_op_class = layerOpt.get(sub_vec, zkLayer)

            data = child.attrib
            del data['type']
            name = parentName + "_" + str(index) + sub_vec
            data['name'] = name

            sub_layer = sub_op_class(name, data)
            total_layer.append(sub_layer)
            if sub_vec == 'ewl':
                llLayer = parseEWL(name, child)
                sub_layer.subLayers = llLayer
        # total_layer.append(mylayer[0])
        index += 1
    return total_layer

# def my_obj_pairs_hook(lst):
#     result={}
#     count={}
#     for key,val in lst:
#         if key in count:
#             count[key]=1+count[key]
#         else:
#             count[key]=1
#
#         if key in result:
#            result[key + str(count[key] - 1)]=val
#         else:
#             result[key]=val
#     return result
#
# def parseJson(data):
#     for idx, vec in enumerate(data):
#         d = data[vec]
#         yield vec, d
#
# def parseJsonFromFile(model):
#     with open(model) as json_file:
#         data = json.load(json_file, object_pairs_hook=my_obj_pairs_hook)
#         vec, data = parseJson(data)
#
#     return vec, data
#
# def create_network(model):
#     layers = list()
#     meta = dict()
#     loss_meta = dict()
#     for vec, data in parseJsonFromFile(model):
#         if vec == "TrainConfig":
#             meta = data
#         elif vec == "LossConfig":
#             loss_meta = data
#         else:
#             for vec, data in parseJson(data):
#                 type_vec = vec[0:3]
#                 op_class = layerOpt.get(type_vec, zkLayer)
#                 layer = op_class(vec, data)
#                 if type_vec == 'ewl':
#                     mylayer = parseEWL(vec, data['merge'])
#                     layer.subLayers = mylayer
#                 layers.append(layer)
#                 if type_vec == "inp":
#                     meta['batch_size'] = layer.batch_size
#                     meta['image_size'] = layer.size
#                     meta['image_channel'] = layer.channel
#
#     return meta, loss_meta, layers
#
# def parseEWL(parentName, mergeNode):
#     index = 0
#     total_layer = []
#     for data in mergeNode:
#         mylayer = []
#         for sub_vec, sub_data in parseJson(data):
#             sub_type_vec = sub_vec[0:3]
#             sub_op_class = layerOpt.get(sub_type_vec, zkLayer)
#             name = parentName + "_" + str(index) + sub_vec
#             sub_layer = sub_op_class(parentName + "_" + str(index) + sub_vec, sub_data)
#             mylayer.append(sub_layer)
#             if sub_type_vec == 'ewl':
#                 llLayer = parseEWL(name, sub_data['merge'])
#                 sub_layer.subLayers = llLayer
#
#         total_layer.append(mylayer)
#         index += 1
#
#     return total_layer