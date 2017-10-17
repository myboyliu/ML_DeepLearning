from zkreader.Reader import Reader, CifarReader, MnistReader, Oxflower17Reader,\
    CatDogReader, Voc2007Reader, STLReader

readOpt = {
    "cifar10" : CifarReader,
    "cifar20" : CifarReader,
    "cifar100" : CifarReader,
    "mnist" : MnistReader,
    "flower" : Oxflower17Reader,
    "catdog" : CatDogReader,
    "voc" : Voc2007Reader,
    "stl" : STLReader
}

def reader_new(meta):
    reader = readOpt.get(meta['dataType'], Reader)
    return reader(meta)