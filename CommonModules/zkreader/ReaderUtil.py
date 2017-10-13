from zkreader.Reader import Reader, CifarReader, MnistReader, Oxflower17Reader, CatDogReader

readOpt = {
    "cifar10" : CifarReader,
    "cifar20" : CifarReader,
    "cifar100" : CifarReader,
    "mnist" : MnistReader,
    "flower" : Oxflower17Reader,
    "catdog" : CatDogReader
}

def reader_new(meta):
    reader = readOpt.get(meta['dataType'], Reader)
    return reader(meta)