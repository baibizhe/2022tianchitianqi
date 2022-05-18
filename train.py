import torch

from predrnn_pytorch import RNN

def train(args):
    a = torch.randn(2, 8, 1, 250, 350)
    shape = [2, 8, 1, 250, 350]
    numlayers = 4
    num_hidden = [1, 1, 1, 1]
    seq_length = 20
    predrnn = RNN(shape=shape,
                  num_layers=numlayers,
                  num_hidden=num_hidden,
                  seq_length=seq_length)
    predict = predrnn(a)
    print("numlayers  {} num_hidden {} seq_length              {}".format(numlayers, num_hidden, seq_length))

    print("input shape {} outputshape {}".format(shape, predict.shape))
