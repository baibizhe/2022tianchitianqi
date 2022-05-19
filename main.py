import argparse
import os

import torch

from train import train
import os
import pandas as pd
from utils import  getTruePathsFromCsv
from utils import CustomTrainImageDataset
from predrnn_pytorch import RNN
def main():
    # seed_everything()
    # torch.autograd.set_detect_anomaly(True)
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg("--batch-size", type=int, default=1

        )
    arg("--epochs", type=int, default=1000)
    arg("--lr", type=float, default=0.000005)
    arg("--optimizer", type=str, default="AdamW")
    arg("--resumePath", type=str, default='')
    arg(
        "--device-ids",
        type=str,
        default="0",
        help="For example 0,1 to run on two GPUs",
    )

    args = parser.parse_args()
    args.dataPath = os.path.join("data","TestA")
    train(args)



if __name__ == '__main__':
    
    dataPath = "data/Train"
    csvPath = os.path.join(dataPath,"train.csv")
    shape = [1, 20, 1, 480, 560]
    numlayers = 4
    num_hidden = [1,1,1,1]
    seq_length= 20
    predrnn= RNN(shape=shape, 
                 num_layers=numlayers, 
                 num_hidden=num_hidden,
                 seq_length=seq_length)
    testAThreeDataPaths = getTruePathsFromCsv(dataPath,csvPath)
    radarDataset = CustomTrainImageDataset(testAThreeDataPaths["radar"])
    for enumerate,(batch) in enumerate(radarDataset):
        img = batch[0].unsqueeze(0).unsqueeze(2)
        print("epoch {} || input shape :{} || output shape{}".format(enumerate,img.shape,predrnn(img).shape))

