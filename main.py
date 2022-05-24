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
    arg("--batch-size", type=int, default=40
        )
    arg("--epochs", type=int, default=1000)
    arg("--lr", type=float, default=0.005)
    arg("--optimizer", type=str, default="AdamW")
    arg("--resumePath", type=str, default='')
    arg(
        "--device-ids",
        type=str,
        default="0",
        help="For example 0,1 to run on two GPUs",
    )
    arg("--workers", type=int, default=6)


    args = parser.parse_args()
    args.dataPath = os.path.join("data","TestA")
    train(args)



if __name__ == '__main__':
    main()

