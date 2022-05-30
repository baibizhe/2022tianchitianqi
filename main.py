import argparse

from config import  config

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
    arg("--batch-size", type=int, default=4
        )
    arg("--epochs", type=int, default=20)
    arg("--lr", type=float, default=0.005)
    arg("--optimizer", type=str, default="AdamW")
    arg("--resumePath", type=str, default='')

    if not os.path.exists("outModels"):
        try:
            os.makedirs("outModels")
        except:
            pass

    args = parser.parse_args()
    config.update(vars(args))
    print(config)

    train(config)



if __name__ == '__main__':
    main()

