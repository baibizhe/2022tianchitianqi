from torch.utils.data import Dataset
from matplotlib.pyplot import  imread
import pandas as pd
import os
import numpy as np
import torch
def read_image(first20Path):

    return  np.asarray([imread(singleImg) for singleImg in first20Path])


def read_label(last20Path):
    return  np.asarray([imread(singleImg) for singleImg in last20Path])
class CustomTrainImageDataset(Dataset):
    def __init__(self, allImagePath, imgTransform=None, labelTransform=None):
        self.allImagePath = allImagePath
        self.imgTransform = imgTransform
        self.labelTransform = labelTransform

    def __len__(self):
        return len(self.allImagePath)

    def __getitem__(self, idx):
        image = read_image(self.allImagePath[idx][0:20])
        label = read_image(self.allImagePath[idx][20:])
        return torch.tensor(image), torch.tensor(label)

def getTruePathsFromCsv(dataPath,csvPath):

    csvContent = pd.read_csv(csvPath,header=None)
    precip,radar,wind = [],[],[]
    for i in range(len(csvContent[1])):
        imagesSingleRow = list(csvContent.iloc[i].values)
        radars = list(map(lambda x:os.path.join(dataPath,"Radar","radar_"+x),imagesSingleRow))
        precips = list(map(lambda x:os.path.join(dataPath,"Precip","precip_"+x),imagesSingleRow))
        winds = list(map(lambda x:os.path.join(dataPath,"Wind","wind_"+x),imagesSingleRow))

        precip.append(np.asarray(precips))
        radar.append(np.asarray(radars))
        wind.append(np.asarray(winds))

    return {"precip":precip,"radar":radar,"wind":wind}