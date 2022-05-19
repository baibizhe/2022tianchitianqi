import os

import torch

from predrnn_pytorch import RNN
from utils import getTruePathsFromCsv, CustomTrainImageDataset
from torch.optim import  AdamW


def train(args):
    print(torch.cuda.is_available())
    dataPath = os.path.join("data","TestA")
    csvPath = os.path.join(dataPath,"TestA.csv")

    shape = [args.batch_size, 20, 1, 480, 560]
    numlayers = 4
    num_hidden = [2, 2, 2, 2]
    seq_length = 20
    device = torch.device("cuda:%d" % 0)

    radarRNN = RNN(shape=shape,
                   num_layers=numlayers,
                   num_hidden=num_hidden,
                   seq_length=seq_length)
    # for m in radarRNN.modules():
    #     m.cuda()
    radarRNN=radarRNN.to(device)
    testAThreeDataPaths = getTruePathsFromCsv(dataPath, csvPath)
    radarDataset = CustomTrainImageDataset(testAThreeDataPaths["radar"])
    train_loader = torch.utils.data.DataLoader(radarDataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, prefetch_factor=4, pin_memory=True)

    warmup_epochs = 3
    T_mult = 2
    optimizer = eval(args.optimizer)(radarRNN.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                     T_mult=T_mult)
    lossFun = torch.nn.MSELoss() # CE 的loss很容易训飞了
    radarRNN.train()
    for i, (img,label) in enumerate(train_loader):
        img = img.unsqueeze(2)
        label = label.unsqueeze(2).cuda()
        img = img.cuda()
        label= label.cuda()
        optimizer.zero_grad()


        output = radarRNN(img)
        loss = lossFun(output,label)
        loss.backward()
        optimizer.step()
        print(loss.item())

