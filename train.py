import os

import torch
from tqdm import trange
from predrnn_pytorch import RNN
from utils import getTruePathsFromCsv, CustomTrainImageDataset
from torch.optim import  AdamW
from utils import  calculate_fid
from timm.utils import  AverageMeter
import torchio as tio
from torch.cuda.amp import autocast, GradScaler

def train(args):
    print(torch.cuda.is_available())
    dataPath = os.path.join("data","Train")
    csvPath = os.path.join(dataPath,"Train.csv")
    seq_length = 20
    img_w,img_h = 224,224
    target_size =(seq_length,img_w,img_h)
    # shape = [2, 8, 1, 250, 350]  batch*seq_length*chann*w*h
    shape = [args.batch_size, seq_length,1,img_w,img_h ]
    shape.extend(list(target_size))
    numlayers = 4
    num_hidden = [1, 1, 1, 1]
    device = torch.device("cuda:%d" % 0)
    radarRNN = RNN(shape=shape,
                   num_layers=numlayers,
                   num_hidden=num_hidden,
                   seq_length=seq_length)
    # for m in radarRNN.modules():
    #     m.cuda()
    resize =    tio.Resize(target_shape=target_size)
    trainTransform = tio.Compose([
        resize,
    ])

    radarRNN=radarRNN.to(device)

    dataPaths = getTruePathsFromCsv(dataPath, csvPath)
    radarDataPaths = dataPaths["radar"]
    splitIndex = int(len(radarDataPaths) * 0.8)

    trainPaths = radarDataPaths[0:splitIndex]
    validPaths = radarDataPaths[splitIndex:]

    radarTrainDataset = CustomTrainImageDataset(trainPaths,imgTransform=trainTransform)
    train_loader = torch.utils.data.DataLoader(radarTrainDataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, prefetch_factor=4, pin_memory=True,drop_last=True)

    radarValidDataset = CustomTrainImageDataset(validPaths,imgTransform=trainTransform)

    valid_loader = torch.utils.data.DataLoader(radarValidDataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=False, prefetch_factor=4, pin_memory=True,drop_last=True)
    print("train length {}  valid length {}".format(len(trainPaths),len(validPaths)))
    warmup_epochs = 3
    T_mult = 2
    optimizer = eval(args.optimizer)(radarRNN.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                     T_mult=T_mult)
    lossFun = torch.nn.MSELoss()
    radarRNN.train()
    epochs=1000
    scaler = GradScaler()
    best_fid = 10000
    with trange(epochs) as t:
        for epoch in t:
            train_losses,train_fids,valid_fids,valid_losses, = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
            t.set_description('Epoch %i' % epoch)
            total_iter = len(train_loader)
            for iter, (img,label) in enumerate(train_loader):
                img = img.unsqueeze(2).to(device)
                label = label.unsqueeze(2).to(device)
                optimizer.zero_grad()
                with autocast():
                    output = radarRNN(img)
                    loss = lossFun(output,label)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                scheduler.step()
                train_losses.update(loss.item(),args.batch_size)
                if iter %  int(total_iter/10) == 0:
                    fid = 0
                    label = label.cpu().detach().squeeze(2).numpy()
                    output = output.cpu().detach().squeeze(2).numpy()
                    for i in range(len(label)):
                        for j in range(len(label[i])):
                            fid += calculate_fid(label[i][j],output[i][j])
                    train_fids.update(fid,args.batch_size)
                    print(" training iter {} / total {}  || epoch {} || loss {} || fid {}".
                          format(iter,len(train_loader),epoch,train_losses.avg,train_fids.avg))
            for iter, (img, label) in enumerate(valid_loader):
                with torch.no_grad():
                    img = img.unsqueeze(2).to(device)
                    label = label.unsqueeze(2).to(device)
                    with autocast():
                        output = radarRNN(img)
                        loss = lossFun(output, label)
                    valid_losses.update(loss.item(), args.batch_size)
                    if iter % int(total_iter / 10) == 0:
                        fid = 0
                        label = label.cpu().detach().squeeze(2).numpy()
                        output = output.cpu().detach().squeeze(2).numpy()
                        for i in range(len(label)):
                            for j in range(len(label[i])):
                                fid += calculate_fid(label[i][j], output[i][j])
                            valid_fids.update(fid,args.batch_size)
                        print("valid iter {} / total {}  || epoch {} || loss {} || fid {}".format(iter, len(valid_loader),
                                                                                            epoch, loss.item(), fid))


            print("Loss : {}".format(loss.item()))
            t.set_postfix({"Train loss ":train_losses.avg,
                           "LR": scheduler.get_last_lr(),
                           })


