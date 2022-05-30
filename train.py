import os

import torch
from tqdm import trange
from predrnn_pytorch import RNN
from utils import getTruePathsFromCsv, CustomTrainImageDataset
from torch.optim import  AdamW
from tqdm import *

from utils import  calculate_fid
from timm.utils import  AverageMeter
import torchio as tio
from torch.cuda.amp import autocast, GradScaler
import  numpy as np
def train(args):

    print(torch.cuda.is_available())
    dataPath = os.path.join("data","Train")
    csvPath = os.path.join(dataPath,"train.csv")

    target_size = (args.seq_length, args.img_w, args.img_h)

    resize =    tio.Resize(target_shape=target_size)
    trainTransform = tio.Compose([
        # resize,
    ])


    dataPaths = getTruePathsFromCsv(dataPath, csvPath)
    for key in dataPaths.keys():

        print("training {}".format(key))
        device, model = get_model(args)
        model = torch.nn.DataParallel(model,args.gpu_ids,args.gpu_ids[0])
        train_loader, valid_loader = getDataLoaders(args, dataPaths, key, trainTransform)
        warmup_epochs = 2
        T_mult = 2
        optimizer = eval(args.optimizer)(model.parameters(), lr=args.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=warmup_epochs,
                                                                         T_mult=T_mult)
        lossFun = torch.nn.MSELoss()
        epochs=args.epochs
        scaler = GradScaler()
        # best_fid = 100000
        best_loss =100
        with trange(epochs) as t:
            for epoch in t:
                train_losses,train_fids,valid_fids,valid_losses, = AverageMeter(),AverageMeter(),AverageMeter(),AverageMeter()
                t.set_description('Epoch %i' % epoch)
                pbar = tqdm(total=len(train_loader))

                for iter, (img,label) in enumerate(train_loader):
                    pbar.set_description('Iter  %i' % iter)
                    img = img.unsqueeze(2).to(device)
                    label = label.unsqueeze(2).to(device)
                    optimizer.zero_grad()
                    with autocast():
                        output = model(img)
                        loss = lossFun(output,label)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # print("input max:{} input min:{}  output manx:{} output min {}".format(torch.max(img),torch.min(img),
                    #                                                                        torch.max(output),torch.min(output)))
                    scheduler.step()
                    train_losses.update(loss.item(),args.batch_size)
                    # if iter %  int(total_iter/10) == 0:
                    #     fid = 0
                        # label = label.cpu().detach().squeeze(2).numpy()
                        # output = output.cpu().detach().squeeze(2).numpy()
                        # for i in range(len(label)):
                        #     for j in range(len(label[i])):
                        #         fid += calculate_fid(label[i][j],output[i][j])
                        # train_fids.update(fid,args.batch_size)
                        # print(" training iter {} / total {}  || epoch {} || loss {} || fid {}".
                        #       format(iter,len(train_loader),epoch,train_losses.avg,train_fids.avg))
                    pbar.set_postfix({"Train loss ": loss.item()})

                    pbar.update(1)
                print(" training iter {} / total {}  || epoch {} || loss {} ".
                      format(iter,len(train_loader),epoch,train_losses.avg))
                for iter, (img, label) in enumerate(valid_loader):
                    with torch.no_grad():
                        img = img.unsqueeze(2).to(device)
                        label = label.unsqueeze(2).to(device)
                        with autocast():
                            output = model(img)
                            loss = lossFun(output, label)
                        valid_losses.update(loss.item(), args.batch_size)
                        # if iter % int(total_iter / 10) == 0:
                        #     fid = 0
                        #     label = label.cpu().detach().squeeze(2).numpy()
                        #     output = output.cpu().detach().squeeze(2).numpy()
                        #     for i in range(len(label)):
                        #         for j in range(len(label[i])):
                        #             fid += calculate_fid(label[i][j], output[i][j])
                        #         valid_fids.update(fid,args.batch_size)
                        if valid_losses.avg < best_loss:
                            torch.save(model.state_dict(),os.path.join("outModels",key+".pth"))
                            best_loss = valid_losses.avg
                            print("model saved")
                        # print("valid iter {} / total {}  || epoch {} || loss {} ".format(iter, len(valid_loader),
                                                                                            # epoch, loss.item()))


                print("Loss : {}".format(loss.item()))
                t.set_postfix({"Train loss ":train_losses.avg,
                               "LR": scheduler.get_last_lr(),
                               })


def get_model(args):
    seq_length = 20
    img_w, img_h = args.img_w, args.img_h
    target_size = (seq_length, img_w, img_h)
    shape = [args.batch_size, seq_length, 1, img_w, img_h]
    shape.extend(list(target_size))
    numlayers = 4
    num_hidden = args.num_hidden
    device = torch.device("cuda:%d" % 0) if torch.cuda.is_available() else torch.device("cpu")
    radarRNN = RNN(shape=shape,
                   num_layers=numlayers,
                   num_hidden=num_hidden,
                   seq_length=seq_length)
    radarRNN = radarRNN.to(device)
    radarRNN.train()

    return device, radarRNN


def getDataLoaders(args, dataPaths, key, trainTransform):
    singleDataPaths = dataPaths[key]
    splitIndex = int(len(singleDataPaths) * 0.8)
    trainPaths = singleDataPaths[0:splitIndex]
    validPaths = singleDataPaths[splitIndex:]
    factorDict={"radar":70,"precip":35,"wind":10}
    factor = factorDict[key]
    radarTrainDataset = CustomTrainImageDataset(trainPaths, imgTransform=trainTransform,factor=factor)
    train_loader = torch.utils.data.DataLoader(radarTrainDataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=True, prefetch_factor=4, pin_memory=True, drop_last=True)
    radarValidDataset = CustomTrainImageDataset(validPaths, imgTransform=trainTransform,factor=factor)
    valid_loader = torch.utils.data.DataLoader(radarValidDataset, batch_size=args.batch_size, num_workers=args.workers,
                                               shuffle=False, prefetch_factor=4, pin_memory=True, drop_last=True)
    print("train length {}  valid length {}".format(len(trainPaths), len(validPaths)))
    return train_loader, valid_loader


