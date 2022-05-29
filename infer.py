import torch

from check_Format import check_submission_format
from train import  get_model
import os
import  numpy as np
from utils import CustomInferImageDataset
import torchio as tio
from  skimage.io import imread
from  imageio import imwrite
from config import  config
from skimage import img_as_ubyte,img_as_float
import  cv2

def getTruePathForInfer():

    # precip,radar,wind = [],[],[]
    allImgDirs = dict()
    precipDir = os.path.join("data","TestB","TestB1","Precip")
    precipImgDirs = os.listdir(precipDir)
    precipImgDirs = list(
        map(lambda x:os.path.join(precipDir,x),precipImgDirs))
    allImgDirs["precip"] = precipImgDirs

    radarDir = os.path.join("data","TestB","TestB1","Radar")
    radarImgDirs = os.listdir(radarDir)
    radarImgDirs = list(
        map(lambda x:os.path.join(radarDir,x),radarImgDirs))
    allImgDirs["radar"] = radarImgDirs

    windDir = os.path.join("data","TestB","TestB1","Wind")
    windirImgDirs = os.listdir(windDir)
    windirImgDirs = list(
        map(lambda x:os.path.join(windDir,x),windirImgDirs))
    allImgDirs["wind"] = windirImgDirs
    return  allImgDirs


def image_write(image: object, write_path, key: object) -> object:
    factorDict={"radar":70,"precip":35,"wind":10}
    image=np.clip(np.array(image),0,factorDict[key])/factorDict[key]*255.
    cv2.imwrite(write_path,image)


def imageWriteVolume(output:np.ndarray,key:str,path:str):

    submitDir = os.path.join("submit",path)

    submitDir = submitDir.split("TestB1")[1][1:]
    submitDir = os.path.join("submit",submitDir)

    if not os.path.exists(submitDir):
        try:
            os.makedirs(submitDir)
        except:
            pass
    for i in range(1,21):
        imgPath = os.path.join(submitDir,key+"_"+str(i).zfill(3)+".png")
        # try:
        image_write(output[i-1,:,:],imgPath,key)
        # except :
        print(imgPath,key)







def infer(args,resize_data):
    if not os.path.exists("submit"):
        try:
            os.makedirs("submit")
        except:
            pass
    args.batch_size = 1
    inferPaths = getTruePathForInfer()
    target_size = (args.seq_length, args.img_w, args.img_h)

    resize =  tio.Resize(target_shape=target_size)
    trainTransform = tio.Compose([
        resize,
    ])
    resizeToOriginal = tio.Resize(target_shape=(20,480,560))

    for key in inferPaths:
        device,model = get_model(args)
        model.eval()
        radarTrainDataset = CustomInferImageDataset(inferPaths[key], imgTransform=trainTransform)

        infer_loader = torch.utils.data.DataLoader(radarTrainDataset, batch_size=1, num_workers=args.workers,
                                                   shuffle=False, prefetch_factor=4, pin_memory=True, drop_last=True)
        model.load_state_dict(torch.load(os.path.join("outModels",key+".pth")),strict = True)
        for (batch,path) in infer_loader:
            with torch.no_grad():
                model.to(device)
                batch=batch.unsqueeze(2).float().to(device)
                output = resizeToOriginal(model(batch).squeeze(2).cpu().numpy()).squeeze(0)
                imageWriteVolume(output,key,path[0])
                print(output.shape)
            # print(model(batch).shape)


    # for batch in infer_loader:

if __name__ == '__main__':

    # infer(args=config,resize_data=(224,224))
    import zipfile


    def zipdir(path, ziph):
        # ziph is zipfile handle
        for root, dirs, files in os.walk(path):
            for file in files:
                ziph.write(os.path.join(root, file),
                           os.path.relpath(os.path.join(root, file),
                                           os.path.join(path, '..')))


    def zipit(dir_list, zip_name):
        zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
        for dir in dir_list:
            zipdir(dir, zipf)
        zipf.close()
    zipit([os.path.join("submit","Radar"),os.path.join("submit","Precip"),os.path.join("submit","Wind")],"submit.zip")
    check_submission_format("submit.zip")



