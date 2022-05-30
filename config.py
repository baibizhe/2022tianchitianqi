import os
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 1337
C.dataPath = os.path.join("data","Train")
C.csvPath = os.path.join(C.dataPath,"train.csv")
C.img_w = 480
C.img_h = 560
C.numlayers = 4
C.num_hidden = [2, 2,2, 2]
C.seq_length = 20
C.warmup_epochs =3
C.T_mult = 2
C.workers = 3
C.gpu_ids = [0]


