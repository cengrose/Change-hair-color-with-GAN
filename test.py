# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:20:37 2023

@author: Omen
"""

import shutil ## kopyalama için kullanılan 
import os
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image as ImageDisplay
from sklearn.model_selection import train_test_split

import glob
import random

from PIL import Image 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

import time
import datetime
import sys

import numpy as np
import itertools

from tqdm import tqdm_notebook as tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from torchvision.utils import save_image
import pandas as pd
from IPython.display import Image as ImageDisplay

from datasets import ImageDataset
from models import Generator
from models import Discriminator

from utils import weights_init_normal
from utils import ReplayBuffer
from utils import LambdaLR

from utils import Logger
import warnings
warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

epoch=0
n_epochs=50
batchSize=1
dataroot='./dataset/preprocessed_dataset_celeba/'
lr=0.0002
decay_epoch=3
size=256
input_nc=3
output_nc=3
cuda=True
n_cpu=8


netG_A2B=Generator(input_nc,output_nc)
netG_B2A=Generator(output_nc,input_nc)
if cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()
    
netG_A2B.load_state_dict(torch.load('./output/netG_A2B.pth'))
netG_B2A.load_state_dict(torch.load('./output/netG_B2A.pth'))  

Tensor=torch.cuda.FloatTensor if cuda else torch.Tensor
input_A=Tensor(batchSize,input_nc,313,313)
input_B=Tensor(batchSize,output_nc,313,313)

transforms_=[transforms.Resize((313,313),Image.BICUBIC),
             transforms.ToTensor(),
             transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
dataloader=DataLoader(ImageDataset(dataroot,transforms_=transforms_,mode='test'),batch_size=batchSize,shuffle=False, num_workers=n_cpu)


save_transforms=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(218,178)),
    transforms.ToTensor()    
    ])
                                   
if not os.path.exists('output/A'):
    os.makedirs('output/A')
if not os.path.exists('output/B'):
    os.makedirs('output/B')
    
for i,batch in enumerate(dataloader):
    print(batch['A'].shape)
    #Set model input
    real_A=Variable(input_A.copy_(batch['A']))
    real_B=Variable(input_B.copy_(batch['B']))
    
    #Generate output
    fake_B=0.5*(netG_A2B(real_A).data +1.0)
    fake_A=0.5*(netG_B2A(real_B).data +1.0)
    
    #save image files
    fake_A=fake_A.detach().cpu()
    fake_A=[save_transforms(x_) for x_ in fake_A]
    
    fake_B=fake_B.detach().cpu()
    fake_B=[save_transforms(x_) for x_ in fake_B]
    save_image(fake_A,'output/A/%04d.png'% (i+1))
    save_image(fake_B,'output/B/%04d.png'% (i+1))
    
    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1,len(dataloader)))

sys.stdout.write('\n')


ImageDisplay('./dataset/preprossed_dataset_celeba/test/A/198674.jpg')
ImageDisplay('./output/B/00010.png')









             

    
    
            
    



