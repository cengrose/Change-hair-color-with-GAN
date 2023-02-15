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



