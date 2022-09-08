from torch.utils.tensorboard import SummaryWriter   

import Models 
from train_test_validat import *
from self_attention import *
from  ecg_get_data import *
import neurokit2
import matplotlib.pyplot as plt

import torch
import torch.utils.data as Data
import random

import time

NET = Models.CNN_ATT()
