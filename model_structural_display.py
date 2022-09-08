import Models 
from train_test_validat import *
from self_attention import *
from  ecg_get_data import *
import neurokit2
import matplotlib.pyplot as plt

import torch
import torch.utils.data as Data
import random
from torchsummary import summary

models = Models.CNN_ATT2()
print(summary(models, (12,5000), device="cpu"))