import Models 
from train_test_validat import *
from self_attention import *
from  ecg_get_data import *

import matplotlib.pyplot as plt

import torch
import torch.utils.data as Data
import random
from torchsummary import summary
class test(nn.Module):
    def __init__(self,DropoutRate = 0.1):
        super(test,self).__init__()
        self.lstmlayer =nn.LSTM(input_size=512, hidden_size=32,num_layers  = 2,batch_first=True)
    def forward(self,x):
        x, (ht,ct) = self.lstmlayer(x)
        return x

models = test()
print(summary(models, (61 ,512), device="cpu"))