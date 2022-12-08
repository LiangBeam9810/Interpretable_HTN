import ECGDataset 
import Models 
import Net
from train_test_validat import *
from self_attention import *
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
import random
import pandas as pd

import time
import os
from torch.utils.tensorboard import SummaryWriter

import sys
import logger

time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
model_path = './model/'+time_str
log_path = './logs/'+  time_str

EcgChannles_num = 12
EcgLength_num = 5000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

BATCH_SIZE = 128

FOLDS = 5
EPOCHS = 100  
PATIENCE = 10
LR = 0.01

num2class = np.array(['NORM', 'MI', 'STTC', 'CD', 'HYP'])
if __name__ == '__main__':

    train_data = np.load('/workspace/data/Preprocess_HTN/PTB_XL/X_train.npy', allow_pickle=True)
    # train_label = np.load('/workspace/data/Preprocess_HTN/PTB_XL/y_train.npy', allow_pickle=True)
    train_label = np.load('/workspace/data/Preprocess_HTN/PTB_XL/y_train_onehot.npy', allow_pickle=True)
    
    test_data = np.load('/workspace/data/Preprocess_HTN/PTB_XL/X_test.npy', allow_pickle=True)
    # test_label = np.load('/workspace/data/Preprocess_HTN/PTB_XL/y_test.npy', allow_pickle=True)    
    test_label = np.load('/workspace/data/Preprocess_HTN/PTB_XL/y_test_onehot.npy', allow_pickle=True)    
    
    # train_label_onehot = np.zeros((len(train_label),5))
    # test_label_onehot = np.zeros((len(test_label),5))
    
    # for index in tqdm(range(len(train_label))):
    #     for i in range(len(train_label[index])):
    #         j = np.where(num2class == train_label[index][i])[0][0]
    #         train_label_onehot[index,j]=1
    
    # for index in tqdm(range(len(test_label))):
    #     for i in range(len(test_label[index])):
    #         j = np.where(num2class == test_label[index][i])[0][0]
    #         test_label_onehot[index,j]=1
            
    # np.save('/workspace/data/Preprocess_HTN/PTB_XL/y_train_onehot.npy', np.array(train_label_onehot))
    # np.save('/workspace/data/Preprocess_HTN/PTB_XL/y_test_onehot.npy', np.array(test_label_onehot))
    print(train_data.max())
    print(train_data.min())
    print(test_data.max())
    print(test_data.min())
    
    
    
    