import ECGDataset 
import Models 
import Net
from train_test_validat import *
from self_attention import *
import matplotlib.pyplot as plt
import ecg_plot

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import random
import pandas as pd

import time
import math
import os
import gc


import sys
import logger


from torch.utils.tensorboard import SummaryWriter

time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
model_path = './model/'+time_str
log_path = './logs/'+  time_str

EcgChannles_num = 12
EcgLength_num = 5000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

BATCH_SIZE = 160

FOLDS = 5
EPOCHS = 100  
PATIENCE = 10
LR = 0.01

if __name__ == '__main__':
    
    ALLDataset = ECGDataset.ECG_Dataset_Init('/workspace/data/Preprocess_HTN/data_like_pxl/',rebuild_flage= False,filter_age = 0)
    ALLDataset.report()  # type: ignore
    
    torch.cuda.empty_cache()# 清空显卡cuda
    NET = [Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),] # type: ignore
    # NET = [Models.ECGNet(input_channel=1, num_classes=2,mark = False, res=True,se = True),
    #        Models.ECGNet(input_channel=1, num_classes=2,mark = False, res=True,se = True),
    #        Models.ECGNet(input_channel=1, num_classes=2,mark = False, res=True,se = True),
    #        Models.ECGNet(input_channel=1, num_classes=2,mark = False, res=True,se = True),
    #        Models.ECGNet(input_channel=1, num_classes=2,mark = False, res=True,se = True),]
    test_dataset = ECGDataset.ECG_Dataset(ALLDataset.testECGs,ALLDataset.testLabels,ALLDataset.testDf,preprocess= True,onehot_lable = False)  # type: ignore
    os.makedirs(model_path, exist_ok=True)  # type: ignore
    writer = SummaryWriter(log_path)  # type: ignore
    # writer.add_graph(NET[0], torch.zeros((1,12,5000)))  #模型及模型输入数据
    sys.stdout = logger.Logger(log_path+'/log.txt')
    torch.cuda.empty_cache()# 清空显卡cuda
    skf = StratifiedKFold(n_splits=FOLDS, random_state=None, shuffle=True)
    fold = 0

    train_loss_sum =[0]*FOLDS
    train_acc_sum = [0]*FOLDS
    validate_loss_sum = [0]*FOLDS
    validate_acc_sum = [0]*FOLDS
    test_loss_sum = [0]*FOLDS
    test_acc_sum = [0]*FOLDS


    print('\nTraining..\n')
    for train_index, val_index in skf.split(ALLDataset.TVECGs, ALLDataset.TVLabels):
        print(" "*10+ "Fold "+str(fold)+" of "+str(FOLDS) + ' :')
        # print("TRAIN:", train_index, "TEST:", val_index)
        train_datas = ALLDataset.TVECGs[train_index]
        train_labels = ALLDataset.TVLabels[train_index]
        train_infos = ALLDataset.TVDf.iloc[train_index].reset_index(drop=True) 
        train_dataset =  ECGDataset.ECG_Dataset(train_datas,train_labels,train_infos,preprocess = True,onehot_lable = False,pair_flag=False)  # type: ignore

        val_datas = ALLDataset.TVECGs[val_index]
        val_labels = ALLDataset.TVLabels[val_index]
        val_infos = ALLDataset.TVDf.iloc[val_index]
        val_dataset =  ECGDataset.ECG_Dataset(val_datas,val_labels,val_infos,preprocess = True,onehot_lable = False)  # type: ignore
        # weights = torch.tensor([1.0/7.0, 8.0/7.0]).to(DEVICE)#device表示GPU显卡
        # criterion = nn.CrossEntropyLoss(weight=weights)  # 设置损失函数
        # criterion = FocalLoss(class_num=2,alpha=torch.Tensor([0.25,0.75]))
        criterion = nn.CrossEntropyLoss()
        train_loss,train_acc,validate_loss,validate_acc,test_loss,test_acc = tarinning_one_flod(fold,NET[fold]
                                                                                                ,train_dataset,val_dataset,test_dataset
                                                                                                ,writer,model_path
                                                                                                ,BATCH_SIZE = BATCH_SIZE,
                                                                                                DEVICE=DEVICE,
                                                                                                criterion = criterion,
                                                                                                EPOCHS = 100,  
                                                                                                PATIENCE = 10,
                                                                                                LR_MAX = 1e-3,
                                                                                                LR_MIN = 1e-6,
                                                                                                onehot_lable= False,
                                                                                                pair_flag=False,
                                                                                                warm_up_iter = 5)
        train_loss_sum[fold] = train_loss
        train_acc_sum[fold] = train_acc
        validate_loss_sum[fold] = validate_loss
        validate_acc_sum[fold] = validate_acc
        test_loss_sum[fold] = test_loss
        test_acc_sum[fold] = test_acc
        torch.cuda.empty_cache()# 清空显卡cuda
        fold = fold + 1
        print(" "*10+'='*50)
        print(" "*10+'='*50)
        print(" "*10+"train_loss",train_loss,
        " "*10+"train_acc",train_acc,
        '\n'+" "*10+"validate_loss",validate_loss,
        " "*10+"validate_acc",validate_acc,
        '\n'+" "*10+"test_loss",test_loss,
        " "*10+"test_acc",test_acc)
        print(" "*10+'='*50)
        print(" "*10+'='*50)
    print(" "*5+'='*50)
    
    print("train_loss",train_loss_sum,
    " "*5+"train_acc",train_acc_sum,
    "\n" + " "*5+"validate_loss",validate_loss_sum,
    " "*5+"validate_acc",validate_acc_sum,
    "\n" + " "*5+"test_loss",test_loss_sum,
    " "*5+"test_acc",test_acc_sum)
    print(" "*5+'='*50)
    print('Training Finished')