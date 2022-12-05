import ECGDataset 
import Models 
import Net
from train_test_validat import *
from self_attention import *
import matplotlib.pyplot as plt
import ecg_plot

import torch
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import random
import pandas as pd

import time
import math
import os
import gc
from torch.utils.tensorboard import SummaryWriter

time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
model_path = './model/'+time_str
log_path = './logs/'+  time_str

EcgChannles_num = 12
EcgLength_num = 5000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

BATCH_SIZE = 128

FOLDS = 2
EPOCHS = 100  
PATIENCE = 10
LR = 0.01

if __name__ == '__main__':
    ALLDataset = ECGDataset.ECG_Dataset_Init('/workspace/data/Preprocess_HTN/data_like_pxl/',filter_age=45)
    ALLDataset.report()  # type: ignore
    torch.cuda.empty_cache()# 清空显卡cuda
    NET = [Net.MLBFNet(True,res = True,se = True,Dropout_rate = 0.25) ]*FOLDS # type: ignore
    test_dataset = ECGDataset.ECG_Dataset(ALLDataset.testECGs,ALLDataset.testLabels,ALLDataset.testDf,preprocess= True,num_classes = 0)  # type: ignore
    os.makedirs(model_path, exist_ok=True)  # type: ignore
    writer = SummaryWriter(log_path)
    # writer.add_graph(NET[0], torch.zeros((1,12,5000)))  #模型及模型输入数据
    torch.cuda.empty_cache()# 清空显卡cuda
    skf = StratifiedKFold(n_splits=FOLDS, random_state=None, shuffle=True)
    fold = 0

    train_loss_sum =[0]*FOLDS
    train_acc_sum = [0]*FOLDS
    validate_loss_sum = [0]*FOLDS
    validate_acc_sum = [0]*FOLDS
    test_loss_sum = [0]*FOLDS
    test_acc_sum = [0]*FOLDS


    print('Training..')
    for train_index, val_index in skf.split(ALLDataset.TVECGs, ALLDataset.TVLabels):
        # print("TRAIN:", train_index, "TEST:", val_index)
        train_datas = ALLDataset.TVECGs[train_index]
        train_labels = ALLDataset.TVLabels[train_index]
        train_infos = ALLDataset.TVDf.iloc[train_index]
        train_dataset =  ECGDataset.ECG_Dataset(train_datas,train_labels,train_infos,preprocess = True,num_classes = 0,)  # type: ignore

        val_datas = ALLDataset.TVECGs[val_index]
        val_labels = ALLDataset.TVLabels[val_index]
        val_infos = ALLDataset.TVDf.iloc[val_index]
        val_dataset =  ECGDataset.ECG_Dataset(val_datas,val_labels,val_infos,preprocess = True,num_classes = 0)  # type: ignore
        
        criterion = torch.nn.CrossEntropyLoss()
        train_loss,train_acc,validate_loss,validate_acc,test_loss,test_acc = tarinning_one_flod(fold,NET[fold],train_dataset,val_dataset,test_dataset,writer,model_path
                                                                                                ,BATCH_SIZE = BATCH_SIZE,
                                                                                                DEVICE=DEVICE,
                                                                                                criterion = criterion,
                                                                                                EPOCHS = 300,  
                                                                                                PATIENCE = 50,
                                                                                                LR_MAX = 1e-3,
                                                                                                LR_MIN = 1e-5,)
        train_loss_sum[fold] = train_loss
        train_acc_sum[fold] = train_acc
        validate_loss_sum[fold] = validate_loss
        validate_acc_sum[fold] = validate_acc
        test_loss_sum[fold] = test_loss
        test_acc_sum[fold] = test_acc
        torch.cuda.empty_cache()# 清空显卡cuda
        fold = fold + 1
        print(" "*20+'='*50)
        print(" "*20+"train_loss",train_loss,
        " "*20+"train_acc",train_acc,
        " "*20+"validate_loss",validate_loss,
        " "*20+"validate_acc",validate_acc,
        " "*20+"test_loss",test_loss,
        " "*20+"test_acc",test_acc)
        print(" "*20+'='*50)
    print(" "*3+'='*50)
    print("\n train_loss",train_loss_sum,
    " "*3+"train_acc",train_acc_sum,
    "\n" + " "*3+"validate_loss",validate_loss_sum,
    " "*3+"validate_acc",validate_acc_sum,
    "\n" + " "*3+"test_loss",test_loss_sum,
    " "*3+"test_acc",test_acc_sum)
    print(" "*3+'='*50)
    print('Training Finished')