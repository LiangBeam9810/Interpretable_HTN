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


from torch.utils.tensorboard import SummaryWriter  # type: ignore

def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss/n, nll, self.epsilon)



time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
model_path = './model/'+time_str
log_path = './logs/'+  time_str

EcgChannles_num = 12
EcgLength_num = 5000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

BATCH_SIZE = 160
FOLDS = 5
EPOCHS = 300  
PATIENCE = 35
LR = 0.001

PAIR =True
notion ="####"*10 +\
        "\n#LabelSmoothingCrossEntropy "  +\
        "\n#ReduceLROnPlateau "  +\
        "\n#The reset and delete list (main in test)" +\
        "\n#qc == 0" +\
        "\n#pair HTN" +\
        "\n#use adam with 0 weight decay" +\
        "\n#Shuffle before k-fold train"+\
        "\n#Use binary F1. "  +\
        "\n#Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3)." +\
        "\n#Sample HTN to fit NHTN numbers (test,val,train)" +\
        "\n"+"####"*10
    
    
if __name__ == '__main__':
    ALLDataset = ECGDataset.ECG_Dataset_Init('/workspace/data/Preprocess_HTN/data_like_pxl//',filter_age= 18,filter_department='外科',rebuild_flage=False)    
    torch.cuda.empty_cache()# 清空显卡cuda
    NET = [Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),] # type: ignore
    # NET = [ Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1),
    #         Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1),
    #         Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1),
    #         Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1),
    #         Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1)]
    os.makedirs(model_path, exist_ok=True)  # type: ignore
    writer = SummaryWriter(log_path)  # type: ignore
    # writer.add_graph(NET[0], torch.zeros((1,12,5000)))  #模型及模型输入数据
    sys.stdout = logger.Logger(log_path+'/log.txt')
    print(notion)
    
    torch.cuda.empty_cache()# 清空显卡cuda
    ALLDataset.report()  # type: ignore    
    fold = 0
    train_loss_sum =[0]*FOLDS
    train_acc_sum = [0]*FOLDS
    validate_loss_sum = [0]*FOLDS
    validate_acc_sum = [0]*FOLDS
    test_loss_sum = [0]*FOLDS
    test_acc_sum = [0]*FOLDS    
    # criterion = nn.CrossEntropyLoss()
    criterion = LabelSmoothingCrossEntropy()
    
    # if not PAIR:
    #     test_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',ALLDataset.testDf)  # type: ignore  
    # else:
    #     test_DF = ALLDataset.testDf.copy().reset_index(drop=True)
    #     test_pair_Df = pair_HTN(test_DF[(test_DF['diagnose']==1)],test_DF[(test_DF['diagnose']==0)],Range_max = 15)  
    #     test_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',test_pair_Df)  # type: ignore   
    test_pair_Df = pair_HTN(ALLDataset.testDf[(ALLDataset.testDf['diagnose']==1)],ALLDataset.testDf[(ALLDataset.testDf['diagnose']==0)],Range_max = 15,shuffle=True) 
    test_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',test_pair_Df)  # type: ignore  
    
    tv_Df = (ALLDataset.tvDf.copy()).reset_index(drop=True)
    tv_Df = tv_Df.sample(frac=1)  #Shuffle before k-fold train
    validaate_size = len(tv_Df[(tv_Df['diagnose']==1)])//FOLDS # validatesize for each fold
    for i in range(FOLDS):
        print(" "*10+ "Fold "+str(fold)+" of "+str(FOLDS) + ' :')
        tv_Df_buffer = tv_Df.copy() 
        validate_pair_Df = pair_HTN(tv_Df_buffer[(tv_Df_buffer['diagnose']==1)].iloc[validaate_size*i:validaate_size*i+validaate_size],tv_Df_buffer[(tv_Df_buffer['diagnose']==0)],Range_max = 15,shuffle=True)
        validate_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',validate_pair_Df)  # type: ignore
        tv_Df_buffer = tv_Df_buffer.drop(index= validate_pair_Df.index)    #删掉validate_pair_Df 用于训练
        train_pair_Df = pair_HTN(tv_Df_buffer[(tv_Df_buffer['diagnose']==1)],tv_Df_buffer[(tv_Df_buffer['diagnose']==0)],Range_max = 15,shuffle=True)
        train_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',train_pair_Df)
        
        train_loss,train_acc,validate_loss,validate_acc,test_loss,test_acc = tarinning_one_flod(fold,NET[fold]
                                                                                                ,train_dataset,validate_dataset,test_dataset
                                                                                                ,writer,model_path
                                                                                                ,BATCH_SIZE = BATCH_SIZE,
                                                                                                DEVICE=DEVICE,
                                                                                                criterion = criterion,
                                                                                                EPOCHS = EPOCHS,  
                                                                                                PATIENCE = PATIENCE,
                                                                                                LR_MAX = LR,
                                                                                                LR_MIN = 1e-6,
                                                                                                onehot_lable= False,
                                                                                                pair_flag= PAIR,
                                                                                                warm_up_iter = 5,
                                                                                                num_workers= 4,
                                                                                                train_Df = tv_Df_buffer
                                                                                                )
    # strKFold = StratifiedKFold(n_splits=FOLDS, shuffle=True)  # shuffle 参数用于确定在分类前是否对数据进行打乱清洗      
    # tv_Df = (ALLDataset.tvDf.copy()).reset_index(drop=True)
    # tv_Df = tv_Df.sample(frac=1)#Shuffle before k-fold train
    # tv_Lables = np.array(tv_Df['diagnose'].tolist())# type: ignore 
    # for train_index, val_index in strKFold.split(np.zeros(len(tv_Lables)),tv_Lables):
    #     print(" "*10+ "Fold "+str(fold)+" of "+str(FOLDS) + ' :')
    #     train_infos = tv_Df.iloc[train_index].reset_index(drop=True) # type: ignore        
    #     val_infos = tv_Df.iloc[val_index].reset_index(drop=True) # type: ignore  
    #     train_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',train_infos) 
    #     validate_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',val_infos) 
    #     criterion = nn.CrossEntropyLoss()
    #     train_loss,train_acc,validate_loss,validate_acc,test_loss,test_acc = tarinning_one_flod(fold,NET[fold]
    #                                                                                             ,train_dataset,validate_dataset,test_dataset
    #                                                                                             ,writer,model_path
    #                                                                                             ,BATCH_SIZE = BATCH_SIZE,
    #                                                                                             DEVICE=DEVICE,
    #                                                                                             criterion = criterion,
    #                                                                                             EPOCHS = 200,  
    #                                                                                             PATIENCE = 10,
    #                                                                                             LR_MAX = 1e-3,
    #                                                                                             LR_MIN = 1e-6,
    #                                                                                             onehot_lable= False,
    #                                                                                             pair_flag= False,
    #                                                                                             warm_up_iter = 5,
    #                                                                                             num_workers= 4
    #                                                                                             )
   
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
        # if(fold >= 2): break
    print(" "*5+'='*50)
    
    print("train_loss",train_loss_sum,
    " "*5+"train_acc",train_acc_sum,
    "\n" + " "*5+"validate_loss",validate_loss_sum,
    " "*5+"validate_acc",validate_acc_sum,
    "\n" + " "*5+"test_loss",test_loss_sum,
    " "*5+"test_acc",test_acc_sum)
    print(" "*5+'='*50)
    print('Training Finished')