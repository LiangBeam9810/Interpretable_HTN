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



def pair_HTN(INPUT_HTN_Df,INPUT_NHTN_Df,Range_max = 10,shuffle = False):
    HTN_Df = ((INPUT_HTN_Df).copy())
    NHTN_Df = ((INPUT_NHTN_Df).copy())#即抽即删,抽出一条删一条
    if(shuffle): #打乱
        HTN_Df = (HTN_Df.sample(frac=1))
        NHTN_Df = (NHTN_Df.sample(frac=1))
    # pair_Df = INFOs_df = pd.DataFrame(index=range(len(HTN_Df)*2),columns=HTN_Df.columns)   #所有的HNT和抽取出来的NHTN都存放入其中
    pair_Df = HTN_Df #先将所有HTN存放入其中
    index = len(HTN_Df)
    for info in HTN_Df.itertuples():
        age = info.ages
        gender = info.gender
        candidate_NHTN_Df = pd.DataFrame()
        
        for Range in range(1,Range_max): # 在 ±Range_max 范围内搜寻ages，且gender相同的NHTN样本
            candidate_NHTN_Df = NHTN_Df[(NHTN_Df['ages']>age-Range)&(NHTN_Df['ages']<age+Range)&(NHTN_Df['gender']==gender)]
            if(len(candidate_NHTN_Df) > 0):
                break
        
        if(len(candidate_NHTN_Df)<1):# ±Range_max 范围内都没有，那么就从所有NHTN样本（删除掉之前被抽到的）中抽一个
            print("lack sample like :",info)
            candidate_NHTN_Df = NHTN_Df
        NHTN_data_buff = candidate_NHTN_Df.sample(n=1) #从candida中随机抽样一个
        # pair_Df.iloc[index] = NHTN_data_buff.iloc[0]
        pair_Df = pair_Df.append(NHTN_data_buff)
        # print(age,',',NHTN_data_buff['ages'])
        # print(NHTN_data_buff.index)
        NHTN_Df = NHTN_Df.drop(index= (NHTN_data_buff.index))
        index = index +1
    return pair_Df


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
    ALLDataset = ECGDataset.ECG_Dataset_Init('/workspace/data/Preprocess_HTN/data_like_pxl//',filter_age= 18,filter_department='外科',rebuild_flage=False)    
    ALLDataset.report()  # type: ignore
    torch.cuda.empty_cache()# 清空显卡cuda
    NET = [Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),
           Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.25),] # type: ignore
    NET = [ Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1),
            Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1),
            Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1),
            Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1),
            Net.MLBFNet_GUR(True,res = True,se = True,Dropout_rate = 0.1)]

    test_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',ALLDataset.testDf)  # type: ignore
    os.makedirs(model_path, exist_ok=True)  # type: ignore
    writer = SummaryWriter(log_path)  # type: ignore
    # writer.add_graph(NET[0], torch.zeros((1,12,5000)))  #模型及模型输入数据
    sys.stdout = logger.Logger(log_path+'/log.txt')
    torch.cuda.empty_cache()# 清空显卡cuda

    fold = 0
    train_loss_sum =[0]*FOLDS
    train_acc_sum = [0]*FOLDS
    validate_loss_sum = [0]*FOLDS
    validate_acc_sum = [0]*FOLDS
    test_loss_sum = [0]*FOLDS
    test_acc_sum = [0]*FOLDS


    print('\nTraining..\n')
    validaate_size = len(ALLDataset.tvDf[(ALLDataset.tvDf['diagnose']==1)])//FOLDS
    for i in range(FOLDS):
        print(" "*10+ "Fold "+str(fold)+" of "+str(FOLDS) + ' :')
        tv_Df = ALLDataset.tvDf.copy()
        validate_pair_Df = pair_HTN(tv_Df[(tv_Df['diagnose']==1)].iloc[validaate_size*i:validaate_size*i+validaate_size],tv_Df[(tv_Df['diagnose']==0)],Range_max = 15)
        train_Df = tv_Df.drop(index= validate_pair_Df.index)    #删掉validate_pair_Df 用于训练
        
        # criterion = nn.CrossEntropyLoss()
        criterion = LabelSmoothingCrossEntropy()
        train_loss,train_acc,validate_loss,validate_acc,test_loss,test_acc = tarinning_one_flod(fold,NET[fold]
                                                                                                ,train_Df,validate_pair_Df,test_dataset
                                                                                                ,writer,model_path
                                                                                                ,BATCH_SIZE = BATCH_SIZE,
                                                                                                DEVICE=DEVICE,
                                                                                                criterion = criterion,
                                                                                                EPOCHS = 200,  
                                                                                                PATIENCE = 50,
                                                                                                LR_MAX = 1e-3,
                                                                                                LR_MIN = 1e-6,
                                                                                                onehot_lable= False,
                                                                                                pair_flag= True,
                                                                                                warm_up_iter = 5
                                                                                                )
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