import ECGDataset 
import Models 
import Net
import res1d
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

def seed_torch(seed=2023):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False 
	torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False

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



# time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
# model_path = './model/'+time_str
# log_path = './logs/'+  time_str

seed_torch(2023)

EcgChannles_num = 12
EcgLength_num = 5000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

BATCH_SIZE = 64
FOLDS = 5
EPOCHS = 200  
PATIENCE = 50
LR = 0.0005

PAIR =True
notion ="####"*10 +\
        "\n#LR = 0.0005" +\
        "\n#pair HTN candidate >0 break " +\
        "\n#delete all have the same name&sex&ages" +\
        "\n# seed_torch(2023),    L2_list = 0.007 BATCH_SIZE = 128 ,5 foldcorss 2 times"+\
        "\n#CrossEntropyLoss "  +\
        "\n#ReduceLROnPlateau "  +\
        "\n#The reset and delete list (main in test)" +\
        "\n#qc == 0" +\
        "\n#pair HTN" +\
        "\n#use adam with 0 weight decay" +\
        "\n#Shuffle before k-fold train"+\
        "\n#Use binary F1. "  +\
        "\n#Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3). lead_branch (3,3). add two relu-fc"  +\
        "\n#Sample HTN to fit NHTN numbers (test,val,train)" +\
        "\n"+"####"*10 +\
        "\n"
        
print(notion) 
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
log_root = './logs/'+  time_str+'/'
model_root =  './model/'+time_str+'/'
if __name__ == '__main__':
    L2_list = [0.007,0.007]
    BS_list = [128,128]
    for i in range(len(L2_list)):
        seed_torch(2023)
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
        model_path = model_root + time_str
        log_path = log_root +  time_str
        
        # epsilon = epsilon_list[i]
        L2 = L2_list[i]
        BATCH_SIZE = BS_list[i]
        # BATCH_SIZE = 128
        # criterion = LabelSmoothingCrossEntropy(epsilon=epsilon)
        criterion =nn.CrossEntropyLoss()
        ALLDataset = ECGDataset.ECG_Dataset_Init('/workspace/data/Preprocess_HTN/data_like_pxl//',filter_age= 18,filter_department='外科',rebuild_flage=False)    
        torch.cuda.empty_cache()# 清空显卡cuda
        NET = [
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3),
            Net.MLBFNet(num_class = 2,mark = True,res = True,se = True,Dropout_rate = 0.3)] # type: ignore
        os.makedirs(model_path, exist_ok=True)  # type: ignore
        writer = SummaryWriter(log_path)  # type: ignore
        # sys.stdout = logger.Logger(log_path+'/log.txt'
        
        print("\n\n L2 = ",L2)
        print("\nBatchsize = ",BATCH_SIZE)
        torch.cuda.empty_cache()# 清空显卡cuda
        ALLDataset.report()  # type: ignore    
        fold = 0
        train_loss_sum =[0]*FOLDS
        train_acc_sum = [0]*FOLDS
        validate_loss_sum = [0]*FOLDS
        validate_acc_sum = [0]*FOLDS
        test_loss_sum = [0]*FOLDS
        test_acc_sum = [0]*FOLDS    
        precision_test_sum = [0]*FOLDS    
        recall_test_sum = [0]*FOLDS    
        
        # if not PAIR:
        #     test_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',ALLDataset.testDf)  # type: ignore  
        # else:
        #     test_DF = ALLDataset.testDf.copy().reset_index(drop=True)
        #     test_pair_Df = pair_HTN(test_DF[(test_DF['diagnose']==1)],test_DF[(test_DF['diagnose']==0)],Range_max = 15)  
        #     test_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',test_pair_Df)  # type: ignore   
        # test_pair_Df = pair_HTN(ALLDataset.testDf[(ALLDataset.testDf['diagnose']==1)],ALLDataset.testDf[(ALLDataset.testDf['diagnose']==0)],Range_max = 15,shuffle=True) 
        all_dataset = ALLDataset.INFOsDf.copy()
        all_dataset = all_dataset.sample(frac=1).reset_index(drop=True) 
        # test_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',ALLDataset.testDf)  # type: ignore  
        test_size = len(all_dataset[(all_dataset['diagnose']==1)])//FOLDS
        test_pair_Df = pair_HTN(all_dataset[(all_dataset['diagnose']==1)].iloc[:test_size],all_dataset[(all_dataset['diagnose']==0)],Range_max = 15,shuffle=True)
        test_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',test_pair_Df)  # type: ignore
        tv_Df = ((all_dataset).drop(index= test_pair_Df.index)).reset_index(drop=True)
        # tv_Df = (ALLDataset.tvDf.copy()).reset_index(drop=True)
        tv_Df = tv_Df.sample(frac=1).reset_index(drop=True)  #Shuffle before k-fold train
        validaate_size = len(tv_Df[(tv_Df['diagnose']==1)])//FOLDS # validatesize for each fold
        for fold in range(FOLDS):
            print(" "*10+ "Fold "+str(fold)+" of "+str(FOLDS) + ' :')
            seed_torch(2023) # reset random seed every fold, keep sequent
            tv_Df_buffer = tv_Df.copy() 
            # validate_pair_Df = pair_HTN(tv_Df_buffer[(tv_Df_buffer['diagnose']==1)].iloc[validaate_size*fold:validaate_size*fold+validaate_size],tv_Df_buffer[(tv_Df_buffer['diagnose']==0)],Range_max = 15,shuffle=True)
            # validate_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',validate_pair_Df)  # type: ignore
            # tv_Df_buffer = tv_Df_buffer.drop(index= validate_pair_Df.index)    #删掉validate_pair_Df 用于训练
            validate_dataset = test_dataset
            train_pair_Df = pair_HTN(tv_Df_buffer[(tv_Df_buffer['diagnose']==1)],tv_Df_buffer[(tv_Df_buffer['diagnose']==0)],Range_max = 15,shuffle=True)
            train_dataset = ECGDataset.ECG_Dataset('/workspace/data/Preprocess_HTN/data_like_pxl//',train_pair_Df)
            
            train_loss,train_acc,validate_loss,validate_acc,test_loss,test_acc,precision_test,recall_test = tarinning_one_flod(fold,NET[fold]
                                                                                                    ,train_dataset,validate_dataset,test_dataset
                                                                                                    ,writer,model_path
                                                                                                    ,log_path
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
                                                                                                    num_workers= 0,
                                                                                                    train_Df = tv_Df_buffer,
                                                                                                    weight_decay= L2,
                                                                                                    )
    
            train_loss_sum[fold] = train_loss
            train_acc_sum[fold] = train_acc
            validate_loss_sum[fold] = validate_loss
            validate_acc_sum[fold] = validate_acc
            test_loss_sum[fold] = test_loss
            test_acc_sum[fold] = test_acc
            precision_test_sum[fold] = precision_test  # type: ignore   
            recall_test_sum[fold] = recall_test    # type: ignore 
            torch.cuda.empty_cache()# 清空显卡cuda
            print(" "*10+'='*50)
            print(" "*10+'='*50)
            print(" "*10+"train_loss",train_loss,
            " "*10+"train_acc",train_acc,
            '\n'+" "*10+"validate_loss",validate_loss,
            " "*10+"validate_acc",validate_acc,
            '\n'+" "*10+"test_loss",test_loss,
            " "*10+"test_acc",test_acc,
            '\n'+" "*10+"test_precision",precision_test,
            '\n'+" "*10+"test_recall",recall_test,)
            print(" "*10+'='*50)
            print(" "*10+'='*50)
            print(" "*10+'Fold %d Training Finished' %(fold))
            # if(fold >= 2): break
            
        print(" "*5+'='*50)
        print("train_loss",train_loss_sum,
        " "*5+"train_acc",train_acc_sum,
        "\n" + " "*5+"validate_loss",validate_loss_sum,
        " "*5+"validate_acc",validate_acc_sum,
        "\n" + " "*5+"test_loss",test_loss_sum,
        " "*5+"test_acc",test_acc_sum,
        '\n'+" "*5+"test_precision",precision_test_sum," mean:",(np.array(precision_test_sum)).mean(),
        '\n'+" "*5+"test_recall",recall_test_sum," mean:",(np.array(recall_test_sum)).mean())
        print(" "*5+'='*50)
        print('Training Finished')
        # sys.stdout.log.close()