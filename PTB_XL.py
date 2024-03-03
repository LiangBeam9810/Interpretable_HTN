import ECGDataset 
import Models 
import Net
from train_test_validat import *
from self_attention import *
from tqdm import tqdm
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import random
import pandas as pd
import shutil

import time
import os
from torch.utils.tensorboard import SummaryWriter

import sys
import logger

time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
model_path = '/workspace/data/Interpretable_HTN/PTX/models/'+time_str
log_path = '/workspace/data/Interpretable_HTN/PTX/logs/'+  time_str

EcgChannles_num = 12
EcgLength_num = 5000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

BATCH_SIZE = 256

FOLDS = 5
EPOCHS = 200  
PATIENCE = 100
LR = 0.015

num2class = np.array(['NORM', 'MI', 'STTC', 'CD', 'HYP'])

def amplitude_limiting(ecg_data,max_bas = 90):
    ecg_data = ecg_data
    ecg_data[ecg_data > max_bas] = max_bas
    ecg_data[ecg_data < (-1*max_bas)] = -1*max_bas
    return ecg_data

def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))


class TensorDataset(Data.Dataset):
    """
    TensorDataset继承Dataset, 重载了__init__(), __getitem__(), __len__()
    实现将一组Tensor数据对封装成Tensor数据集
    能够通过index得到数据集的数据，能够通过len，得到数据集大小
    """
    def __init__(self, data, label):
        data = amplitude_limiting(data)
        self.data = torch.FloatTensor(data)
        self.labels = (torch.from_numpy(np.array(label))).float()
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return self.labels.size(0)


if __name__ == '__main__':

    train_data = np.load('/workspace/data/Preprocess_HTN/PTB_XL/X_train.npy', allow_pickle=True)
    train_label = np.load('/workspace/data/Preprocess_HTN/PTB_XL/y_train_onehot.npy', allow_pickle=True)
    
    test_data = np.load('/workspace/data/Preprocess_HTN/PTB_XL/X_test.npy', allow_pickle=True)
    test_label = np.load('/workspace/data/Preprocess_HTN/PTB_XL/y_test_onehot.npy', allow_pickle=True)
    
    train_data = train_data.swapaxes(1,2)
    test_data = test_data.swapaxes(1,2)
    print(train_data.shape)
    print(train_label.shape)
    torch.cuda.empty_cache()# 清空显卡cuda '/workspace/data/Interpretable_HTN/PTX/
    NET = [Net.MLBFNet_GUR_o(num_class = 5,mark = True,res = True,se = True,Dropout_rate = 0.2),
           Net.MLBFNet_GUR_o(num_class = 5,mark = True,res = True,se = True,Dropout_rate = 0.2),
           Net.MLBFNet_GUR_o(num_class = 5,mark = True,res = True,se = True,Dropout_rate = 0.2),
           Net.MLBFNet_GUR_o(num_class = 5,mark = True,res = True,se = True,Dropout_rate = 0.2),
           Net.MLBFNet_GUR_o(num_class = 5,mark = True,res = True,se = True,Dropout_rate = 0.2)] # type: ignore
    os.makedirs(model_path, exist_ok=True)  # type: ignore
    writer = SummaryWriter(log_path)  # type: ignore

    sys.stdout = logger.Logger(log_path+'/log.txt')
    torch.cuda.empty_cache()# 清空显卡cuda
    skf = KFold(n_splits=FOLDS, random_state=None, shuffle=True)
    
    fold = 0
    train_loss_sum =[0]*FOLDS
    train_acc_sum = [0]*FOLDS
    validate_loss_sum = [0]*FOLDS
    validate_acc_sum = [0]*FOLDS
    test_loss_sum = [0]*FOLDS
    test_acc_sum = [0]*FOLDS
    print('\nTraining..\n')
    test_dataset = TensorDataset(test_data,test_label)
    for train_index, val_index in KFold(5).split(train_data):
        print(" "*10+ "Fold "+str(fold)+" of "+str(FOLDS) + ' :')
        train_datas = train_data[train_index]
        train_labels = train_label[train_index]

        train_dataset =  TensorDataset(train_datas,train_labels)

        val_datas = train_data[val_index]
        val_labels = train_label[val_index]

        val_dataset =  TensorDataset(val_datas,val_labels)
        
        criterion = torch.nn.CrossEntropyLoss()
        train_loss,train_acc,validate_loss,validate_acc,test_loss,test_acc = tarinning_one_flod_mutilabels(fold,NET[fold]
                                                                            ,train_dataset,val_dataset,test_dataset
                                                                            ,writer,model_path
                                                                            ,BATCH_SIZE = BATCH_SIZE
                                                                            ,DEVICE=DEVICE
                                                                            ,criterion = criterion
                                                                            ,EPOCHS = 500
                                                                            ,PATIENCE = 20
                                                                            ,LR_MAX = 5*1e-3
                                                                            ,LR_MIN = 1e-5)
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

        print("train_loss",train_loss_sum," mean:",(np.array(train_loss_sum)).mean(),
        "\n" + " "*5+"train_acc",train_acc_sum," mean:",(np.array(train_acc_sum)).mean(),
        "\n" + "\n" + " "*5+"validate_loss",validate_loss_sum," mean:",(np.array(validate_loss_sum)).mean(),
        "\n" + " "*5+"validate_acc",validate_acc_sum," mean:",(np.array(validate_acc_sum)).mean(),

        "\n" + "\n" + " "*5+"test_loss",test_loss_sum," mean:",(np.array(test_loss_sum)).mean(),
        "\n" + " "*5+"test_acc",test_acc_sum," mean:",(np.array(test_acc_sum)).mean())
        print(" "*5+'='*50)
        print('Training Finished')
        # sys.stdout.log.close()
        
        for ttt in range(5):
            print("train_loss",train_loss_sum," mean:",(np.array(train_loss_sum)).mean(),
            "\n" + " "*5+"train_acc",train_acc_sum," mean:",(np.array(train_acc_sum)).mean(),
            "\n" + "\n" + " "*5+"validate_loss",validate_loss_sum," mean:",(np.array(validate_loss_sum)).mean(),
            "\n" + " "*5+"validate_acc",validate_acc_sum," mean:",(np.array(validate_acc_sum)).mean(),
            "\n" + "\n" + " "*5+"test_loss",test_loss_sum," mean:",(np.array(test_loss_sum)).mean(),
            "\n" + " "*5+"test_acc",test_acc_sum," mean:",(np.array(test_acc_sum)).mean())
            print(" "*5+'='*50)
    
        mycopyfile('./log.log',log_path)
    
    