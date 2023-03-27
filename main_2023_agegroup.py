import ECGDataset 
import Models 
import Net
import res1d
from train_test_validat import *
from self_attention import *
import matplotlib.pyplot as plt
import ecg_plot
import ECGHandle
import VGG
import resnet
import torch

import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import random
import pandas as pd

import time
import os


import os
import shutil

from torch.utils.tensorboard import SummaryWriter  # type: ignore


def mycopyfile(srcfile,dstpath):                       # 复制函数
    if not os.path.isfile(srcfile):
        print ("%s not exist!"%(srcfile))
    else:
        fpath,fname=os.path.split(srcfile)             # 分离文件名和路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)                       # 创建路径
        shutil.copy(srcfile, dstpath + fname)          # 复制文件
        print ("copy %s -> %s"%(srcfile, dstpath + fname))

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
    
seed_torch(2023)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)

BATCH_SIZE = 256
L2 = 0.07
FOLDS = 5
EPOCHS = 200  
PATIENCE = 30
LR = 0.0005
PAIR =True

notion ="####"*10 +\
        "\n#Net.MLBFNet_GUR(True,True,True,2,0.3),position embedding   " +\
        "\n#LR = 0.0005" +\
        "\n#pair HTN candidate >0 break " +\
        "\n#delete all have the same name&sex&ages" +\
        "\n# seed_torch(2023),    L2_list = 0.007 BATCH_SIZE = 128 ,5 foldcorss 1 times"+\
        "\n#CrossEntropyLoss "  +\
        "\n#ReduceLROnPlateau "  +\
        "\n#The reset and delete list (main in test)" +\
        "\n#qc == 0" +\
        "\n#pair HTN" +\
        "\n#use adam with 0 weight decay" +\
        "\n#Shuffle before k-fold train"+\
        "\n#Use binary F1. "  +\
        "\n#Net.MLBFNet_GUR(mark = True,res = True,se = True,Dropout_rate = 0.3). lead_branch (3,3). add two relu-fc"  +\
        "\n#Sample HTN to fit NHTN numbers (test,val,train)" +\
        "\n"+"####"*10 +\
        "\n"
        
print(notion) 
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
log_root = './logs/'+  time_str+'/'
model_root =  './model/'+time_str+'/'
data_root = '/workspace/data/Preprocess_HTN/datas_/'

if __name__ == '__main__':
    L2_list = [0.007,0.007,0.007,0.007,0.007,0.007,0.007,0.007]
    BS_list = [64,64,64,64,64,64,64,64,64,64]
    random_seed_list = [2023,2022,2021,2020,2019,2018,3407,115114]
    for i in range(len(L2_list)):
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
        model_path = model_root + time_str
        log_path = log_root +  time_str
        
        random_seed = random_seed_list[i]
        L2 = L2_list[i]
        BATCH_SIZE = BS_list[i]
        
        criterion =nn.CrossEntropyLoss()
        
        ALL_data = pd.read_csv(data_root+'/All_data_handled_ID_range_age_IDimputate.csv',low_memory=False)
        
        
        ALL_data = ECGHandle.change_label(ALL_data)
        ALL_data = ECGHandle.filter_ID(ALL_data)
        ALL_data = ECGHandle.filter_QC(ALL_data)
        
        ALL_data = ECGHandle.filter_ages(ALL_data,18)
        ALL_data = ECGHandle.filter_departmentORlabel(ALL_data,'外科')
        
        ALL_data = ECGHandle.correct_label(ALL_data)
        ALL_data = ECGHandle.correct_age(ALL_data)
        ALL_data = ECGHandle.filter_diagnose(ALL_data,'起搏')
        ALL_data = ECGHandle.filter_diagnose(ALL_data,'房颤')
        ALL_data = ECGHandle.filter_diagnose(ALL_data,'左束支传导阻滞')
        ALL_data = ECGHandle.filter_diagnose(ALL_data,'左前分支阻滞')
        # ALL_data = ECGHandle.filter_diagnose(ALL_data,'阻滞')
        # ALL_data = ECGHandle.remove_duplicated(ALL_data)
        
        ALL_data = ALL_data.rename(columns={'住院号':'ID','年龄':'age','性别':'gender','姓名':'name'}) 
        # 按照指定的年龄区间和性别进行分组
        bins = [18, 30, 40, 50, 60, 70, 80,110]
        labels = ['18-30', '31-40', '41-50', '51-60', '61-70', '71-80','81-110']
        ALL_data['agegroup'] = pd.cut(ALL_data['age'], bins=bins, labels=labels)
        ALL_data.hist(column='agegroup',by = 'label',sharex=True)
        
        
        fold = 0
        train_loss_sum =[0]*FOLDS
        train_acc_sum = [0]*FOLDS
        validate_loss_sum = [0]*FOLDS
        validate_acc_sum = [0]*FOLDS
        precision_valid_sum = [0]*FOLDS    
        recall_valid_sum = [0]*FOLDS   
        validate_auc_sum = [0]*FOLDS
        
        test_loss_sum = [0]*FOLDS
        test_acc_sum = [0]*FOLDS       
        precision_test_sum = [0]*FOLDS    
        recall_test_sum = [0]*FOLDS   
        test_auc_sum = [0]*FOLDS
        ALL_data_buffer = ALL_data.copy()
        np.random.seed(2023)
        ALL_data_buffer = ALL_data_buffer.sample(frac=1).reset_index(drop=True) #打乱顺序
        ######################################################################################################################################
        NHTN_df = ALL_data_buffer[(ALL_data_buffer['label']==0) ].drop_duplicates(subset=['ID'],keep = 'first').copy()
        HTN_df = ALL_data_buffer[(ALL_data_buffer['label']==1) ].drop_duplicates(subset=['ID'],keep = 'first').copy()
        TV_HTN_df, test_HTN_df = train_test_split(HTN_df, test_size=0.2, stratify=HTN_df[['agegroup', 'gender']])
        test_ID_list = pair_HTN(test_HTN_df.drop_duplicates(['ID'],keep='first'),NHTN_df.drop_duplicates(['ID'],keep='first'),
                                    Range_max=2,
                                    pair_num=1,
                                    shuffle=True)['ID'].tolist()#按照年龄和性别对每个ID号去配对 (先去除重复ID)
        test_index = ALL_data_buffer[[True if i in test_ID_list else False for i in ALL_data_buffer['ID']]].index
        test_df = ALL_data_buffer.loc[test_index].copy()
        left_index = ALL_data_buffer[[False if i in test_ID_list else True for i in ALL_data_buffer['ID']]].index #不在test_ID_list的ID 即为tv的
        TV_df = ALL_data_buffer.loc[left_index].copy()
        test_dataset = ECGHandle.ECG_Dataset(data_root,test_df,preprocess = True)
        ##########选出每一轮的valid
        TV_NHTN_df = TV_df[(TV_df['label']==0) ].drop_duplicates(subset=['ID'],keep = 'first').copy()
        TV_HTN_df = TV_df[(TV_df['label']==1) ].drop_duplicates(subset=['ID'],keep = 'first').copy()
        fold_len= float((TV_HTN_df.__len__())//5) #每一fold的HTN的长度
        TV_HTN_buffer = TV_HTN_df.copy()

        validat_HTN_df_subsets = []
        for i in range(4):
            TV_HTN_buffer, subset = train_test_split(TV_HTN_buffer, test_size=fold_len/(TV_HTN_buffer.__len__()), stratify=TV_HTN_buffer[['agegroup', 'gender']])
            print(subset.__len__())
            validat_HTN_df_subsets.append(subset)
        print(TV_HTN_buffer.__len__())
        validat_HTN_df_subsets.append(TV_HTN_buffer) #每一轮的valida的HTN的df
        
        validat_ID_list_subsets = [] #每一轮的valida的ID list
        for i in range(validat_HTN_df_subsets.__len__()):
            i_ID_list = pair_HTN(validat_HTN_df_subsets[i].drop_duplicates(['ID'],keep='first'),
                                TV_NHTN_df.drop_duplicates(['ID'],keep='first'),
                                    Range_max=5,
                                    pair_num=1,
                                    shuffle=True)['ID'].tolist()#按照年龄和性别对每个ID号去配对 (先去除重复ID)
            validat_ID_list_subsets.append(i_ID_list)
        ######################################################################################################################################
        
        torch.cuda.empty_cache()# 清空显卡cuda
        NET = [
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3),] # type: ignore

        os.makedirs(model_path, exist_ok=True)  # type: ignore
        writer = SummaryWriter(log_path)  # type: ignore
        # sys.stdout = logger.Logger(log_path+'/log.txt'
        
        print("\n\n L2 = ",L2)
        print("\nBatchsize = ",BATCH_SIZE)
        torch.cuda.empty_cache()# 清空显卡cuda  
        
        for fold in range(FOLDS):
            print(" "*10+ "Fold "+str(fold)+" of "+str(FOLDS) + ' :')
            
        #######################################################################
            pair_index = TV_df[[True if i in validat_ID_list_subsets[fold] else False for i in TV_df['ID']]].index
            validate_df = TV_df.loc[pair_index].copy()
            left_index = TV_df[[False if i in validat_ID_list_subsets[fold] else True for i in TV_df['ID']]].index #不在test_ID_list的ID 即为tv的
            train_df = TV_df.loc[left_index].copy()
            train_pair_df,_ = Pair_ID(train_df,1,star_index=0,Range_max=15,pair_num=1,shuffle=True)
        #######################################################################
        
            validate_dataset = ECGHandle.ECG_Dataset(data_root,validate_df,preprocess = True)
            train_dataset = ECGHandle.ECG_Dataset(data_root,train_pair_df ,preprocess = True)
            
            validate_dataset.infos.to_csv(log_path+'/randomseed'+str(random_seed)+'_fold'+str(fold)+'_valida.csv')
            train_dataset.infos.to_csv(log_path+'/randomseed'+str(random_seed)+'_fold'+str(fold)+'_train.csv')
            test_dataset.infos.to_csv(log_path+'/randomseed'+str(random_seed)+'_fold'+str(fold)+'_test.csv')
            
            
            train_loss,train_acc,validate_loss,validate_acc,precision_valid,recall_valid,auc_valid,test_loss,test_acc,precision_test,recall_test,auc_test = tarinning_one_flod(fold,NET[fold]
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
                                                                                                    train_Df = train_df,
                                                                                                    weight_decay= L2,
                                                                                                    data_path= data_root
                                                                                                    )
    
            train_loss_sum[fold] = train_loss
            train_acc_sum[fold] = train_acc
            
            validate_loss_sum[fold] = validate_loss
            validate_acc_sum[fold] = validate_acc
            precision_valid_sum[fold] = precision_valid  # type: ignore        
            recall_valid_sum[fold] = recall_valid  # type: ignore        
            validate_auc_sum[fold] = auc_valid # type: ignore        
            
            test_loss_sum[fold] = test_loss
            test_acc_sum[fold] = test_acc
            precision_test_sum[fold] = precision_test  # type: ignore   
            recall_test_sum[fold] = recall_test    # type: ignore 
            test_auc_sum[fold] = auc_test  # type: ignore            
            
            
            torch.cuda.empty_cache()# 清空显卡cuda
            print(" "*10+'='*50)
            print(" "*10+'Fold %d Training Finished' %(fold))
            print('\n')
            # if(fold >= 3): break
            
        print(" "*5+'='*50)
        print("train_loss",train_loss_sum," mean:",(np.array(train_loss_sum)).mean(),
        "\n" + " "*5+"train_acc",train_acc_sum," mean:",(np.array(train_acc_sum)).mean(),
        "\n" + "\n" + " "*5+"validate_loss",validate_loss_sum," mean:",(np.array(validate_loss_sum)).mean(),
        "\n" + " "*5+"validate_acc",validate_acc_sum," mean:",(np.array(validate_acc_sum)).mean(),
        "\n" + " "*5+"validate_precision",precision_valid_sum," mean:",(np.array(precision_valid_sum)).mean(),
        "\n" + " "*5+"validate_recall",recall_valid_sum," mean:",(np.array(recall_valid_sum)).mean(),   
        "\n" + " "*5+"validate_auc",validate_auc_sum," mean:",(np.array(validate_auc_sum)).mean(),
        "\n" + "\n" + " "*5+"test_loss",test_loss_sum," mean:",(np.array(test_loss_sum)).mean(),
        "\n" + " "*5+"test_acc",test_acc_sum," mean:",(np.array(test_acc_sum)).mean(),
        '\n'+" "*5+"test_precision",precision_test_sum," mean:",(np.array(precision_test_sum)).mean(),
        "\n" + " "*5+"test_recall",recall_test_sum," mean:",(np.array(recall_test_sum)).mean(),
        "\n" + " "*5+"test_auc",test_auc_sum," mean:",(np.array(test_auc_sum)).mean())
        print(" "*5+'='*50)
        print('Training Finished')
        # sys.stdout.log.close()
        
        for ttt in range(5):
            print("train_loss",train_loss_sum," mean:",(np.array(train_loss_sum)).mean(),
        "\n" + " "*5+"train_acc",train_acc_sum," mean:",(np.array(train_acc_sum)).mean(),
        "\n" + "\n" + " "*5+"validate_loss",validate_loss_sum," mean:",(np.array(validate_loss_sum)).mean(),
        "\n" + " "*5+"validate_acc",validate_acc_sum," mean:",(np.array(validate_acc_sum)).mean(),
        "\n" + " "*5+"validate_precision",precision_valid_sum," mean:",(np.array(precision_valid_sum)).mean(),
        "\n" + " "*5+"validate_recall",recall_valid_sum," mean:",(np.array(recall_valid_sum)).mean(),   
        "\n" + " "*5+"validate_auc",validate_auc_sum," mean:",(np.array(validate_auc_sum)).mean(),
        "\n" + "\n" + " "*5+"test_loss",test_loss_sum," mean:",(np.array(test_loss_sum)).mean(),
        "\n" + " "*5+"test_acc",test_acc_sum," mean:",(np.array(test_acc_sum)).mean(),
        '\n'+" "*5+"test_precision",precision_test_sum," mean:",(np.array(precision_test_sum)).mean(),
        "\n" + " "*5+"test_recall",recall_test_sum," mean:",(np.array(recall_test_sum)).mean(),
        "\n" + " "*5+"test_auc",test_auc_sum," mean:",(np.array(test_auc_sum)).mean())
            print(" "*5+'='*50)
        #重复打印几次 等待.log 打印完
        mycopyfile('./log.log',log_root)