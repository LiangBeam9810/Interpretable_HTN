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
import random
import pandas as pd

import time
import os
from sklearn.model_selection import KFold

import os
import shutil

from torch.utils.tensorboard import SummaryWriter  # type: ignore


def filter_diagnose(df_input,remove_diagnose = ''): 
    df_filter = df_input.copy()
    if(remove_diagnose):
        fitler_ID_list = df_filter[(df_filter['concat'].str.contains(remove_diagnose) == True)]['ID'].tolist()
        fitler_index = df_filter[[True if i in fitler_ID_list else False for i in df_filter['ID']]].index #选取出所有含有该ID的样本
        df_remove = df_filter.loc[(fitler_index)]
        no_fitler_index = df_filter[[False if i in fitler_ID_list else True for i in df_filter['ID']]].index #选取出所有不含有该ID的样本
        df_filter = df_filter.loc[(no_fitler_index)]
        
        print('\n')
        print("{:^10} {:^10} {:^20}".format('  ','orginal','remove diagnose' + remove_diagnose))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_filter)))
        print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_filter[(df_filter['label']==1)]),len(df_filter[(df_filter['label']==0)])))
        print("{:^10} {:^10} {:^20}".format('  ','remove HTN','remove NHTN'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_remove[(df_remove['label']==1)]),len(df_remove[(df_remove['label']==0)])))
        return df_filter

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
FOLDS = 1
EPOCHS = 100  
PATIENCE = 10
LR = 0.0001
PAIR = False

notion ="####"*10 +\
        "\n# correct at frist, no set ramdom seed each fold   " +\
        "\n"+"####"*10 +\
        "\n"
        
print(notion) 
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
log_root = './logs/'+  time_str+'/'
model_root =  './model/'+time_str+'/'
data_root = '/workspace/data/Preprocess_HTN/datas_/'

if __name__ == '__main__':
    seed_torch(2023)
    supplement_diagnose = pd.read_csv('./补充诊断.csv',encoding='utf-8-sig')
    supplement_diagnose['ID'] = supplement_diagnose['ID'].astype(str)
    # 使用groupby方法按照ID分组，然后使用agg方法将data列拼接在一起
    supplement_diagnose = supplement_diagnose.groupby('ID')['住院所有诊断'].agg(lambda x: ','.join(x.astype(str))).reset_index()
    data_root = '/workspace/data/Preprocess_HTN/datas_/'
    ALL_data = pd.read_csv(data_root+'/All_data_handled_ID_range_age_IDimputate.csv',low_memory=False)
    # ALL_data = ECGHandle.correct_age(ALL_data)
    ALL_data = ALL_data.rename(columns={'住院号':'ID','年龄':'age','性别':'gender','姓名':'name'}) 
    
    ALL_data = ALL_data[(ALL_data['申请部门'].str.contains('重症') == False)]#删除重症病房的样本
    
    ALL_data = ALL_data[(~ALL_data['ID'].isnull())] #ID NULL
    ALL_data = ALL_data[(~ALL_data['gender'].isnull())] #ID NULL
    ALL_data = ALL_data[(ALL_data['Q']<1)&(~ALL_data['Q'].isnull())]#q_sum<qc_threshold
    ALL_data = ALL_data[((ALL_data['age'].apply(int))>17) ]# 选年龄

    ALL_data['诊断'] = ALL_data['诊断'].fillna(value='')
    ALL_data['临床诊断'] = ALL_data['临床诊断'].fillna(value='')

    #添加补充诊断
    ALL_data = pd.merge(ALL_data,supplement_diagnose,how='inner',on='ID').reset_index()
    # 使用groupby和agg函数将具有相同'ID'的行的data、data1和data2列拼接在一起
    # 使用lambda函数和join方法将每个分组的值用逗号分隔
    # 保留所有的行和列，使用merge方法将拼接后的结果与原始数据框合并
    # 重置索引并重命名新的列为'concat'
    df_concat = ALL_data.groupby('ID')[[ '诊断','住院所有诊断', '临床诊断',]].agg(lambda x: ','.join(x)).reset_index()
    df_concat['concat'] = df_concat.apply(lambda x: ','.join([x['诊断'], x['住院所有诊断'], x['临床诊断']]), axis=1)
    df_concat.drop(['诊断','住院所有诊断', '临床诊断'], axis=1, inplace=True) 
    df_merge = ALL_data.merge(df_concat, on='ID', how='left') 

    df_merge.loc[(df_merge['concat'].str.contains('高血压')==True),'label']= 1 # concat diagnose含有高血压的label为1
    df_merge.loc[~(df_merge['label']==1),'label']= 0 #diagnose不含有高血压的label为0

    # Replace '男' with 1 and '女' with 2 in the 'gender' column of df
    df_merge['gender'].replace({'男': 1, '女': 2}, inplace=True)
    df_merge['label'] = pd.to_numeric(df_merge['label'],errors='coerce', downcast='integer') #把label（diagnose）改成数值型
    df_merge['age'] = pd.to_numeric(df_merge['age'],errors='coerce', downcast='integer') #把label（diagnose）改成数值型
    df_merge['gender'] = pd.to_numeric(df_merge['gender'],errors='coerce', downcast='integer') #把label（diagnose）改成数值型

    df_merge.drop_duplicates(subset=['ID'],keep='last')#删除重复的ID

    print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_merge[(df_merge['label']==1)]),len(df_merge[(df_merge['label']==0)])))

    df_merge = ECGHandle.filter_departmentORlabel(df_merge,'外')

    df_merge = filter_diagnose(df_merge,'起搏')
    df_merge = filter_diagnose(df_merge,'除颤')
    df_merge = filter_diagnose(df_merge,'电解质')
    df_merge = filter_diagnose(df_merge,'钙血')
    df_merge = filter_diagnose(df_merge,'钾血')
    df_merge = filter_diagnose(df_merge,'镁血')

    ALL_data_buffer = df_merge.copy()
    
    ALL_data_buffer = ALL_data_buffer.sample(frac=1).reset_index(drop=True) #打乱顺序
    
    test_df = ALL_data_buffer[ALL_data_buffer['year']==22].reset_index(drop=True)
    test_dataset = ECGHandle.ECG_Dataset(data_root,test_df,preprocess = True)
    
    TV_df = ALL_data_buffer[ALL_data_buffer['year']!=22].reset_index(drop=True)
    

    L2_list = [0.01,0.007,0.005,0.0005]
    BS_list = [128,128,128,128]
    random_seed_list = [2020,2021,2022,2023]
    
    for i in range(len(L2_list)):
        time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
        model_path = model_root + time_str
        log_path = log_root +  time_str
        random_seed = random_seed_list[i]
        seed_torch(random_seed)
        L2 = L2_list[i]
        BATCH_SIZE = BS_list[i]
        
        criterion =nn.CrossEntropyLoss()
        torch.cuda.empty_cache()# 清空显卡cuda
        NET = [
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.2),
            
] # type: ignore
        # NET = [res1d.resnet50(input_channels=12, inplanes=64, num_classes=2),
        #        res1d.resnet50(input_channels=12, inplanes=64, num_classes=2),
        #        res1d.resnet50(input_channels=12, inplanes=64, num_classes=2),
        #        res1d.resnet50(input_channels=12, inplanes=64, num_classes=2),
        #        res1d.resnet50(input_channels=12, inplanes=64, num_classes=2),
        #        res1d.resnet50(input_channels=12, inplanes=64, num_classes=2),
        #        ] # type: ignore
        os.makedirs(model_path, exist_ok=True)  # type: ignore
        writer = SummaryWriter(log_path)  # type: ignore
        # sys.stdout = logger.Logger(log_path+'/log.txt'
        
        print("\n\n L2 = ",L2)
        print("\nBatchsize = ",BATCH_SIZE)
        torch.cuda.empty_cache()# 清空显卡cuda  
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
         
        if(True): # 调用split方法切分数据
            
            
            tarin_df,validate_df = TV_df,test_df # 根据索引获取训练集和测试集的特征
            validate_dataset = ECGHandle.ECG_Dataset(data_root,validate_df,preprocess = True)
            train_dataset = ECGHandle.ECG_Dataset(data_root,tarin_df ,preprocess = True)
            print(" "*10+ "Fold "+str(fold)+" of "+str(FOLDS) + ' :')
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
                                                                                                    LR_MIN = LR/100000,
                                                                                                    onehot_lable= False,
                                                                                                    pair_flag= PAIR,
                                                                                                    warm_up_iter = 5,
                                                                                                    num_workers= 0,
                                                                                                    train_Df = tarin_df,
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
            fold = fold+1
            
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
        mycopyfile('./log_copy.log',log_root)