import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import random
from scipy.signal import butter, lfilter

reset_list = [
                '848023',
                '848316',
                '578148',
                '736912',
                '847065',
                '849400',
                '839738',
                '309738',
                '496939',
                '847724',
                '847377',
                '848996',
                '850179',
                '844904',
                '848473',
]



def change_ages(df_input): #把年龄改成 数值型
    df = df_input.copy()
    df = df.dropna(subset=['年龄']) #删除ages== nan
    df.loc[~(df['年龄'].str.contains('岁')==True),'年龄']='0岁' #不含有岁的（天周月）改为"0岁"
    df['年龄'].replace(regex=True,inplace=True,to_replace=r'岁',value=r'') #删除"岁"
    df["年龄"] = pd.to_numeric(df["年龄"],errors='coerce') #把年龄改成数值型
    print('\n')
    print("{:^10} {:^10} {:^20}".format('  ','orginal','removed ages NaN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df)))
    return df

def change_label(df_input):#把临床诊断改成 数值型（0/1） 作为标签
    df = df_input.copy()
    df.insert(loc=48, column='label', value=0) 
    df = df.dropna(subset=['临床诊断']) #删除diagnose== nan
    df.loc[(df['临床诊断'].str.contains('高血压')==True),'label']= 1 #diagnose含有高血压的label为1
    df.loc[~(df['label']==1),'label']= 0 #diagnose不含有高血压的label为0
    df['label'] = pd.to_numeric(df['label'],errors='coerce') #把label（diagnose）改成数值型
    # print(df['diagnose'].value_counts())
    print('\n')
    print("{:^10} {:^10} {:^20}".format('  ','orginal','removed diagnose NaN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df)))
    print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df[(df['label']==1)]),len(df[(df['label']==0)])))
    return df 

def filter_QC(df_input,qc_threshold = 1):#剔除QC为NAN和qc值大于等于qc_threshold的样本
    df = df_input.copy()       
    df = df[(df['Q']<qc_threshold)&(~df['Q'].isnull())]#q_sum<qc_threshold
    print('\n')
    print("{:^10} {:^10} {:^20}".format('  ','orginal','QC'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df)))
    print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df[(df['label']==1)]),len(df[(df['label']==0)])))
    return df 

def filter_ID(df_input):# 剔除住院号为NAN的样本
    df = df_input.copy()       
    df = df[(~df['住院号'].isnull())]#q_sum<qc_threshold
    print('\n')
    print("{:^10} {:^10} {:^20}".format('  ','orginal','removed ID NaN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df)))
    print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df[(df['label']==1)]),len(df[(df['label']==0)])))
    return df 

#department符合条件 或者 诊断为高血压的样本 
def filter_departmentORlabel(df_input,filter_department = ''):  # type: ignore
    df_filter = df_input.copy()
    if(filter_department):
        df_filter = df_filter[
            ( (df_filter['label'] == 1)| (df_filter['申请部门'].str.contains(filter_department) == True))
            ]#只选择外科
        print('\n')
        print("{:^10} {:^10} {:^20}".format('  ','orginal','filtered department'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_filter)))
        print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_filter[(df_filter['label']==1)]),len(df_filter[(df_filter['label']==0)])))
    return df_filter

#年龄符合条件 >filter_age
def filter_ages(df_input,filter_age = 0): 
    df_filter = df_input.copy()
    if(filter_age):
        df_filter = df_filter[
                ((df_filter['年龄'].apply(int))>filter_age) 
            ]#删选年龄
        print('\n')
        print("{:^10} {:^10} {:^20}".format('  ','orginal','filtered ages'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_filter)))
        print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_filter[(df_filter['label']==1)]),len(df_filter[(df_filter['label']==0)])))
    return df_filter

def correct_label(df_input): #更改错误的label
    print('\n')
    df_filter = df_input.copy()
    df_filter = df_filter.sort_values(by=['label'], ascending=[False]) #按照诊断排序，HTN在前
    #####################################################test
    #将确认为HTN的错误样本的标签重置###reset lable 
    reset_index = df_filter[[True if i in reset_list else False for i in df_filter['住院号']]].index
    df_filter.loc[reset_index,'label'] = 1
    print("{:^20} {:^5}".format("reset num:",len(reset_index)))
    #####################################################test 把住院号相同但诊断标签不一致的全部改为HTN
    df0 = df_filter[df_filter['label']==0] #所有非高血压
    df1 = df_filter[df_filter['label']==1] #所有高血压
    duplicated_index = df0[[True if i in df1['住院号'].tolist() else False for i in df0['住院号']]].index
    df_filter.loc[duplicated_index,'label'] = 1   
    print("{:^20} {:^5}".format("ERR labels num:",len(duplicated_index))) 
    #####################################################test
    print("{:^10} {:^10} {:^20}".format('  ','orginal','correct label'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_filter)))
    print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_filter[(df_filter['label']==1)]),len(df_filter[(df_filter['label']==0)])))
    return df_filter


def correct_age(df_input): # 把住院号相同但年龄不一致的全部改为统一值
    correct_age_counts = 0;
    print('\n')
    df_filter = df_input.copy()
    df_filter = df_filter.sort_values(by=['姓名'], ascending=[False])#按照姓名排序
    #####################################################test 把住院号相同但年龄不一致的全部改为一直
    for index,row in df_filter.iterrows():#遍历所有行
        age = row['年龄']
        ID = row['住院号']
        df_buffer = df_filter[(df_filter['住院号'] == ID)&(~(df_filter['年龄'] == age))]#住院号相同 但年龄不同
        if(df_buffer.__len__()>0):
            for index_,row_ in df_buffer.iterrows():#遍历所有符合条件的样本，赋予相同年龄
                df_filter.loc[index_,'年龄'] = age
                correct_age_counts=correct_age_counts+ 1
    print("{:^20} {:^5}".format("ERR ages num:",correct_age_counts)) 
    #####################################################test
    print("{:^10} {:^10} {:^20}".format('  ','orginal','correct age'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_filter)))
    print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
    print("{:^10} {:^10} {:^20}".format('nums',len(df_filter[(df_filter['label']==1)]),len(df_filter[(df_filter['label']==0)])))
    return df_filter

#年龄符合条件 >filter_age
def filter_diagnose(df_input,remove_diagnose = ''): 
    df_filter = df_input.copy()
    if(remove_diagnose):
        df_filter = df_filter[
            (  (df_filter['诊断'].str.contains(remove_diagnose) == False) )#不含有该诊断
            ]#只选择外科
        print('\n')
        print("{:^10} {:^10} {:^20}".format('  ','orginal','remove diagnose'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_filter)))
        print("{:^10} {:^10} {:^20}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_filter[(df_filter['label']==1)]),len(df_filter[(df_filter['label']==0)])))
    return df_filter

class ECG_Dataset(Dataset):
    def __init__(self,data_root,infos,preprocess = True,onehot_lable=False):
        self.ECGs_path = data_root+'/ECG/'
        self.Qualitys_path = data_root+'/Q/'
        
        self.infos = infos
        self.datas = self.get_ECGs_form_FilesList(self.infos['ECGFilename'].tolist())
        self.labels = self.infos['label'].tolist()
        self.len = len(self.datas)
        if(preprocess):
            self.preprocess()
        self.datas[np.isnan(self.datas)]=0
        self.datas = torch.FloatTensor(self.datas)
        # num_classes = len(torch.bincount(self.labels))
        self.labels = torch.from_numpy(np.array(self.labels)).long()
        if(onehot_lable):
            self.labels = torch.nn.functional.one_hot(self.labels).float()
    def get_ECGs_form_FilesList(self,FilesList):
        data = np.zeros((len(FilesList),12,5000))
        i = 0
        for file in (FilesList):
            data[i] = np.load(self.ECGs_path + file+'.npy')
            i = i+1
        return  data
    def preprocess(self):
        # filter_lowcut = 1.0
        # filter_highcut = 47.0
        # filter_order = 1
        # for i in tqdm(range(len(self.datas))):
        #     self.datas[i] = bandpass_filter(self.datas[i], lowcut=filter_lowcut, highcut=filter_highcut, signal_freq=500, filter_order=filter_order)# type: ignore   
        self.datas = self.amplitude_limiting(self.datas,5000)
        # for i in range(12):
        #     mean =  self.datas[:,i,:].mean()
        #     var = self.datas[:,i,:].var()
        #     self.datas[:,i,:] = (self.datas[:,i,:] - mean)/(self.datas[:,i,:].var()+1e-6)
    def __getitem__(self,index):
        return self.datas[index],self.labels[index]
    def __len__(self):
        return self.len
    
    def amplitude_limiting(self,ecg_data,max_bas = 3500):
        ecg_data = ecg_data*4.88
        ecg_data[ecg_data > max_bas] = max_bas
        ecg_data[ecg_data < (-1*max_bas)] = -1*max_bas
        return ecg_data/max_bas
    def report(self):
        print("{:^10} {:^10} {:^10}".format('  ','ECGs','Labels'))
        print("{:^10} {:^10} {:^10}".format('TestShape', self.datas.shape,self.labels.shape))
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format('Nums', np.all(self.labels == 1),np.all(self.labels == 0)))
    def info(self,index):
        return self.infos.iloc[index]