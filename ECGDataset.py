import numpy as np
import os
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import random
from scipy.signal import butter, lfilter
class ECG_Dataset_Init():
    def __init__(self,data_root:str,filter_age = 0,filter_department = None,rebuild_flage = False): 
        
        self.ECGs_path = data_root+'/ECG/'
        self.INFOs_path = data_root+'/INFO/'
        self.Qualitys_path = data_root+'/Q/'
        self.filter_age = filter_age
        self.filter_department = filter_department
        if((not rebuild_flage) and (os.path.exists(data_root+'/INFOs.pkl'))):
            self.INFOsDf = pd.read_pickle(data_root+'/INFOs.pkl')
            print(len(self.INFOsDf))
        else: #INFOs_df文件不存在/rebuild_flage == Ture，则重构INFOs_df
            self.INFOsDf = self.rebuild_INFOs_Df()
            self.INFOsDf.to_pickle(data_root+'/INFOs.pkl')
        self.INFOsDf = self.filter_INFOs_Df(self.INFOsDf)
        self.testDf,self.tvDf = self.splite_TVandT(self.INFOsDf)                   
    def filter_INFOs_Df(self,df_input):
        df = df_input.copy()
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format( 'nums',len(df[(df['diagnose'].str.contains('高血压') == True)]),len(df[~(df['diagnose'].str.contains('高血压') == True)]) ))
        df = self.__filter__quality__(df)
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format( 'nums',len(df[(df['diagnose'].str.contains('高血压') == True)]),len(df[~(df['diagnose'].str.contains('高血压') == True)]) ))
        df = self.__filter__departmentORlabel__(df,)
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format( 'nums',len(df[(df['diagnose'].str.contains('高血压') == True)]),len(df[~(df['diagnose'].str.contains('高血压') == True)]) ))
        df = self. __change_ages__(df)
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format( 'nums',len(df[(df['diagnose'].str.contains('高血压') == True)]),len(df[~(df['diagnose'].str.contains('高血压') == True)]) ))
        df = self.__filter__agesORlabel__(df)
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format( 'nums',len(df[(df['diagnose'].str.contains('高血压') == True)]),len(df[~(df['diagnose'].str.contains('高血压') == True)]) ))
        df = self.__change__label__(df)
        df = self.__remove_duplicated__(df)
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format( 'nums',len(df[(df['diagnose']==1)]),len(df[~(df['diagnose']==0)]) ))
        
        return df
    # 分割00、20->train&validate set; 21 ->testset
    def splite_TVandT(self,df_input):
        df = df_input.copy()
        df = df[['num','name','ages','gender','department','diagnose','ID','date','ecgFN','q_sum']]
        test_df = df[(df['ecgFN'].str[:3]=='21-')]
        TV_df = df[(df['ecgFN'].str[:3]=='00-') | (df['ecgFN'].str[:3]=='20-')]
        # df1 = test_df.drop_duplicates(subset=['ID'],keep='last')#有ID号的
        # df2 = TV_df.drop_duplicates(subset=['ID'],keep='last')#有ID号的
        # intersected_df = pd.merge(df1, df2, on=['ID','name'], how='inner')
        # print(intersected_df[(intersected_df['diagnose_x'] == 0) | (intersected_df['diagnose_y'] == 0)] )
        print('\n')
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format('ALL',len(df[(df['diagnose']==1)]),len(df[(df['diagnose']==0)])))
        print("{:^10} {:^10} {:^10}".format('testset',len(test_df[(test_df['diagnose']==1)]),len(test_df[(test_df['diagnose']==0)])))
        print("{:^10} {:^10} {:^10}".format('T&V set',len(TV_df[(TV_df['diagnose']==1)]),len(TV_df[(TV_df['diagnose']==0)])))
        return test_df,TV_df
    def get_ECGs_form_FilesList(self,FilesList):
        data = np.zeros((len(FilesList),12,5000))
        i = 0
        for file in tqdm(FilesList):
            data[i] = np.load(self.ECGs_path + file)
            i = i+1
        return    data
    def __change_ages__(self,df_input):
        df = df_input.copy()
        df = df.dropna(subset=['ages']) #删除ages== nan
        df.loc[~(df['ages'].str.contains('岁')==True),'ages']='0岁' #不含有岁的（天周月）改为"0岁"
        df['ages'].replace(regex=True,inplace=True,to_replace=r'岁',value=r'') #删除"岁"
        df["ages"] = pd.to_numeric(df["ages"],errors='coerce') #把年龄改成数值型
        print('\n')
        print("{:^10} {:^10} {:^20}".format('  ','orginal','removed ages NaN'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df)))
        return df    
    def __change__label__(self,df_input):
        df = df_input.copy()
        df = df.dropna(subset=['diagnose']) #删除diagnose== nan
        df.loc[(df['diagnose'].str.contains('高血压')==True),'diagnose']= 1 #diagnose含有高血压的label为1
        df.loc[~(df['diagnose']==1),'diagnose']= 0 #diagnose不含有高血压的label为0
        df['diagnose'] = pd.to_numeric(df['diagnose'],errors='coerce') #把label（diagnose）改成数值型
        # print(df['diagnose'].value_counts())
        print('\n')
        print("{:^10} {:^10} {:^20}".format('  ','orginal','removed diagnose NaN'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df)))
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format('nums',len(df[(df['diagnose']==1)]),len(df[(df['diagnose']==0)])))
        
        return df  
    #QC by q file 
    def __filter__quality__(self,df_input):
        df = df_input.copy()       
        df = df[
                ( (df['q_sum'] == 0))
                ]#q_sum<=1
        print('\n')
        print("{:^10} {:^10} {:^20}".format('  ','orginal','QC'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df)))
        return df
    #department符合条件 或者 诊断为高血压的样本 
    def __filter__departmentORlabel__(self,df_input):  # type: ignore
        df_filter = df_input.copy()
        if(self.filter_department):
            df_filter = df_filter[
                ( (df_filter['diagnose'].str.contains('高血压') == True)| (df_filter['department'].str.contains(self.filter_department) == True))
                ]#只选择外科
            print('\n')
            print("{:^10} {:^10} {:^20}".format('  ','orginal','filtered department'))
            print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_filter)))
        return df_filter
    #年龄符合条件 或者 诊断为高血压的样本 
    def __filter__agesORlabel__(self,df_input):
        df_filter = df_input.copy()
        if(self.filter_age):
            # df_filter = self.__change_ages__(df_filter)#把年龄改为数值型
            df_filter = df_filter[
                    ((df_filter['ages'].apply(int))>self.filter_age) 
                ]#删选年龄
            print('\n')
            print("{:^10} {:^10} {:^20}".format('  ','orginal','filtered ages'))
            print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_filter)))
        return df_filter
    #删除重复出现的样本 只保留最后一个
    def __remove_duplicated__(self,df_input):
        df_remove = df_input.copy()
        print('\n')
        df1 = df_remove[df_remove['ID']=='']
        df1 = df1.sort_values(by=['diagnose'], ascending=[False]) #按照诊断排序，HTN在前
        # print('df1 the same name&ages&gender:',len(df1[df1.duplicated(subset=['name','ages','gender'],keep=False ) ]))
        # print('df1 the same name&ages&gender & diagnose:',len(df1[df1.duplicated(subset=['name','ages','gender','diagnose'],keep=False )]))
        df1 = df1.drop_duplicates(subset=['name','ages','gender'],keep='first')#没有ID号的，剔除姓名和年龄都一样的，即认为是同一个人，保存第一个（尽可能保留高血压患者）
        
        df2 = df_remove[~(df_remove['ID']=='')] #有ID号的 ，相同ID号，保存第一个
        df2 = df2.sort_values(by=['diagnose'], ascending=[False])
        # print('df2 the same name&ages&gender:',len(df2[df2.duplicated(subset=['ID'],keep=False) ]))
        # print('df2 the same name&ages&gender & diagnose:',len(df2[df2.duplicated(subset=['ID','diagnose'],keep=False )]))
        df2 = df2.drop_duplicates(subset=['ID'],keep='first')
        
        df_remove =pd.concat([df1,df2],axis=0)
        print('\n')
        print("{:^10} {:^10} {:^20}".format('  ','orginal','removed duplicated'))
        print("{:^10} {:^10} {:^20}".format('nums',len(df_input),len(df_remove)))
        return df_remove
    #创建INFOs_Df 包含样本信息、文件名、ECG各个质量情况
    def rebuild_INFOs_Df(self):
        INFOsList = os.listdir(self.INFOs_path)
        INFOs_df = pd.DataFrame(index=range(len(INFOsList)),columns=['num','name','ages','gender','diagnose','department','ID','date','ecgFN','q_sum']) 
        i = 0
        for infoFN in tqdm(INFOsList):
            year = infoFN[:2] 
            info = ((pd.read_pickle(self.INFOs_path+infoFN))[0]).tolist()[1:]#取datafram的第0行转list，去掉第一行的index，因为保存时候不能不保存index

            ecgFN = str(infoFN[:-4]+'.npy')
            if(year == '00'):
                info.extend(['','','',ecgFN])
            # elif(year == '21'):##################################################################################################################################################################
            #     info[4], info[5] = info[5], info[4]  #交换4、5，使得 department 和 diagnose对应正确 ['序号','姓名','年龄','性别','申请部门']-> ['序号','姓名','年龄','性别','临床诊断']############
            #     info.extend([ecgFN])#############################################################################################################################################################
            elif(year == '20' or year == '21'):
                info.extend([ecgFN])
            if(os.path.exists(self.Qualitys_path+infoFN)):
                try:
                    q = pd.read_pickle(self.Qualitys_path+infoFN)
                    q_sum = int (np.array(q)[:,:12].sum())
                except:

                    print('QC data err :', infoFN)
                    q_sum = 36
            else:
                print(infoFN," Not Find")
            info.append(q_sum)
            INFOs_df.iloc[i] = info  # type: ignore
            i = i + 1
        # INFOs_df.to_pickle(data_root+'/INFOs.pkl')
        return INFOs_df
    def report(self):
        print("{:^10} {:^10} {:^10}".format('  ','HTN','NHTN'))
        print("{:^10} {:^10} {:^10}".format('TestSet',len(self.testDf[(self.testDf['diagnose']==1)]),len(self.testDf[(self.testDf['diagnose']==0)])))
        print("{:^10} {:^10} {:^10}".format('TVSet',len(self.tvDf[(self.tvDf['diagnose']==1)]),len(self.tvDf[(self.tvDf['diagnose']==0)])))
        print("{:^10} {:^10} {:^10}".format('ALL',len(self.INFOsDf[(self.INFOsDf['diagnose']==1)]),len(self.INFOsDf[(self.INFOsDf['diagnose']==0)])))
    
        
class ECG_Dataset(Dataset):
    def __init__(self,data_root,infos,preprocess = True,onehot_lable=False):
        self.ECGs_path = data_root+'/ECG/'
        self.INFOs_path = data_root+'/INFO/'
        self.Qualitys_path = data_root+'/Q/'
        
        self.infos = infos
        self.datas = self.get_ECGs_form_FilesList(self.infos['ecgFN'].tolist())
        self.labels = self.infos['diagnose'].tolist()
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
        for file in tqdm(FilesList):
            data[i] = np.load(self.ECGs_path + file)
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
    
    
def bandpass_filter(data, lowcut, highcut, signal_freq, filter_order):
        """
        Method responsible for creating and applying Butterworth filter.
        :param deque data: raw data
        :param float lowcut: filter lowcut frequency value
        :param float highcut: filter highcut frequency value
        :param int signal_freq: signal frequency in samples per second (Hz)
        :param int filter_order: filter order
        :return array: filtered data
        """
        nyquist_freq = 0.5 * signal_freq
        low = lowcut / nyquist_freq
        high = highcut / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, data)
        return y
    
if __name__ == '__main__':
    ALLDataset = ECG_Dataset_Init('/workspace/data/Preprocess_HTN/data_like_pxl/',filter_age= 18,filter_department='外科',rebuild_flage=False)
    print(ALLDataset.testDf.head())
    
        
        