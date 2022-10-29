import pandas as pd
import random
import numpy as np
import os
from pip import main
from sklearn.utils import resample
from tqdm import tqdm
class splite_dataset():
    def __init__(self,folder,drop_duplicates = True):
        self.folder =folder
        self.drop_duplicates = drop_duplicates
        
        self.test_HTN_path= (self.folder+'/test_HTN.pkl')
        self.test_NHTN_path = (self.folder+'/test_NHTN.pkl')
        self.VT_HTN_path = (self.folder+'/VT_HTN.pkl')
        self.VT_NHTN_path =(self.folder+'/VT_NHTN.pkl')
        self.test_list = list()
        
        self.val_list = list()
        self.train_list = list()
        # if(shuffer):
        #     random.shuffle(self.HTN_files_list)
        #     random.shuffle(self.NHTN_files_list)
        
        self.val_list = list()
        self.train_list = list()
        self.addition_train_list = list()
    
    def __get_test_file_list__(self,shuffer = True):
        test_HTN_df = pd.read_pickle(self.test_HTN_path)
        # test_NHTN_df = pd.read_pickle(self.folder+'/validate_NHTN.pkl')
        test_NHTN_df = self.__filter_NHTN_df__(pd.read_pickle(self.test_NHTN_path))# type: ignore #按照年龄和科室进行过滤
        
        if(self.drop_duplicates):
            test_HTN_df =self.__remove_duplicated__(test_HTN_df)  # type: ignore
            test_NHTN_df =self.__remove_duplicated__(test_NHTN_df)  # type: ignore
        
        self.test_list = list()
        self.test_list.extend(test_HTN_df['ECG_path'].tolist())#先把所有HTN添加到列表中
        test_HTN_lenth = len(self.test_list) 
        
        NHTN_list = test_NHTN_df['ECG_path'].tolist()
        
        
        if(shuffer):
            random.shuffle(NHTN_list) #随机打乱
            self.test_list.extend(NHTN_list[:test_HTN_lenth])#取前test_HTN_lenth个NHTN样本，保证样本平衡
            random.shuffle(self.test_list) #随机打乱
        else:
            self.test_list.extend(NHTN_list)#不打乱,则取所有NHTN样本
        print('\t')
        print("{:^5} {:^5} {:^5}".format(' ','HTN','NHTN'))
        print("{:^5} {:^5} {:^5}".format('test',test_HTN_lenth,(len(self.test_list)-test_HTN_lenth)))
        return  self.test_list
    
    
    def __get_VT_file_list__(self,rate=0.3,shuffer = True):
        VT_HTN_df = pd.read_pickle(self.VT_HTN_path)
        # VT_NHTN_df = pd.read_pickle(self.folder+'/VT_NHTN.pkl')
        VT_NHTN_df = self.__filter_NHTN_df__(pd.read_pickle(self.VT_NHTN_path))# type: ignore #按照年龄和科室进行过滤
        
        if(self.drop_duplicates):
            VT_HTN_df =self.__remove_duplicated__(VT_HTN_df)# type: ignore
            VT_NHTN_df =self.__remove_duplicated__(VT_NHTN_df)# type: ignore
        
        
        # self.val_list = list()
        # self.train_list = list()
        # self.addition_train_list = list()
        
        HTN_list = VT_HTN_df['ECG_path'].tolist()
        NHTN_list = VT_NHTN_df['ECG_path'].tolist()
        if(shuffer):
            random.shuffle(HTN_list) #打乱
            val_size = int(len(HTN_list)*rate) #按照rate 设置 test的样本数量
            self.val_list.extend(HTN_list[:val_size])#前val_size个为测试集
            val_HTN_size = len(self.val_list)
            self.train_list.extend(HTN_list[val_size:])#其余的为训练集
            train_HTN_size = len(self.train_list)
            
            random.shuffle(NHTN_list) #打乱
            self.val_list.extend(NHTN_list[:val_size])#前val_size个为测试集
            self.train_list.extend(NHTN_list[val_size:len(HTN_list)])#其余从val_size:len(HTN_list)为训练集（保持和HTN的样本1：1的个数）
            self.addition_train_list.extend(NHTN_list[len(HTN_list):])#从len(HTN_list)往后全为附加集
        
            print('\t')
            print("{:^5} {:^5} {:^5}".format('','HTN','NHTN'))
            print("{:^5} {:^5} {:^5}".format('train',train_HTN_size,len(self.train_list)-train_HTN_size))
            print("{:^5} {:^5} {:^5}".format('valid',val_HTN_size,len(self.val_list)-val_HTN_size))
            print("{:^5} {:^5} {:^5}".format('add',0,(len(self.addition_train_list))))
            
        
            return self.val_list,self.train_list,self.addition_train_list
        else:
            val_size = int(len(HTN_list)*rate) #按照rate 设置 test的样本数量
            self.val_list.extend(HTN_list[:val_size])#前val_size个为测试集
            val_HTN_size = len(self.val_list)
            self.train_list.extend(HTN_list[val_size:])#其余的为训练集
            train_HTN_size = len(self.train_list)
            
            val_size = int(len(NHTN_list)*rate) #按照rate 设置 test的样本数量
            self.val_list.extend(NHTN_list[:val_size])#前val_size个为测试集
            self.train_list.extend(NHTN_list[val_size:])#其余的为训练集
            
            print('\t')
            print("{:^5} {:^5} {:^5}".format('','HTN','NHTN'))
            print("{:^5} {:^5} {:^5}".format('train',train_HTN_size,len(self.train_list)-train_HTN_size))
            print("{:^5} {:^5} {:^5}".format('valid',val_HTN_size,len(self.val_list)-val_HTN_size))
            print("{:^5} {:^5} {:^5}".format('add',0,(len(self.addition_train_list))))
            
            return self.val_list,self.train_list,self.addition_train_list
    
    #删除重复的ID
    def __remove_duplicated__(self,df):
        df_remove = df.copy()
        df1 = df_remove[df_remove['ID']=='']
        df2 = df_remove[~(df_remove['ID']=='')]
        df_remove =pd.concat([df1,df2.drop_duplicates(subset=['ID'],keep='last')],axis=0)
        print('\t')
        print("{:^10} {:^10}".format('orginal','fliterID'))
        print("{:^10} {:^10}".format(len(df),len(df_remove)))
        return df_remove
        
    #通过年龄与科室过滤NHTN   
    def __filter_NHTN_df__(self,df):
        df_filter = df.copy()
        df_filter = df_filter.dropna(subset=['years']) #删除years== nan
        df_filter.loc[~(df_filter['years'].str.contains('岁')),'years']='0岁' #不含有岁的（天周月）改为"0岁"
        df_filter['years'].replace(regex=True,inplace=True,to_replace=r'岁',value=r'') #删除"岁"
        # df_filter = df_filter[
        #     (((df_filter['years'].apply(int))<55) &(df_filter['department'].str.contains('外科')))|
        #     (((df_filter['years'].apply(int))<50) &(df_filter['department']==''))
        #    ]#两种条件
        df_filter = df_filter[
            (df_filter['department'].str.contains('外科'))==True
           ]#只选择外科
        print('\t')
        print("{:^10} {:^10}".format('orginal','fliteryears'))
        print("{:^10} {:^10}".format(len(df),len(df_filter)))
        return df_filter
        
    def __read_info__(self,FN): #按照ECG文件名FN查找info信息
        type = FN.split('_')[1]
        if(type == "HTN.npy"):
            info_df = pd.concat([pd.read_pickle(self.test_HTN_path),pd.read_pickle(self.VT_HTN_path)])#把
        else:
            info_df = pd.concat([pd.read_pickle(self.test_NHTN_path),pd.read_pickle(self.VT_NHTN_path)])
        info = info_df[(info_df['ECG_path']) == FN]
        #print(info)
        return info
    def __read_infos__(self,FN_list):
        info_df = pd.concat([pd.read_pickle(self.test_HTN_path),pd.read_pickle(self.VT_HTN_path),\
            pd.read_pickle(self.test_NHTN_path),pd.read_pickle(self.VT_NHTN_path)])#把所有的数据都拼接起来
        df_filted = info_df[info_df['ECG_path'].isin(FN_list)]
        return df_filted
    def get_gender_years(self,FN):
        info = self.__read_info__(FN)
        gender = info['gender'].tolist()
        years = info['years'].tolist()
        return gender,years
                     
        

class assemble_dataset():
    def __init__(self,folder):
        self.folder =folder
        self.HTN_folder = folder+'/pkl/HTN/'
        self.NHTN_folder = folder+'/pkl/NHTN/'
        self.ECG_folder = folder+'/ECG/'
            
        self.HTN_files_list = os.listdir(self.HTN_folder)#返回指定的文件夹包含的文件或文件夹的名字的列表
        self.NHTN_files_list = os.listdir(self.NHTN_folder)
        
        # if(shuffer):
        #     random.shuffle(self.HTN_files_list)
        #     random.shuffle(self.NHTN_files_list)
            
    def __read_info__(self,FN_without_extension):
        type = FN_without_extension.split('_')[1]
        if(type == "HTN"):
            path = self.HTN_folder+FN_without_extension+'.pkl'
        else:
            path = self.NHTN_folder+FN_without_extension+'.pkl'
        info = ((pd.read_pickle(path))[0]).tolist()
        return info
    def __split___(self):
        test_HTN_df = pd.DataFrame(columns=['num','name','years','gender','department','diagnose','ID','date','ECG_path']) 
        test_NHTN_df = pd.DataFrame(columns=['num','name','years','gender','department','diagnose','ID','date','ECG_path']) 
        VT_HTN_df = pd.DataFrame(columns=['num','name','years','gender','department','diagnose','ID','date','ECG_path']) 
        VT_NHTN_df = pd.DataFrame(columns=['num','name','years','gender','department','diagnose','ID','date','ECG_path']) 

        for file in tqdm(self.HTN_files_list): 
            year = file[:2] 
            info = list()
            if(year == '21'):
                info = (self.__read_info__(file[:-4]))[1:]#去掉第一行的标号，因为保存的时候忘了不保存标号了
                str_1 = str(file[:-4]+'.npy')
                info.append(str_1)#添加ECG地址
                test_HTN_df.loc[len(test_HTN_df.index)] = info  # type: ignore
            elif(year == '00'):
                
                info = (self.__read_info__(file[:-4]))[1:]#去掉第一行的标号，
                info.extend(['','','','',str(file[:-4]+'.npy')])#添加空缺的空信息为''#添加ECG地址
                VT_HTN_df.loc[len(VT_HTN_df.index)] = info  # type: ignore
            elif(year == '20'):
                info = (self.__read_info__(file[:-4]))[1:]#去掉第一行的标号，
                str_1 = str(file[:-4]+'.npy')
                info.append(str_1)#添加ECG地址
                VT_HTN_df.loc[len(VT_HTN_df.index)] = info  # type: ignore
                
        for file in tqdm(self.NHTN_files_list):  
            year = file[:2] 
            info = list()
            if(year == '21'):
                info = (self.__read_info__(file[:-4]))[1:]#去掉第一行的标号，
                str_1 = str(file[:-4]+'.npy')
                info.append(str_1)#添加ECG地址
                test_NHTN_df.loc[len(test_NHTN_df.index)] = info  # type: ignore
            elif(year == '00'):
                
                info = (self.__read_info__(file[:-4]))[1:]#去掉第一行的标号，
                info.extend(['','','','',str(file[:-4]+'.npy')])#添加空缺的空信息为''#添加ECG地址
                VT_NHTN_df.loc[len(VT_NHTN_df.index)] = info  # type: ignore
            elif(year == '20'):
                info = (self.__read_info__(file[:-4]))[1:]#去掉第一行的标号
                str_1 = str(file[:-4]+'.npy')
                info.append(str_1)#添加ECG地址
                VT_NHTN_df.loc[len(VT_NHTN_df.index)] = info  # type: ignore       
                
        
        test_HTN_df.to_pickle(self.folder+'/validate_HTN.pkl')
        test_NHTN_df.to_pickle(self.folder+'/validate_NHTN.pkl')
        VT_HTN_df.to_pickle(self.folder+'/VT_HTN.pkl')
        VT_NHTN_df.to_pickle(self.folder+'/VT_NHTN.pkl')


if __name__ == '__main__':
    print("   ")
    data = splite_dataset('/workspace/data/Preprocess_HTN/data/',True)
    test_list = data.__get_test_file_list__(True)
    # print(valid_list)
    valid_list,train_list,addition_train_list = data.__get_VT_file_list__(0.3,True)
    x = [k for k in test_list if k in addition_train_list]
    print(x)
# print(test_list,train_list)
# if __name__ == 'main':
#     data = assemble_dataset('/workspace/data/Preprocess_HTN/data/')
#     info = data.__read_info__('20-10046_HTN')[1:]
#     print(info)
#     data.__split___()
