import pandas as pd
import random
import numpy as np
import os
from pip import main
from sklearn.utils import resample
from tqdm import tqdm
import pandas as pd
import random
import numpy as np
import os
from tqdm import tqdm
import time

class splite_dataset():                                                                                                                                           
    def __init__(self,folder,drop_duplicates = True):
        self.folder =folder
        self.drop_duplicates = drop_duplicates
        
        self.test_HTN_path= (self.folder+'/test_HTN.pkl')
        self.test_NHTN_path = (self.folder+'/test_NHTN.pkl')
        self.VT_HTN_path = (self.folder+'/VT_HTN.pkl')
        self.VT_NHTN_path =(self.folder+'/VT_NHTN.pkl')
        
        self.info_df = pd.concat([pd.read_pickle(self.test_HTN_path),pd.read_pickle(self.VT_HTN_path),\
            pd.read_pickle(self.test_NHTN_path),pd.read_pickle(self.VT_NHTN_path)])#把所有的数据都拼接起来,用于查询数据时使用
        self.info_df = self.__change_years__(self.info_df)
        
    def __get_test_file_list__(self,shuffer = True,filter_department:str = None,filter_age:int = None):  # type: ignore
        test_HTN_df = self.__change_years__(pd.read_pickle(self.test_HTN_path))  #把HTN_df中的'xx岁'->'xx' 不满'1岁'->'0'  并把年龄改成数值型 
        test_NHTN_df = self.__filter_NHTN_df__(pd.read_pickle(self.test_NHTN_path),filter_department,filter_age)# type: ignore #按照年龄和科室进行过滤 并把年龄改成数值型 
        
        if(self.drop_duplicates):
            test_HTN_df =self.__remove_duplicated__(test_HTN_df)  # type: ignore
            test_NHTN_df =self.__remove_duplicated__(test_NHTN_df)  # type: ignore
        
        self.test_list = list()
        test_HTN_lenth = len(test_HTN_df['ECG_path']) 
        
        if(shuffer):
            test_HTN_df = test_HTN_df.sample(frac=1)#打乱
            test_df = self.__pair_HTNs_(test_HTN_df,test_NHTN_df)
            self.test_list = test_df['ECG_path'].tolist()
            random.shuffle(self.test_list)
        else:
            self.test_list = (test_HTN_df['ECG_path'].tolist())#不打乱,则取所有HTN样本
            self.test_list.extend((test_NHTN_df['ECG_path'].tolist())[:test_HTN_lenth])#不打乱,则取所有NHTN样本
        print('\t')
        print("{:^5} {:^5} {:^5}".format(' ','HTN','NHTN'))
        print("{:^5} {:^5} {:^5}".format('test',test_HTN_lenth,(len(self.test_list)-test_HTN_lenth)))
        return  self.test_list
    
    def __get_VT_file_list__(self,rate=0.3,shuffer = True,filter_department:str = None,filter_age:int = None):  # type: ignore
        VT_HTN_df = self.__change_years__(pd.read_pickle(self.VT_HTN_path))  #把HTN_df中的'xx岁'->'xx' 不满'1岁'->'0'
        VT_NHTN_df = self.__filter_NHTN_df__(pd.read_pickle(self.VT_NHTN_path),filter_department,filter_age)# type: ignore #按照年龄和科室进行过滤 并把年龄改成数值型
        
        if(self.drop_duplicates):#删除重复的个体，只保留最后一个
            VT_HTN_df =self.__remove_duplicated__(VT_HTN_df)# type: ignore
            VT_NHTN_df =self.__remove_duplicated__(VT_NHTN_df)# type: ignore
        
        self.val_list = list()
        self.train_list = list()
        self.addition_train_list = list()
        
        
        if(shuffer):
            VT_HTN_df = VT_HTN_df.sample(frac=1)#打乱
            VT_HTN_size = len(VT_HTN_df['ECG_path'].tolist())
            VT_df = self.__pair_HTNs_(VT_HTN_df,VT_NHTN_df)#根据HTN配对选择NHTN
            VT_list = VT_df['ECG_path'].tolist()
            HTN_list = VT_list[:VT_HTN_size]# 前VT_HTN_size是HTN
            if(VT_HTN_df['ECG_path'].tolist() != HTN_list): #确认一下 前VT_HTN_size是HTN
                print("err!")
                print(len(VT_HTN_df['ECG_path'].tolist()))
                print(len(HTN_list))
                return 
            NHTN_list = VT_list[VT_HTN_size:]
            
            train_size = int(len(HTN_list)*rate) #按照rate 设置 train的样本数量
            self.train_list.extend(HTN_list[:train_size])#前train_size个为训练集
            self.val_list.extend(HTN_list[train_size:])#其余的为验证集
            val_HTN_size = len(self.val_list)
                
            self.train_list.extend(NHTN_list[:train_size])#前train_size个为训练集
            self.val_list.extend(NHTN_list[train_size:])#其余为验证集
            #father_list中的，却没在VT_list中的都为附加集
            father_list =  VT_NHTN_df['ECG_path'].tolist() #VT所有的样本集合
            self.addition_train_list =  [x for x in father_list if x not in VT_list]
            # random.shuffle(self.train_list)
            # random.shuffle(self.val_list)
            # random.shuffle(self.addition_train_list)
            
        else:
            HTN_list = VT_HTN_df['ECG_path'].tolist()
            NHTN_list = VT_NHTN_df['ECG_path'].tolist()
            
            train_size = int(len(HTN_list)*rate) #按照rate 设置 train的样本数量
            self.train_list.extend(HTN_list[:train_size])#前train_size个为训练集    
            self.val_list.extend(HTN_list[train_size:])#其余的为测试集
            val_HTN_size = len(self.val_list)
                
            self.train_list.extend(NHTN_list[:train_size])#前train_size个为训练集
            self.val_list.extend(NHTN_list[train_size:len(HTN_list)])#其余从train_size:len(HTN_list)为训练集（保持和HTN的样本1：1的个数）
            self.addition_train_list.extend(NHTN_list[len(HTN_list):])#从len(HTN_list)往后全为附加集
        
        print('\t')
        print("{:^5} {:^5} {:^5}".format('','HTN','NHTN'))
        print("{:^5} {:^5} {:^5}".format('train',train_size,len(self.train_list)-train_size))
        print("{:^5} {:^5} {:^5}".format('valid',val_HTN_size,len(self.val_list)-val_HTN_size))
        print("{:^5} {:^5} {:^5}".format('add',0,(len(self.addition_train_list))))
            
        
        return self.val_list,self.train_list,self.addition_train_list
    
    # def __get_n_fold_VT_file_list__(self,pair = True,Folds = 5,filter_department:str = None,filter_age:int = None):  # type: ignore
    #     VT_HTN_df = self.__change_years__(pd.read_pickle(self.VT_HTN_path))  #把HTN_df中的'xx岁'->'xx' 不满'1岁'->'0'
    #     VT_NHTN_df = self.__filter_NHTN_df__(pd.read_pickle(self.VT_NHTN_path),filter_department,filter_age)# type: ignore #按照年龄和科室进行过滤 并把年龄改成数值型
        
    #     if(self.drop_duplicates):#删除重复的个体，只保留最后一个
    #         VT_HTN_df =self.__remove_duplicated__(VT_HTN_df)# type: ignore
    #         VT_NHTN_df =self.__remove_duplicated__(VT_NHTN_df)# type: ignore
    #     HTN_list = list()
    #     NHTN_list =list()
    #     addition_list = list()
        
    #     if(True):
    #         VT_HTN_df = VT_HTN_df.sample(frac=1)#打乱
    #         VT_HTN_size = len(VT_HTN_df['ECG_path'].tolist())
    #         VT_df = self.__pair_HTNs_(VT_HTN_df,VT_NHTN_df)#根据HTN配对选择NHTN
    #         VT_list = VT_df['ECG_path'].tolist()
    #         HTN_list = VT_list[:VT_HTN_size]# 前VT_HTN_size是HTN
    #         if(VT_HTN_df['ECG_path'].tolist() != HTN_list): #确认一下 前VT_HTN_size是HTN
    #             print("err!")
    #             print(len(VT_HTN_df['ECG_path'].tolist()))
    #             print(len(HTN_list))
    #             return 
    #         NHTN_list = VT_list[VT_HTN_size:]
    #         #father_list中的，却没在VT_list中的都为附加集
    #         father_list =  VT_NHTN_df['ECG_path'].tolist() #VT所有的样本集合
    #         addition_list =  [x for x in father_list if x not in VT_list]
            
    #     print('\t')
    #     print("{:^5} {:^5} {:^5}".format('','HTN','NHTN'))
    #     print("{:^5} {:^5} {:^5}".format('train',len(HTN_list),len(NHTN_list)))
    #     print("{:^5} {:^5} {:^5}".format('add',0,(len(addition_list))))
            
    #     return HTN_list,NHTN_list,addition_list
    #删除nan年龄，并将年龄转为数值型
    def __change_years__(self,df_input):
        df = df_input.copy()
        df = df.dropna(subset=['years']) #删除years== nan
        df.loc[~(df['years'].str.contains('岁')==True),'years']='0岁' #不含有岁的（天周月）改为"0岁"
        df['years'].replace(regex=True,inplace=True,to_replace=r'岁',value=r'') #删除"岁"
        df["years"] = pd.to_numeric(df["years"],errors='coerce') #把年龄改成数值型
        return df    
    
    #根据年龄、性别抽取出HTN条件相同的NHTN样本，力求正负标签年龄和性别分布相同
    def __pair_HTNs_(self,HTN_df,NHTN_df,ageRang = 10):
        ALL_df = HTN_df.copy() #所有的HNT和抽取出来的NHTN都存放入其中
        NHTN_df_popl = NHTN_df.copy()#即抽即删,抽出一条删一条
        
        for data in HTN_df.itertuples():
            age = int(data[3])
            gender = str(data[4])
            #按条件（年龄、性别等）抽取候选组
            Rang = 1
            NHTN_data_buff = NHTN_df_popl[(((NHTN_df_popl['years'].apply(int))<age+1) & ((NHTN_df_popl['years'].apply(int))>age-1))&(NHTN_df_popl['gender']==gender)]#按条件（年龄、性别等）抽取候选组
            while((len(NHTN_data_buff)<2) and (Rang<ageRang)):#年龄匹配的没有超过3个的话，则扩大搜索范围在ageRang范围内搜索到尽可能和data年龄相近的样本
                NHTN_data_buff = NHTN_df_popl[(((NHTN_df_popl['years'].apply(int))<age+Rang) & ((NHTN_df_popl['years'].apply(int))>age-Rang))&(NHTN_df_popl['gender']==gender)]#按条件（年龄、性别等）抽取候选组
                Rang = Rang+1
            if(len(NHTN_data_buff)<1):
                print("lack sample like :",data)
                NHTN_data_buff = NHTN_df_popl #如果实在没有 就从剩下所有的里面随机抽一个吧
            NHTN_data_buff = NHTN_data_buff.sample(n=1) #随机抽样一个
            ALL_df = ALL_df.append(NHTN_data_buff.iloc[0], ignore_index = True) #抽其中一个添加到ALL_df中
            NHTN_df_popl = NHTN_df_popl[~((NHTN_df_popl['name']==NHTN_data_buff.iloc[0]['name'])&(NHTN_df_popl['num']==NHTN_data_buff.iloc[0]['num']))] #删除抽取出来的行
        return ALL_df
    
    def pair_HTN_by_list_(self,HTN_list,NHTN_list,ageRang = 10):#HTN为需配对的样本 可以不是HTN类
        HTN_size = len(HTN_list)
        HTN_df = self.__read_infos__(HTN_list)
        HTN_df = HTN_df.sample(frac=1) #打乱       
        NHTN_df_popl = self.__read_infos__(NHTN_list)
        NHTN_df_popl = NHTN_df_popl.sample(frac=1) #打乱
        
        ALL_df = self.__pair_HTNs_(HTN_df,NHTN_df_popl,ageRang)
        
        pair_list =(ALL_df['ECG_path'].tolist())
        return pair_list
    #删除重复的样本,先通过ID号筛选，并再用姓名和年龄都一致的筛选
    def __remove_duplicated__(self,df):
        df_remove = df.copy()
        df_remove = df_remove.drop_duplicates(subset=['name','years'],keep='last')#剔除姓名和年龄都一样的，即认为是同一个人
        df1 = df_remove[df_remove['ID']=='']
        df2 = df_remove[~(df_remove['ID']=='')] #有ID号的
        df2 = df2.drop_duplicates(subset=['ID'],keep='last')
        df_remove =pd.concat([df1,df2],axis=0)
        print('\t')
        print("{:^10} {:^10}".format('orginal','fliter duplicated'))
        print("{:^10} {:^10}".format(len(df),len(df_remove)))
        return df_remove
    
    #通过年龄与科室过滤NHTN   
    def __filter_NHTN_df__(self,df,filter_department:str = None,filter_age:int = None):  # type: ignore
        df_filter = df.copy()
        df_filter = self.__change_years__(df_filter) #把df中的'xx岁'->'xx' 不满'1岁'->'0'
        if(filter_department):
            df_filter = df_filter[
                (df_filter['department'].str.contains(filter_department) == True)
            ]#只选择外科
        if(filter_age):
            df_filter = df_filter[
                ((df_filter['years'].apply(int))<filter_age)
            ]#删选年龄
        print('\t')
        print("{:^10} {:^10}".format('orginal','fliter department&age'))
        print("{:^10} {:^10}".format(len(df),len(df_filter)))
        return df_filter
        
    def __read_info__(self,FN): #按照ECG文件名FN查找info信息
        info = self.info_df[(self.info_df['ECG_path']) == FN]
        #print(info)
        return info
    def __read_infos__(self,FN_list):
        df_filted = self.info_df[self.info_df['ECG_path'].isin(FN_list)]
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
        type = FN_without_extension.split('_')[-1]
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
    valid_list,train_list,addition_train_list = data.__get_VT_file_list__(0.3,True)  # type: ignore
    x = [k for k in test_list if k in addition_train_list]
    print(x)
    x = [k for k in test_list if k in valid_list]
    print(x)
    x = [k for k in test_list if k in train_list]
    print(x)
    x = [k for k in train_list if k in test_list]
    print(x)
    x = [k for k in train_list if k in addition_train_list]
    print(x)
# print(test_list,train_list)
# if __name__ == 'main':
#     data = assemble_dataset('/workspace/data/Preprocess_HTN/data/')
#     info = data.__read_info__('20-10046_HTN')[1:]
#     print(info)
#     data.__split__