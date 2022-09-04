from random import sample
from wsgiref import validate
import numpy as np
import os
from tqdm import tqdm
import torch
import torch.utils.data as Data
from sklearn import preprocessing
from sklearn.model_selection import train_test_split



def load_data(npy_folder,EcgChannles_num = 12 ,EcgLength_num =5000):
    seq_files = os.listdir(npy_folder)#返回指定的文件夹包含的文件或文件夹的名字的列表
    seq_files.sort(key=lambda x:int(x.split('_')[0])) #按“.”分割，并把分割结果的[0]转为整形并排序
    sample_num = len(seq_files)
    ECGs = np.empty([sample_num,EcgChannles_num,EcgLength_num], dtype = float)
    for i in tqdm(range(sample_num)):
        file_path = os.path.join(npy_folder,seq_files[i])
        X = np.load(file_path)
        ECGs[i] = X[:EcgChannles_num,:EcgLength_num]
    return ECGs

def load_label(lable_file_path):
    return np.load(lable_file_path)

'''
# input： x(sample_nums,changnal,timesteps)
# function : normalize each feacture for each sample
'''
def MAX_MIN_normalization_by_feactures(x,feature_rangetuple=(-1,1) ):
    x_normalized = x.copy()
    for i,data in enumerate(x,0):
        data_swap = data.swapaxes(0,1) # transfor to (timesteps,changnal), fit the MinMaxScaler(),who's input shape is (samples_nums,features) 
        min_max_scaler = preprocessing.MinMaxScaler()#默认为范围0~1，拷贝操作
        x_normalized[i] = (min_max_scaler.fit_transform(data_swap)).swapaxes(0,1)  # turn shape back to (changnal,timesteps)
    return x_normalized

def get_k_fold_dataset(fold,x,y,k = 5):
    if(fold<=0):#防止fold过小
        fold = 1
    x = MAX_MIN_normalization_by_feactures(x)
    x = torch.FloatTensor(x)  #turn numpy to tensor
    y = torch.LongTensor(y)

    data_len_for_each_fold_nums = int(len(y)/k)
    validate_end_index = int(data_len_for_each_fold_nums*fold)

    if(validate_end_index >len(y)): #防止超长度
        validate_end_index = len(y)
    elif(validate_end_index<data_len_for_each_fold_nums):#防止超长度
        validate_end_index = data_len_for_each_fold_nums
    validate_start_index = int(validate_end_index - data_len_for_each_fold_nums)
    
    INDEX_LIST = torch.arange(0,len(y))
    validate_INDEX_LIST = torch.arange(validate_start_index,validate_end_index) # 左闭右开[validate_start_index,validate_end_index)
    validate_MARK_INDEX = torch.isin(INDEX_LIST,validate_INDEX_LIST) # 把validate对应的INDEX位置用Ture标志出来
                                                                  # train对应的位置就是False
    validate_x = x[validate_MARK_INDEX]
    validate_y = y[validate_MARK_INDEX]
    train_x = x[~validate_MARK_INDEX]
    train_y = y[~validate_MARK_INDEX]
    print(('*'*50+"The %2d fold" % fold)+'*'*50)
    validate_end_index = int(data_len_for_each_fold_nums*fold)
    print("validate form %5d to %5d , all nums is %5d" % (validate_start_index,validate_end_index,len(validate_y)))
    validate_dataset = Data.TensorDataset(validate_x, validate_y)
    train_dataset = Data.TensorDataset(train_x, train_y)
    return train_dataset,validate_dataset



def load_numpy_dataset_to_tensor_dataset(x,y,random_seed,train_rate = 0.8):
    torch.manual_seed(random_seed) 
    x = MAX_MIN_normalization_by_feactures(x)
    x = torch.FloatTensor(x)  #turn numpy to tensor
    y = torch.LongTensor(y)
    dataset = Data.TensorDataset(x, y)
    print(dataset.__len__())
    train_size = int(train_rate * len(y))
    valid_size = int(len(y) - train_size)
    train_dataset,valid_dataset = Data.random_split(dataset,[train_size,valid_size],generator=torch.Generator().manual_seed(random_seed))
    print("               HTN  NHTN")
    print("valid_dataset: %d  %d" % ( (valid_dataset[:][1]).sum(),valid_dataset.__len__()-(valid_dataset[:][1]).sum() ) )
    print("train_dataset: %d  %d" % ( (train_dataset[:][1]).sum(),train_dataset.__len__()-(valid_dataset[:][1]).sum() ) )
    return  train_dataset,valid_dataset

