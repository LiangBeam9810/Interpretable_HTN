from cProfile import label
import numpy as np
import os
from tqdm import tqdm
import random
import torch
import torch.utils.data as Data
from torch.utils.data.dataset import Dataset
import pywt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import xml.dom.minidom as dm


class ECG_Dataset(Dataset):
    def __init__(self,npy_folder,EcgChannles_num,EcgLength_num,shadow_npy_folder = None,xml_folder = None):
    
        self.npy_root = npy_folder
        npy_files_list = os.listdir(npy_folder)#返回指定的文件夹包含的文件或文件夹的名字的列表
        npy_files_list.sort(key=lambda x:int(x.split('_')[0])) #按“_”分割，并把分割结果的[0]转为整形并排序
        self.npys = npy_files_list

        self.Channles_size = EcgChannles_num
        self.Length_size = EcgLength_num

        self.xml_root = xml_folder
        # for i in range(len(self.npys)):
        #     self.filter_outlier_npy(i)
        #     if(i >= len(self.npys)-1):
        #         break
        print('npys:{%d}',len(self.npys))
        self.shadow_npy_root = shadow_npy_folder #存放了比正样本多出来很多的负样本
        if(self.shadow_npy_root):
            shadow_npy_files_list = os.listdir(self.shadow_npy_root)
            shadow_npy_files_list.sort(key=lambda x:int(x.split('_')[0])) #按“_”分割，并把分割结果的[0]转为整形并排序
            self.shadow_npys = shadow_npy_files_list
            # for i in range(len(self.shadow_npys)):
            #     self.filter_outlier_shadow(i)
            #     if(i >= len(self.shadow_npys)-1):
            #         break
            print('shadow_npys:{%d}',len(self.shadow_npys))
    def __getitem__(self, item):
        label = 1 if (((((self.npys[item]).split('.'))[0]).split('_'))[1]) =='HTN' else 0 #先按“.”分割，并把分割结果的[0]再按“_"分割，结果的[-1](最后一个)即为
        
        if((self.shadow_npy_root == None) or (label == 1)):#如果是没有开启负样本抽样/正样本的话，正常读取
            npy_path = os.path.join(self.npy_root,self.npys[item])
            ECG =  (np.load(npy_path))[:self.Channles_size,:self.Length_size]
        else:#如果是负样本，就从所有的shadow_npy负样本中随机抽一个
            #print("reselect the other normal sample. ")
            if random.random()>0.5:
                item_ = random.randint(0,len(self.shadow_npys)-1)
                while( (((((self.shadow_npys[item_]).split('.'))[0]).split('_'))[-1]) =='HTN'):
                    item_ = random.randint(0,len(self.shadow_npys)-1)
                npy_path = os.path.join(self.shadow_npy_root,self.shadow_npys[item_])
            else :
                npy_path = os.path.join(self.npy_root,self.npys[item])
            
            ECG =  (np.load(npy_path))[:self.Channles_size,:self.Length_size]
        #ECG = denoise(ECG)
        ECG = amplitude_limiting(ECG,3500) #幅值
        ECG[np.isnan(ECG)]=0
        ECG = torch.FloatTensor(ECG)
        label = torch.from_numpy(np.array(label))
        #print(self.npys[item])
        return ECG, label

    def filter_outlier_npy(self, item):
        npy_path = os.path.join(self.npy_root,self.npys[item])
        ECG =  (np.load(npy_path))[:self.Channles_size,:self.Length_size]
        if((ECG.min() ==  -32768) or (ECG.max() ==  32768)):
            self.deleteitem_npys(item)
            print(npy_path)
            return True
        return False
    def filter_outlier_shadow(self, item):
        npy_path = os.path.join(self.shadow_npy_root,self.shadow_npys[item])
        ECG =  (np.load(npy_path))[:self.Channles_size,:self.Length_size]
        if((np.sum(ECG == -32768) +np.sum(ECG == 32768))>=5000):
        #if((ECG.min() ==  -32768) or (ECG.max() ==  32768)):
            self.deleteitem_shadow_npys(item)
            print(npy_path)
            return True
        return False

    def deleteitem_npys(self, item):
        del self.npys[item]
        return 

    def deleteitem_shadow_npys(self, item):
        del self.shadow_npys[item]
        return 

    def npy_path(self, item):
        return os.path.join(self.npy_root,self.npys[item])

    def sample_name(self,item):#获取没有后缀的文件名
        return str(self.npys[item].split('.')[0])

    def __len__(self):
        return len(self.npys)

    def size(self):
        return len(self.npys), self.Channles_size,self.Length_size

    def xml_path(self, item):
        if(self.xml_root):
            xml_path = self.sample_name(item)+'.xml'
            return os.path.join(self.xml_root,xml_path)
        else:
            print('xml_path is None.')

    def name_date(self, item):
        if(self.xml_root):
            xml_path = self.xml_path(item)
            xml_doc = dm.parse(xml_path) #打开该xml文件
            try:
                name =  (xml_doc.getElementsByTagName('name'))[1].childNodes[0].data
            except :
                name = "error"
            try:
                date = (xml_doc.getElementsByTagName('center'))[0].getAttribute("value")
            except :
                date = "error"
            return name,date
        else:
            print('xml_path is None.')


def amplitude_limiting(ecg_data,max_bas = 3500):
    ecg_data = ecg_data*4.88
    ecg_data[ecg_data > max_bas] = max_bas
    ecg_data[ecg_data < (-1*max_bas)] = -1*max_bas
    return ecg_data/max_bas

# 小波去噪预处理
def denoise(data):
    # 小波变换
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # 阈值去噪
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # 小波反变换,获取去噪后的信号
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    rdata[np.isnan(rdata)]=0
    return rdata


def get_ECG_form_xml(xml_path,EcgChannles_num,EcgLength_num):
    xml_doc = dm.parse(xml_path) #打开该xml文件
    nope_root = xml_doc.getElementsByTagName('digits')#寻找名为digits的子类
    ECG = np.empty([EcgChannles_num,EcgLength_num], dtype = int)
    for channle_i in (range(EcgChannles_num)):  #遍历通道
        list_buf = nope_root[channle_i].childNodes[0].data.split(" ") ##第i导联数据第i组序列中 取出text序列，并以‘ ’分割，得到list
        len_of_buf = len(list_buf)
        list_buf = list_buf[:len_of_buf-1] #去掉最后一个空位
        len_of_buf = len(list_buf)
        #print('Channel: '+str(channle_i)+" Length: "+str(len_of_buf))
        for j in range(EcgLength_num):        #取前int_sequence_num个数据点
            if(j > (len_of_buf - 1)):         #过短时补零
                ECG[channle_i,j] = 0
            else:
                ECG[channle_i,j] = eval(list_buf[j]) #转化为数值型
    return ECG

def mark_input_numpy(input,lable,mark_time = 1):
    sample_num,channelsize,sqenlenth = input.shape
    x = np.zeros((sample_num*mark_time,channelsize,sqenlenth))
    for i in range(sample_num*mark_time):
        if(i<sample_num):
            x[i]=input[i]
        else:
            original_index = int(i%sample_num) #copy the original_index th ecg form original input, ready to be marked
            x[i]=input[original_index]
            mark_lenth = torch.randint(int(sqenlenth/15),int(sqenlenth/10),[1]) 
            mark = np.zeros([mark_lenth])
            mark_index = np.random.randint(mark_lenth,sqenlenth-mark_lenth, size=[1])
            for j in range(channelsize):
                x[i,j,mark_index[0]:mark_index[0]+mark_lenth]=mark
            
    return x,np.tile(lable,mark_time)

def sliding_window(input,lable,sliding_lenth = 1000,stride_factor = 2,sequence_size = 5000):
    samlpe_num,channel_size,_ = input.shape
    stride = int(sliding_lenth/stride_factor)
    print(stride)
    ECG_sliding = np.zeros(((int(((sequence_size/sliding_lenth))*stride_factor)-stride_factor+1)*(samlpe_num),channel_size,sliding_lenth))
    lable_sliding = []
    ECG_buff = np.zeros((channel_size,sliding_lenth))
    num = 0
    for i in tqdm((range(samlpe_num))):
        for start_index in range(0,sequence_size,stride):
            #print(i,":",start_index)
            if(start_index+sliding_lenth > sequence_size):
                break;
            ECG_buff = input[i,:,start_index:start_index+sliding_lenth]
            #print(ECG_buff.shape)
            #print(ECG_sliding.shape)
            ECG_sliding[num] = ECG_buff
            #ECG_sliding = np.concatenate((ECG_sliding,ECG_buff),axis=0)
            lable_sliding.append(lable[i])
            num+=1
    print(num,(int(((sequence_size/sliding_lenth))*stride_factor)-stride_factor+1)*(samlpe_num))
    #ECG_sliding =np.delete(ECG_sliding, 0, 0)#删除掉第0行，即最开始的空行
    lable_sliding = np.array(lable_sliding)
    return ECG_sliding,lable_sliding

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
# function : normalize each feacture for all sample
'''
def MAX_MIN_normalization_by_feactures(x,feature_range=(-1,1) ):
    sample_nums,changnal,timesteps = x.shape
    x_swap = x.swapaxes(1,2) #(sample_nums,timesteps,changnal)
    x_swap = x_swap.reshape(-1,changnal)#(sample_nums*timestep,,changnal)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))#默认为范围0~1，拷贝操作
    x_swap = min_max_scaler.fit_transform(x_swap)
    x_swap = x_swap.reshape(-1,timesteps,changnal) #(-1,timesteps,changnal)
    x_swap = x_swap.swapaxes(1,2) #(sample_nums,changnal,timesteps)
    return x_swap

def z_score_normalization_by_feactures(x ):
    sample_nums,changnal,timesteps = x.shape
    x_swap = x.swapaxes(1,2) #(sample_nums,timesteps,changnal)
    x_swap = x_swap.reshape(-1,changnal)#(sample_nums*timestep,,changnal)
    z_score_scaler = preprocessing.StandardScaler()#默认为范围0~1，拷贝操作
    x_swap = z_score_scaler.fit_transform(x_swap)
    x_swap = x_swap.reshape(-1,timesteps,changnal) #(-1,timesteps,changnal)
    x_swap = x_swap.swapaxes(1,2) #(sample_nums,changnal,timesteps)
    return x_swap


def get_k_fold_dataset(fold,x,y,k = 5 ,random_seed =1):
    if(k <= 1): #当k = 1时，就按照8：2的比列分配训练集和测试集
        k = 1
        train_dataset,validate_dataset = load_numpy_dataset_to_tensor_dataset(x,y,random_seed=random_seed)
        return train_dataset,validate_dataset
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

def load_numpy_dataset_to_tensor_dataset(x,y):
    x = z_score_normalization_by_feactures(x)
    x = torch.FloatTensor(x)  #turn numpy to tensor
    y = torch.LongTensor(y)
    return Data.TensorDataset(x, y)  


def load_numpy_dataset_to_tensor_dataset_split(x,y,random_seed,train_rate = 0.8):
    torch.manual_seed(random_seed) 
    #x = MAX_MIN_normalization_by_feactures(x)
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

