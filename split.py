from http.client import CONTINUE
from macpath import split
import numpy as np
from tqdm import tqdm
from ecgdetectors import Detectors
import os 
import shutil


fs = 500 #采样率 sample rate
detectors = Detectors(fs)
data_folder = './data/shadow'
output_folder = './data_split/shadow'

Channles_size = 12
Length_size = 5000
split_len = 1500

npy_files_list = os.listdir(data_folder)
npy_files_list.sort(key=lambda x:int(x.split('_')[0])) #按“_”分割，并把分割结果的[0]转为整形并排序

def RemoveDir(filepath):
    '''
    如果文件夹不存在就创建，如果文件存在就清空！
    
    '''
    if not os.path.exists(filepath):
        os.mkdir(filepath)
    else:
        shutil.rmtree(filepath)
        os.mkdir(filepath)


RemoveDir(output_folder)

for npy in tqdm(npy_files_list):
    
    npy_path = os.path.join(data_folder,npy)
    ECG =  (np.load(npy_path))[:Channles_size,:Length_size]
    if((ECG.min() ==  -32768) or (ECG.max() ==  32768)):
        continue
    r_peaks = detectors.hamilton_detector(ECG[0,])
    for i in range(len(r_peaks)-2):
        ecg_splite_np = np.zeros((Channles_size,split_len))

        if r_peaks[i]<50 or r_peaks[i+2]>Length_size-50: break #

        rr_len = r_peaks[i+2]-r_peaks[i]

        if(rr_len > 1200 or rr_len < 500):#rr间隔过长或者过短
            continue
        ecg_splite_np[:,0:rr_len+100] = ECG[:,r_peaks[i]-50:r_peaks[i+2]+50]
        output_name = str(npy.split('.')[0])+"_"+str(i)+'.npy'
        np.save(os.path.join(output_folder,output_name),ecg_splite_np)


