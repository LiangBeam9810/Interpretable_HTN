import ECGDataset 
import Models 
import Net
from train_test_validat import *
from self_attention import *
import matplotlib.pyplot as plt
import ecg_plot
import cam
import ECGplot
import ECGHandle
import torch
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import random
import pandas as pd
from tqdm import tqdm

import time
import math
import os
from captum.attr import Occlusion


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

EcgChannles_num = 12
EcgLength_num = 5000
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
DEVICE = "cpu"
seed_torch(2023)

data_root = '/workspace/data/Preprocess_HTN/datas_/'
Models_path = '/workspace/data/Interpretable_HTN/model/20230212_031236/20230212_031236/parameter_EarlyStoping_4.pt'
NET = [Net.MLBFNet_GUR_o(True,True,True,2,Dropout_rate=0.3) ] # type: ignore

def get_occlusion_value(index):
        occlusion = Occlusion(testmodel)
        inputs,labels = test_dataset.__getitem__(index)
        info =test_dataset.infos.iloc[index]
        inputs = inputs.unsqueeze(0)
        ECGfile_name = info['ECGFilename']
        attributions_occ = occlusion.attribute(inputs,
                                            strides = (1, 10), # 遮挡滑动移动步长
                                            target=labels, # 目标类别
                                            sliding_window_shapes=(12, 200), # 遮挡滑块尺寸
                                            baselines=0)
        Occlusion_vlue = attributions_occ.squeeze(0).numpy()
        np.save(Occlusion_root+ECGfile_name,Occlusion_vlue)

if __name__ == '__main__':
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
    # ALL_data = ECGHandle.filter_diagnose(ALL_data,'阻滞')
    ALL_data = ECGHandle.remove_duplicated(ALL_data)
    ALL_data = ALL_data.rename(columns={'住院号':'ID','年龄':'age','性别':'gender','姓名':'name'}) 
    ALL_data_buffer = ALL_data.copy()
    seed_torch(2023)
    ALL_data_buffer = ALL_data_buffer.sample(frac=1).reset_index(drop=True) #打乱顺序
    # all_dataset = ECGHandle.ECG_Dataset(data_root,ALL_data_buffer,preprocess = True)
    ####################################################################随机选取test
    test_df,tv_df = Pair_ID(ALL_data,0.2,Range_max=15,pair_num=1)
    test_dataset = ECGHandle.ECG_Dataset(data_root,test_df,preprocess = True)
    testmodel = NET[0].to(DEVICE)

    
    save_root = Models_path[:-3]+'/'
    Occlusion_root = save_root+'/Occlusion/'
    if(not(os.path.exists(save_root))): os.mkdir(save_root)
    if(not(os.path.exists(Occlusion_root))): os.mkdir(Occlusion_root)
    for index in tqdm(range(test_dataset.__len__())):#test_dataset.__len__()
        get_occlusion_value(index)