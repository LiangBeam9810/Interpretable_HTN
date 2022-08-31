import numpy as np
import os 
import ecg_plot
from tqdm import tqdm

def load_data(npy_path):
    return np.load(npy_path)

EcgLength_num = 12000
EcgChannles_num = 12
data_path =  '/workspace/data/OneDrive - mail.hfut.edu.cn/ECG/Interpretable_HTN/npy_ECG/' #路径
lable_path = '/workspace/data/OneDrive - mail.hfut.edu.cn/ECG/Interpretable_HTN/label.npy'
seq_files = os.listdir(data_path)#返回指定的文件夹包含的文件或文件夹的名字的列表
seq_files.sort(key=lambda x:int(x.split('_')[0])) #按“.”分割，并把分割结果的[0]转为整形并排序
label = np.load(lable_path)

for ecg_file_index in tqdm(range(len(seq_files))):
    ecg = load_data(os.path.join(data_path,seq_files[ecg_file_index]))
    ecg = ecg[:,:6000]*(4.88/1000.0)
    ecg_plot.plot(ecg, sample_rate = 500, title = seq_files[ecg_file_index],row_height= 8,show_grid=True,show_separate_line=True)
    ecg_plot.save_as_png(seq_files[ecg_file_index][:-4],'/workspace/data/OneDrive - mail.hfut.edu.cn/ECG/Interpretable_HTN//PNG_ECG/',dpi = 100)




