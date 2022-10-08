import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
import ecg_plot
from tqdm import tqdm
import os
#----------------------------------------------------------------------------------#
#input:
#   fig 
#   x:(timesteps,) 0.2s/step
#   y:(leads,timesteps) unit:uV
#
#----------------------------------------------------------------------------------#
def plot_multicolored_line(fig,axs,x,y,color_depend,cmap = "jet",y_name = "Voltage(mV)",title=""):
    
    points = np.array([x, y],dtype=object).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(color_depend.min(), color_depend.max()) # type: ignore #normalizer
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    #lc = LineCollection(segments, cmap=cmap)
    # Set the values used for colormapping
    lc.set_array(color_depend)
    lc.set_linewidth(1)
    line = axs.add_collection(lc)
    #fig.colorbar(line, ax=axs)
    axs.set_xlim(x.min(), x.max())
    axs.set_ylim(-3500, +3500)

    axs.set_aspect(0.2)#用于设置轴缩放的方面，即y-unit与x-unit的比率
    axs.xaxis.set_major_locator(plt.MultipleLocator(100))# type: ignore # 100*0.002s=0.2s = 5格
    axs.xaxis.set_minor_locator(plt.MultipleLocator(20)) # type: ignore # 20*0.002=0.004S = 1格
    axs.yaxis.set_major_locator(plt.MultipleLocator(500))# type: ignore # 0.1uv*500 = 0.5ms = 5格
    axs.yaxis.set_minor_locator(plt.MultipleLocator(100))# type: ignore # 0.1uv*100 =0.1ms = 1格 

    #axs.xaxis.set_major_formatter(plt.NullFormatter()) #x轴不显示刻度值/lable per 0.2s
    axs.xaxis.set_major_formatter(lambda x, pos: str(round(0.2*(x/100.0),2))) #x轴 lable per 0.2s
    axs.yaxis.set_major_formatter(lambda x, pos: str(x/1000.0)) # label per '0.5 mv'，turn uV to mv

    axs.grid(which='major', axis='x', linewidth=0.5, linestyle='-', color='b')
    axs.grid(which='minor', axis='x', linewidth=0.2, linestyle='-', color='b')
    axs.grid(which='major', axis='y', linewidth=0.5, linestyle='-', color='b')
    axs.grid(which='minor', axis='y', linewidth=0.2, linestyle='-', color='b')
    axs.set_ylabel(y_name)
    axs.set_title(title)
    axs.grid(True, which='both')


def plot_power():
    return 0


if __name__ == '__main__':
    EcgLength_num = 12000
    EcgChannles_num = 12
    data_path =  '/workspace/data/OneDrive - mail.hfut.edu.cn/ECG/Interpretable_HTN/npy_ECG/' #路径
    lable_path = '/workspace/data/OneDrive - mail.hfut.edu.cn/ECG/Interpretable_HTN/label.npy'
    seq_files = os.listdir(data_path)#返回指定的文件夹包含的文件或文件夹的名字的列表
    seq_files.sort(key=lambda x:int(x.split('_')[0])) #按“.”分割，并把分割结果的[0]转为整形并排序
    label = np.load(lable_path)

    for ecg_file_index in tqdm(range(len(seq_files))):
        ecg = np.load(os.path.join(data_path,seq_files[ecg_file_index]))
        ecg = ecg[:,:6000]*(4.88/1000.0)
        ecg_plot.plot(ecg, sample_rate = 500, title = seq_files[ecg_file_index],row_height= 8,show_grid=True,show_separate_line=True)
        ecg_plot.save_as_png(ecg_file_index,'/workspace/data/OneDrive - mail.hfut.edu.cn/ECG/Interpretable_HTN//PNG_ECG/',dpi = 100)
