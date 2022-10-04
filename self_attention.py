from tkinter import E
import torch
import torch.nn as nn
from math import gamma, sqrt
import numpy as np

class Attention_1D_tanh(nn.Module):
    def __init__(self, in_channels,class_num):
        super().__init__()
        self.in_channels = in_channels
        self.class_num = class_num
        self.fc1 = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.fc2   = nn.Conv1d(self.in_channels, self.class_num, kernel_size = 1, stride = 1)
        self.fc3 = nn.Conv1d(self.in_channels, self.class_num, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(in_channels)
        
    def forward(self, input):
        batch_size, channels,seq_len = input.shape

        x1 = self.relu(self.fc1(input)) # bs,class,time
        x2 = (self.relu(self.fc2(input)))# bs,class,time
        #print(x2)
        x3 = self.relu(self.fc3(input))# bs,class,time
        #print(x1.shape,x2.shape,x3.shape)
        x = x2*x3
        out = x.mean(dim=2)
        return out,x



def init_conv(conv, glu=True):
    nn.init.xavier_uniform_(conv.weight)
    if conv.bias is not None:
        conv.bias.data.zero_()

class self_Attention_1D_for_timestep_without_relu(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.bn = nn.BatchNorm1d(self.in_channels)
        self.query = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.key   = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.value = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)

        init_conv(self.query)
        init_conv(self.key)
        init_conv(self.value)
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        sita = np.array(sqrt(seq_len))
        q = self.query(input)
        k = self.key(input)
        v = self.value(input)

        attn_matrix = (torch.bmm(q.permute(0,2 ,1), k))/torch.from_numpy(sita)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        #print(attn_matrix)
        #print("attention:",attn_matrix.shape)
        attn_matrix = self.softmax(attn_matrix)
        #self.attention_value = attn_matrix #输出attention值
        out = (((torch.bmm(attn_matrix,v.permute(0,2 ,1))).permute(0,2 ,1)))+  input

        #print("out:",out.shape)
        return out,attn_matrix

class self_Attention_1D_for_timestep(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.key   = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.value = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        sita = np.array(sqrt(seq_len))
        q = self.dropout(self.bn(self.relu(self.query(input))))
        k = self.dropout(self.bn(self.relu(self.key(input))))
        v = self.dropout(self.bn(self.relu(self.value(input))))

        attn_matrix = (torch.bmm(q.permute(0,2 ,1), k))/torch.from_numpy(sita)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        #print(attn_matrix)
        #print("attention:",attn_matrix.shape)
        attn_matrix = self.softmax(attn_matrix)
        #self.attention_value = attn_matrix #输出attention值
        out = (self.gamma*((torch.bmm(attn_matrix,v.permute(0,2 ,1))).permute(0,2 ,1)))+  input
        #print("out:",out.shape)
        return out,attn_matrix

class self_Attention_1D_for_timestep_position(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.key   = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.value = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(in_channels)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input + (create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)
        sita = np.array(sqrt(seq_len))
        q = self.dropout(self.bn(self.relu(self.query(input))))
        k = self.dropout(self.bn(self.relu(self.key(input))))
        v = self.dropout(self.bn(self.relu(self.value(input))))
        attn_matrix = (torch.bmm(q.permute(0,2 ,1), k))/torch.from_numpy(sita)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)
        out = (self.gamma*torch.bmm(attn_matrix,v.permute(0,2 ,1))).permute(0,2 ,1)+  input
        #print("{:.5}".format(self.gamma[0]))
        return out,attn_matrix

def create_1d_absolute_sin_cos_embedding(batch_size,dim,pos_len):
    #assert dim % 2 == 0, "wrong dimension!"
        dim_backup = dim
        if dim % 2 != 0:
            dim = dim+1
        position_emb = torch.zeros(batch_size,pos_len, dim, dtype=torch.float)
        # i矩阵
        i_matrix = torch.arange(dim//2, dtype=torch.float)
        i_matrix /= dim / 2
        i_matrix = torch.pow(10000, i_matrix)
        i_matrix = 1 / i_matrix
        i_matrix = i_matrix.to(torch.long)
        # pos矩阵
        pos_vec = torch.arange(pos_len).to(torch.long)
        # 矩阵相乘，pos变成列向量，i_matrix变成行向量
        out = pos_vec[:, None] @ i_matrix[None, :]
        # 奇/偶数列
        emb_cos = torch.cos(out)
        emb_sin = torch.sin(out)
        # 赋值
        position_emb[:,:, 0::2] = emb_sin.repeat(batch_size,1,1)
        position_emb[:,:, 1::2] = emb_cos.repeat(batch_size,1,1)
        if dim_backup % 2 != 0:
            return position_emb[:,:,:-1].permute(0,2 ,1)
        return position_emb.permute(0,2 ,1)    

class self_Attention_1D_for_leads(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.query = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.key   = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.value = nn.Conv1d(self.in_channels, self.in_channels, kernel_size = 1, stride = 1)
        self.gamma = nn.Parameter(torch.zeros(1))  #gamma为一个衰减参数，由torch.zero生成，nn.Parameter的作用是将其转化成为可以训练的参数.
        self.softmax = nn.Softmax(dim = -1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.bn = nn.BatchNorm1d(in_channels)
        
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        sita = np.array(sqrt(channels))
        q = self.dropout(self.bn(self.relu(self.query(input))))
        k = self.dropout(self.bn(self.relu(self.key(input))))
        v = self.dropout(self.bn(self.relu(self.value(input))))
        attn_matrix = (torch.bmm(q, k.permute(0,2 ,1)))/torch.from_numpy(sita)  #torch.bmm进行tensor矩阵乘法,q与k相乘得到的值为attn_matrix.
        attn_matrix = self.softmax(attn_matrix)
        #print("{:.5}".format(self.gamma[0]))
        out = self.gamma *  (torch.bmm(attn_matrix,v))+input
        return out,attn_matrix






