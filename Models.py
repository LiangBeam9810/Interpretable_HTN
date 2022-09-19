import torch
import torch.nn as nn
from self_attention import *

class ATICNN(nn.Module):
    def __init__(self,DropoutRate = 0.1):
        super(ATICNN,self).__init__()
        self.ConvUnit1 = nn.Sequential(
            nn.Conv1d(in_channels = 12,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            #nn.Dropout(DropoutRate)
            nn.Conv1d(in_channels = 64,out_channels = 64,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DropoutRate)
        )

        self.ConvUnit2 = nn.Sequential(
            nn.Conv1d(in_channels = 64,out_channels = 128,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            #nn.Dropout(DropoutRate)
            nn.Conv1d(in_channels = 128,out_channels = 128,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DropoutRate)
        )

        self.ConvUnit3 = nn.Sequential(
            nn.Conv1d(in_channels = 128,out_channels = 256,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            #nn.Dropout(DropoutRate)
            nn.Conv1d(in_channels = 256,out_channels = 256,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels = 256,out_channels = 256,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DropoutRate)
        )

        self.ConvUnit4 = nn.Sequential(
            nn.Conv1d(in_channels = 256,out_channels = 512,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(DropoutRate)
            nn.Conv1d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DropoutRate)
        )

        self.ConvUnit5 = nn.Sequential(
            nn.Conv1d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            #nn.Dropout(DropoutRate)
            nn.Conv1d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Conv1d(in_channels = 512,out_channels = 512,kernel_size = 3,stride = 1,padding = 1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(DropoutRate)
        )
        self.pool = nn.MaxPool1d(3)
        self.LSTM_layer = nn.LSTM(input_size=512, hidden_size=32,num_layers  = 2,batch_first=True)
        self.attn = self_Attention_1D_for_timestep_position(32)
        self.linear_unit = nn.Sequential(
            nn.Linear(1952,512),
            nn.ReLU(),
            nn.Linear(512,2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
    def forward(self,x):
        x = self.ConvUnit1(x)
        x = self.pool(x)
        x = self.ConvUnit2(x)
        x = self.pool(x)
        x = self.ConvUnit3(x)
        x = self.pool(x)
        x = self.ConvUnit4(x)
        x = self.pool(x)
        x = x.permute(0,2 ,1)#[bs,  61 ,512]
        x , _ = self.LSTM_layer(x)#[bs,  61 ,32,]
        #print(x.shape)
        x = x.permute(0,2 ,1) #[bs, 32, 61]
        #print(x.shape)
        x,self.attention_value = self.attn(x)#[bs, 32, 61]
        x = x.contiguous().reshape(x.size(0),-1)
        x = self.linear_unit(x)
        return x

class CNN_ATT(nn.Module):
    def __init__(self):
        super(CNN_ATT,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 5,stride = 1,padding = 2)
        self.conv2 = nn.Conv1d(32,64,5,1,2)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.attn = self_Attention_1D_for_timestep(64)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.linear_unit = nn.Sequential(
            nn.Linear(79872,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.bn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.maxpool(x)
        x,self.attention_value = self.attn(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x)) # bs,128,1250
        x = self.bn4(x)
        x = self.relu(self.conv4(x)) # bs,256,1250
        #print(x.size())
        
        #print(x.size())
        x = self.maxpool(x) # bs,256,312
        x = self.dropout(x)
        x = x.contiguous().reshape(x.size(0),-1)
        x = self.linear_unit(x)
        return x


def mark_input(input,mark_lenth=500):
    batchsize,channelsize,sqenlenth = input.shape
    mark = torch.zeros([mark_lenth]).to(input.device)
    for i in range(batchsize):
        mark_index = torch.randint(mark_lenth,sqenlenth-mark_lenth,[1])
        #print(mark_index)
        for j in range(channelsize):
            input[i,j,mark_index:mark_index+mark_lenth]=mark
    return input

class CNN_ATT_Mark(nn.Module):
    def __init__(self):
        super(CNN_ATT_Mark,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,32,5,1,2)
        self.attn = self_Attention_1D_for_timestep_position(32)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(32)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.adtivepooling = nn.AdaptiveAvgPool1d(1024)
        self.linear_unit = nn.Sequential(
            nn.Linear(1024,256),
            nn.ReLU(),
            nn.Linear(256,2),
            nn.Softmax(dim=1)
        )

    def forward(self,input):
        batch_size, channels,seq_len = input.shape
        x = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)
        mark_lenth = torch.randint(int(seq_len/15),int(seq_len/10),[1])
        x = mark_input(x,mark_lenth=mark_lenth[0])
        x = self.bn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.bn3(x)
        x = self.maxpool(x)
        x,self.attention_value = self.attn(x)
        #print(x.size())
        x = self.maxpool(x) # bs,256,312
        x = self.dropout(x)
        x = x.contiguous().reshape(x.size(0),-1)
        x = self.adtivepooling(x)
        x = self.linear_unit(x)
        return x

class CNN_ATT2(nn.Module):
    def __init__(self):
        super(CNN_ATT2,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,64,11,1,5)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.attn = self_Attention_1D_for_leads(12)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            nn.Linear(79872,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.bn1(x)
        x,self.attention_value = self.attn(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x)) # bs,128,1250
        x = self.bn4(x)
        x = self.relu(self.conv4(x)) # bs,256,1250
        #print(x.size())
        #print(x.size())
        x = self.maxpool(x) # bs,256,312
        x = self.dropout(x)
        x = x.contiguous().reshape(x.size(0),-1)
        x = self.linear_unit(x)
        return x

class CNN_ATT3(nn.Module):
    def __init__(self):
        super(CNN_ATT3,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,64,11,1,5)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.attn = self_Attention_1D_for_timestep_position(256)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            nn.Linear(79872,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.bn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x)) # bs,128,1250
        x = self.bn4(x)
        x = self.relu(self.conv4(x)) # bs,256,1250
        #print(x.size())
        #print(x.size())
        x = self.maxpool(x) # bs,256,312
        x,self.attention_value = self.attn(x)
        x = self.dropout(x)
        x = x.contiguous().reshape(x.size(0),-1)
        x = self.linear_unit(x)
        return x

class CNN_ATT4(nn.Module):
    def __init__(self):
        super(CNN_ATT4,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,64,11,1,5)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.attn1 = self_Attention_1D_for_leads(256)
        self.attn2 = self_Attention_1D_for_timestep_position(256)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            nn.Linear(79872,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.bn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x)) # bs,128,1250
        x = self.bn4(x)
        x = self.relu(self.conv4(x)) # bs,256,1250
        #print(x.size())
        #print(x.size())
        x = self.maxpool(x) # bs,256,312
        x1,self.attention_value1 = self.attn1(x)
        x2,self.attention_value2 = self.attn2(x)
        x = x1+x2
        x = self.dropout(x)
        x = x.contiguous().reshape(x.size(0),-1)
        x = self.linear_unit(x)
        return x

class CNN_ATT5(nn.Module):
    def __init__(self):
        super(CNN_ATT5,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,64,11,1,5)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.conv5 = nn.Conv1d(256,256,3,1,1)
        self.attn1 = self_Attention_1D_for_leads(256)
        self.attn2 = self_Attention_1D_for_timestep_position(256)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            nn.Linear(79872,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.bn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x)) # bs,128,1250
        x = self.bn4(x)
        x = self.relu(self.conv4(x)) # bs,256,1250
        #print(x.size())
        #print(x.size())
        x = self.maxpool(x) # bs,256,312
        x1,self.attention_value1 = self.attn1(x)
        x2,self.attention_value2 = self.attn2(x)
        x = self.relu(self.conv5(x1))+self.relu(self.conv5(x2))
        x = self.dropout(x)
        x = x.contiguous().reshape(x.size(0),-1)
        x = self.linear_unit(x)
        return x

class CNN_ATT6(nn.Module):
    def __init__(self):
        super(CNN_ATT6,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,64,11,1,5)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.conv5 = nn.Conv1d(256,256,3,1,1)
        self.attn1 = self_Attention_1D_for_leads(12)
        self.attn2 = self_Attention_1D_for_timestep_position(256)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            nn.Linear(79872,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.bn1(x)
        print(x.shape)
        x,self.attention_value1 = self.attn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.maxpool(x)
        print(x.shape)
        x = self.bn3(x)
        x = self.relu(self.conv3(x)) # bs,128,1250
        x = self.bn4(x)
        x = self.relu(self.conv4(x)) # bs,256,1250
        #print(x.size())
        #print(x.size())
        x = self.maxpool(x) # bs,256,312
        x,self.attention_value2 = self.attn2(x)
        x = self.dropout(x)
        print(x.shape)
        x = x.contiguous().reshape(x.size(0),-1)
        print(x.shape)
        x = self.linear_unit(x)
        return x

class CNN_ATT7(nn.Module):
    def __init__(self):
        super(CNN_ATT7,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 5,stride = 1,padding = 2)
        self.conv2 = nn.Conv1d(32,64,3,1,1)
        self.attn = self_Attention_1D_for_timestep_without_relu(64)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear_unit = nn.Sequential(
            nn.Linear(19968,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        batch_size, channels,seq_len = x.shape
        x = x+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(x.device)
        x = self.bn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.maxpool(x)          # bs,32,1250
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,1250
        x = self.maxpool(x) # bs,64,312
        x = self.bn3(x)
        #print(x.shape)
        x,self.attention_value = self.attn(x)
        x = self.dropout(x)
        x = x.contiguous().reshape(x.size(0),-1)
        #print(x.shape)
        x = self.linear_unit(x)
        return x

class ECGiCOVIDNet(nn.Module):
    def __init__(self,DropoutRate = 0.3):
        super(ECGiCOVIDNet,self).__init__()
        self.ConvUnit12 = nn.Sequential(
            nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 11,stride = 1,padding = 5),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(DropoutRate)
        )
        self.ConvUnit32 = nn.Sequential(
            nn.Conv1d(in_channels = 32,out_channels = 32,kernel_size = 3,stride = 1,padding = 1),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(DropoutRate)
        )
        self.linear_unit = nn.Sequential(
            nn.Linear(1000,512),
            nn.ReLU(),
            nn.Linear(512,2),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        self.poolling = nn.AdaptiveAvgPool1d(1000)
        self.attn = self_Attention_1D_for_leads(32)
    def forward(self,x):
        x = self.ConvUnit12(x)
        x = self.ConvUnit32(x)
        x = self.ConvUnit32(x)+x
        x = self.poolling(x)
        x,self.attention_value1 = self.attn1(x)
        x = x.contiguous().reshape(x.size(0),-1)
        x = self.poolling(x)
        x = self.linear_unit(x)
        return x

# 定义模型结构
class VGG_6(nn.Module):

    def __init__(self):
        super(VGG_6,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 64,kernel_size = 3,stride = 1,padding = 1)
        self.conv2 = nn.Conv1d(64,64,3,1,1)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,128,3,1,1)
        self.conv5 = nn.Conv1d(128,256,3,1,1)
        self.conv6 = nn.Conv1d(256,256,3,1,1)
        self.conv7 = nn.Conv1d(256,512,3,1,1)
        self.conv8 = nn.Conv1d(512,512,3,1,1)

        self.bn64 = nn.BatchNorm1d(64)
        self.bn128 = nn.BatchNorm1d(128)
        self.bn256 = nn.BatchNorm1d(256)
        self.bn512 = nn.BatchNorm1d(512)

        self.pool = nn.AvgPool1d(3)
        self.relu = nn.RReLU()
        self.dropout = nn.Dropout(0.2)
        self.linear_unit = nn.Sequential(
            nn.Linear(10240,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )


    def forward(self,x):
        x = self.relu(self.conv1(x)) # (batchsize,12,5000) -> (batchsize,64,5000)
        x = self.bn64(x)
        x = self.relu(self.conv2(x))
        #x = self.bn64(x)
        x = self.pool(x)  # (batchsize,64,5000/3=1666)

        x = self.relu(self.conv3(x)) # (batchsize,64,5000) --> (batchsize,128,5000)
        #x = self.bn128(x)
        x = self.relu(self.conv4(x)) 
        #x = self.bn128(x)
        x = self.pool(x)  # (batchsize,64,1666/3=555)

        x = self.relu(self.conv5(x)) # (batchsize,128,5000) --> (batchsize,256,5000)
        #x = self.bn256(x)
        x = self.relu(self.conv6(x)) 
        #x = self.bn256(x)    
        x = self.relu(self.conv6(x)) 
        #x = self.bn256(x)   
        x = self.pool(x)  # (batchsize,64,555/3=185)

        x = self.relu(self.conv7(x)) # (batchsize,256,5000) --> (batchsize,512,5000)
        #x = self.bn512(x)
        x = self.relu(self.conv8(x)) 
        #x = self.bn512(x)    
        x = self.relu(self.conv8(x)) 
        #x = self.bn512(x)   
        x = self.pool(x)  # (batchsize,64,185/3=61)

        x = self.relu(self.conv8(x)) 
        #x = self.bn512(x)    
        x = self.relu(self.conv8(x)) 
        #x = self.bn512(x)    
        x = self.relu(self.conv8(x)) 
        #x = self.bn512(x)    
        x = self.pool(x)  # (batchsize,64,61/3=20)
        x = x.view(x.size(0),-1)

        x = self.linear_unit(x)
        return x

class CNN(nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Conv1d(in_channels = 12,out_channels = 32,kernel_size = 11,stride = 1,padding = 5)
        self.conv2 = nn.Conv1d(32,64,11,1,5)
        self.conv3 = nn.Conv1d(64,128,3,1,1)
        self.conv4 = nn.Conv1d(128,256,3,1,1)
        self.bn1 = nn.BatchNorm1d(12)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.maxpool = nn.MaxPool1d(4)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.linear_unit = nn.Sequential(
            nn.Linear(79872,1024),
            nn.ReLU(),
            nn.Linear(1024,128),
            nn.ReLU(),
            nn.Linear(128,2),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.bn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.maxpool(x)
        x = self.bn3(x)
        x = self.relu(self.conv3(x)) # bs,128,1250
        x = self.bn4(x)
        x = self.relu(self.conv4(x)) # bs,256,1250
        x = self.maxpool(x) # bs,256,312
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = self.linear_unit(x)
        return x