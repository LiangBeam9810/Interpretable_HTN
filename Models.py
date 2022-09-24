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
        self.dropout = nn.Dropout(0.2)
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

def shift(input,interval = 20):
    batchsize,channelsize,sqenlenth = input.shape
    for i in range(batchsize):
        for j in range(channelsize):
            offset = torch.random.choice

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
            nn.Linear(9984,256),
            nn.ReLU(),
            nn.Linear(256,2),
            nn.Softmax(dim=1)
        )

    def forward(self,input):
        batch_size, channels,seq_len = input.shape
        
        x = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if self.training:
            mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
            x = mark_input(x,mark_lenth=mark_lenth[0])

        x = self.bn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,32,5000
        x = self.bn3(x)
        x = self.maxpool(x)# bs,64,1250
        x,self.attention_value = self.attn(x)
        #print(x.size())
        x = self.maxpool(x) # bs,32,312
        x = self.dropout(x)
        x = x.contiguous().reshape(x.size(0),-1) # bs,32*312
        #x = self.adtivepooling(x)
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

class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet1d(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=9):
        super(ResNet1d, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(BasicBlock1d, 512, layers[3], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x1 = self.adaptiveavgpool(x)
        x2 = self.adaptivemaxpool(x)
        x = torch.cat((x1, x2), dim=1)
        x = x.view(x.size(0), -1)
        return self.fc(x)

def resnet18(input_channels=12, inplanes=64, num_classes=9):
    model = ResNet1d(BasicBlock1d, [2, 2, 2, 2], input_channels, inplanes, num_classes)
    return model

def resnet34(input_channels=12, inplanes=64, num_classes=9):
    model = ResNet1d(BasicBlock1d, [3, 4, 6, 3], input_channels, inplanes, num_classes)
    return model


class channels_split_CNN(nn.Module):
    def __init__(self):
        super(channels_split_CNN, self).__init__()
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=11, stride=1, padding=5, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv1 = nn.Conv1d(384, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(4)

        self.linear_unit = nn.Sequential(
            nn.Linear(1984,1024),
            nn.ReLU(),
            nn.Linear(1024,512),
            nn.ReLU(),
            nn.Linear(512,2),
            nn.Softmax(dim=1)
        )
        self.att = Attention_1D_tanh(64,2)
        self.softmax =nn.Softmax(dim=-1)
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if self.training:
            mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
            input = mark_input(input,mark_lenth=mark_lenth[0])

        x1 = self.channels_unit1(input[:,0:1,:]) #提取channel_i的数据
        x2 = self.channels_unit2(input[:,1:2,:]) #提取channel_i的数据
        x3 = self.channels_unit3(input[:,2:3,:]) #提取channel_i的数据
        x4 = self.channels_unit4(input[:,3:4,:]) #提取channel_i的数据
        x5 = self.channels_unit5(input[:,4:5,:]) #提取channel_i的数据
        x6 = self.channels_unit6(input[:,5:6,:]) #提取channel_i的数据
        x7 = self.channels_unit7(input[:,6:7,:]) #提取channel_i的数据
        x8 = self.channels_unit8(input[:,7:8,:]) #提取channel_i的数据
        x9 = self.channels_unit9(input[:,8:9,:]) #提取channel_i的数据
        x10 = self.channels_unit10(input[:,9:10,:]) #提取channel_i的数据
        x11 = self.channels_unit11(input[:,10:11,:]) #提取channel_i的数据
        x12 = self.channels_unit12(input[:,11:,:]) #提取channel_i的数据  
        #print(x1.shape)
        x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12),1) #按照第1维度(channel)合并
        #print(x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        #out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool(out)
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out