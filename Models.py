import torch
import torch.nn as nn
from self_attention import *


from informer.encoder import Encoder, EncoderLayer, ConvLayer, EncoderStack
from informer.decoder import Decoder, DecoderLayer
from informer.attn import FullAttention, ProbAttention, AttentionLayer
from informer.embed import DataEmbedding


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
        #print(x.shape)
        x,self.attention_value1 = self.attn1(x)
        x = self.relu(self.conv1(x)) # bs,32,5000
        x = self.bn2(x)
        x = self.relu(self.conv2(x)) # bs,64,5000
        x = self.maxpool(x)
        #print(x.shape)
        x = self.bn3(x)
        x = self.relu(self.conv3(x)) # bs,128,1250
        x = self.bn4(x)
        x = self.relu(self.conv4(x)) # bs,256,1250
        #print(x.size())
        #print(x.size())
        x = self.maxpool(x) # bs,256,312
        x,self.attention_value2 = self.attn2(x)
        x = self.dropout(x)
        #print(x.shape)
        x = x.contiguous().reshape(x.size(0),-1)
        #print(x.shape)
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
        batch_size, channels,seq_len = x.shape
        # x = x+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(x.device)#位置编码
        if self.training:
            mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
            x = mark_input(x,mark_lenth=int(mark_lenth[0]))
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
            nn.Linear(4992,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
        self.att = self_Attention_1D_for_timestep(64)
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
        out,self.att_value = self.att(out)
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.reshape(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out

class Informer(nn.Module):
    def __init__(self, enc_in, 
                factor=5, d_model=12, n_heads=12, e_layers=3, d_layers=2, d_ff=12, 
                dropout=0.0, attn='prob', embed='fixed', freq='h', activation='relu', 
                output_attention = False, distil=True, mix=True,
                device=torch.device('cuda:0')):
        super(Informer, self).__init__()
        self.attn = attn
        self.output_attention = output_attention

        # Encoding
        self.enc_embedding1 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding2 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding3 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding4 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding5 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding6 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding7 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding8 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding9 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding10 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding11 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
        self.enc_embedding12 = DataEmbedding(enc_in, d_model, embed, freq, dropout)
       
        # Attention
        Attn = ProbAttention if attn=='prob' else FullAttention
        #Attn = FullAttention
        # Encoder
        self.encoder1 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )


        self.encoder2 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder3 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder4 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder5 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder6 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder7 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder8 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder9 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder10 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder11 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.encoder12 = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(Attn(False, factor, attention_dropout=dropout, output_attention=output_attention), 
                                d_model, n_heads, mix=False),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation
                ) for l in range(e_layers)
            ],
            [
                ConvLayer(
                    d_model
                ) for l in range(e_layers-1)
            ] if distil else None,
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        self.conv1 = nn.Conv1d(144, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool = nn.MaxPool1d(5)
        self.linear_unit = nn.Sequential(
            nn.Linear(3200,1024),
            nn.ReLU(),
            nn.Linear(1024,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x_enc,enc_self_mask=None):
        B,C,L=x_enc.shape
        x_enc = x_enc.permute(0,2,1)
        x1 = self.enc_embedding1(x_enc[:,:,0:1]) #提取channel_i的数据
        x2 = self.enc_embedding2(x_enc[:,:,1:2]) #提取channel_i的数据
        x3 = self.enc_embedding3(x_enc[:,:,2:3]) #提取channel_i的数据
        x4 = self.enc_embedding4(x_enc[:,:,3:4]) #提取channel_i的数据
        x5 = self.enc_embedding5(x_enc[:,:,4:5]) #提取channel_i的数据
        x6 = self.enc_embedding6(x_enc[:,:,5:6]) #提取channel_i的数据
        x7 = self.enc_embedding7(x_enc[:,:,6:7]) #提取channel_i的数据
        x8 = self.enc_embedding8(x_enc[:,:,7:8]) #提取channel_i的数据
        x9 = self.enc_embedding9(x_enc[:,:,8:9]) #提取channel_i的数据
        x10 = self.enc_embedding10(x_enc[:,:,9:10]) #提取channel_i的数据
        x11 = self.enc_embedding11(x_enc[:,:,10:11]) #提取channel_i的数据
        x12 = self.enc_embedding12(x_enc[:,:,11:]) #提取channel_i的数据

        x1, _ = self.encoder1(x1, attn_mask=enc_self_mask)
        x2, _ = self.encoder2(x2, attn_mask=enc_self_mask)
        x3, _ = self.encoder3(x3, attn_mask=enc_self_mask)
        x4, _ = self.encoder4(x4, attn_mask=enc_self_mask)
        x5, _ = self.encoder5(x5, attn_mask=enc_self_mask)
        x6, _ = self.encoder6(x6, attn_mask=enc_self_mask)
        x7, _ = self.encoder7(x7, attn_mask=enc_self_mask)
        x8, _ = self.encoder8(x8, attn_mask=enc_self_mask)
        x9, _ = self.encoder9(x9, attn_mask=enc_self_mask)
        x10, _ = self.encoder10(x10, attn_mask=enc_self_mask)
        x11, _ = self.encoder11(x11, attn_mask=enc_self_mask)
        x12, _ = self.encoder12(x12, attn_mask=enc_self_mask)


        enc_out = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12),2) #按照第2维度(channel)合并 [B,625,12*12=144]
        enc_out = enc_out.permute(0,2,1)   #[B,144,375]
        enc_out = self.bn1(self.relu(self.conv1(enc_out)))#[B,128,375]
        enc_out = self.maxpool(enc_out)    #[B,128,75]

        enc_out = self.bn2(self.relu(self.conv2(enc_out)))#[B,64,75]
        enc_out = self.maxpool(enc_out)    #[B,64,15]

        enc_out = enc_out.view(enc_out.size(0),-1) #[B,64,15]
        enc_out = self.linear_unit(enc_out)
        return enc_out

class channels_split_ATT_CNN(nn.Module):
    def __init__(self,mark = False):
        super(channels_split_ATT_CNN, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_without_relu(32)
        self.att2 = self_Attention_1D_for_timestep_without_relu(32)

        self.maxpool = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(4)
        self.conv1 = nn.Conv1d(768, 128, kernel_size=3, stride=1, padding=1)#1250
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(5)#b,64,250
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)#200
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1_1,self.att1v_1 = self.att1(x1)
        x1_2,self.att1v_2 = self.att2(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2_1,self.att2v_1 = self.att1(x2)
        x2_2,self.att2v_2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3_1,self.att3v_1 = self.att1(x3)
        x3_2,self.att3v_2 = self.att2(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4_1,self.att4v_1 = self.att1(x4)
        x4_2,self.att4v_2 = self.att2(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5_1,self.att5v_1 = self.att1(x5)
        x5_2,self.att5v_2 = self.att2(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6_1,self.att6v_1 = self.att1(x6)
        x6_2,self.att6v_2 = self.att2(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7_1,self.att7v_1 = self.att1(x7)
        x7_2,self.att7v_2 = self.att2(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8_1,self.att8v_1 = self.att1(x8)
        x8_2,self.att8v_2 = self.att2(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9_1,self.att9v_1 = self.att1(x9)
        x9_2,self.att9v_2 = self.att2(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10_1,self.att10v_1 = self.att1(x10)
        x10_2,self.att10v_2 = self.att2(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11_1,self.att11v_1 = self.att1(x11)
        x11_2,self.att11v_2 = self.att2(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12_1,self.att12v_1 = self.att1(x12)
        x12_2,self.att12v_2 = self.att2(x12)
        
        #print(x1.shape)
        x = torch.cat((x1_1,x1_2,x2_1,x2_2,x3_1,x3_2,x4_1,x4_2,x5_1,x5_2,x6_1,x6_2,x7_1,x7_2,x8_1,x8_2,x9_1,x9_2,x10_1,x10_2,x11_1,x11_2,x12_1,x12_2),1) #按照第1维度(channel)合并 # B,32*12,200
        #print(x.shape)
        out = self.conv1(x)# B,128,200
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)# B,128,50
        #out = self.dropout(out)
        out = self.conv2(out)# B,64,25
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out

class channels_split_ATT_CNN_linear_relu(nn.Module):
    def __init__(self,mark = False,extract_dim = 32, hdim = 32):
        super(channels_split_ATT_CNN_linear_relu, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_with_relu_Linear(extract_dim,hdim)
        self.att2 = self_Attention_1D_for_timestep_with_relu_Linear(extract_dim,hdim)

        self.maxpool = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(4)
        self.conv1 = nn.Conv1d(extract_dim*12*2, 128, kernel_size=3, stride=1, padding=1)#1250
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(5)#b,64,250
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)#200
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1_1,self.att1v_1 = self.att1(x1)
        x1_2,self.att1v_2 = self.att2(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2_1,self.att2v_1 = self.att1(x2)
        x2_2,self.att2v_2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3_1,self.att3v_1 = self.att1(x3)
        x3_2,self.att3v_2 = self.att2(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4_1,self.att4v_1 = self.att1(x4)
        x4_2,self.att4v_2 = self.att2(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5_1,self.att5v_1 = self.att1(x5)
        x5_2,self.att5v_2 = self.att2(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6_1,self.att6v_1 = self.att1(x6)
        x6_2,self.att6v_2 = self.att2(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7_1,self.att7v_1 = self.att1(x7)
        x7_2,self.att7v_2 = self.att2(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8_1,self.att8v_1 = self.att1(x8)
        x8_2,self.att8v_2 = self.att2(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9_1,self.att9v_1 = self.att1(x9)
        x9_2,self.att9v_2 = self.att2(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10_1,self.att10v_1 = self.att1(x10)
        x10_2,self.att10v_2 = self.att2(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11_1,self.att11v_1 = self.att1(x11)
        x11_2,self.att11v_2 = self.att2(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12_1,self.att12v_1 = self.att1(x12)
        x12_2,self.att12v_2 = self.att2(x12)
        
        #print(x1.shape)
        x = torch.cat((x1_1,x1_2,x2_1,x2_2,x3_1,x3_2,x4_1,x4_2,x5_1,x5_2,x6_1,x6_2,x7_1,x7_2,x8_1,x8_2,x9_1,x9_2,x10_1,x10_2,x11_1,x11_2,x12_1,x12_2),1) #按照第1维度(channel)合并 # B,32*12,200
        #print(x.shape)
        out = self.conv1(x)# B,128,200
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)# B,128,50
        #out = self.dropout(out)
        out = self.conv2(out)# B,64,25
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out

class channels_split_ATT_CNN_linear(nn.Module):
    def __init__(self,mark = False,extract_dim = 32, hdim = 32):
        super(channels_split_ATT_CNN_linear, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)
        self.att2 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)

        self.maxpool = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(4)
        self.conv1 = nn.Conv1d(extract_dim*12*2, 128, kernel_size=3, stride=1, padding=1)#1250
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(5)#b,64,250
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)#200
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1_1,self.att1v_1 = self.att1(x1)
        x1_2,self.att1v_2 = self.att2(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2_1,self.att2v_1 = self.att1(x2)
        x2_2,self.att2v_2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3_1,self.att3v_1 = self.att1(x3)
        x3_2,self.att3v_2 = self.att2(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4_1,self.att4v_1 = self.att1(x4)
        x4_2,self.att4v_2 = self.att2(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5_1,self.att5v_1 = self.att1(x5)
        x5_2,self.att5v_2 = self.att2(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6_1,self.att6v_1 = self.att1(x6)
        x6_2,self.att6v_2 = self.att2(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7_1,self.att7v_1 = self.att1(x7)
        x7_2,self.att7v_2 = self.att2(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8_1,self.att8v_1 = self.att1(x8)
        x8_2,self.att8v_2 = self.att2(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9_1,self.att9v_1 = self.att1(x9)
        x9_2,self.att9v_2 = self.att2(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10_1,self.att10v_1 = self.att1(x10)
        x10_2,self.att10v_2 = self.att2(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11_1,self.att11v_1 = self.att1(x11)
        x11_2,self.att11v_2 = self.att2(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12_1,self.att12v_1 = self.att1(x12)
        x12_2,self.att12v_2 = self.att2(x12)
        
        #print(x1.shape)
        x = torch.cat((x1_1,x1_2,x2_1,x2_2,x3_1,x3_2,x4_1,x4_2,x5_1,x5_2,x6_1,x6_2,x7_1,x7_2,x8_1,x8_2,x9_1,x9_2,x10_1,x10_2,x11_1,x11_2,x12_1,x12_2),1) #按照第1维度(channel)合并 # B,32*12,1250
        #print(x.shape)
        out = self.conv1(x)# B,128,1250
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)# B,128,250
        #out = self.dropout(out)
        out = self.conv2(out)# B,64,250
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)# B,64,50
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out

class channels_split_ATT_CNN_linear_ATT(nn.Module):
    def __init__(self,mark = False,extract_dim = 32, hdim = 32):
        super(channels_split_ATT_CNN_linear_ATT, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)
        self.att2 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)


        self.CNN_unit1 = nn.Sequential(
            nn.Conv1d(extract_dim*12*2, 128, kernel_size=3, stride=1, padding=1),#1250,
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        self.CNN_unit2 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, stride=1, padding=1),#1250,
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self.maxpool = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(4)
        self.avgPool1 = nn.MaxPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1_1,self.att1v_1 = self.att1(x1)
        x1_2,self.att1v_2 = self.att2(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2_1,self.att2v_1 = self.att1(x2)
        x2_2,self.att2v_2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3_1,self.att3v_1 = self.att1(x3)
        x3_2,self.att3v_2 = self.att2(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4_1,self.att4v_1 = self.att1(x4)
        x4_2,self.att4v_2 = self.att2(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5_1,self.att5v_1 = self.att1(x5)
        x5_2,self.att5v_2 = self.att2(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6_1,self.att6v_1 = self.att1(x6)
        x6_2,self.att6v_2 = self.att2(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7_1,self.att7v_1 = self.att1(x7)
        x7_2,self.att7v_2 = self.att2(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8_1,self.att8v_1 = self.att1(x8)
        x8_2,self.att8v_2 = self.att2(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9_1,self.att9v_1 = self.att1(x9)
        x9_2,self.att9v_2 = self.att2(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10_1,self.att10v_1 = self.att1(x10)
        x10_2,self.att10v_2 = self.att2(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11_1,self.att11v_1 = self.att1(x11)
        x11_2,self.att11v_2 = self.att2(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12_1,self.att12v_1 = self.att1(x12)
        x12_2,self.att12v_2 = self.att2(x12)
        
        #print(x1.shape)
        x = torch.cat((x1_1,x1_2,x2_1,x2_2,x3_1,x3_2,x4_1,x4_2,x5_1,x5_2,x6_1,x6_2,x7_1,x7_2,x8_1,x8_2,x9_1,x9_2,x10_1,x10_2,x11_1,x11_2,x12_1,x12_2),1) #按照第1维度(channel)合并 # B,extract_dim*12*head,200
        #print(x.shape)
        out = self.avgPool1(x)
        out = self.CNN_unit1(out)# B,128,250
        out = self.avgPool1(out)
        out = self.CNN_unit2(out)# B,64,50
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out

class channels_split_ATT_CNN_linear_channels(nn.Module):
    def __init__(self,mark = False,extract_dim = 32, hdim = 32):
        super(channels_split_ATT_CNN_linear_channels, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)
        self.att2 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)
        self.attleads = self_Attention_1D_for_leads(extract_dim*12*2)
        self.maxpool = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(4)
        self.conv1 = nn.Conv1d(extract_dim*12*2, 128, kernel_size=3, stride=1, padding=1)#1250
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(5)#b,64,250
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)#200
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1_1,self.att1v_1 = self.att1(x1)
        x1_2,self.att1v_2 = self.att2(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2_1,self.att2v_1 = self.att1(x2)
        x2_2,self.att2v_2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3_1,self.att3v_1 = self.att1(x3)
        x3_2,self.att3v_2 = self.att2(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4_1,self.att4v_1 = self.att1(x4)
        x4_2,self.att4v_2 = self.att2(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5_1,self.att5v_1 = self.att1(x5)
        x5_2,self.att5v_2 = self.att2(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6_1,self.att6v_1 = self.att1(x6)
        x6_2,self.att6v_2 = self.att2(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7_1,self.att7v_1 = self.att1(x7)
        x7_2,self.att7v_2 = self.att2(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8_1,self.att8v_1 = self.att1(x8)
        x8_2,self.att8v_2 = self.att2(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9_1,self.att9v_1 = self.att1(x9)
        x9_2,self.att9v_2 = self.att2(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10_1,self.att10v_1 = self.att1(x10)
        x10_2,self.att10v_2 = self.att2(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11_1,self.att11v_1 = self.att1(x11)
        x11_2,self.att11v_2 = self.att2(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12_1,self.att12v_1 = self.att1(x12)
        x12_2,self.att12v_2 = self.att2(x12)
        
        #print(x1.shape)
        x = torch.cat((x1_1,x1_2,x2_1,x2_2,x3_1,x3_2,x4_1,x4_2,x5_1,x5_2,x6_1,x6_2,x7_1,x7_2,x8_1,x8_2,x9_1,x9_2,x10_1,x10_2,x11_1,x11_2,x12_1,x12_2),1) #按照第1维度(channel)合并 # B,32*12,200
        x,_ = self.attleads(x)
        #print(x.shape)
        out = self.conv1(x)# B,128,200
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)# B,128,50
        #out = self.dropout(out)
        out = self.conv2(out)# B,64,25
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out

class channels_split_ATT_CNN_linear_avgpool(nn.Module):
    def __init__(self,mark = False,extract_dim = 32, hdim = 32):
        super(channels_split_ATT_CNN_linear_avgpool, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)
        self.att2 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)

        self.maxpool = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(4)
        self.conv1 = nn.Conv1d(extract_dim*12*2, 128, kernel_size=3, stride=1, padding=1)#1250
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.AvgPool1d(5)#b,64,250
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)#200
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.AvgPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1_1,self.att1v_1 = self.att1(x1)
        x1_2,self.att1v_2 = self.att2(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2_1,self.att2v_1 = self.att1(x2)
        x2_2,self.att2v_2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3_1,self.att3v_1 = self.att1(x3)
        x3_2,self.att3v_2 = self.att2(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4_1,self.att4v_1 = self.att1(x4)
        x4_2,self.att4v_2 = self.att2(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5_1,self.att5v_1 = self.att1(x5)
        x5_2,self.att5v_2 = self.att2(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6_1,self.att6v_1 = self.att1(x6)
        x6_2,self.att6v_2 = self.att2(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7_1,self.att7v_1 = self.att1(x7)
        x7_2,self.att7v_2 = self.att2(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8_1,self.att8v_1 = self.att1(x8)
        x8_2,self.att8v_2 = self.att2(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9_1,self.att9v_1 = self.att1(x9)
        x9_2,self.att9v_2 = self.att2(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10_1,self.att10v_1 = self.att1(x10)
        x10_2,self.att10v_2 = self.att2(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11_1,self.att11v_1 = self.att1(x11)
        x11_2,self.att11v_2 = self.att2(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12_1,self.att12v_1 = self.att1(x12)
        x12_2,self.att12v_2 = self.att2(x12)
        
        #print(x1.shape)
        x = torch.cat((x1_1,x1_2,x2_1,x2_2,x3_1,x3_2,x4_1,x4_2,x5_1,x5_2,x6_1,x6_2,x7_1,x7_2,x8_1,x8_2,x9_1,x9_2,x10_1,x10_2,x11_1,x11_2,x12_1,x12_2),1) #按照第1维度(channel)合并 # B,32*12,1250
        #print(x.shape)
        out = self.conv1(x)# B,128,1250
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)# B,128,250
        #out = self.dropout(out)
        out = self.conv2(out)# B,64,250
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)# B,64,50
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out

class channels_split_ATT_CNN_linear_avgpool_for_grad(nn.Module):
    def __init__(self,mark = False,extract_dim = 32, hdim = 32):
        super(channels_split_ATT_CNN_linear_avgpool_for_grad, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, extract_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(extract_dim),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)
        self.att2 = self_Attention_1D_for_timestep_without_relu_Linear(extract_dim,hdim)

        self.maxpool = nn.MaxPool1d(4)
        self.avgPool = nn.AvgPool1d(4)
        self.conv1 = nn.Conv1d(extract_dim*12*2, 128, kernel_size=3, stride=1, padding=1)#1250
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.AvgPool1d(5)#b,64,250
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)#200
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.AvgPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2)
        )
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1_1,self.att1v_1 = self.att1(x1)
        x1_2,self.att1v_2 = self.att2(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2_1,self.att2v_1 = self.att1(x2)
        x2_2,self.att2v_2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3_1,self.att3v_1 = self.att1(x3)
        x3_2,self.att3v_2 = self.att2(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4_1,self.att4v_1 = self.att1(x4)
        x4_2,self.att4v_2 = self.att2(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5_1,self.att5v_1 = self.att1(x5)
        x5_2,self.att5v_2 = self.att2(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6_1,self.att6v_1 = self.att1(x6)
        x6_2,self.att6v_2 = self.att2(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7_1,self.att7v_1 = self.att1(x7)
        x7_2,self.att7v_2 = self.att2(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8_1,self.att8v_1 = self.att1(x8)
        x8_2,self.att8v_2 = self.att2(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9_1,self.att9v_1 = self.att1(x9)
        x9_2,self.att9v_2 = self.att2(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10_1,self.att10v_1 = self.att1(x10)
        x10_2,self.att10v_2 = self.att2(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11_1,self.att11v_1 = self.att1(x11)
        x11_2,self.att11v_2 = self.att2(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12_1,self.att12v_1 = self.att1(x12)
        x12_2,self.att12v_2 = self.att2(x12)
        
        #print(x1.shape)
        x = torch.cat((x1_1,x1_2,x2_1,x2_2,x3_1,x3_2,x4_1,x4_2,x5_1,x5_2,x6_1,x6_2,x7_1,x7_2,x8_1,x8_2,x9_1,x9_2,x10_1,x10_2,x11_1,x11_2,x12_1,x12_2),1) #按照第1维度(channel)合并 # B,32*12,1250
        #print(x.shape)
        out = self.conv1(x)# B,128,1250
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)# B,128,250
        #out = self.dropout(out)
        out = self.conv2(out)# B,64,250
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)# B,64,50
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        self.last_out = self.linear_unit(out)
        out = self.softmax(self.last_out)
        
        
        return out

# ECGNet_201911091150
def conv_2d(in_planes, out_planes, stride=(1,1), size=3):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,size), stride=stride,
                     padding=(0,(size-1)//2), bias=False)

def conv_1d(in_planes, out_planes, stride=1, size=3):
    """3x3 convolution with padding"""
    return nn.Conv1d(in_planes, out_planes, kernel_size=size, stride=stride,
                     padding=(size-1)//2, bias=False)
                     
class BasicBlock1d_(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, size=3, res=True,se = True):
        super(BasicBlock1d_, self).__init__()
        self.conv1 = conv_1d(inplanes, planes, stride, size=size)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_1d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = conv_1d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm1d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res
        self.se=se
        if(self.se):
            self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
            self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)   
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out) 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.se:
            original_out = out
            out = self.globalAvgPool(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            out = out.view(out.size(0), out.size(1), 1)
            out = out * original_out
        
        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        #out = self.relu(out)
        
        return out

class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=(1,1), downsample=None, size=3, res=True,se = True):
        super(BasicBlock2d, self).__init__()
        self.conv1 = conv_2d(inplanes, planes, stride, size=size) #只有第一个卷积是使用给定的步长，意味着所有的
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv_2d(planes, planes, size=size)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv_2d(planes, planes, size=size)
        self.bn3 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(.2)
        self.downsample = downsample
        self.stride = stride
        self.res = res
        self.se = se
        if(se):
            self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(in_features=planes, out_features=round(planes / 16))
            self.fc2 = nn.Linear(in_features=round(planes / 16), out_features=planes)
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)   
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out) 
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out) 
        if self.se:
            original_out = out
            out = self.globalAvgPool(out)
            out = out.view(out.size(0), -1)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            out = out.view(out.size(0), out.size(1), 1, 1)
            out = out * original_out
        if self.res:
            if self.downsample is not None:
                residual = self.downsample(x)
            out += residual
        #out = self.relu(out)
        
        return out

class ECGNet(nn.Module):
    def __init__(self, input_channel=1, num_classes=2,mark = False, res=True,se = True):#, layers=[2, 2, 2, 2, 2, 2]
        self.mark = mark
        sizes = [
            [3,3,3,3,3,3],
            [5,5,5,5,3,3],
            [7,7,7,7,3,3],
                ]
        self.sizes = sizes
        layers = [
            [3,3,2,2,2,2],
            [3,2,2,2,2,2],
            [2,2,2,2,2,2]
            ]
           

        super(ECGNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_channel, 32, kernel_size=(1,50), stride=(1,2), padding=(0,0),
                               bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.res = res
        self.se = se
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
        #                        bias=False)
        # self.bn2 = nn.BatchNorm2d(32)


        self.inplanes = 32
        self.layers = nn.Sequential()
        self.layers.add_module('layer_1',self._make_layer2d(BasicBlock2d,32,1,stride=(1,2),size=15, res=self.res,se = self.se))
        self.layers.add_module('layer_2',self._make_layer2d(BasicBlock2d,32,1,stride=(1,2),size=15, res=self.res,se = self.se))
        self.layers.add_module('layer_3',self._make_layer2d(BasicBlock2d,32,1,stride=(1,2),size=15, res=self.res,se = self.se))
        #self.conv3 = nn.Conv2d(32, 32, kernel_size=(1,16), stride=(1,2), padding=(0,0),
        #                       bias=False)
        #print(self.conv2)
        #self.bn3 = nn.BatchNorm2d(32)
        #self.dropout = nn.Dropout(.2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(-1)   

        self.layers1_list = nn.ModuleList()
        self.layers2_list = nn.ModuleList()
        for i,size in enumerate(sizes):
            self.inplanes = 32 
            self.layers1 = nn.Sequential()
            self.layers2 = nn.Sequential()
            self.layers1.add_module('layer{}_1_1'.format(size), self._make_layer2d(BasicBlock2d, 32, layers[i][0], stride=(1,1), size=sizes[i][0], res=self.res,se = self.se))
            self.layers1.add_module('layer{}_1_2'.format(size), self._make_layer2d(BasicBlock2d, 32, layers[i][1], stride=(1,1), size=sizes[i][1], res=self.res,se = self.se))
            self.inplanes *= 12
            self.layers2.add_module('layer{}_2_1'.format(size), self._make_layer1d(BasicBlock1d_, 384 , layers[i][2], stride=2, size=sizes[i][2], res=self.res,se = self.se))
            self.layers2.add_module('layer{}_2_2'.format(size), self._make_layer1d(BasicBlock1d_, 384 , layers[i][3], stride=2, size=sizes[i][3], res=self.res,se = self.se))
            self.layers2.add_module('layer{}_2_3'.format(size), self._make_layer1d(BasicBlock1d_, 384 , layers[i][4], stride=2, size=sizes[i][4], res=self.res,se = self.se))
            self.layers2.add_module('layer{}_2_4'.format(size), self._make_layer1d(BasicBlock1d_, 384 , layers[i][5], stride=2, size=sizes[i][5], res=self.res,se = self.se))
            
            self.layers1_list.append(self.layers1)
            self.layers2_list.append(self.layers2)

        self.dropout = nn.Dropout(p=0.1)
        self.fc = nn.Linear(384 *len(sizes), num_classes)
        
    def _make_layer1d(self, block, planes, blocks, stride=2, size=3, res=True,se = True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res,se = se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res,se = se))

        return nn.Sequential(*layers)
    
    def _make_layer2d(self, block, planes, blocks, stride=(1,2), size=3, res=True,se = True):
        downsample = None
        if stride != (1,1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=(1,1), padding=(0,0), stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, size=size, res=res, se = se))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, size=size, res=res, se = se))

        return nn.Sequential(*layers)
    

    def forward(self, x0):
        batch_size, channels,seq_len = x0.shape
        # x0 = x0+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(x0.device)#位置编码

        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                x0 = mark_input(x0,mark_lenth=mark_lenth[0])  # type: ignore
        x0 = x0.unsqueeze(1)
        x0 = self.conv1(x0) 
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.layers(x0)
        
        #x0 = self.bn2(x0)
        #x0 = self.relu(x0)
        x0 = self.dropout(x0)  # type: ignore
        

        xs = []
        for i in range(len(self.sizes)):
            #print(self.layers1_list[i])
            x = self.layers1_list[i](x0)
            x = torch.flatten(x, start_dim=1,end_dim=2)
            x = self.layers2_list[i](x)
            x = self.avgpool(x)
            # x = self.dropout(x)  # type: ignore
            xs.append(x)
        out = torch.cat(xs, dim=2)
        out = out.view(out.size(0), -1)
        self.out = self.fc(out)
        out = self.softmax(self.out)

        return out

class channels_split_ATT_CNN__linear(nn.Module):
    def __init__(self,mark = False):
        super(channels_split_ATT_CNN__linear, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att2 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att3 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att4 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att5 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att6 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att7 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att8 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att9 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att10 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att11 = self_Attention_1D_for_timestep_without_relu_Linear(16)
        self.att12 = self_Attention_1D_for_timestep_without_relu_Linear(16)

        self.maxpool = nn.MaxPool1d(4)
        self.conv1 = nn.Conv1d(192, 128, kernel_size=3, stride=1, padding=1)#1250
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(5)#b,64,250
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)#200
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        
        if(self.mark):
            input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))  

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1,self.attv1 = self.att1(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2,self.attv2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3,self.attv3 = self.att3(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4,self.attv4 = self.att4(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5,self.attv5 = self.att5(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6,self.attv6 = self.att6(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7,self.attv7 = self.att7(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8,self.attv8 = self.att8(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9,self.attv9 = self.att9(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10,self.attv10 = self.att10(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11,self.attv11 = self.att11(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12,self.attv12 = self.att12(x12)
        
        #print(x1.shape)
        x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12),1) #按照第1维度(channel)合并 # B,32*12,200
        #print(x.shape)
        out = self.conv1(x)# B,128,200
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)# B,128,50
        #out = self.dropout(out)
        out = self.conv2(out)# B,64,25
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out

class channels_split_ATT_CNN_(nn.Module):
    def __init__(self,mark = False):
        super(channels_split_ATT_CNN_, self).__init__()
        self.mark = mark
        self.channels_unit1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit2 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()

        )
        self.channels_unit3 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit4 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit5 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit6 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit7 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit8 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit9 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit10 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit11 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )
        self.channels_unit12 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU()
        )

        self.att1 = self_Attention_1D_for_timestep_without_relu(16)
        self.att2 = self_Attention_1D_for_timestep_without_relu(16)
        self.att3 = self_Attention_1D_for_timestep_without_relu(16)
        self.att4 = self_Attention_1D_for_timestep_without_relu(16)
        self.att5 = self_Attention_1D_for_timestep_without_relu(16)
        self.att6 = self_Attention_1D_for_timestep_without_relu(16)
        self.att7 = self_Attention_1D_for_timestep_without_relu(16)
        self.att8 = self_Attention_1D_for_timestep_without_relu(16)
        self.att9 = self_Attention_1D_for_timestep_without_relu(16)
        self.att10 = self_Attention_1D_for_timestep_without_relu(16)
        self.att11 = self_Attention_1D_for_timestep_without_relu(16)
        self.att12 = self_Attention_1D_for_timestep_without_relu(16)

        self.maxpool = nn.MaxPool1d(4)
        self.conv1 = nn.Conv1d(192, 128, kernel_size=3, stride=1, padding=1)#1250
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool1d(5)#b,64,250
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2)#200
        self.bn2 = nn.BatchNorm1d(64)
        self.maxpool2 = nn.MaxPool1d(5)

        self.linear_unit = nn.Sequential(
            nn.Linear(3200,512),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.Softmax(dim=1)
        )
    
    def forward(self, input):
        batch_size, channels,seq_len = input.shape
        input = input+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(input.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                input = mark_input(input,mark_lenth=int(mark_lenth[0]))  

        x1 = self.maxpool(self.channels_unit1(input[:,0:1,:]) + input[:,0:1,:])#提取channel_i的数据
        x1,self.attv1 = self.att1(x1)

        x2 = self.maxpool(self.channels_unit2(input[:,1:2,:]) + input[:,0:1,:])#提取channel_i的数据
        x2,self.attv2 = self.att2(x2)

        x3 = self.maxpool(self.channels_unit3(input[:,2:3,:]) + input[:,2:3,:])#提取channel_i的数据
        x3,self.attv3 = self.att3(x3)
        
        x4 = self.maxpool(self.channels_unit4(input[:,3:4,:]) + input[:,3:4,:])#提取channel_i的数据
        x4,self.attv4 = self.att4(x4)

        x5 = self.maxpool(self.channels_unit5(input[:,4:5,:]) + input[:,4:5,:])#提取channel_i的数据
        x5,self.attv5 = self.att5(x5)

        x6 = self.maxpool(self.channels_unit6(input[:,5:6,:]) + input[:,5:6,:])#提取channel_i的数据
        x6,self.attv6 = self.att6(x6)

        x7 = self.maxpool(self.channels_unit7(input[:,6:7,:]) + input[:,6:7,:])#提取channel_i的数据
        x7,self.attv7 = self.att7(x7)

        x8 = self.maxpool(self.channels_unit8(input[:,7:8,:]) + input[:,7:8,:])#提取channel_i的数据
        x8,self.attv8 = self.att8(x8)

        x9 = self.maxpool(self.channels_unit9(input[:,8:9,:]) + input[:,8:9,:])#提取channel_i的数据
        x9,self.attv9 = self.att9(x9)

        x10 = self.maxpool(self.channels_unit10(input[:,9:10,:]) + input[:,9:10,:])#提取channel_i的数据
        x10,self.attv10 = self.att10(x10)

        x11 = self.maxpool(self.channels_unit11(input[:,10:11,:]) + input[:,10:11,:])#提取channel_i的数据
        x11,self.attv11 = self.att11(x11)

        x12 = self.maxpool(self.channels_unit12(input[:,11:,:]) + input[:,11:,:])#提取channel_i的数据  
        x12,self.attv12 = self.att12(x12)
        
        #print(x1.shape)
        x = torch.cat((x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12),1) #按照第1维度(channel)合并 # B,32*12,200
        #print(x.shape)
        out = self.conv1(x)# B,128,200
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool1(out)# B,128,50
        #out = self.dropout(out)
        out = self.conv2(out)# B,64,25
        out = self.bn2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
        
        #out,self.att_value = self.att(out)
        #out = self.softmax(out)
        #print(out)
        out = out.view(out.size(0),-1)
        out = self.linear_unit(out)
        
        
        return out


