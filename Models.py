import torch
import torch.nn as nn
import torch.nn.functional as F
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
def scaler_input(input):  # type: ignore
    factor_max = (1.0/input.max())
    factor = torch.rand(1).to(input.device)
    if factor > factor_max:
        factor = factor_max 
    input = input*factor
    return input
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


# MLBFNet
class block1d_1(nn.Module):
    def __init__(self,inplanes, outplanes = 12,dropout = 0.2,res=True,se = True):
        super(block1d_1,self).__init__()
        self.res = res
        self.se=se
        self.conv1 = nn.Conv1d(in_channels = inplanes,out_channels = outplanes,kernel_size = 3,stride = 1,padding=1, bias=False)
        self.conv2 = nn.Conv1d(outplanes,outplanes, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv1d(outplanes,outplanes,25, 2,12, bias=False) 
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.bn3 = nn.BatchNorm1d(outplanes)
        self.dropout = nn.Dropout(p = dropout)     
        self.relu = nn.LeakyReLU(inplace=True)
        if(self.res):
            self.downsammpler = nn.Sequential(
                nn.Conv1d(inplanes, outplanes,
                          kernel_size=1, padding=0, stride=2, bias=False),
                nn.BatchNorm1d(outplanes),
            )
        if(se):
            self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(in_features=outplanes, out_features=round(outplanes / 16))
            self.fc2 = nn.Linear(in_features=round(outplanes / 16), out_features=outplanes)
            self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
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
        out = self.dropout(out)
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
            residual = self.downsammpler(x)
            out += residual
            
        return out

class block1d_2(nn.Module):
    def __init__(self,inplanes, outplanes = 12,dropout = 0.2,res=True,se = True):
        super(block1d_2,self).__init__()
        self.res = res
        self.se=se
        self.conv1 = nn.Conv1d(in_channels = inplanes,out_channels = outplanes,kernel_size = 3,stride = 1,padding=1, bias=False)
        self.conv2 = nn.Conv1d(outplanes,outplanes, 3, 1, 1, bias=False)
        self.conv3 = nn.Conv1d(outplanes,outplanes,49, 2,24, bias=False) 
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.bn3 = nn.BatchNorm1d(outplanes)
        self.dropout = nn.Dropout(p = dropout)     
        self.relu = nn.LeakyReLU(inplace=True)
        if(self.res):
            self.downsammpler = nn.Sequential(
                nn.Conv1d(inplanes, outplanes,
                          kernel_size=1, padding=0, stride=2, bias=False),
                nn.BatchNorm1d(outplanes),
            )
        if(se):
            self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(in_features=outplanes, out_features=round(outplanes / 16))
            self.fc2 = nn.Linear(in_features=round(outplanes / 16), out_features=outplanes)
            self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
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
        out = self.dropout(out)
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
            residual = self.downsammpler(x)
            out += residual
            
        return out


        
        return out

class att(nn.Module):
    def __init__(self,inplanes = 24, outplanes = 24,dropout = 0.2):
        super(att,self).__init__()
        self.u = nn.Linear(inplanes,outplanes)
        self.a = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.rand(outplanes,1))  
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        u = self.u(x)
        a = self.a((u.permute(0,2 ,1))*gamma)
        faat = x*a
        faat = faat.squeeze(-1)
        faat = self.drop(faat)
        return faat
        



class LSTNet(nn.Module):
    
    def __init__(self):
        super(LSTNet, self).__init__()
        self.num_features = 12
        self.conv1_out_channels = 32 
        self.conv1_kernel_height = 7
        self.recc1_out_channels = 64 
        self.skip_steps = [4, 24] 
        self.skip_reccs_out_channels = [4, 4] 
        self.output_out_features = 2
        self.ar_window_size = 0
        self.dropout = nn.Dropout(p = 0.2)
       
        
        self.conv1 = nn.Conv2d(1, self.conv1_out_channels, 
                               kernel_size=(self.conv1_kernel_height, self.num_features))
        self.recc1 = nn.GRU(self.conv1_out_channels, self.recc1_out_channels, batch_first=True)
        self.skip_reccs = nn.Sequential()
        for i in range(len(self.skip_steps)):
            self.skip_reccs.append(nn.GRU(self.conv1_out_channels, self.skip_reccs_out_channels[i], batch_first=True))
        self.output_in_features = self.recc1_out_channels + np.dot(self.skip_steps, self.skip_reccs_out_channels)
        self.output = nn.Linear(self.output_in_features, self.output_out_features)
        if self.ar_window_size > 0:
            self.ar = nn.Linear(self.ar_window_size, 1)
        
    def forward(self, X):
        """
        Parameters:
        X (tensor) [batch_size, time_steps, num_features]
        """
        batch_size = X.size(0)
        X = X.permute(0,2 ,1)
        # Convolutional Layer
        C = X.unsqueeze(1) # [batch_size, num_channels=1, time_steps, num_features]
        C = F.relu(self.conv1(C)) # [batch_size, conv1_out_channels, shrinked_time_steps, 1]
        C = self.dropout(C)
        C = torch.squeeze(C, 3) # [batch_size, conv1_out_channels, shrinked_time_steps]
        
        # Recurrent Layer
        R = C.permute(0, 2, 1) # [batch_size, shrinked_time_steps, conv1_out_channels]
        out, hidden = self.recc1(R) # [batch_size, shrinked_time_steps, recc_out_channels]
        R = out[:, -1, :] # [batch_size, recc_out_channels]
        R = self.dropout(R)
        #print(R.shape)
        
        # Skip Recurrent Layers
        shrinked_time_steps = C.size(2)
        for i in range(len(self.skip_steps)):
            skip_step = self.skip_steps[i]
            skip_sequence_len = shrinked_time_steps // skip_step
            # shrinked_time_steps shrinked further
            S = C[:, :, -skip_sequence_len*skip_step:] # [batch_size, conv1_out_channels, shrinked_time_steps]
            S = S.view(S.size(0), S.size(1), skip_sequence_len, skip_step) # [batch_size, conv1_out_channels, skip_sequence_len, skip_step=num_skip_components]
            # note that num_skip_components = skip_step
            S = S.permute(0, 3, 2, 1).contiguous() # [batch_size, skip_step=num_skip_components, skip_sequence_len, conv1_out_channels]
            S = S.view(S.size(0)*S.size(1), S.size(2), S.size(3))  # [batch_size*num_skip_components, skip_sequence_len, conv1_out_channels]
            out, hidden = self.skip_reccs[i](S) # [batch_size*num_skip_components, skip_sequence_len, skip_reccs_out_channels[i]]
            S = out[:, -1, :] # [batch_size*num_skip_components, skip_reccs_out_channels[i]]
            S = S.view(batch_size, skip_step*S.size(1)) # [batch_size, num_skip_components*skip_reccs_out_channels[i]]
            S = self.dropout(S)
            R = torch.cat((R, S), 1) # [batch_size, recc_out_channels + skip_reccs_out_channels * num_skip_components]
            #print(S.shape)
        #print(R.shape)
        
        # Output Layer
        O = F.softmax(self.output(R),dim=-1) # [batch_size, output_out_features=1]
        
        return O