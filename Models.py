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
        
class MLBFNet(nn.Module):
    def __init__(self,inplanes, outplanes,dropout = 0.2,res=True,se = True,mark = True):
        super(MLBFNet,self).__init__()
        self.res = res
        self.se=se
        self.mark = mark
        self.layers0 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers1 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers2 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers3 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers4 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers5 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers6 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
        )
        self.layers7 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers8 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers9 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers10 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),

        )
        self.layers11 = nn.Sequential(
            block1d_1(inplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_1(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
            block1d_2(outplanes, outplanes, dropout=dropout, res=self.res, se=self.se),
        )
        self.GRU = nn.GRU(outplanes, outplanes, 1, batch_first=True,bidirectional=True)
        self.conv1 = nn.Conv2d(1,32,(5,5),(5,5),(2,2))
        self.conv2 = nn.Conv2d(32,32,(5,5),(5,5),(2,2))
        self.conv3 = nn.Conv2d(32,32,(3,3),(2,2),(1,1))
        self.conv4 = nn.Conv2d(32,32,(3,3),(2,2),(1,1))
        self.dorp = nn.Dropout(dropout)
        self.att =att(outplanes,outplanes,dropout)
        self.fc = nn.Linear(192,2)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        batch_size, channels,seq_len = x.shape
        # x0 = x0+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(x0.device)#位置编码
        if(self.mark):
            if self.training:
                mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                x = mark_input(x,mark_lenth=mark_lenth[0])  # type: ignore
        x0 = self.layers0(x[:,:1,:])
        x1 = self.layers1(x[:,1:2,:])
        x2 = self.layers2(x[:,2:3,:])
        x3 = self.layers3(x[:,3:4,:])
        x4 = self.layers4(x[:,4:5,:])
        x5 = self.layers5(x[:,5:6,:])
        x6 = self.layers6(x[:,6:7,:])
        x7 = self.layers7(x[:,7:8,:])
        x8 = self.layers8(x[:,8:9,:])
        x9 = self.layers9(x[:,9:10,:])
        x10 = self.layers10(x[:,10:11,:])
        x11 = self.layers11(x[:,11:,:])
        
        x0,_ = (self.GRU(x0.permute(0,2 ,1)))
        x1,_ = (self.GRU(x1.permute(0,2 ,1)))
        x2,_ = (self.GRU(x2.permute(0,2 ,1)))
        x3,_ = (self.GRU(x3.permute(0,2 ,1)))
        x4,_ = (self.GRU(x4.permute(0,2 ,1)))
        x5,_ = (self.GRU(x5.permute(0,2 ,1)))
        x6,_ = (self.GRU(x6.permute(0,2 ,1)))
        x7,_ = (self.GRU(x7.permute(0,2 ,1)))
        x8,_ = (self.GRU(x8.permute(0,2 ,1)))
        x9,_ = (self.GRU(x9.permute(0,2 ,1)))
        x10,_ = (self.GRU(x10.permute(0,2 ,1)))
        x11,_ = (self.GRU(x11.permute(0,2 ,1)))
        
        x0 = (self.dorp(x0.permute(0,2 ,1)))
        x1 = (self.dorp(x1.permute(0,2 ,1)))
        x2 = (self.dorp(x2.permute(0,2 ,1)))
        x3 = (self.dorp(x3.permute(0,2 ,1)))
        x4 = (self.dorp(x4.permute(0,2 ,1)))
        x5 = (self.dorp(x5.permute(0,2 ,1)))
        x6 = (self.dorp(x6.permute(0,2 ,1)))
        x7 = (self.dorp(x7.permute(0,2 ,1)))
        x8 = (self.dorp(x8.permute(0,2 ,1)))
        x9 = (self.dorp(x9.permute(0,2 ,1)))
        x10 = (self.dorp(x10.permute(0,2 ,1)))
        x11 = (self.dorp(x11.permute(0,2 ,1)))
        
        x = torch.cat((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11),dim=1)
        x = x.unsqueeze(dim = 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(batch_size, -1)
        x = self.fc(x)
        x = self.softmax(x)
        return x