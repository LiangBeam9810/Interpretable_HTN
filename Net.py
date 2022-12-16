import torch
import torch.nn as nn
import Models
from self_attention import *

class ResSeBlock1d(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, kernel_size = (3,3),res = True,se=True):
        super(ResSeBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, outplanes, kernel_size= kernel_size[0], stride=stride, 
                               padding=(kernel_size[0]-1)//2, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(outplanes, outplanes, kernel_size=kernel_size[1], stride=1, 
                               padding=(kernel_size[1]-1)//2, bias=False)
        self.bn2 = nn.BatchNorm1d(outplanes)
        self.conv3 = nn.Conv1d(outplanes, outplanes, kernel_size=kernel_size[1], stride=1, 
                               padding=(kernel_size[1]-1)//2, bias=False)
        self.bn3 = nn.BatchNorm1d(outplanes)
        
        if (stride != 1 or inplanes != outplanes): #
            self.downsample = nn.Sequential(
                nn.Conv1d(inplanes, outplanes, kernel_size= 1, stride=stride, 
                          padding=0, bias=False),
                nn.BatchNorm1d(outplanes)
            )
        else:    
            self.downsample = None

        self.res = res
        self.se = se
        if(se):
            self.globalAvgPool = nn.AdaptiveAvgPool1d(1)
            self.fc1 = nn.Linear(in_features=outplanes, out_features=round(outplanes / 16))
            self.fc2 = nn.Linear(in_features=round(outplanes / 16), out_features=outplanes)
            self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        residual = x
       
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dropout(out)
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
            
        out = self.relu(out)
        return out

class ResSeBlock2d(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, kernel_size =(3,3),dilation=(1,1),res = True,se=True):
        super(ResSeBlock2d, self).__init__()
        self.conv12d = nn.Conv2d(inplanes, outplanes, kernel_size, stride=(1,stride), dilation=dilation,
                               padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2), bias=False)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv22d = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=(1,1), dilation=(1,1),
                               padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2), bias=False)
        self.bn2 = nn.BatchNorm2d(outplanes)
        self.conv32d = nn.Conv2d(outplanes, outplanes, kernel_size=kernel_size, stride=(1,1), dilation=(1,1),
                               padding=((kernel_size[0]-1)//2,(kernel_size[1]-1)//2), bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        
        if (stride != 1 or inplanes != outplanes): #
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, outplanes, kernel_size= 1, stride=(1,stride), 
                          padding=0, bias=False),
                nn.BatchNorm2d(outplanes)
            )
        else:    
            self.downsample = None

        self.res = res
        self.se = se
        if(se):
            self.globalAvgPool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Linear(in_features=outplanes, out_features=round(outplanes / 16))
            self.fc2 = nn.Linear(in_features=round(outplanes / 16), out_features=outplanes)
            self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        residual = x
       
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv12d(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv22d(out)
        out = self.dropout(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv32d(out)

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
            
        out = self.relu(out)
        return out

class channels_branch_CNN(nn.Module):
    def __init__(self,mark = True,res = True,se=True,Dropout_rate = 0.1,size = [[3,3,3,3,3,3],
                                                                                [7,7,7,7,3,3],
                                                                                [5,5,5,5,3,3]]):
        
        
        super(channels_branch_CNN, self).__init__()
        self.mark = mark
        self.res = res
        self.se = se
        self.conv0 = nn.Conv2d(1,32,(1,51),(1,2),(0,25))
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ResSeBlock2d(inplanes=32,outplanes=32,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        self.conv2 = ResSeBlock2d(inplanes=32,outplanes=32,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        self.conv3 = ResSeBlock2d(inplanes=32,outplanes=32,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        
        # self.self_att = self_Attention_1D_for_timestep_without_relu_Linear(32,32)
        self.self_att_t = self_Attention_1D_for_timestep_without_relu(384)
        self.sizes = size
        
        self.layers_list_2d = nn.ModuleList()
        for i,size in enumerate(self.sizes):
            self.layers = nn.Sequential()
            self.inplanes = 32 
            layers = nn.Sequential()
            layers.append(ResSeBlock2d(inplanes=self.inplanes,outplanes=32,stride=1, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=32,outplanes=32,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=32,outplanes=32,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=32,outplanes=32,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            self.layers_list_2d.append(layers)
    
        self.layers_list_1d = nn.ModuleList()
        for i,size in enumerate(self.sizes):
            self.layers = nn.Sequential()
            self.inplanes = 32*12
            layers = nn.Sequential()
            layers.append(ResSeBlock1d(inplanes=self.inplanes,outplanes=384,stride=2, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=2, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=2, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=2, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            
            self.layers_list_1d.append(layers)    
        self.dorp = nn.Dropout(p = Dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.inplanes*len(self.sizes),2)
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        batch_size, channels,seq_len = x.shape
        #x = x+(Models.create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(x.device)#位置编码
        if(self.mark):
            if self.training:
                if(torch.rand(1)>0.5):
                    mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                    x = Models.mark_input(x,mark_lenth=int(mark_lenth[0]))
        
        x = x.unsqueeze(1)
        x = self.conv0(x)
        x = self.bn(x)    
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)#b,32,12,313
        # x1 = torch.flatten(x, start_dim=1,end_dim=2)#[N,384,313]
        # x1,self.att1 = self.self_att_t(x1)
        # x1 = torch.reshape(x1,x.shape)
        x = self.dorp(x)
        xs = []
        for i in range(len(self.sizes)):
            x1 = self.layers_list_2d[i](x)#[N,D,12,L]
            x1 = torch.flatten(x1, start_dim=1,end_dim=2)#[N,D*12,L]
            x1 = self.layers_list_1d[i](x1)#[N,D*12,L]
            x1 = self.avgpool(x1)#[N,D,1]
            x1 = self.dorp(x1)
            xs.append(x1) #[N,D*12,L]
        out = torch.cat(xs, dim=1)#[N,3*D,L]
        
        out = out.view(out.size(0), -1)
        self.last_out = self.fc(out)
        out = self.softmax(self.last_out)
        return out
    
    
class MLBFNet(nn.Module):
    def __init__(self,num_class,mark = True,res = True,se=True,Dropout_rate = 0.2,size = [[3,3,3,3,3,3],
                                                                                [7,7,7,7,3,3],
                                                                                [5,5,5,5,3,3]]):
        super(MLBFNet, self).__init__()
        self.mark = mark
        self.res = res
        self.se = se
        self.Dropout_rate = Dropout_rate
        self.sizes = size
        
        self.conv0 = nn.Conv2d(1,16,(1,51),(1,2),(0,25))
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ResSeBlock2d(inplanes=16,outplanes=16,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        self.conv2 = ResSeBlock2d(inplanes=16,outplanes=16,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        self.conv3 = ResSeBlock2d(inplanes=16,outplanes=16,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        
        self.layers0 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
        )
        self.layers1 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers2 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers3 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers4 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers5 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers6 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
        )
        self.layers7 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers8 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers9 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers10 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),

        )
        self.layers11 = nn.Sequential(
            ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
        )
        
        self.layers_list_2d = nn.ModuleList()
        for i,size in enumerate(self.sizes):
            self.layers = nn.Sequential()
            self.inplanes = 32 
            layers = nn.Sequential()
            layers.append(ResSeBlock2d(inplanes=self.inplanes,outplanes=32,stride=1, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=32,outplanes=32,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=32,outplanes=32,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=32,outplanes=32,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            self.layers_list_2d.append(layers)
    
        self.layers_list_1d = nn.ModuleList()
        for i,size in enumerate(self.sizes):
            self.layers = nn.Sequential()
            self.inplanes = 32*12
            layers = nn.Sequential()
            layers.append(ResSeBlock1d(inplanes=self.inplanes,outplanes=384,stride=2, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            # layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=2, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            # layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=2, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            # layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            
            layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=2, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            # layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            
            self.layers_list_1d.append(layers)    
        self.dorp = nn.Dropout(p = Dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.inplanes*len(self.sizes),num_class)
        self.softmax = nn.Softmax(-1)
    def forward(self, x):
        batch_size, channels,seq_len = x.shape
        #x = x+(Models.create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(x.device)#位置编码
        if(self.mark):
            if self.training:
                if(torch.rand(1)>0.5):
                    mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                    x = Models.mark_input(x,mark_lenth=int(mark_lenth[0]))
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
        x0 = torch.cat((x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),\
                x4.unsqueeze(1),x5.unsqueeze(1),x6.unsqueeze(1),x7.unsqueeze(1),\
                x8.unsqueeze(1),x9.unsqueeze(1),x10.unsqueeze(1),x11.unsqueeze(1)),dim=1)
        x0 = x0.permute(0,2,1,3)#B 16 12 L/2/2/2/2
        
        
        x = x.unsqueeze(1)
        x = self.conv0(x)
        x = self.bn(x)    
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)#b,32,12,313
        # x1 = torch.flatten(x, start_dim=1,end_dim=2)#[N,384,313]
        # x1,self.att1 = self.self_att_t(x1)
        # x1 = torch.reshape(x1,x.shape)
        x = torch.cat((x,x0),dim = 1) #B 32 12 L/2/2/2/2
        x = self.dorp(x)
        xs = []
        for i in range(len(self.sizes)):
            x1 = self.layers_list_2d[i](x)#[N,D,12,L]
            x1 = torch.flatten(x1, start_dim=1,end_dim=2)#[N,D*12,L]
            x1 = self.layers_list_1d[i](x1)#[N,D*12,L]
            x1 = self.avgpool(x1)#[N,D,1]
            x1 = self.dorp(x1)
            xs.append(x1) #[N,D*12,L]
        out = torch.cat(xs, dim=1)#[N,3*D,L]
        
        out = out.view(out.size(0), -1)
        self.last_out = self.fc(out)
        out = self.softmax(self.last_out)
        return out
    
  
class MLBFNet_GUR(nn.Module):
    def __init__(self,mark = True,res = True,se=True,Dropout_rate = 0.2,size = [[3,3,3,3,3,3],
                                                                                [5,5,5,5,3,3],
                                                                                [7,7,7,7,3,3]]):
        super(MLBFNet_GUR, self).__init__()
        self.mark = mark
        self.res = res
        self.se = se
        self.Dropout_rate = Dropout_rate
        self.sizes = size
        
        self.conv0 = nn.Conv2d(1,16,(1,51),(1,2),(0,25))
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ResSeBlock2d(inplanes=16,outplanes=16,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        self.conv2 = ResSeBlock2d(inplanes=16,outplanes=16,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        self.conv3 = ResSeBlock2d(inplanes=16,outplanes=16,stride=2,kernel_size=(1,15),res=self.res,se=self.se)
        
        self.layers0 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
        )
        self.layers1 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers2 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers3 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers4 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers5 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers6 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
        )
        self.layers7 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers8 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers9 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers10 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),

        )
        self.layers11 = nn.Sequential(
            # ResSeBlock1d(inplanes=1,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            # ResSeBlock1d(inplanes=16,outplanes=16,stride=2, kernel_size=(3,3), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=1,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
            ResSeBlock1d(inplanes=16,outplanes=16,stride=4, kernel_size=(15,15), res=self.res,se=self.se),
        )
        
        self.layers_list_2d = nn.ModuleList()
        for i,size in enumerate(self.sizes):
            self.layers = nn.Sequential()
            self.inplanes = 32+32 
            layers = nn.Sequential()
            layers.append(ResSeBlock2d(inplanes=self.inplanes,outplanes=64,stride=2, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=64,outplanes=64,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=64,outplanes=64,stride=2, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            layers.append(ResSeBlock2d(inplanes=64,outplanes=64,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            self.layers_list_2d.append(layers)
    
        self.layers_list_1d = nn.ModuleList()
        for i,size in enumerate(self.sizes):
            self.layers = nn.Sequential()
            self.inplanes = 64*12
            layers = nn.Sequential()
            layers.append(ResSeBlock1d(inplanes=self.inplanes,outplanes=512,stride=2, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            layers.append(ResSeBlock1d(inplanes=512,outplanes=512,stride=1, kernel_size=(self.sizes[i][0],self.sizes[i][1]), res=res, se = se))
            
            layers.append(ResSeBlock1d(inplanes=512,outplanes=512,stride=2, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            layers.append(ResSeBlock1d(inplanes=512,outplanes=512,stride=1, kernel_size=(self.sizes[i][2],self.sizes[i][3]), res=res, se = se))
            
            # layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=2, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            # layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            
            # layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=2, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            # layers.append(ResSeBlock1d(inplanes=384,outplanes=384,stride=1, kernel_size=(self.sizes[i][4],self.sizes[i][5]), res=res, se = se))
            
            self.layers_list_1d.append(layers)    
        self.dorp = nn.Dropout(p = Dropout_rate)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512*len(self.sizes),2)
        self.softmax = nn.Softmax(-1)
        self.GRU0 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU1 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU2 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU3 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU4 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU5 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU6 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU7 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU8 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU9 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU10 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        self.GRU11 = nn.GRU(16, 16, 2, batch_first=True,bidirectional=True)
        
        self.GRU = nn.GRU(16,16,2,batch_first=True,bidirectional=True)
        
    def forward(self, x):
        batch_size, channels,seq_len = x.shape
        #x = x+(Models.create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(x.device)#位置编码
        if(self.mark):
            if self.training:
                if(torch.rand(1)>0.5): #mark
                    mark_lenth = torch.randint(int(seq_len/10),int(seq_len/5),[1])
                    x = Models.mark_input(x,mark_lenth=int(mark_lenth[0]))

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
        
        x0,_ = (self.GRU0(x0.permute(0,2 ,1)))
        x1,_ = (self.GRU1(x1.permute(0,2 ,1)))
        x2,_ = (self.GRU2(x2.permute(0,2 ,1)))
        x3,_ = (self.GRU3(x3.permute(0,2 ,1)))
        x4,_ = (self.GRU4(x4.permute(0,2 ,1)))
        x5,_ = (self.GRU5(x5.permute(0,2 ,1)))
        x6,_ = (self.GRU6(x6.permute(0,2 ,1)))
        x7,_ = (self.GRU7(x7.permute(0,2 ,1)))
        x8,_ = (self.GRU8(x8.permute(0,2 ,1)))
        x9,_ = (self.GRU9(x9.permute(0,2 ,1)))
        x10,_ = (self.GRU10(x10.permute(0,2 ,1)))
        x11,_ = (self.GRU11(x11.permute(0,2 ,1)))
        
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
        

        x0 = torch.cat((x0.unsqueeze(1),x1.unsqueeze(1),x2.unsqueeze(1),x3.unsqueeze(1),\
                x4.unsqueeze(1),x5.unsqueeze(1),x6.unsqueeze(1),x7.unsqueeze(1),\
                x8.unsqueeze(1),x9.unsqueeze(1),x10.unsqueeze(1),x11.unsqueeze(1)),dim=1)
        x0 = x0.permute(0,2,1,3)#B 16 12 L/2/2/2/2
        x0 = self.dorp(x0)
        
        x = x.unsqueeze(1)
        x = self.conv0(x)
        x = self.bn(x)    
        x = self.relu(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.shape[0],x.shape[1],x.shape[2]*x.shape[3])# b,16,12,313
        x,_ = self.GRU(x.permute(0,2 ,1))
        x = (self.dorp(x.permute(0,2 ,1)))
        x = x.view(x.shape[0],32,12,313)
        x = torch.cat((x,x0),dim = 1) #B 32 12 L/2/2/2/2
        xs = []
        for i in range(len(self.sizes)):
            x1 = self.layers_list_2d[i](x)#[N,D,12,L]
            x1 = torch.flatten(x1, start_dim=1,end_dim=2)#[N,D*12,L]
            x1 = self.layers_list_1d[i](x1)#[N,D*12,L]
            x1 = self.avgpool(x1)#[N,D,1]
            x1 = self.dorp(x1)
            xs.append(x1) #[N,D*12,L]
        out = torch.cat(xs, dim=1)#[N,3*D,L]
        out = out.view(out.size(0), -1)
        self.last_out = self.dorp(self.fc(out))
        out = self.softmax(self.last_out)
        return out