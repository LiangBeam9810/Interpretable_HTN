import imp
import numpy as np
import torch
import csv
import random
from Models import mark_input,create_1d_absolute_sin_cos_embedding
import torch.utils.data as Data

# 定义训练函数
def train_model(train_loader,model,criterion,optimizer,device):
    
    train_loss = []
    train_acc = []   

    for i,data in enumerate(train_loader,0):
        model.train()
        # inputs,labels = data[0].cuda(),data[1].cuda()
        inputs,labels = data[0].to(device),data[1].to(device) # 获取数据
        #batch_size, channels,seq_len = inputs.shape

        #inputs = inputs+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(inputs.device)
        optimizer.zero_grad() # 梯度清0
        outputs = model(inputs) # 预测结果
        loss = criterion(outputs,labels) # 计算loss

        loss.backward() # 反向传播
        optimizer.step() # 更新系数
        #print(outputs)
        
        #print("labels:",labels)
        _,pred = outputs.max(1) # 求概率最大值对应的标签
        #print(pred)
        num_correct = (pred == labels).sum().item()
        acc = num_correct/len(labels) # 计算准确率
        train_loss.append(loss.item())
        train_acc.append(acc)

    return np.mean(train_loss),np.mean(train_acc)

# 定义测试函数，具体结构与训练函数相似
def test_model(test_loader,criterion,model,device):
    
    test_loss = []
    test_acc = []   
    for i,data in enumerate(test_loader,0):
        model.eval()
        inputs,labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        #print("output:",outputs)
        #print("labels:",labels)
        _,pred = outputs.max(1) # 求概率最大值对应的标签
        #print("pred:",pred)
        num_correct = (pred == labels).sum().item()
        acc = num_correct/len(labels)
        test_loss.append(loss.item())
        test_acc.append(acc)

    return np.mean(test_loss),np.mean(test_acc)

class EarlyStopping:
    
    def __init__(self,patience=7, verbose=True,model_path = "./",delta = 0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.model_path = model_path

    def __call__(self, val_acc,model):

        #score = -val_loss
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if(score>0.8):
                self.save_checkpoint(val_acc, model)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            print(" "*20+'-'*50+'\n')
        torch.save(model, self.model_path+'/EarlyStoping.pt')                 # 这里会存储迄今最优的模型
        self.val_loss_min = val_loss