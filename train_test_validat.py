import numpy as np
import torch
import csv
import random
import torch.utils.data as Data

# 定义训练函数
def train_model(train_loader,model,criterion,optimizer,device):
    model.train()
    train_loss = []
    train_acc = []   

    for i,data in enumerate(train_loader,0):

        # inputs,labels = data[0].cuda(),data[1].cuda()
        inputs,labels = data[0].to(device),data[1].to(device) # 获取数据
        outputs = model(inputs) # 预测结果
        loss = criterion(outputs,labels) # 计算loss

        optimizer.zero_grad() # 梯度清0
        loss.backward() # 反向传播
        optimizer.step() # 更新系数

        #print("labels:",labels)
        outputs_ = outputs.clone()
        _,pred = outputs_.max(1) # 求概率最大值对应的标签
        num_correct = (pred == labels).sum().item()
        acc = num_correct/len(labels) # 计算准确率
        train_loss.append(loss.item())
        train_acc.append(acc)

    return np.mean(train_loss),np.mean(train_acc)

# 定义测试函数，具体结构与训练函数相似
def test_model(test_loader,criterion,model,device):
    model.eval()
    test_loss = []
    test_acc = []   

    for i,data in enumerate(test_loader,0):

        inputs,labels = data[0].to(device),data[1].to(device)
        outputs = model(inputs)
        loss = criterion(outputs,labels)

        #print("output:",outputs)
        print("labels:",labels)
        outputs_ = outputs.clone()
        _,pred = outputs_.max(1) # 求概率最大值对应的标签
        print("pred:",pred)
        num_correct = (pred == labels).sum().item()
        acc = num_correct/len(labels)
        test_loss.append(loss.item())
        test_acc.append(acc)

    return np.mean(test_loss),np.mean(test_acc)
