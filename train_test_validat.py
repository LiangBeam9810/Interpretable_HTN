import numpy as np
import torch
from sklearn.metrics import f1_score

# 定义训练函数
def train_model(train_loader,model,criterion,optimizer,device):
    
    train_loss = []
    train_acc = []   
    y_ture = []
    y_pred = []
    for i,data in enumerate(train_loader,0):
        model.train()
        # inputs,labels = data[0].cuda(),data[1].cuda()
        inputs,labels = data[0].to(device),data[1].to(device) # 获取数据
        #print(model)
        #batch_size, channels,seq_len = inputs.shape

        #inputs = inputs+(create_1d_absolute_sin_cos_embedding(batch_size,channels,seq_len)).to(inputs.device)
        optimizer.zero_grad() # 梯度清0
        outputs = model(inputs) # 预测结果
        loss = criterion(outputs,labels) # 计算loss
        # # L2 #################################
        # # Replaces pow(2.0) with abs() for L1 regularization
        # l2_lambda = 0.001
        # l2_norm = sum(p.pow(2.0).sum()
        #           for p in model.parameters())
        # loss = loss + l2_lambda * l2_norm
        # ######################################
        loss.backward() # 反向传播
        optimizer.step() # 更新系数
        #print(outputs)
        
        #print("labels:",labels)
        _,pred = outputs.max(1) # 求概率最大值对应的标签
        #print(pred)
        _,label = labels.max(1) #
        num_correct = (pred == label).sum().item()
        acc = num_correct/len(label) # 计算准确率
        train_loss.append(loss.item())
        train_acc.append(acc)
        y_ture.extend((label.to('cpu').detach().numpy().flatten()).tolist())
        y_pred.extend((pred.to('cpu').detach().numpy().flatten()).tolist())
    return y_ture,y_pred,np.mean(train_loss),np.mean(train_acc)

# 定义测试函数，具体结构与训练函数相似
def test_model(test_loader,criterion,model,device):
    
    test_loss = []
    test_acc = []   
    for i,data in enumerate(test_loader,0):
        model.eval()
        with torch.no_grad():
            inputs,labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)

            #print("output:",outputs)
            #print("labels:",labels)
            _,pred = outputs.max(1) # 求概率最大值对应的标签
            _,label = labels.max(1) #
            #print("pred:",pred)
            num_correct = (pred == label).sum().item()
            acc = num_correct/len(label)
            test_loss.append(loss.item())
            test_acc.append(acc)

    return np.mean(test_loss),np.mean(test_acc)

def eval_model(test_loader,criterion,model,device):
    
    test_loss = []
    test_acc = []   
    y_ture = []
    y_pred = []
    for i,data in enumerate(test_loader,0):
        model.eval()
        with torch.no_grad():
            inputs,labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            #print("output:",outputs)
            #print("labels:",labels)
            _,pred = outputs.max(1) # 求概率最大值对应的标签
            _,label = labels.max(1) #
            #print("pred:",pred)
            num_correct = (pred == label).sum().item()
            acc = num_correct/len(label)
            test_loss.append(loss.item())
            test_acc.append(acc)
            y_ture.extend((label.to('cpu').detach().numpy().flatten()).tolist())
            y_pred.extend((pred.to('cpu').detach().numpy().flatten()).tolist())
    return y_ture,y_pred,np.mean(test_loss),np.mean(test_acc),

def eval_model_possibility(test_loader,criterion,model,device):
    test_loss = []
    test_acc = []   
    y_ture = []
    y_pred = []
    possibility = []
    for i,data in enumerate(test_loader,0):
        model.eval()
        with torch.no_grad():
            inputs,labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            #print("output:",outputs)
            #print("labels:",labels)
            _,pred = outputs.max(1) # 求概率最大值对应的标签
            
            #print("pred:",pred)
            outputs_= outputs.to('cpu')
            num_correct = (pred == labels).sum().item()
            acc = num_correct/len(labels)
            test_loss.append(loss.item())
            test_acc.append(acc)
            y_ture.extend((labels.to('cpu').detach().numpy().flatten()).tolist())
            y_pred.extend((pred.to('cpu').detach().numpy().flatten()).tolist())
            possibility.extend((outputs.to('cpu').detach().numpy()).tolist())
    return possibility,y_ture,y_pred,np.mean(test_loss),np.mean(test_acc)





class EarlyStopping:
    
    def __init__(self,patience=7, verbose=True,model_path = "./",delta = 0,positive=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.last_best_score = np.Inf
        self.delta = delta
        self.model_path = model_path
        self.positive= positive

    def __call__(self, score,model,fold = 0):

        #score = -val_loss
        if(self.positive):
            score = score
        else:
            score = -score
            self.delta = -self.delta
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(score, model,fold)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model,fold)
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, score, model,fold = 0):
        if self.verbose:
            print(f'Validation F1 score  increase to ({self.last_best_score:.8f} --> {score:.8f}).  Saving model ...')
            print(" "*20+'-'*50+'\n')
        # torch.save(model, self.model_path+'/all_EarlyStoping_'+str(fold)+'.pt')                 # 这里会存储迄今最优的模型
        torch.save(model.state_dict(), self.model_path+'/parameter_EarlyStoping_' + str(fold) + '.pt')
        self.last_best_score = score


        
def train_model_mixup(train_loader,model,criterion,optimizer,device,alpha=0.2):
    
    train_loss = []
    train_acc = []   
    y_ture = []
    y_pred = []
    for i,data in enumerate(train_loader,0):
        model.train()
        
        inputs,labels = data[0].to(device),data[1].to(device) # 获取数据
        lam = np.random.beta(alpha, alpha)#生成符合Beta分布的值
        index = torch.randperm(inputs.size(0)).cuda()
        mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
        mixed_labels = lam * labels + (1 - lam) * labels[index, :]
        # Mixup loss.    
        optimizer.zero_grad() # 梯度清0
        outputs = model(mixed_inputs) # 预测结果
        loss = criterion(outputs,mixed_labels) # 计算loss
        loss.backward() # 反向传播
        optimizer.step() # 更新系数
        _,pred = outputs.max(1) # 求概率最大值对应的标签
        _,label = mixed_labels.max(1) #
        num_correct = (pred == label).sum().item()
        acc = num_correct/len(label) # 计算准确率
        train_loss.append(loss.item())
        train_acc.append(acc)
        y_ture.extend((label.to('cpu').detach().numpy().flatten()).tolist())
        y_pred.extend((pred.to('cpu').detach().numpy().flatten()).tolist())
    return y_ture,y_pred,np.mean(train_loss),np.mean(train_acc)