import numpy as np
import torch
from sklearn.metrics import f1_score
import math
import time
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def tarinning_one_flod(fold,Model,train_dataset,val_dataset,test_dataset,writer,save_model_path,BATCH_SIZE,DEVICE,
                        criterion = torch.nn.CrossEntropyLoss(),
                        EPOCHS = 100,  
                        PATIENCE = 10,
                        LR_MAX = 1e-2,
                        LR_MIN = 1e-5,
                        warm_up_iter = 5,
                        weight_decay=1e-3,
                        num_workers = 0,
                        shuffle = True,
                        ):
    target = train_dataset.labels
    class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in target])
    samples_weight = torch.from_numpy(samples_weight)
    samples_weight = samples_weight.double()
    sampler = Data.WeightedRandomSampler(samples_weight, len(samples_weight))  # type: ignore

    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=num_workers,pin_memory=True,sampler = sampler)
    valid_dataloader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=num_workers,pin_memory=True)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=num_workers,pin_memory=True)
    early_stopping = EarlyStopping(PATIENCE, verbose=True, model_path=save_model_path, delta=0, positive=False)
    optimizer  = torch.optim.Adam(Model.parameters(), lr=LR_MAX,weight_decay=weight_decay) 
    criterion =  criterion.to(DEVICE)
    
    warm_up_iter = warm_up_iter
    T_max = EPOCHS//2	# 周期
    lr_max = LR_MAX	# 最大值
    lr_min = LR_MIN	# 最小值
    lambda0 = lambda cur_iter: lr_min if  cur_iter < warm_up_iter else \
        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    best_valida_acc = 0
    Model.to(DEVICE)
    for epoch in range(1,EPOCHS):
        time_all=0
        start_time = time.time()
        
        y_true,y_pred,train_loss,train_acc = train_model(train_dataloader, Model, criterion, optimizer,DEVICE) # type: ignore # 训练模型
        
        # F1_score_train =f1_score(y_true, y_pred, average='macro')#F1分数
        # C0 = confusion_matrix(y_true,y_pred)
        
        y_true,y_pred,validate_loss,validate_acc = eval_model(valid_dataloader,criterion,Model,DEVICE) # 验证模型
        time_all = time.time()-start_time
        # F1_score_valid =f1_score(y_true, y_pred, average='macro')#F1分数
        # C1 = confusion_matrix(y_true,y_pred)
        
        writer.add_scalars(main_tag=str(fold)+'_Loss',tag_scalar_dict={'train': train_loss,'validate': validate_loss},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_Accuracy',tag_scalar_dict={'train': train_acc,'validate': validate_acc},global_step=epoch)
        # writer.add_scalars(main_tag=str(fold)+'_LearningRate',tag_scalar_dict={'LR': optimizer.state_dict()['param_groups'][0]['lr']},global_step=epoch)
        # writer.add_scalars(main_tag=str(fold)+'_F1_score',tag_scalar_dict={'train':F1_score_train,'validate': F1_score_valid},global_step=epoch)        
        print('- Epoch: %d - Train_loss: %.5f - Train_acc: %.5f -  - Val_loss: %.5f - Val_acc: %.5f  - T_Time: %.5f' %(epoch,train_loss,train_acc,validate_loss,validate_acc,time_all))
        print('当前学习率：%.8f' %optimizer.state_dict()['param_groups'][0]['lr'])
        # print('train:\n',C0)
        # print('validate:\n',C1)
        
        if(validate_acc>best_valida_acc):
            
            best_valida_acc = validate_acc
            F1_score_valid =f1_score(y_true, y_pred, average='macro')#F1分数
            C1 = confusion_matrix(y_true,y_pred)
            print('Get best valida acc !')
            print('validate: ',F1_score_valid,'\n',C1)
            
        scheduler.step() # 学习率迭代
        
        #是否满足早停法条件
        if(early_stopping(validate_loss,Model,fold)):
            print("Early stopping")
            break
        
    # 计算此flod 在testset上的效果
    best_model_path = save_model_path+'/parameter_EarlyStoping_' + str(fold) + '.pt' #此fold最优参数
    Model.load_state_dict(torch.load(best_model_path))
    
    y_true,y_pred,train_loss,train_acc = train_model(train_dataloader, Model, criterion, optimizer,DEVICE) # type: ignore # 模型
    y_true,y_pred,validate_loss,validate_acc = eval_model(valid_dataloader,criterion,Model,DEVICE) # 验证模型
    F1_score_valid =f1_score(y_true, y_pred, average='macro')#F1分数
    C1 = confusion_matrix(y_true,y_pred)
    print('validate: ',F1_score_valid,'\n',C1)
    
    y_true,y_pred,test_loss,test_acc = eval_model(test_dataloader,criterion,Model,DEVICE) # 验证模型
    F1_score_test =f1_score(y_true, y_pred, average='macro')#F1分数
    C = confusion_matrix(y_true,y_pred)
    print('test: ',F1_score_test,'\n',C)
    print('Fold %d Training Finished' %(fold))
    return train_loss,train_acc,validate_loss,validate_acc,test_loss,test_acc



# 定义训练函数
def train_model(train_loader,model,criterion,optimizer,device,onehot_lable = False):
    
    train_loss = []
    train_acc = []   
    y_ture = []
    y_pred = []
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
        if(onehot_lable):
            _,taget = labels.max(1)
        else:
            taget = labels
        #print(pred)
        num_correct = (pred == taget).sum().item()
        acc = num_correct/len(taget) # 计算准确率
        train_loss.append(loss.item())
        train_acc.append(acc)
        # y_ture.extend((taget.to('cpu').detach().numpy().flatten()).tolist())
        # y_pred.extend((pred.to('cpu').detach().numpy().flatten()).tolist())
    return y_ture,y_pred,np.mean(train_loss),np.mean(train_acc)

def eval_model(test_loader,criterion,model,device,onehot_lable=False):
    
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
            if(onehot_lable):
                _,taget = labels.max(1)
            else:
                taget = labels
            #print("pred:",pred)
            num_correct = (pred == taget).sum().item()
            acc = num_correct/len(taget)
            test_loss.append(loss.item())
            test_acc.append(acc)
            y_ture.extend((taget.to('cpu').detach().numpy().flatten()).tolist())
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
            print(" "*20+'-'*50+'\n')
            print(f'Validation  score to ({self.last_best_score:.8f} --> {score:.8f}).  Saving model ...')
            print(" "*20+'-'*50+'\n')
        # torch.save(model, self.model_path+'/all_EarlyStoping_'+str(fold)+'.pt')                 # 这里会存储迄今最优的模型
        torch.save(model.state_dict(), self.model_path+'/parameter_EarlyStoping_' + str(fold) + '.pt')
        self.last_best_score = score