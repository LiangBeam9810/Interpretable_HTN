import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score,recall_score,roc_auc_score,fbeta_score 
import math
import time
import torch.utils.data as Data
import ECGDataset
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
import pandas as pd
import ECGHandle

def pair_HTN(INPUT_HTN_Df:pd.DataFrame,INPUT_NHTN_Df:pd.DataFrame,Range_max:int = 5,shuffle:bool = False,pair_num:int = 3):
    HTN_Df = ((INPUT_HTN_Df).copy())
    NHTN_Df = ((INPUT_NHTN_Df).copy())#即抽即删,抽出一条删一条
    if(shuffle): #打乱
        HTN_Df = (HTN_Df.sample(frac=1))
        NHTN_Df = (NHTN_Df.sample(frac=1))
    pair_Df = HTN_Df #先将所有HTN存放入其中
    for info in HTN_Df.itertuples(): #遍历每一个HTN样本 寻找与之配对的样本
        age = info.age
        gender = info.gender
        candidate_NHTN_Df = pd.DataFrame()
        for Range in range(1,Range_max): #在 ±Range_max 范围内搜寻ages，且gender相同的NHTN样本
            candidate_NHTN_Df = NHTN_Df[(NHTN_Df['age']>age-Range)&(NHTN_Df['age']<age+Range)&(NHTN_Df['gender']==gender)]
            if(len(candidate_NHTN_Df)>=pair_num):#候选>0时，即使跳出，确保抽取出的样本年龄尽可能与其相近
                break
        if(len(candidate_NHTN_Df)<pair_num):# ±Range_max 范围内都没有，那么就从所有NHTN样本（已经删除了之前被抽到的）中抽 pair_num 个
            print("lack sample like :",info)
            candidate_NHTN_Df = NHTN_Df
        NHTN_data_buff = candidate_NHTN_Df.sample(n=pair_num) #从candidate中随机抽样 pair_num 个样本 ,带来了随机性！！！
        pair_Df = pair_Df.append(NHTN_data_buff)
        NHTN_Df = NHTN_Df.drop(index= (NHTN_data_buff.index))#删除抽中的那个样本
    return pair_Df    #输出配对完之后的DF

# 将Input_DF中从star_index:HTN_pair_size 个label==1的ID 进行配对
def Pair_ID(Input_DF,HTN_pair_rate:float,star_index:int = 0,Range_max:int = 5,shuffle:bool = False,pair_num:int = 3, ):
    HTN_data = Input_DF[(Input_DF['label']==1) ].copy()
    NHTN_data = Input_DF[(Input_DF['label']==0) ].copy()
    HTN_ID_list = HTN_data['ID'].unique().tolist() #所有的HTN的ID号
    HTN_size = HTN_data['ID'].unique().__len__()
    HTN_pair_size = int(HTN_size*HTN_pair_rate) 
    HTN_ID_pair_list = HTN_ID_list[star_index:star_index+HTN_pair_size]#选取用来pair的ID号 从star_index 到star_index+HTN_pair_size
    HTN_pair_index = HTN_data[[True if i in HTN_ID_pair_list else False for i in HTN_data['ID']]].index
    HTN_pair_data = HTN_data.loc[HTN_pair_index].copy()
    pair_ID_list = pair_HTN(HTN_pair_data.drop_duplicates(['ID'],keep='first'),NHTN_data.drop_duplicates(['ID'],keep='first'),
                            Range_max=Range_max,
                            pair_num=pair_num,
                            shuffle=shuffle)['ID'].tolist()#按照年龄和性别对每个ID号去配对 (先去除重复ID)
    pair_index = Input_DF[[True if i in pair_ID_list else False for i in Input_DF['ID']]].index
    pair_df = Input_DF.loc[pair_index].copy()
    left_index = Input_DF[[False if i in pair_ID_list else True for i in Input_DF['ID']]].index #不在test_ID_list的ID 即为tv的
    left_df = Input_DF.loc[left_index].copy()
    return pair_df,left_df




def tarinning_one_flod(fold,Model,train_dataset:ECGHandle.ECG_Dataset ,val_dataset:ECGHandle.ECG_Dataset,test_dataset:ECGHandle.ECG_Dataset,writer,save_model_path,log_path,BATCH_SIZE,DEVICE,
                        criterion = torch.nn.CrossEntropyLoss(),
                        EPOCHS = 100,  
                        PATIENCE = 10,
                        LR_MAX = 1e-2,
                        LR_MIN = 1e-5,
                        warm_up_iter = 5,
                        weight_decay=1e-3,
                        num_workers = 0,
                        shuffle = True,
                        onehot_lable = False,
                        pair_flag = False,
                        train_Df:pd.DataFrame = None,# type: ignore      
                        data_path = ''                  
                        ):
    
    
    
    
    if(not pair_flag):
        target = train_dataset.labels
        class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in target])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        sampler_train = Data.WeightedRandomSampler(samples_weight, 2*len(samples_weight))  # type: ignore
        train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=num_workers,pin_memory=True,sampler=sampler_train)#
    
        # valid_dataloader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        # test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=num_workers,pin_memory=True)   
    else:
        train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle,num_workers=num_workers,pin_memory=True)
        
    valid_dataloader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=num_workers,pin_memory=True)  
    
    early_stopping = EarlyStopping(PATIENCE, verbose=True, model_path=save_model_path, delta=0, positive=False)
    optimizer  = torch.optim.Adam(Model.parameters(), lr=LR_MAX,weight_decay=weight_decay) 
    criterion =  criterion.to(DEVICE)
    
    warm_up_iter = warm_up_iter
    # T_max = EPOCHS//4	# 周期
    lr_max = LR_MAX	# 最大值
    lr_min = LR_MIN	# 最小值
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    best_scoret = 0
    Model.to(DEVICE)
    for epoch in range(1,EPOCHS):
        time_all=0
        start_time = time.time()
        
        y_true,y_pred,train_loss,train_acc = train_model(train_dataloader, Model, criterion, optimizer,DEVICE,onehot_lable=onehot_lable) # type: ignore # 训练模型       
        y_true,y_pred,y_out,validate_loss,validate_acc = eval_model(valid_dataloader,criterion,Model,DEVICE,onehot_lable=onehot_lable) # 验证模型
        # F1_score_valid =f1_score(y_true, y_pred, average='binary')#F1分数
        F1_score_valid =fbeta_score(y_true, y_pred, average='binary',beta=1.2)#F1-β分数
        p_valid = precision_score(y_true, y_pred, average='binary')
        r_valid = recall_score(y_true, y_pred, average='binary')   
        auc_valid = roc_auc_score(y_true,y_score=((np.array(y_out))[:,1]))
        C1 = confusion_matrix(y_true,y_pred)
        print(" "*20+'Validate: ',F1_score_valid,'\n'+" "*20,C1[0],'\n'+" "*20,C1[1],'\n'+" "*20,"precision: ",p_valid,"recall: ",r_valid,'AUC',auc_valid)
        # y_true,y_pred,test_loss,test_acc = eval_model(test_dataloader,criterion,Model,DEVICE,onehot_lable=onehot_lable) # 验证模型
        # F1_score_test =f1_score(y_true, y_pred, average='binary')#F1分数
        # p_test = precision_score(y_true, y_pred, average='binary')
        # r_test = recall_score(y_true, y_pred, average='binary') 
        # C = confusion_matrix(y_true,y_pred)
        # print(" "*20+'test: ',F1_score_test,'\n'+" "*20,C[0],'\n'+" "*20,C[1],'\n'+" "*20,"precision: ",p_test,"recall: ",r_test)
        time_all = time.time()-start_time
        writer.add_scalars(main_tag=str(fold)+'_Loss',tag_scalar_dict={'train': train_loss,'validate': validate_loss},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_Accuracy',tag_scalar_dict={'train': train_acc,'validate': validate_acc},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_LearningRate',tag_scalar_dict={'LR': optimizer.state_dict()['param_groups'][0]['lr']},global_step=epoch)      
        print(" "*20+'- Epoch: %d - Train_loss: %.5f - Train_acc: %.5f -  - Val_loss: %.5f - Val_acc: %.5f  - T_Time: %.5f' %(epoch,train_loss,train_acc,validate_loss,validate_acc,time_all),'LR：%.10f' %optimizer.state_dict()['param_groups'][0]['lr'])
        
        if(float(F1_score_valid) > float(best_scoret)):
            best_scoret = F1_score_valid
            print(" "*20+'-- -- The best model for validate (F1= . ',best_scoret,') -- --')
            torch.save(Model.state_dict(), save_model_path+'/BestF1_' + str(fold) + '.pt')
            
        scheduler.step(metrics=validate_loss) # 学习率迭代
        
        #是否满足早停法条件
        if(early_stopping(validate_loss,Model,fold)):
            print(" "*20+"Early stopping...")
            break
        
        if(pair_flag ):# 每次重新抽取train_pair_Df (train_Df 是已经除去了val_Df的tv_Df)
                train_pair_df,_ = Pair_ID(train_Df,1,star_index=0,Range_max=15,pair_num=1,shuffle=True)
                train_dataset = ECGHandle.ECG_Dataset(data_path,train_pair_df ,preprocess = True)
                train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=shuffle,num_workers=num_workers,pin_memory=True)
    
    
    
    
    print(" "*5+'Best Loss:')
    best_model_path = save_model_path+'/parameter_EarlyStoping_' + str(fold) + '.pt' #此fold最优参数
    Model.load_state_dict(torch.load(best_model_path))
    
    y_true,y_pred,train_loss,train_acc = train_model(train_dataloader, Model, criterion, optimizer,DEVICE,onehot_lable=onehot_lable) # type: ignore # 模型
    Model.load_state_dict(torch.load(best_model_path))
    
    y_true,y_pred,y_out,validate_loss,validate_acc = eval_model(valid_dataloader,criterion,Model,DEVICE,onehot_lable=onehot_lable) # 验证模型
    F1_score_valid =f1_score(y_true, y_pred, average='binary')#F1分数
    auc_valid = roc_auc_score(y_true,(np.array(y_out))[:,1])
    C = confusion_matrix(y_true,y_pred)
    precision_valid = precision_score(y_true, y_pred, average='binary')
    recall_valid = recall_score(y_true, y_pred, average='binary') 
    save_test_infos(log_path+'/Validate_answer_'+str(fold)+'.csv',val_dataset, y_true,y_pred,y_out)
    print(" "*10+'valid')
    print(" "*10,C[0],'\n'+" "*10,C[1])
    print(" "*10+'F1: ',F1_score_valid,)
    print(" "*10+'AUC: ',auc_valid)
    print(" "*10+'precision: ',precision_valid)
    print(" "*10+'recall: ',recall_valid)
    
    y_true,y_pred,y_out,test_loss,test_acc = test_model(test_dataloader,criterion,Model,DEVICE,onehot_lable=onehot_lable) # 验证模型
    F1_score_test =f1_score(y_true, y_pred, average='binary')#F1分数
    auc_test = roc_auc_score(y_true,(np.array(y_out))[:,1])
    C = confusion_matrix(y_true,y_pred)
    precision_test = precision_score(y_true, y_pred, average='binary')
    recall_test = recall_score(y_true, y_pred, average='binary') 
    save_test_infos(log_path+'/Test_answer_'+str(fold)+'.csv',test_dataset, y_true,y_pred,y_out)
    print(" "*10+'test')
    print(" "*10,C[0],'\n'+" "*10,C[1])
    print(" "*10+'F1: ',F1_score_test)
    print(" "*10+'AUC: ',auc_test)
    print(" "*10+'precision: ',precision_test)
    print(" "*10+'recall: ',recall_test)
    print('\n')
    
    
    
    
    print('')
    print(" "*5+'Best Sore:')
    best_model_path = save_model_path+'/BestF1_' + str(fold) + '.pt' #此fold最优参数
    Model.load_state_dict(torch.load(best_model_path))
    
    y_true,y_pred,train_loss,train_acc = train_model(train_dataloader, Model, criterion, optimizer,DEVICE,onehot_lable=onehot_lable) # type: ignore # 模型
    Model.load_state_dict(torch.load(best_model_path))
    
    y_true,y_pred,y_out,validate_loss,validate_acc = eval_model(valid_dataloader,criterion,Model,DEVICE,onehot_lable=onehot_lable) # 验证模型
    F1_score_valid =f1_score(y_true, y_pred, average='binary')#F1分数
    auc_valid = roc_auc_score(y_true,(np.array(y_out))[:,1])
    C = confusion_matrix(y_true,y_pred)
    precision_valid = precision_score(y_true, y_pred, average='binary')
    recall_valid = recall_score(y_true, y_pred, average='binary') 
    save_test_infos(log_path+'/Validate_answer_'+str(fold)+'.csv',val_dataset, y_true,y_pred,y_out)
    print(" "*10+'valid')
    print(" "*10,C[0],'\n'+" "*10,C[1])
    print(" "*10+'F1: ',F1_score_valid,)
    print(" "*10+'AUC: ',auc_valid)
    print(" "*10+'precision: ',precision_valid)
    print(" "*10+'recall: ',recall_valid)
    
    y_true,y_pred,y_out,test_loss,test_acc = test_model(test_dataloader,criterion,Model,DEVICE,onehot_lable=onehot_lable) # 验证模型
    F1_score_test =f1_score(y_true, y_pred, average='binary')#F1分数
    auc_test = roc_auc_score(y_true,(np.array(y_out))[:,1])
    C = confusion_matrix(y_true,y_pred)
    precision_test = precision_score(y_true, y_pred, average='binary')
    recall_test = recall_score(y_true, y_pred, average='binary') 
    save_test_infos(log_path+'/Test_answer_'+str(fold)+'.csv',test_dataset, y_true,y_pred,y_out)
    print(" "*10+'test')
    print(" "*10,C[0],'\n'+" "*10,C[1])
    print(" "*10+'F1: ',F1_score_test)
    print(" "*10+'AUC: ',auc_test)
    print(" "*10+'precision: ',precision_test)
    print(" "*10+'recall: ',recall_test)
    return train_loss,train_acc,validate_loss,validate_acc,precision_valid,recall_valid,auc_valid,test_loss,test_acc,precision_test,recall_test,auc_test

def save_test_infos(csv_path,test_dataset:ECGHandle.ECG_Dataset,y_true:list,y_pred:list,y_out:list):
    ouputs_Df = pd.DataFrame(np.array(y_out),columns=["out0","out1"])
    preds_Df = pd.DataFrame(np.array(y_pred),columns=["pred"])
    target_Df = pd.DataFrame(np.array(y_true),columns=["target"])
    test_infors_Df = (test_dataset.infos.copy()).reset_index()
    test_infors_Df = pd.concat([test_infors_Df,target_Df,preds_Df,ouputs_Df],axis = 1)
    test_infors_Df.to_csv(csv_path)
    return 


def tarinning_one_flod_mutilabels(fold,Model,train_dataset,val_dataset,test_dataset,writer,save_model_path,BATCH_SIZE,DEVICE,
                        criterion = torch.nn.CrossEntropyLoss(),
                        EPOCHS = 100,  
                        PATIENCE = 10,
                        LR_MAX = 1e-2,
                        LR_MIN = 1e-5,
                        warm_up_iter = 5,
                        weight_decay=1e-3,
                        num_workers = 0,
                        shuffle = True,
                        onehot_lable = False
                        ):
    # target = train_dataset.labels
    # class_sample_count = np.array([len(np.where(target == t)[0]) for t in np.unique(target)])
    # weight = 1. / class_sample_count
    # samples_weight = np.array([weight[t] for t in target])
    # samples_weight = torch.from_numpy(samples_weight)
    # samples_weight = samples_weight.double()
    # sampler = Data.WeightedRandomSampler(samples_weight, len(samples_weight))  # type: ignore

    train_dataloader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=num_workers,pin_memory=True)
    valid_dataloader = Data.DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=num_workers,pin_memory=True)
    test_dataloader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=num_workers,pin_memory=True)
    early_stopping = EarlyStopping(PATIENCE, verbose=True, model_path=save_model_path, delta=0, positive=False)
    optimizer  = torch.optim.Adam(Model.parameters(), lr=LR_MAX,weight_decay=weight_decay) 
    criterion =  criterion.to(DEVICE)
    
    warm_up_iter = warm_up_iter
    T_max = EPOCHS	# 周期
    lr_max = LR_MAX	# 最大值
    lr_min = LR_MIN	# 最小值
    lambda0 = lambda cur_iter: lr_min if  cur_iter < warm_up_iter else \
        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    best_valida_loss = np.inf
    Model.to(DEVICE)
    for epoch in range(1,EPOCHS):
        time_all=0
        start_time = time.time()
        
        y_true,y_pred,train_loss,train_acc = train_model_mutilabels(train_dataloader, Model, criterion, optimizer,DEVICE,onehot_lable=onehot_lable) # type: ignore # 训练模型
        
        y_true,y_pred,validate_loss,validate_acc = eval_model_mutilabels(valid_dataloader,criterion,Model,DEVICE,onehot_lable=onehot_lable) # 验证模型
        time_all = time.time()-start_time
        # F1_score_valid =f1_score(y_true, y_pred, average='macro')#F1分数
        # C1 = confusion_matrix(y_true,y_pred)
        # print(" "*20+'Validate: ',F1_score_valid,'\n'+" "*20,C1[0],'\n'+" "*20,C1[1])
        
        writer.add_scalars(main_tag=str(fold)+'_Loss',tag_scalar_dict={'train': train_loss,'validate': validate_loss},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_Accuracy',tag_scalar_dict={'train': train_acc,'validate': validate_acc},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_LearningRate',tag_scalar_dict={'LR': optimizer.state_dict()['param_groups'][0]['lr']},global_step=epoch)      
        print(" "*20+'- Epoch: %d - Train_loss: %.5f - Train_acc: %.5f -  - Val_loss: %.5f - Val_acc: %.5f  - T_Time: %.5f' %(epoch,train_loss,train_acc,validate_loss,validate_acc,time_all),'LR：%.8f' %optimizer.state_dict()['param_groups'][0]['lr'])
        
        if(best_valida_loss>validate_loss):
            best_valida_loss = validate_loss
            F1_score_valid =f1_score(y_true, y_pred, average='macro')#F1分数
            C1 = multilabel_confusion_matrix(y_true,y_pred)
            print(" "*20+'Validate: ',F1_score_valid,'\n'+" "*20,C1)
            
        scheduler.step() # 学习率迭代
        
        #是否满足早停法条件
        if(early_stopping(validate_loss,Model,fold)):
            print(" "*20+"Early stopping...")
            break
        
    # 计算此flod 在testset上的效果
    best_model_path = save_model_path+'/parameter_EarlyStoping_' + str(fold) + '.pt' #此fold最优参数
    Model.load_state_dict(torch.load(best_model_path))
    
    y_true,y_pred,train_loss,train_acc = train_model_mutilabels(train_dataloader, Model, criterion, optimizer,DEVICE) # type: ignore # 模型
    Model.load_state_dict(torch.load(best_model_path))
    y_true,y_pred,validate_loss,validate_acc = eval_model_mutilabels(valid_dataloader,criterion,Model,DEVICE) # 验证模型
    F1_score_valid =f1_score(y_true, y_pred, average='macro')#F1分数
    C1 = multilabel_confusion_matrix(y_true,y_pred)
    print(" "*10+'validate: ',F1_score_valid,'\n'+" "*10,C1)
    
    y_true,y_pred,test_loss,test_acc = eval_model_mutilabels(test_dataloader,criterion,Model,DEVICE) # 验证模型
    F1_score_test =f1_score(y_true, y_pred, average='macro')#F1分数
    C = multilabel_confusion_matrix(y_true,y_pred)
    print(" "*10+'test: ',F1_score_test,'\n'+" "*10,C)
    print(" "*10+'Fold %d Training Finished' %(fold))
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
            _,target = labels.max(1)
        else:
            target = labels
        #print(pred)
        num_correct = (pred == target).sum().item()
        acc = num_correct/len(target) # 计算准确率
        train_loss.append(loss.item())
        train_acc.append(acc)
        # y_ture.extend((target.to('cpu').detach().numpy().flatten()).tolist())
        # y_pred.extend((pred.to('cpu').detach().numpy().flatten()).tolist())
    return y_ture,y_pred,np.mean(train_loss),np.mean(train_acc)

def eval_model(test_loader,criterion,model,device,onehot_lable=False):
    
    test_loss = []
    test_acc = []   
    y_ture = []
    y_pred = []
    y_output = [] 
    for i,data in enumerate(test_loader,0):
        model.eval()
        with torch.no_grad():
            inputs,labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            _,pred = outputs.max(1) # 求概率最大值对应的标签
            if(onehot_lable):
                _,target = labels.max(1)
            else:
                target = labels
            #print("pred:",pred)
            num_correct = (pred == target).sum().item()
            acc = num_correct/len(target)
            test_loss.append(loss.item())
            test_acc.append(acc)
            y_ture.extend((target.to('cpu').detach().numpy().flatten()).tolist())
            y_pred.extend((pred.to('cpu').detach().numpy().flatten()).tolist())
            y_output.extend((torch.nn.functional.softmax(outputs,dim=1).to('cpu').detach().numpy().tolist()))
    return y_ture,y_pred,y_output,np.mean(test_loss),np.mean(test_acc),

def test_model(test_loader,criterion,model,device,onehot_lable=False):
    
    test_loss = []
    test_acc = []   
    y_ture = []
    y_pred = []
    y_output = [] 
    for i,data in enumerate(test_loader,0):
        model.eval()
        with torch.no_grad():
            inputs,labels = data[0].to(device),data[1].to(device)
            outputs = model(inputs)
            loss = criterion(outputs,labels)
            _,pred = outputs.max(1) # 求概率最大值对应的标签
            if(onehot_lable):
                _,target = labels.max(1)
            else:
                target = labels
            #print("pred:",pred)
            num_correct = (pred == target).sum().item()
            acc = num_correct/len(target)
            test_loss.append(loss.item())
            test_acc.append(acc)
            y_ture.extend((target.to('cpu').detach().numpy()).tolist())
            y_pred.extend((pred.to('cpu').detach().numpy()).tolist())
            y_output.extend((torch.nn.functional.softmax(outputs,dim=1).to('cpu').detach().numpy().tolist()))
    return y_ture,y_pred,y_output,np.mean(test_loss),np.mean(test_acc)



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

# 定义多标签训练函数
def train_model_mutilabels(train_loader,model,criterion,optimizer,device,onehot_lable = False,threshold = (0.5)):
    threshold = torch.tensor(threshold).float()
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
        pred = outputs.ge(threshold).int() # 概率超过阈值 则判断为该类确证 置为1
        #print(pred)
        acc = (sum(row.all().int().item() for row in (pred == labels)))/(len(labels)*1.0) # 计算准确率(全对才算对)
        train_loss.append(loss.item())
        train_acc.append(acc)
        # y_ture.extend((target.to('cpu').detach().numpy().flatten()).tolist())
        # y_pred.extend((pred.to('cpu').detach().numpy().flatten()).tolist())
    return y_ture,y_pred,np.mean(train_loss),np.mean(train_acc)

def eval_model_mutilabels(test_loader,criterion,model,device,onehot_lable=False,threshold = (0.5)):
    threshold = torch.tensor(threshold).float()
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
            pred = outputs.ge(threshold).int() # 概率超过阈值 则判断为该类确证 置为1
            #print(pred)
            acc = (sum(row.all().int().item() for row in (pred == labels)))/(len(labels)*1.0) # 计算准确率(全对才算对)
            test_loss.append(loss.item())
            test_acc.append(acc)
            y_ture.extend((labels.to('cpu').detach().numpy()).tolist())
            y_pred.extend((pred.to('cpu').detach().numpy()).tolist())
    return y_ture,y_pred,np.mean(test_loss),np.mean(test_acc),




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
            print(" "*20+f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
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
            print(" "*20+f'Validation  score to ({self.last_best_score:.8f} --> {score:.8f}).  Saving model ...')
            print(" "*20+'-'*50+'\n')
        # torch.save(model, self.model_path+'/all_EarlyStoping_'+str(fold)+'.pt')                 # 这里会存储迄今最优的模型
        torch.save(model.state_dict(), self.model_path+'/parameter_EarlyStoping_' + str(fold) + '.pt')
        self.last_best_score = score