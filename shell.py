import select_dataset
import Models 
import Net
from train_test_validat import *
from self_attention import *
import ecg_get_data 
import math

import torch
import torch.utils.data as Data
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import random
from torch.utils.tensorboard import SummaryWriter # type: ignore
import torch.nn.functional as F
import time
import os

# random_seed = 2
# torch.manual_seed(random_seed)    # reproducible
# torch.cuda.manual_seed_all(random_seed)
# random.seed(random_seed)
# np.random.seed(random_seed)
def linear_combination(x, y, epsilon): 
    return epsilon*x + (1-epsilon)*y
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon:float=0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
    
    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)#就是交叉熵loss
        return linear_combination(loss/n, nll, self.epsilon)


def train_fold(fold,NET,test_Dataset,valid_Dataset,train_Dataset):
    torch.cuda.empty_cache()
    #每个人fold都重新抽取
    NET[fold].to(DEVICE)
    train_dataloader = Data.DataLoader(dataset=train_Dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=16,pin_memory=True)
    valid_dataloader = Data.DataLoader(dataset=valid_Dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=16,pin_memory=True)
    test_dataloader = Data.DataLoader(dataset=test_Dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=16,pin_memory=True)
    early_stopping = EarlyStopping(PATIENCE, verbose=True, model_path=model_path, delta=0, positive=False)
    optimizer  = torch.optim.Adam(NET[fold].parameters(), lr=LR,weight_decay=1e-2)  
    warm_up_iter = 10
    T_max = 500	# 周期
    lr_max = 1e-3	# 最大值
    lr_min = 1e-5	# 最小值
    lambda0 = lambda cur_iter: lr_min if  cur_iter < warm_up_iter else \
        (lr_min + 0.5*(lr_max-lr_min)*(1.0+math.cos( (cur_iter-warm_up_iter)/(T_max-warm_up_iter)*math.pi)))/0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda0)
    criterion = torch.nn.CrossEntropyLoss()   
    # criterion = LabelSmoothingCrossEntropy()
    best_test_F1 = 0
    for epoch in range(1,EPOCHS):
        time_all=0
        start_time = time.time()
        y_true,y_pred,train_loss,train_acc = train_model(train_dataloader, NET[fold], criterion, optimizer,DEVICE) # type: ignore # 训练模型
        time_all = time.time()-start_time
        F1_score_train =f1_score(y_true, y_pred, average='macro')#F1分数
        C0 = confusion_matrix(y_true,y_pred)
        y_true,y_pred,validate_loss,validate_acc = eval_model(valid_dataloader,criterion,NET[fold],DEVICE) # 验证模型
        F1_score_valid =f1_score(y_true, y_pred, average='macro')#F1分数
        C1 = confusion_matrix(y_true,y_pred)
        y_true,y_pred,test_loss,test_acc = eval_model(test_dataloader,criterion,NET[fold],DEVICE) # 验证模型
        F1_score_test =f1_score(y_true, y_pred, average='macro')#F1分数
        C2 = confusion_matrix(y_true,y_pred)

        # writer.add_scalars(main_tag=str(fold)+'_Loss',tag_scalar_dict={'train': train_loss,'validate': validate_loss},global_step=epoch)
        # writer.add_scalars(main_tag=str(fold)+'_Accuracy',tag_scalar_dict={'train': train_acc,'validate': validate_acc},global_step=epoch)
        # writer.add_scalars(main_tag=str(fold)+'_LearningRate',tag_scalar_dict={'LR': optimizer.state_dict()['param_groups'][0]['lr']},global_step=epoch)
        # writer.add_scalars(main_tag=str(fold)+'_F1_score',tag_scalar_dict={'train':F1_score_train,'validate': F1_score_valid},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_Loss',tag_scalar_dict={'train': train_loss,'validate': validate_loss,'test':test_loss},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_Accuracy',tag_scalar_dict={'train': train_acc,'validate': validate_acc,'test':test_acc},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_LearningRate',tag_scalar_dict={'LR': optimizer.state_dict()['param_groups'][0]['lr']},global_step=epoch)
        writer.add_scalars(main_tag=str(fold)+'_F1_score',tag_scalar_dict={'train':F1_score_train,'validate': F1_score_valid,'test':F1_score_test},global_step=epoch)        
        print('- Epoch: %d - Train_loss: %.5f - Train_acc: %.5f - F1 score: %.5f - Val_loss: %.5f - Val_acc: %.5f - F1 score: %.5f - T_Time: %.5f' %(epoch,train_loss,train_acc,F1_score_train,validate_loss,validate_acc,F1_score_valid,time_all))
        print('当前学习率：%.8f' %optimizer.state_dict()['param_groups'][0]['lr'])
        print('train:\n',C0)
        print('validate:\n',C1)
        print('test:\n',C2)
        
        if(F1_score_test>best_test_F1):
            best_test_F1 = F1_score_test
            torch.save(NET[fold].state_dict(), model_path+'/parameter_best_test_' + str(fold) + '.pt')
        
        scheduler.step() # 学习率迭代
        #是否满足早停法条件
        if(early_stopping(validate_loss,NET[fold],fold)):
            print("Early stopping")
            break
        if(epoch>T_max):
            break
    
    


EcgChannles_num = 12
EcgLength_num = 5000
BATCH_SIZE = 128
EPOCHS = 5000  
PATIENCE = 150
LR = 0.01
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)
time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
model_path = './model/'+time_str
log_path = './log/'+  time_str
ECG_root = '/workspace/data/Preprocess_HTN/data/ECG'
os.makedirs(model_path, exist_ok=True)
writer = SummaryWriter(log_path)

def main(): 
    FOLDS = 4 
    torch.cuda.empty_cache()# 清空显卡cuda
    NET = [Net.MLBFNet(True,res = True,se = True,Dropout_rate = 0.1),Net.MLBFNet(True,res = True,se = True,Dropout_rate = 0.2),Net.MLBFNet(True,res = True,se = True,Dropout_rate = 0.3),Net.MLBFNet(True,res = True,se = True,Dropout_rate = 0.5) ] # type: ignore
    data = select_dataset.splite_dataset('/workspace/data/Preprocess_HTN/data/',True)
    for fold in range(FOLDS):
        test_list = data.__get_test_file_list__(True)
        test_Dataset = ecg_get_data.ECG_Dataset(ECG_root,test_list,EcgChannles_num,EcgLength_num)  # type: ignore
        valid_list,train_list,addition_train_list = data.__get_VT_file_list__(0.9,True)
        valid_Dataset = ecg_get_data.ECG_Dataset(ECG_root,valid_list,EcgChannles_num,EcgLength_num)
        train_Dataset = ecg_get_data.ECG_Dataset(ECG_root,train_list,EcgChannles_num,EcgLength_num)
    
        torch.cuda.empty_cache()# 清空显卡cuda
        train_fold(fold,NET,test_Dataset,valid_Dataset,train_Dataset)
        print('Fold %d Training Finished' %(fold+1))
        torch.cuda.empty_cache()# 清空显卡cuda
    print('Training Finished')
    
    

if __name__ == '__main__':
    main()