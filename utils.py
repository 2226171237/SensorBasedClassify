import  numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import os
from tqdm import tqdm



def smooth_line(x,beta=0.5):
    '''波形滑动平均'''
    # x :[T,feat_len]
    smooth_x=np.zeros_like(x)
    T,feat_dim=x.shape
    smooth_x[0,:]=x[0,:]
    for t in range(1,T):
        smooth_x[t,:]=(smooth_x[t-1,:]*beta+(1-beta)*x[t,:])/(1-beta**t)
    return smooth_x

def extract_feat(dataloader,model,save_path):
    '''提取模型特征'''
    features=[]
    model.eval()
    for batch_x, batch_x_info, _, batch_len in dataloader:
        batch_x = torch.tensor(batch_x).float().cuda()
        batch_len = torch.tensor(batch_len).long().cuda()
        _, feat = model(batch_x, batch_len)
        feat=feat.data.cpu().numpy()
        feat=np.concatenate((feat,batch_x_info.cpu().numpy()),axis=-1)
        features.append(feat)
    features=np.concatenate(features,axis=0)
    np.save(save_path,features)


def getInfo(trainfile,test_file):
    '''获取数据统计信息'''
    train_data=pd.read_csv(trainfile)
    test_data=pd.read_csv(test_file)
    test_data['fragment_id'] += 10000
    label = 'behavior_id'
    data=pd.concat([train_data,test_data],axis=0,sort=False)
    df = data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]
    data['acc'] = (data['acc_x'] ** 2 + data['acc_y'] ** 2 + data['acc_z'] ** 2) ** 0.5
    data['accg'] = (data['acc_xg'] ** 2 + data['acc_yg'] ** 2 + data['acc_zg'] ** 2) ** 0.5
    for f in tqdm([f for f in data.columns if 'acc' in f]):
        for stat in ['min', 'max', 'mean', 'median', 'std', 'skew']:
            df[f + '_' + stat] = data.groupby('fragment_id')[f].agg(stat).values
    train_df = df[df[label].isna() == False].reset_index(drop=True)
    test_df = df[df[label].isna() == True].reset_index(drop=True)

    drop_feat = []
    used_feat = [f for f in train_df.columns if f not in (['fragment_id', label] + drop_feat)]
    print(len(used_feat))
    print(used_feat)
    # 48个统计特征
    train_x = train_df[used_feat].values
    test_x = test_df[used_feat].values
    np.save('./data/sensor_train/sensor_train_info.npy',train_x)
    np.save('./data/sensor_test/sensor_test_info.npy', test_x)

def getTrainOriginData(filename):
    '''获取原始数据'''
    resdata=[]
    label=[]
    time_point=[]
    segmend_id=[]
    dataset=pd.read_csv(filename)
    dataset['acc']=(dataset['acc_x'] ** 2 + dataset['acc_y'] ** 2 + dataset['acc_z'] ** 2) ** 0.5
    dataset['accg'] = (dataset['acc_xg'] ** 2 + dataset['acc_yg'] ** 2 + dataset['acc_zg'] ** 2) ** 0.5

    for b_id,data in dataset.groupby('behavior_id'):
        for seg_id,d in data.groupby('fragment_id'):
            d.sort_values('time_point',inplace=True)
            time_point.append(d.time_point.values)
            segmend_id.append(seg_id)
            label.append(b_id)
            d=d.drop(['fragment_id','time_point','behavior_id'],axis=1)
            values=d.values
            # values=(values-values.mean(axis=0,keepdims=True))/values.var(axis=0,keepdims=True)
            resdata.append(values)
    return resdata,label,time_point,segmend_id

def getTestOriginData(filename):
    '''获取原始数据'''
    resdata=[]
    time_point=[]
    segmend_id=[]
    dataset=pd.read_csv(filename)
    dataset['acc']=(dataset['acc_x'] ** 2 + dataset['acc_y'] ** 2 + dataset['acc_z'] ** 2) ** 0.5
    dataset['accg'] = (dataset['acc_xg'] ** 2 + dataset['acc_yg'] ** 2 + dataset['acc_zg'] ** 2) ** 0.5

    for seg_id,d in dataset.groupby('fragment_id'):
        d.sort_values('time_point',inplace=True)
        time_point.append(d.time_point.values)
        segmend_id.append(seg_id)
        d=d.drop(['fragment_id','time_point'],axis=1)
        values=d.values
        resdata.append(values)
    return resdata,None,time_point,segmend_id

def split_train_test(data,data_info,test_size,random_state=42):
    '''数据切分'''
    lens=len(data[1])
    index=np.arange(lens)
    np.random.seed(random_state)
    np.random.shuffle(index)
    test_begin=int(lens*(1-test_size))
    train_data=[[],[],[],[]]
    train_info=[]

    test_data = [[], [], [], []]
    test_info=[]

    for id in index[:test_begin]:
        train_data[0].append(data[0][id])
        train_data[1].append(data[1][id])
        train_data[2].append(data[2][id])
        train_data[3].append(data[3][id])
        train_info.append(data_info[id])
    for id in index[test_begin:]:
        test_data[0].append(data[0][id])
        test_data[1].append(data[1][id])
        test_data[2].append(data[2][id])
        test_data[3].append(data[3][id])
        test_info.append(data_info[id])
    return train_data,np.array(train_info),test_data,np.array(test_info)


def drawCurve(train_loss,val_loss,save_path,labels):
    '''绘制学习曲线'''
    fig=plt.figure()
    plt.plot(train_loss,label=labels[0])
    plt.plot(val_loss,label=labels[1])
    plt.xlabel('epochs')
    plt.legend()
    plt.savefig(save_path)


class SensorData(Dataset):
    '''数据加载'''
    def __init__(self,data,data_info,max_len,transform=None):
        self.data=data[0]
        self.data_info=data_info
        self.labels=data[1]
        self.times=data[2]
        self.max_len=max_len
        self.transform=transform

    def __getitem__(self, item):
        t=self.data[item]
        x_info=self.data_info[item]
        if self.transform:
           t=self.transform(t)

        x=np.zeros((self.max_len,t.shape[-1]))
        length=len(self.times[item])
        x[:min(length,self.max_len),:]=t[:min(length,self.max_len),:]
        # x=smooth_line(x)
        label= self.labels[item] if self.labels else None
        label=1 if label==None else label
        res=[x, x_info, label, min(length, self.max_len)]
        return res

    def __len__(self):
        return len(self.times)

def acc_combo(y, y_pred):
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0

def score(y,y_pred):
    '''batch 打分，官方评测指标'''
    total_score=0.
    for y_i,y_pred_i in zip(y,y_pred):
        total_score+=acc_combo(y_i,y_pred_i)
    return total_score/len(y)

if __name__ == '__main__':
    # getInfo('./data/sensor_train/sensor_train.csv','./data/sensor_test/sensor_test.csv')
    data=getTrainOriginData('./data/sensor_train/sensor_train.csv')
    with open('./data/sensor_train/sensor_train.pkl','wb') as f:
        pickle.dump(data,f)
    data=getTestOriginData('./data/sensor_test/sensor_test.csv')
    with open('./data/sensor_test/sensor_test.pkl','wb') as f:
        pickle.dump(data,f)



