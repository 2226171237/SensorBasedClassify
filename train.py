from utils import *
import numpy as np
import torch
from torch import nn
from torch import optim
from models import RNNClassifier,CNNClassifier,CNNRNNClassifier
from models import Config
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import argparse
import warnings
from sklearn.svm import SVC
warnings.filterwarnings('ignore')

class Trainer:
    def __init__(self,data,data_info,arg):
        self.arg=arg
        self.cfg = Config()
        train_data,train_info,val_data,val_info=split_train_test(data,data_info,test_size=0.2)
        train_data=SensorData(train_data,train_info,self.cfg.max_len)
        val_data=SensorData(val_data,val_info,self.cfg.max_len)
        dataloader=SensorData(data,data_info,self.cfg.max_len)
        self.train_loader=DataLoader(train_data,batch_size=self.cfg.batch_size,shuffle=True,drop_last=False)
        self.val_loader=DataLoader(val_data,batch_size=self.cfg.batch_size,shuffle=False,drop_last=False)
        self.dataloader=DataLoader(dataloader,batch_size=self.cfg.batch_size,shuffle=False,drop_last=False)
        if arg.model == 'rnn':
            self.model=RNNClassifier(self.cfg).cuda()
        elif arg.model=='cnn':
            self.model=CNNClassifier().cuda()
        elif arg.model=='cnn_rnn':
            self.model=CNNRNNClassifier().cuda()
        else:
            raise ValueError

        # self.optimizer=optim.Adam(self.model.parameters(),lr=self.cfg.lr)
        self.optimizer=optim.SGD(self.model.parameters(),lr=self.cfg.lr,momentum=0.9,nesterov=True)
        self.lr_sche=optim.lr_scheduler.StepLR(self.optimizer,step_size=10,gamma=0.8)
        self.loss_fn=nn.CrossEntropyLoss()

    def train(self):
        train_loss=[]
        val_loss=[]
        train_acc=[]
        test_acc=[]
        for epoch in tqdm(range(self.cfg.epochs)):
            self.model.train()
            loss=[]
            train_total_count,train_right_score=0,0
            for batch_x,batch_x_info,batch_y,batch_len in self.train_loader:
                batch_x=torch.tensor(batch_x).float().cuda()
                batch_y=torch.tensor(batch_y).long().cuda()
                batch_len=torch.tensor(batch_len).long().cuda()
                logits,_=self.model(batch_x,batch_len)
                trainloss=self.loss_fn(logits,batch_y)
                self.optimizer.zero_grad()
                trainloss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    pred_y=logits.argmax(dim=-1)
                train_score=score(batch_y.cpu().numpy(),pred_y.long().cpu().numpy())
                train_right_score+=train_score*batch_x.shape[0]
                train_total_count+=batch_x.shape[0]
                loss.append(trainloss.item())
            self.lr_sche.step()
            t_acc=train_right_score/train_total_count
            train_acc.append(t_acc)
            avg_train_loss=np.mean(loss)
            train_loss.append(avg_train_loss)
            # test
            loss=[]
            total_count,test_right_score=0,0
            with torch.no_grad():
                self.model.eval()
                for batch_x,batch_x_info,batch_y,batch_len in self.val_loader:
                    batch_x=torch.tensor(batch_x).float().cuda()
                    batch_y=torch.tensor(batch_y).long().cuda()
                    batch_len=torch.tensor(batch_len).long().cuda()
                    logits,_=self.model(batch_x,batch_len)
                    test_score=score(logits.argmax(dim=1).long().cpu().numpy(),batch_y.cpu().numpy())
                    test_right_score+=test_score*batch_x.shape[0]
                    total_count+=batch_x.shape[0]
                    valloss=self.loss_fn(logits,batch_y)
                    loss.append(valloss.item())
            acc=test_right_score/total_count
            test_acc.append(acc)
            avg_val_loss=np.mean(loss)
            val_loss.append(avg_val_loss)
            drawCurve(train_loss,val_loss,'./loss/loss.png',labels=['train loss','val loss'])
            drawCurve(train_acc,test_acc,'./loss/acc.png',labels=['train acc','val acc'])
            print('>Epoch: %d/%d train loss:%f  val loss:%f train acc=%f  acc=%f' %
                  (epoch+1,self.cfg.epochs,avg_train_loss,avg_val_loss,t_acc,acc))
        torch.save(self.model.state_dict(),'./save/%s_model.pth' % self.arg.model)
        save_path = './data/sensor_train/sensor_train_%s_feat.npy' % self.arg.model
        extract_feat(self.dataloader,self.model,save_path)


def main(arg):
    with open(os.path.join(arg.train_data_path,'sensor_train.pkl'),'rb') as f:
         data=pickle.load(f)
    data_info=np.load(os.path.join(arg.train_data_path,'sensor_train_info.npy'))
    data_info=(data_info-data_info.mean(axis=0,keepdims=True))/data_info.var(axis=0,keepdims=True)
    trainer=Trainer(data,data_info,arg)
    trainer.train()
    # trainer.model.load_state_dict(torch.load('./save/%s_model.pth' % arg.model))
    with open(os.path.join(arg.test_data_path,'sensor_test.pkl'),'rb') as f:
        testdata=pickle.load(f)
    test_data_info=np.load(os.path.join(arg.test_data_path,'sensor_test_info.npy'))
    dataloader = SensorData(testdata, test_data_info, trainer.cfg.max_len)
    dataloader = DataLoader(dataloader, batch_size=64, shuffle=False, drop_last=False)
    save_path='./data/sensor_test/sensor_test_%s_feat.npy' % arg.model
    extract_feat(dataloader,trainer.model,save_path)

if __name__ == '__main__':
    arg=argparse.ArgumentParser()
    arg.add_argument('--train_data_path',default='./data/sensor_train/')
    arg.add_argument('--test_data_path', default='./data/sensor_test/')
    arg.add_argument('--model', default='cnn_rnn')
    arg=arg.parse_args()
    main(arg)




