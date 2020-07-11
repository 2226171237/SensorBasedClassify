import torch
from torch import nn
from torch.nn import functional as F

class Config:
    def __init__(self):
        self.batch_size=64
        self.input_dim=8
        self.input_info_dim=48
        self.hidden_dim=100
        self.lr=1e-2
        self.epochs=300
        self.dropout=0.5
        self.layers=1
        self.bidirectional=False
        self.max_len=60

class BilinearPooling(nn.Module):
    '''双线性池化，做特征融合，使用低秩矩阵分解'''
    def __init__(self,x_dim,y_dim,deep):
        super(BilinearPooling, self).__init__()
        self.embed1=nn.Linear(x_dim,deep,bias=False)
        self.embed2=nn.Linear(y_dim,deep,bias=False)

    def forward(self, x,y):
        return F.relu(self.embed1(x))*F.relu(self.embed2(y))

class RNNClassifier(nn.Module):
    '''序列到分类器'''
    def __init__(self,options:Config):
        super(RNNClassifier,self).__init__()
        self.input_dim=options.input_dim
        self.hidden_dim=options.hidden_dim
        self.num_layer=options.layers
        self.bidirectional=options.bidirectional
        self.embed_layer=nn.Linear(self.input_dim,self.hidden_dim)

        self.embed_info_layer=nn.Linear(options.input_info_dim,self.hidden_dim)

        self.lstm=nn.LSTM(self.hidden_dim,self.hidden_dim,batch_first=True,
                          dropout=options.dropout,bidirectional=self.bidirectional,num_layers=self.num_layer)
        self.dropout=nn.Dropout(options.dropout)

        # self.bilinear=BilinearPooling(self.hidden_dim* (2 if self.bidirectional else 1),self.hidden_dim,self.hidden_dim)

        self.logits=nn.Linear(self.hidden_dim* (2 if self.bidirectional else 1) ,19)

    def forward(self, x,lengths):
        '''
        :param x: [batch,T,input_dim]
        :param x_info: [batch,48] 统计信息
        :param lengths : [batch]
        :return:
        '''
        batchsize=x.shape[0]
        lengths=torch.tensor(lengths).long().cuda()
        idx_sort=torch.argsort(lengths,descending=True)
        idx_unsort=torch.argsort(idx_sort)
        x=torch.index_select(x,index=idx_sort,dim=0)
        x=F.relu(self.embed_layer(x)) # [batch,T,hidden_dim]
        x=torch.nn.utils.rnn.pack_padded_sequence(x,lengths[idx_sort],batch_first=True)
        output,_=self.lstm(x)  # [batch,T,num_directions*hidden_size]
        output=torch.nn.utils.rnn.pad_packed_sequence(output,batch_first=True)
        batch_idx=torch.range(0,batchsize-1).long()
        feats=torch.index_select(output[0],index=idx_unsort,dim=0)[batch_idx,lengths-1,:]
        logits=self.logits(feats)
        return logits,feats

class CNNClassifier(nn.Module):
    '''全CNN网络'''
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1=nn.Conv2d(1,64,kernel_size=(3,3),padding=(1,1))
        self.conv2=nn.Conv2d(64,128,kernel_size=(3,3),padding=(1,1))
        self.maxpooling=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv3=nn.Conv2d(128,256,kernel_size=(3,3),padding=(1,1))
        self.conv4=nn.Conv2d(256,512,kernel_size=(3,3),padding=(1,1))
        self.globalpooling=nn.AdaptiveMaxPool2d(output_size=(1,1))
        self.dropout=nn.Dropout(0.3)
        self.fc1=nn.Linear(512,19)

        self.bn1=nn.BatchNorm2d(64)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(256)
        self.bn4=nn.BatchNorm2d(512)

    def forward(self,x,*args):
        x=torch.unsqueeze(x,dim=1)
        x=F.relu(self.bn1(self.conv1(x)))
        x=F.relu(self.bn2(self.conv2(x)))
        x=self.maxpooling(x)
        x=F.relu(self.bn3(self.conv3(x)))
        x=F.relu(self.bn4(self.conv4(x)))
        x=self.globalpooling(x)
        batch=x.shape[0]
        feats=x.view((batch,-1))
        x=self.dropout(feats)
        logits=self.fc1(x)
        return logits,feats

class Conv1dInOneSigle(nn.Module):
    '''单条波形的一维(时间维)卷积处理'''
    def __init__(self):
        super(Conv1dInOneSigle, self).__init__()
        self.conv1d_list = nn.ModuleList()
        for _ in range(8):
            self.conv1d_list.append(nn.Sequential(
                nn.Conv1d(1,64,kernel_size=3,padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Conv1d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                )
            ) # 输出大小一样

    def forward(self,x):
        x=torch.transpose(x,1,2)
        out=[]
        for i in range(8):
            input=x[:,i,:].unsqueeze(dim=1)
            out.append(self.conv1d_list[i](input).unsqueeze(dim=1)) # [N,1,Cout,Lout]
        out=torch.cat(out,dim=1)
        return out #[N,8,64,Lout]

class CNNRNNClassifier(nn.Module):
    '''CNN+RNN+分类器'''
    def __init__(self):
        super(CNNRNNClassifier, self).__init__()
        self.conv1=Conv1dInOneSigle()
        self.conv2 = nn.Conv2d(8, 128, kernel_size=(64, 3),padding=(0, 1)) # 直接时间维移动
        # self.conv2=nn.Conv2d(8,128,kernel_size=(3,3),stride=(2,1),padding=(0,1))
        # self.conv3 = nn.Conv2d(128, 256, kernel_size=(31, 3), padding=(0, 1))
        self.bn2=nn.BatchNorm2d(128)
        #self.bn3 = nn.BatchNorm2d(256)

        self.lstm=nn.LSTM(128,256,dropout=0.5,batch_first=True)
        self.logits=nn.Linear(256,19)
        self.dropout=nn.Dropout(0.5)

    def forward(self,x,lengths):
        lengths = torch.tensor(lengths).long().cuda()
        x=F.relu(self.conv1(x))
        x=F.relu(self.bn2(self.conv2(x)))  # [N,128,Lout]
        # x=F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))

        N,Fd,_,Lout=x.shape
        x=x.view(N,Fd,-1)
        x=torch.transpose(x,1,2)
        batchsize = x.shape[0]
        idx_sort = torch.argsort(lengths, descending=True)
        idx_unsort = torch.argsort(idx_sort)
        x = torch.index_select(x, index=idx_sort, dim=0)
        x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths[idx_sort], batch_first=True)
        output, _ = self.lstm(x)  # [batch,T,num_directions*hidden_size]
        output = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        batch_idx = torch.range(0, batchsize - 1).long()
        feats = torch.index_select(output[0], index=idx_unsort, dim=0)[batch_idx, lengths - 1, :]
        logits = self.logits(self.dropout(feats))
        return logits, feats


if __name__ == '__main__':
    opt=Config()
    model=CNNClassifier()
    x=torch.rand(32,60,8)
    x=model(x)
    print(x.shape)








