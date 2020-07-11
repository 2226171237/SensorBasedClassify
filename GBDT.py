from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from utils import score

with open('./data/sensor_train/sensor_train.pkl', 'rb') as f:
    data = pickle.load(f)

# 训练集特征和标签
labels=data[1]
feats=np.load('./data/sensor_train/sensor_train_cnn_feat.npy')

# 网格参数搜索 使用交叉验证
param_gird={'n_estimators':[100]}

kflod=KFold(n_splits=3,shuffle=True,random_state=42)
filter=GridSearchCV(GradientBoostingClassifier(random_state=42),param_grid=param_gird,cv=kflod,verbose=True,n_jobs=4)
# filter=SVC(C=0.1,kernel='linear')
filter.fit(feats,labels)
pred_y=filter.predict(feats)
print('score:',score(labels,pred_y),'best_param:',filter.best_params_)

# 测试集预测
with open('./data/sensor_test/sensor_test.pkl','rb') as f:
    test_data=pickle.load(f)
fragment_id=test_data[-1]
test_feats=np.load('./data/sensor_test/sensor_test_cnn_feat.npy')
pred_y=filter.predict(test_feats)

# 结果保存
fragment=pd.Series(fragment_id,dtype=int)
pred=pd.Series(pred_y,dtype=int)
result=pd.DataFrame({'fragment_id':fragment,'behavior_id':pred_y})
result.to_csv('./data/sensor_test/test_gbdt_result.csv',index=False)



