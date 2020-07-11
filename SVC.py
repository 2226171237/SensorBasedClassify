from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pandas as pd

import pickle
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from utils import score

with open('./data/sensor_train/sensor_train.pkl', 'rb') as f:
    data = pickle.load(f)
labels=data[1]

feats=np.load('./data/sensor_train/sensor_train_cnn_feat.npy')

# param_gird={'C':[0.1,1,5,10,15,20],'kernel':['linear','rbf']}
#
# kflod=KFold(n_splits=5,shuffle=True,random_state=42)
# filter=GridSearchCV(SVC(),param_grid=param_gird,cv=kflod,verbose=True,n_jobs=8)
# # filter=SVC(C=0.1,kernel='linear')
# filter.fit(feats,labels)
# print(filter.best_params_,filter.best_score_)
# pred_y=filter.predict(feats)
# print('score:',score(labels,pred_y))

filter=SVC(C=0.1,kernel='linear')
x_train,x_test,y_train,y_test=train_test_split(feats,labels,test_size=0.8)
filter.fit(x_train,y_train)
print('train acc:',filter.score(x_train,y_train))
pred_y=filter.predict(x_train)
print('train score:',score(y_train,pred_y))
pred_y=filter.predict(x_test)
print('test acc:',filter.score(x_test,y_test))
print('test score:',score(y_test,pred_y))

with open('./data/sensor_test/sensor_test.pkl','rb') as f:
    test_data=pickle.load(f)
fragment_id=test_data[-1]
test_feats=np.load('./data/sensor_test/sensor_test_cnn_feat.npy')
pred_y=filter.predict(test_feats)

fragment=pd.Series(fragment_id,dtype=int)
pred=pd.Series(pred_y,dtype=int)
result=pd.DataFrame({'fragment_id':fragment,'behavior_id':pred_y})
result.to_csv('./data/sensor_test/test_svm_result.csv',index=False)


# 尝试t-SNE降维可视化分析
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib import cm

km=KMeans(n_clusters=19).fit(test_feats)
y_test=km.predict(test_feats)
X_train_embed=TSNE(n_components=2).fit_transform(feats)
X_test_embed=TSNE(n_components=2).fit_transform(test_feats)

fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(X_train_embed[:,0],X_train_embed[:,1],s=5,c=labels,alpha=0.5)
plt.savefig('tsne_train.png')
fig=plt.figure()
ax=fig.add_subplot(111)
ax.scatter(X_test_embed[:,0],X_test_embed[:,1],s=5,alpha=0.5,c=y_test)
plt.savefig('tsne_test.png')



