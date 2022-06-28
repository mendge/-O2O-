import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import log_loss, roc_auc_score, auc,roc_curve
from  sklearn.model_selection import train_test_split
# 使用GridSearchCV进行参数搜索
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import GridSearchCV
# 绘制特征得分图
import matplotlib.pyplot as plt
from xgboost import plot_importance

#特征提取后数据存放路径
afterPath = r'C:\Users\TTTime\Desktop\创新实践2\code\venv\code\data\after'

dataset1 = pd.read_csv(afterPath+r'\df1.csv',index_col=0)
dataset2 = pd.read_csv(afterPath+r'\df2.csv',index_col=0)
dataset3 = pd.read_csv(afterPath+r'\df3.csv',index_col=0)

# 其实并无重复项需要去除
# print(dataset1.shape)
# dataset1.drop_duplicates(inplace=True)
# print(dataset1.shape)
# print(dataset2.shape)
# dataset2.drop_duplicates(inplace=True)
# print(dataset2.shape)
# print(dataset3.shape)
# dataset3.drop_duplicates(inplace=True)
# print(dataset3.shape)

dataset3_preds = dataset3[['User_id','Coupon_id','Date_received']].copy()
dataset3_x = dataset3.drop(['User_id','Date_received','Coupon_id'],axis=1)


dataset_12 = pd.concat([dataset1,dataset2],axis=0)
dataset_12_y = dataset_12.Label
dataset_12_x = dataset_12.drop(['Label'],axis=1)

dataTrain = xgb.DMatrix(dataset_12_x, label=dataset_12_y)
dataTest = xgb.DMatrix(dataset3_x)
print('---data prepare over---')

def myauc(test):
    testgroup = test.groupby(['Coupon_id'])
    aucs = []
    for i in testgroup:
        # i为tuple，i[0]为分组的标签，i[1]为每组中的元素
        tmpdf = i[1]
        if i in testgroup:
            # label必须有两类，如果只有一类，roc曲线中有一个数分母为0(tpr或fpr)
            if len(tempdf['label'].unique()) != 2:
                continue
            # y_ture是实际提取出来的label，y_score是预测出的label
            fpr, tpr, thresholds = roc_curve(y_true=tmpdf['label'],y_score=tempdf['pred'],pos_label=1)
            aucs.append(auc(x=fpr,y=tpr))
    return np.average(aucs)

params = {'booster': 'gbtree',
          'objective': 'rank:pairwise',
          'eval_metric': 'auc',
          'gamma': 0.1,
          'min_child_weight': 1.1,
          'max_depth': 5,
          'lambda': 10,
          'subsample': 0.7,
          'colsample_bytree': 0.7,
          'colsample_bylevel': 0.7,
          'eta': 0.01,
          'tree_method': 'exact',
          'seed': 0,
          'nthread': 12
          }

# 直接预测
# watchlist = [(dataTrain, 'train')]
# print('start trainning')
# model = xgb.train(params,dataTrain,num_boost_round=5867, evals=watchlist)
# print('train over')
# model.save_model(afterPath+r'\model_5867')
#
model = xgb.Booster(params=params,model_file=afterPath+r'\exploremodel')

dataset3_preds1 = dataset3_preds
print('start predict')
dataset3_preds1['Label'] = model.predict(dataTest)
print('predict over')
print(type(dataset3_preds1.Label))
dataset3_preds1['Label'] = MinMaxScaler(copy=True, feature_range=(0,1)).fit_transform(dataset3_preds1['Label'].values.reshape(-1, 1))
dataset3_preds1.sort_values(by=['Coupon_id','Label'],inplace=True)
dataset3_preds1.to_csv(afterPath+r'\submission8250.csv',index=None,header=None)
print('write over')
# print(dataset3_preds1.describe())






# 使用GridSearchCV进行参数搜索
param_test = {'max_depth':range(3,10,2), 'min_child_weight':range(1,6,2)}
gscv = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1,
                                            n_estimators=5,
                                            max_depth=5,
                                            min_child_weight=1,
                                            gamma=0,
                                            subsample=0.8,
                                            colsample_bytree=0.8,
                                            objective='binary:logistic',
                                            scale_pos_weight=1,
                                            seed=0),
                    param_grid=param_test,
                    scoring='roc_auc',
                    iid=False,
                    cv=3)
# gscv.fit:根据模型和输入数据，拟合参数
gscv = gscv.fit(dataset_12_x,dataset_12_y)
# 显示参数
gscv_means = gscv.cv_results_['mean_test_score']
gscv_params = gscv.cv_results_['params']
# zip:将两个列表对应位置拟合成一个列表，返回列表
for param,mean in zip(gscv_params,gscv_means):
    print('%s with: %s'%(param, mean))
print('The best params is: %s , its auc is: %s' %(gscv.best_params_,gscv.best_score_))
# 运行结果：





# 最大迭代次数调优：使用xgb.cv
cvresult = xgb.cv(params=params,dtrain=dataTrain,num_boost_round=30000,nfold=4,metrics='auc',seed=0,callbacks=[
        xgb.callback.print_evaluation(show_stdv=False),
        xgb.callback.early_stop(50)
    ])
# cvresult的行数为从头的最优迭代次数（可能训练到了11739次，但是返回的是最优的11689行，中间正好差50）
num_round_best = cvresult.shape[0] - 1
print('Best round num: ', num_round_best)

watchlist = [(dataTrain,'train')]
exploremodel = xgb.train(params,dataTrain,num_boost_round=num_round_best,evals=watchlist)
exploremodel.save_model(afterPath+r'\exploremodel')
print('---train explore model done---')


# xgb特征筛选功能
exploremodel = xgb.Booster()
exploremodel.load_model(afterPath+r'exploremodel')
# exploremodel.get_score():
# weight - 该特征在所有树中被用作分割样本的特征的次数
# gain - 在所有树中的平均增益
# cover - 在树中使用该特征时的平均覆盖范围
featurescore = exploremodel.get_score()
featurescore = sorted(featurescore.items(),key=lambda x: x[1], reverse=False) # 升序

# 将特征评分写入文件
fs = []
for (key, value) in featurescore:
    fs.append('{0},{1}\n'.format(key, value))
with open(afterPath+r'\fetures_score', 'w') as f:
    f.write(fs)

df = pd.DataFrame(featurescore,columns=['Feature','Score'])
df['Score'] = df['Score'] / df['Score'].sum()

plt.figure()
df.plot(kind='barh',x='Feature',y='Score',legend=False,figsize=(6,10))
plt.title('XGBoost Feature Importance')
plt.xlabel('relative importance')
plt.show()
#
# # plot_importance:xgboost内置绘制特征函数
# plt.figure()
# plot_importance(exploremodel)
# plt.show()














