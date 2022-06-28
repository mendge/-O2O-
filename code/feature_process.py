import pandas as pd
import numpy as np
import datetime as dt
from datetime import date
from sklearn.preprocessing import MinMaxScaler
# 设置df.head()显示所有行、所有列
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

#源数据路径
beforePath = r'C:\Users\TTTime\Desktop\创新实践2\code\venv\code\data\before'
#特征提取后数据存放路径
afterPath = r'C:\Users\TTTime\Desktop\创新实践2\code\venv\code\data\after'

# keep_default_na = False参数使得读进来的含空值列都是object类型，即str类型
dftrain = pd.read_csv(beforePath+r'\ccf_offline_stage1_train.csv',header=0,keep_default_na=False)
dftest = pd.read_csv(beforePath+r'\ccf_offline_stage1_test_revised.csv',header=0,keep_default_na=False)

# print(dftrain.dtypes)
# print(dftest.dtypes)
# print(dftrain.head(20))
# print(dftest.head(20))

# 确实有重复行可以删除
dftrain.drop_duplicates(inplace=True)

# 滑窗法分割dataset，分别是优惠券接收日期为4/14~5/14、5/14~6/14、需要预测数据7.1~7.15
# 滑窗法分割feature,分别是4.14前三个半月，5.14前三个半月，6.14前三个半月，分别用来提取三个dataset的userFeature、merchantFeature、user_merchantFeature
# 最后再用前两个dataset数据样本集成到的特征去预测最后一个dataset即测试数据

dataset1 = dftrain[(dftrain['Date_received']>='20160414') & (dftrain['Date_received']<='20160514')]
feature1 = dftrain[(dftrain['Date_received']>='20160101') & (dftrain['Date_received']<='20160413')]

dataset2 = dftrain[(dftrain['Date_received']>='20160514') & (dftrain['Date_received']<='20160614')]
feature2 = dftrain[(dftrain['Date_received']>='20160201') & (dftrain['Date_received']<='20160513')]

dataset3 = dftest
feature3 = dftrain[(dftrain['Date_received']>='20160301') & (dftrain['Date_received']<='20160614')]

# 由于dftest中Date_received列无空值null，读进来会归为int64类型，但是后面特征值的提取需要该加和拼接该字段，这里提前转化为str类型，debug好久才找到的...
dataset3['Date_received'] = dataset3['Date_received'].astype(str)


def getReceivedUseGap(dates):
    dates = dates.values
    # print(dates)
    receive,use = dates[0],dates[1]
    return (date(int(use[0:4]),int(use[4:6]),int(use[6:8])) - date(int(receive[0:4]),int(receive[4:6]),int(receive[6:8]))).days

def getUserRelatedFeature(feature):
    # 用来连接的df
    t = feature['User_id'].copy()
    t.drop_duplicates(inplace=True)

    # 特征：用户购买商家类数（类数不重复）
    t1 = feature[feature['Date']!='null'][['User_id','Merchant_id']].copy()
    t1.drop_duplicates(inplace=True)
    t1['Merchant_id'] = 1
    t1 = t1.groupby('User_id').agg('sum').reset_index()
    t1.rename(columns={'Merchant_id':'User_buy_merchant_count'},inplace=True)

    t2 = feature[(feature['Date']!='null') & (feature['Coupon_id']!='null')][['User_id','Distance']].copy()
    t2.replace('null',-1,inplace=True)
    t2['Distance'] = t2['Distance'].astype(float)
    t2.replace(-1,np.nan,inplace=True)
    # 特征：用户距离已用消费券消费店铺的最大、最小、平均、中位距离
    t2_1 = t2.groupby('User_id').agg('max').reset_index()
    t2_1.rename(columns={'Distance':'User_max_distance'},inplace=True)
    t2_2 = t2.groupby('User_id').agg('min').reset_index()
    t2_2.rename(columns={'Distance':'User_min_distance'},inplace=True)
    t2_3 = t2.groupby('User_id').agg('mean').reset_index()
    t2_3.rename(columns={'Distance':'User_mean_distance'},inplace=True)
    t2_4 = t2.groupby('User_id').agg('median').reset_index()
    t2_4.rename(columns={'Distance':'User_median_distance'},inplace=True)

    #特征：用户使用优惠券消费次数
    t3 = feature[(feature['Coupon_id']!='null') & (feature['Date']!='null')][['User_id']].copy()
    t3['User_buy_use_coupon_count'] = 1
    t3 = t3.groupby('User_id').agg('sum').reset_index()

    # 特征：用户消费次数
    t4 = feature[(feature['Date']!='null')][['User_id']].copy()
    t4['User_buy_count'] = 1
    t4 = t4.groupby('User_id').agg('sum').reset_index()

    # 特征：用户接收优惠券数目
    t5 = feature[(feature['Coupon_id'] != 'null')][['User_id']].copy()
    t5['User_received_count'] = 1
    t5 = t5.groupby('User_id').agg('sum').reset_index()

    # 特征：用户接收并使用消费券间隔天数
    t6 = feature[(feature['Coupon_id'] != 'null') & (feature['Date'] != 'null')][['User_id', 'Date_received', 'Date']].copy()
    t6['User_received_use_gap'] = t6[['Date_received', 'Date']].apply(getReceivedUseGap, axis=1)
    t6 = t6[['User_id', 'User_received_use_gap']]

    # 特征：用户接收并使用优惠券的最大、最小、平均间隔天数
    t7 = t6.copy()
    t7_1 = t7.groupby('User_id').agg('max').reset_index()
    t7_1.rename(columns={'User_received_use_gap':'User_received_use_max_gap'},inplace=True)
    t7_2 = t7.groupby('User_id').agg('min').reset_index()
    t7_2.rename(columns={'User_received_use_gap':'User_received_use_min_gap'},inplace=True)
    t7_3 = t7.groupby('User_id').agg('mean').reset_index()
    t7_3.rename(columns={'User_received_use_gap':'User_received_use_mean_gap'},inplace=True)

    # 特征：用户浏览总数
    t8 = feature[['User_id']].copy()
    t8['User_browser_count'] = 1
    t8 = t8.groupby('User_id').agg('sum').reset_index()

    userFeature = pd.merge(t,t1,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t2_1,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t2_2,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t2_3,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t2_4,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t3,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t4,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t5,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t6,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t7_1,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t7_2,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t7_3,on='User_id',how='left')
    userFeature = pd.merge(userFeature,t8,on='User_id',how='left')

    # 特征：用户消费总量中使用优惠券占比
    userFeature['User_buy_use_coupon_rate'] = userFeature['User_buy_use_coupon_count']/userFeature['User_buy_count']
    # 特征：用户接收总量中使用优惠券占比
    userFeature['user_received_coupon_use_rate'] = userFeature['User_buy_use_coupon_count']/userFeature['User_received_count']

    # 对于次数或者数目或占比，将Nan转换为0
    userFeature['User_buy_merchant_count'].replace(np.nan,0,inplace=True)
    userFeature['User_buy_use_coupon_count'].replace(np.nan,0,inplace=True)
    userFeature['User_buy_count'].replace(np.nan,0,inplace=True)
    userFeature['User_received_count'].replace(np.nan,0,inplace=True)
    userFeature['User_buy_use_coupon_rate'].replace(np.nan,0,inplace=True)
    userFeature['user_received_coupon_use_rate'].replace(np.nan,0,inplace=True)

    # print(userFeature.dtypes)
    # print(userFeature.head(10))

    return userFeature

def getMerchantRelatedFeature(feature):

    t = feature['Merchant_id'].copy()
    t.drop_duplicates(inplace=True)

    # 特征：商家卖出数目
    t1 = feature[(feature['Date']!='null')][['Merchant_id']].copy()
    t1['Merchant_sale_count'] = 1
    t1 = t1.groupby('Merchant_id').agg('sum').reset_index()

    # 特征：商家核销数目
    t2 = feature[(feature['Coupon_id']!='null') & (feature['Date']!='null')][['Merchant_id']].copy()
    t2['Merchant_sale_use_coupon_count'] = 1
    t2 = t2.groupby('Merchant_id').agg('sum').reset_index()

    # 特征：商家优惠券的总数量
    t3 = feature[(feature['Coupon_id']!='null')][['Merchant_id']].copy()
    t3['Merchant_give_count'] = 1
    t3 = t3.groupby('Merchant_id').agg('sum').reset_index()

    t4 = feature[(feature['Coupon_id']!='null') & (feature['Date']!='null')][['Merchant_id','Distance']].copy()
    t4['Distance'].replace('null',-1,inplace=True)
    t4['Distance'] = t4['Distance'].astype(float)
    t4['Distance'].replace(-1,np.nan,inplace=True)
    # 特征：商家已核销优惠券中距离的最小\最大\平均\中值
    t4_1 = t4.groupby('Merchant_id').agg('max').reset_index()
    t4_1.rename(columns={'Distance':'Merchant_max_distance'},inplace=True)
    t4_2 = t4.groupby('Merchant_id').agg('min').reset_index()
    t4_2.rename(columns={'Distance':'Merchant_min_distance'},inplace=True)
    t4_3 = t4.groupby('Merchant_id').agg('mean').reset_index()
    t4_3.rename(columns={'Distance':'Merchant_mean_distance'},inplace=True)

    merchantFeature = pd.merge(t,t1,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t2,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t3,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t4_1,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t4_2,on='Merchant_id',how='left')
    merchantFeature = pd.merge(merchantFeature,t4_3,on='Merchant_id',how='left')

    # 特征：商家卖出总量中优惠券的核销占比
    merchantFeature['Merchant_sale_use_coupon_rate'] = merchantFeature['Merchant_sale_use_coupon_count']/merchantFeature['Merchant_sale_count']
    # 特征：商家发放总量中优惠券的核销占比
    merchantFeature['Merhcant_give_coupon_use_rate'] = merchantFeature['Merchant_sale_use_coupon_count']/merchantFeature['Merchant_give_count']

    # 次数项目和占比类型数据，Nan用0替代(之所以最后转化，是防止上两个特征提取时出现分母为零溢出)（另外，上两个特征值的计算，只要分子分母一个为pd.nan结果就为nd.nan）
    merchantFeature['Merchant_sale_use_coupon_count'].replace(np.nan,0,inplace=True)
    merchantFeature['Merchant_sale_count'].replace(np.nan,0,inplace=True)
    merchantFeature['Merchant_give_count'].replace(np.nan,0,inplace=True)
    merchantFeature['Merchant_sale_use_coupon_rate'].replace(np.nan,0,inplace=True)
    merchantFeature['Merhcant_give_coupon_use_rate'].replace(np.nan,0,inplace=True)

    # print(merchantFeature.shape)
    # print(merchantFeature.dtypes)
    # print(merchantFeature.head(10))

    return merchantFeature

def getDiscountType(s):
    if ':' in s:
        return 1
    else:
        return 0
# 暂没用到
def getDiscountRate(s):
    if ':' in s:
        x = s.split(s)
        return float(x[1])/float(x[0])
    else:
        return float(s)

def getDiscountMan(s):
    if ':' in s:
        x = s.split(':')
        return int(s[0])
    else :
        return 0

def getDiscountJian(s):
    if ':' in s:
        x = s.split(':')
        return int(x[1])
    else:
        return 0

def getDiscountRate(s):
    if ':' in s:
        rate =  1 - float(s.split(":")[1])/float(s.split(":")[0])
        return rate
    else:
        return float(s)


def getCouponRelatedFeature(dataset):

    t = dataset.copy()
    # 这里dataset无重复值，不用drop_duplicates()

    # 特征：消费券发放的周号\月份
    t['Coupon_give_weekday'] = t['Date_received'].astype(str).apply(lambda x: date(int(x[0:4]),int(x[4:6]),int(x[6:8])).weekday()+1)
    t['Coupon_give_monthday'] = t['Date_received'].astype(str).apply(lambda x: int(x[6:8]))

    t['Discount_rate'] = t['Discount_rate'].astype(str)
    # 特征：消费券是否是满减类型
    t['Coupon_discount_type'] = t['Discount_rate'].apply(getDiscountType)
    # 特征：消费券满减的满\减
    t['Coupon_discount_man'] = t['Discount_rate'].apply(getDiscountMan)
    t['Coupon_discount_jian'] = t['Discount_rate'].apply(getDiscountJian)
    # 特征：优惠券打折类型力度
    t['Coupon_discount_rate'] = t['Discount_rate'].apply(getDiscountRate)
    # t['Coupon_discount_rate'] = t['Discount_rate'].apply(lambda x: float(x) if ':' not in x else np.nan)



    # 特征：每种优惠券的数目
    t1 = dataset[['Coupon_id']].copy()
    t1['Coupon_count'] = 1
    t1 = t1.groupby('Coupon_id').agg('sum').reset_index()

    couponFeature = pd.merge(t,t1,on='Coupon_id',how='left')

    # print(couponFeature.shape)
    # print(couponFeature.dtypes)
    # print(couponFeature.head(10))

    return couponFeature


def getUserMerchantRelatedFeature(feature):

    t = feature[['User_id','Merchant_id']].copy()
    t.drop_duplicates(inplace=True)

    # 特征1：一个客户在一个商家一共买的次数
    t1 = feature[(feature['Date']!='null')][['User_id','Merchant_id']].copy()
    t1['User_Merchant_buy_count'] = 1
    t1 = t1.groupby(['User_id','Merchant_id']).agg('sum').reset_index()

    # 特征2：一个客户在一个商家一共收到的优惠券
    t2 = feature[(feature['Coupon_id']!='null')][['User_id','Merchant_id']].copy()
    t2['User_Merchant_received_count'] = 1
    t2 = t2.groupby(['User_id','Merchant_id']).agg('sum').reset_index()

    # 特征3：一个客户在一个商家使用优惠券购买的次数
    t3 = feature[(feature['Date_received']!='null') & (feature['Date']!='null')][['User_id','Merchant_id']].copy()
    t3['User_Merchant_buy_use_coupon_count'] = 1
    t3 = t3.groupby(['User_id','Merchant_id']).agg('sum').reset_index()

    # 特征4：一个客户在一个商家浏览的次数
    t4 = feature[['User_id','Merchant_id']].copy()
    t4['User_Merchant_browse_count'] = 1
    t4 = t4.groupby(['User_id','Merchant_id']).agg('sum').reset_index()

    # 特征5：一个客户在一个商家没有使用优惠券购买的次数（暂留）
    t5 = feature[(feature['Date_received']=='null') & (feature['Date']!='null')][['User_id','Merchant_id']].copy()
    t5['User_Merchant_buy_not_use_coupon_count'] = 1
    t5 = t5.groupby(['User_id','Merchant_id']).agg('sum').reset_index()

    userMerchantFeature = pd.merge(t,t1,on=['User_id','Merchant_id'],how='left')
    userMerchantFeature = pd.merge(userMerchantFeature,t2,on=['User_id','Merchant_id'],how='left')
    userMerchantFeature = pd.merge(userMerchantFeature,t3,on=['User_id','Merchant_id'],how='left')
    userMerchantFeature = pd.merge(userMerchantFeature,t4,on=['User_id','Merchant_id'],how='left')
    userMerchantFeature = pd.merge(userMerchantFeature,t5,on=['User_id','Merchant_id'],how='left')

    # 特征6：用户在某商家接收优惠券中使用的占比
    userMerchantFeature['User_Merchant_received_use_coupon_rate'] = userMerchantFeature['User_Merchant_buy_use_coupon_count']/userMerchantFeature['User_Merchant_received_count']
    # 特征7：用户在某商家消费中使用优惠券支付的占比
    userMerchantFeature['User_Merchant_buy_use_coupon_rate'] = userMerchantFeature['User_Merchant_buy_use_coupon_count']/userMerchantFeature['User_Merchant_buy_count']
    # 特征8：用户在某商家浏览次数中消费的占比
    userMerchantFeature['User_Merchant_browse_use_coupon_rate'] = userMerchantFeature['User_Merchant_buy_count']/userMerchantFeature['User_Merchant_browse_count']
    # 特征9：用户在某商家浏览中使用优惠券消费占比
    userMerchantFeature['User_Merchant_browser_use_coupon_rate'] = userMerchantFeature['User_Merchant_buy_use_coupon_count']/userMerchantFeature['User_Merchant_browse_count']

    # print(userMerchantFeature.shape)
    # print(userMerchantFeature.dtypes)
    # print(userMerchantFeature.head(100))

    return userMerchantFeature

def isFirstReceived(s):
    a,b = s.split('_')
    b = b.split(':')
    if(int(a)-min([int(t) for t in b]))==0:
        return 1
    return 0

def isLastReceived(s):
    a,b = s.split('_')
    b = b.split(':')
    if(int(a)-max([int(t) for t in b]))==0:
        return 1
    return 0

def getBeforeReceivedGap(s):
    a,b = s.split('_')
    b = b.split(':')
    list = []
    for i in b:
        gap = (date(int(a[0:4]),int(a[4:6]),int(a[6:8])) - date(int(i[0:4]),int(i[4:6]),int(i[6:8]))).days
        if gap>0:
            list.append(gap)
    if len(list)==0:
        return -1
    return min(list)

def getAfterReceivedGap(s):
    a,b = s.split('_')
    b = b.split(':')
    list = []
    for i in b:
        gap = (date(int(i[0:4]),int(i[4:6]),int(i[6:8])) - date(int(a[0:4]),int(a[4:6]),int(a[6:8]))).days
        if gap>0:
            list.append(gap)
    if len(list)==0:
        return -1
    return min(list)


# 赛题leakage
def getOtherRelatedFeature(dataset):

    # 特征：某用户所有领取优惠券数量
    t1 = dataset[(dataset['Coupon_id']!='null')][['User_id']].copy()
    t1['Other_user_received_count'] = 1
    t1 = t1.groupby('User_id').agg('sum').reset_index()

    # 特征：某用户领取特定优惠券数量
    t2 = dataset[(dataset['Coupon_id']!='null')][['User_id','Coupon_id']].copy()
    t2['Ohter_user_coupon_received_count'] = 1
    t2 =t2.groupby(['User_id','Coupon_id']).agg('sum').reset_index()


    t3 = dataset[(dataset['Coupon_id']!='null')][['User_id','Coupon_id','Date_received']].copy()
    t3_ = t3.copy()
    # 只对'Date_received'列实行加和函数，但是整体返回的是分组列和选中的‘Date_received‘列
    t3 = t3.groupby(['User_id','Coupon_id'])[['Date_received']].agg(lambda x: ':'.join(x)).reset_index()
    t3.rename(columns={'Date_received':'Date_received_concat'},inplace=True)
    t3 = pd.merge(t3_,t3,on=['User_id','Coupon_id'],how='inner')
    # 特征：用户特定的优惠券领取是否是第一次\最后一次领取
    t3['Other_user_is_first_received'] = pd.DataFrame(t3['Date_received'].astype(str).apply(lambda x:x+'_')+t3['Date_received_concat']).applymap(isFirstReceived)

    t3['Other_user_is_last_received'] = pd.DataFrame(t3['Date_received'].astype(str).apply(lambda x:x+'_')+t3['Date_received_concat']).applymap(isLastReceived)
    # print(t3.head(100))
    t3.drop('Date_received_concat',axis=1,inplace=True)



    # 特征：一个用户某天所接收到的所有优惠券的数量
    t4 = dataset[(dataset['Coupon_id']!='null')][['User_id','Date_received']].copy()
    t4['Other_user_oneday_received_count'] = 1
    t4 = t4.groupby(['User_id','Date_received']).agg('sum').reset_index()


    # 特征：一个用户某天接收到特定优惠券的数量
    t5 = dataset[(dataset['Coupon_id']!='null')][['User_id','Coupon_id','Date_received']].copy()
    t5['Other_user_oneday_Coupon_received_count'] = 1
    t5 = t5.groupby(['User_id','Coupon_id','Date_received']).agg('sum').reset_index()


    # 特征：用户领取某优惠券日期与上次\下次领取相同优惠券的日期间最小天数间隔，没有则 - 1
    t6 = dataset[(dataset['Coupon_id'] != 'null')][['User_id','Coupon_id' , 'Date_received']].copy()
    t6_ = t6.copy()
    t6 = t6.groupby(['User_id','Coupon_id'])[['Date_received']].agg(lambda x: ':'.join(x)).reset_index()
    t6.rename(columns={'Date_received':'Date_received_concat'},inplace=True)
    t6 = pd.merge(t6,t6_,on=['User_id','Coupon_id'],how='inner')
    t6['Other_user_coupon_before_received_gap'] = pd.DataFrame(t6['Date_received']+"_"+t6['Date_received_concat']).applymap(getBeforeReceivedGap)
    t6['Other_user_coupon_after_received_gap'] = pd.DataFrame(t6['Date_received']+"_"+t6['Date_received_concat']).applymap(getAfterReceivedGap)
    t6.drop(['Date_received_concat'],axis=1,inplace=True)


    otherFeature = pd.merge(t1,t2,on=['User_id'],how='left')
    otherFeature = pd.merge(otherFeature,t3,on=['User_id','Coupon_id'],how='left')
    otherFeature = pd.merge(otherFeature,t4,on=['User_id','Date_received'],how='left')
    otherFeature = pd.merge(otherFeature,t5,on=['User_id','Coupon_id','Date_received'],how='left')
    otherFeature = pd.merge(otherFeature,t6,on=['User_id','Coupon_id','Date_received'],how='left')


    # print('return_ohterFeature_shape:',otherFeature.shape)
    # print(otherFeature.dtypes)
    # print(otherFeature.head(100))

    return otherFeature



def isWeekend(day):
    if day>=1 and day<=5:
        return 0
    else:
        return 1

def getLabel(row):
    row = row.values
    a = str(row[0])
    b = str(row[1])
    # print(a,b)
    if a=='null' or b=='null':
        return 0
    elif (date(int(b[0:4]),int(b[4:6]),int(b[6:8])) - date(int(a[0:4]),int(a[4:6]),int(a[6:8]))).days <= 15:
        return 1
    else:
        return 0



def featureProcess(dataset,feature,processFlag):
    user = getUserRelatedFeature(feature)
    merchant = getMerchantRelatedFeature(feature)
    coupon = getCouponRelatedFeature(dataset)
    userMerchant = getUserMerchantRelatedFeature(feature)
    other = getOtherRelatedFeature(dataset)

    allFeature = pd.merge(coupon,user,on='User_id',how='left')
    allFeature = pd.merge(allFeature,merchant,on='Merchant_id',how='left')
    allFeature = pd.merge(allFeature,userMerchant,on=['User_id','Merchant_id'],how='left')
    allFeature = pd.merge(allFeature,other,on=['User_id','Coupon_id','Date_received'],how='left')

    allFeature['Coupon_give_weekday_is_weekend'] = allFeature['Coupon_give_weekday'].apply(isWeekend)
    weekday_dummies = pd.get_dummies(allFeature['Coupon_give_weekday'])
    weekday_dummies.columns = ['Coupon_give_weekday_' + str(i) for i in range(1,weekday_dummies.shape[1]+1)]
    allFeature = pd.concat([allFeature,weekday_dummies],axis=1)
    allFeature.drop('Coupon_give_weekday',axis=1,inplace=True)

    if processFlag:
        allFeature['Label'] = allFeature[['Date_received','Date']].apply(getLabel,axis=1)
        allFeature.drop(['User_id','Date_received','Coupon_id','Merchant_id','Discount_rate','Date'],axis=1,inplace=True)
    else:
        # 'User_id','Date_received','Coupon_id'字段需要在提交文档中,先留下，复制完后到训练时再删
        allFeature.drop(['Merchant_id','Discount_rate'],axis=1,inplace=True)
    allFeature.replace('null',np.nan,inplace=True)


    # print(allFeature.shape)
    # print(allFeature.dtypes)
    # print(allFeature.head(100))


    return allFeature

# test
# test_user = getUserRelatedFeature(feature1)
# test_merchant = getMerchantRelatedFeature(feature1)
# test_coupon = getCouponRelatedFeature(feature2)
# test_u_m = getUserMerchantRelatedFeature(feature1)
# test_other = getOtherRelatedFeature(dataset3)
# test_all = featureProcess(dataset1,feature1,True)
# print(test_all.Label.head(20))

df1 = featureProcess(dataset1,feature1,True)
df1.to_csv(afterPath+r'\df1.csv')
print('df1 write over')

df2 = featureProcess(dataset2,feature2,True)
df2.to_csv(afterPath+r'\df2.csv')
print('df2 write over')

df3 = featureProcess(dataset3,feature3,False)
df3.to_csv(afterPath+r'\df3.csv')
print('df3 write over')















