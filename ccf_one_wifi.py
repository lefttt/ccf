# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import date
from sklearn import neighbors, datasets
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
import lightgbm as lgb
import re

'''-------------------读取数据-------------------------------'''
def read_data_on_mac():
    shop_info=pd.read_csv('/Users/wtt/ccf/ccf_first_round_shop_info.csv')
    user_shop_behavior=pd.read_csv('/Users/wtt/ccf/ccf_first_round_user_shop_behavior.csv')
    on_test=pd.read_csv('/Users/wtt/ccf/evaluation_public.csv')
    mall_info=shop_info[['mall_id']].drop_duplicates()
    mall_info['mall_num']=list(range(mall_info.shape[0]))

    shop_in_mall=shop_info[['shop_id','mall_id']]
    shop_in_mall['shop_num']=list(range(shop_in_mall.shape[0])) #店名列表 shop_num为序号 'shop_id','mall_id','shop_num'

    all_info=pd.merge(user_shop_behavior,shop_in_mall,on=['shop_id'])
    all_info=pd.merge(all_info,mall_info,on=['mall_id']).reset_index(drop=True)
    return all_info,on_test,mall_info,shop_in_mall

def read_data_on_win():
    shop_info=pd.read_csv(r'C:\Users\lancer\cff\ccf_first_round_shop_info.csv')
    user_shop_behavior=pd.read_csv(r'C:\Users\lancer\cff\ccf_first_round_user_shop_behavior.csv')
    on_test=pd.read_csv(r'C:\Users\lancer\cff\evaluation_public.csv')

    shop_in_mall=shop_info[['shop_id','mall_id']]
#    shop_in_mall['shop_num']=list(range(shop_in_mall.shape[0])) #店名列表 shop_num为序号 'shop_id','mall_id','shop_num'

    all_info=pd.merge(user_shop_behavior,shop_info[['shop_id','mall_id']],on=['shop_id'])
    return all_info,on_test,shop_in_mall

'''------------构建特征--------------'''
'''把一个wifi_info变为最强wifi的名字和强度'''
def the_wifi(a):
    each_wifi=sorted([wifi.split('|') for wifi in a.split(';')],key=lambda x:int(x[1]),reverse=True)[0]
    return [int(each_wifi[0][2:]),int(each_wifi[1])/100]

'''取出强度最高wifi的名字和强度'''
def wifi_feature(all_info):
    wifi_feature_pd=pd.DataFrame(list(map(the_wifi,all_info['wifi_infos'])),columns=['wifi_name','wifi_strong'])
#    wifi_feature_pd.to_csv('wifi_feature.csv',index=None)
    return wifi_feature_pd
#wifi_feature()


def wifi_name_list(all_info):
    wifi_list=[]
    for wifis in all_info['wifi_infos']:
        for wifi in wifis.split(';'):
            s=wifi.split('|')[0]
            wifi_list.append(s)        
            wifi_list=list(set(wifi_list))
    print(len(wifi_list))
    return(wifi_list)

def wifi_features(wifi_info,wifi_list):
    wifi_max=[[]]*5
    for wifis in wifi_info['wifi_infos']: 
        wifi_dict={}
        for wifi in wifis.split(';'):
            wifi_one=wifi.split('|')
            wifi_dict[int(wifi_one[1])]=wifi_list.index(wifi_one[0])
        wifi_dict=sorted(wifi_dict.items(),key=lambda d:d[0],reverse=True)
#        print(wifi_dict)
#        break
        temp=[]
        for i in range(5):
            if i>=len(wifi_dict):
                temp.append(temp[0])
                continue
            temp.append(wifi_dict[i][0])
        temp=sorted(temp)
        for i in range(5):
            wifi_max[i].append(temp[i])
        
    wifi_info['wifi_1']=wifi_max[0]
    wifi_info['wifi_2']=wifi_max[1]
    wifi_info['wifi_3']=wifi_max[2]
    wifi_info['wifi_4']=wifi_max[3]
    wifi_info['wifi_5']=wifi_max[4]
    print(wifi_info)
    return wifi_info

def wifi_to_list(all_info,wifi_list):
    wifi_name=[]
    wifi_strong=[]
    i=0;j=all_info.shape[0]
    for wifis in all_info['wifi_infos']:
        wifi=sorted([wifi.split('|') for wifi in wifis.split(';')],key=lambda x:int(x[1]),reverse=True)[:1]
        wifi_name.append(wifi_list.index(wifi[0][0]))
        wifi_strong.append(int(wifi[0][1])/100)
        i=i+1
        print(i,j)
    all_info['wifi_name']=wifi_name
    all_info['wifi_strong']=wifi_strong
    all_info.to_csv(r'C:\Users\lancer\cff\one_wifi.csv',index=None)

# =============================================================================
# all_info=pd.read_csv(r'C:\Users\lancer\cff\all_info.csv')
# wifi_name_list(all_info)
# =============================================================================
#wifi_to_list(user_shop_behavior,wifi_name_list(user_shop_behavior))

#wifi_feature(all_info,wifi_list))

def wifi_string_to_list(wifi_string):
    wifi_name=[];wifi_strong=[]
    for wifi in wifi_string.split(';'):
        wifi_one=wifi.split('|')
        wifi_name.append(wifi_one[0])
        wifi_strong.append(int(wifi_one[1]))
    return wifi_name,wifi_strong



#将时间戳转化为小时
#结果返回原表加一列column='hour'
def timestamp_to_hour(df_user_shop_timestamp):
    regex=re.compile('[0-9]{2}')
    hour = []
    for timestamp in df_user_shop_timestamp['time_stamp']:
        hour.append(int(regex.findall(timestamp)[-2]))
    df_user_shop_timestamp['hour'] = hour
    #df_user_shop_timestamp.to_csv('all_info_with_hour.csv',index=None)
    return df_user_shop_timestamp
#对用户的小时进行分段
def user_hour_feature(df_user_hour):
    index=['23-09','10-13','14-16','17-22']
    dic={index[0]:[],index[1]:[],index[2]:[],index[3]:[]}
    for hour in df_user_hour['hour'].values:
        i = int(hour)
        if i >= 0 and i <= 9 or i==23:
            dic[index[0]].append(1)
            dic[index[1]].append(0)
            dic[index[2]].append(0)
            dic[index[3]].append(0)
        elif i==10 or i ==11 or i==12 or i ==13:
            dic[index[0]].append(0)
            dic[index[1]].append(1)
            dic[index[2]].append(0)
            dic[index[3]].append(0)
        elif i== 14 or i==15 or i==16:
            dic[index[0]].append(0)
            dic[index[1]].append(0)
            dic[index[2]].append(1)
            dic[index[3]].append(0)
        else:
            dic[index[0]].append(0)
            dic[index[1]].append(0)
            dic[index[2]].append(0)
            dic[index[3]].append(1)
    df_user_hour.loc[index[0]]=dic[index[0]]
    df_user_hour[index[1]]=dic[index[1]]
    df_user_hour[index[2]]=dic[index[2]]
    df_user_hour[index[3]]=dic[index[3]]
    return df_user_hour

    
#加上时间特征，并
def hour(all_info):
    all_info=user_hour_feature(timestamp_to_hour(all_info))

    data_25=all_info[all_info.time_stamp<'2017-08-25 23:51']
    test_data_25=all_info[all_info.time_stamp>'2017-08-25 23:51']
    test_data_25['row_id']=list(range(test_data_25.shape[0]))
    result_data_25=test_data_25[['row_id','shop_id','mall_num']]
    return all_info,data_25,test_data_25,result_data_25
'''
data_1_20__25_30=user_shop_behavior[(user_shop_behavior.time_stamp<'2017-08-20 23:51')|(user_shop_behavior.time_stamp>'2017-08-25 23:51')][['shop_id','longitude','latitude']]
train_1_20__25_30=pd.merge(data_1_20__25_30,shop_in_mall,on=['shop_id'])
test_data_20_25=user_shop_behavior[(user_shop_behavior.time_stamp>'2017-08-20 23:51')&(user_shop_behavior.time_stamp<'2017-08-25 23:51')][['shop_id','longitude','latitude']]
test_data_20_25=pd.merge(test_data_20_25,shop_in_mall,on=['shop_id'])        #'shop_id','mall_id','shop_num''longitude','latitude'
test_data_20_25['row_id']=list(range(test_data_20_25.shape[0]))
result_data_20_25=test_data_20_25[['row_id','shop_id']]
test_data_20_25=test_data_20_25[['mall_id','row_id','longitude','latitude']]
'''

def train_knn_with_hour(train_data,test,n_neighbors=15):
    clf= neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    X=train_data[['longitude','latitude','mall_num','23-09','10-13','14-16','17-22']]
    Y=train_data[['shop_num']]
    clf.fit(X,Y)

    x=user_hour_feature(timestamp_to_hour(test))[['longitude','latitude','mall_num','23-09','10-13','14-16','17-22']]
    result=test[['row_id']]
    result['shop_num']=clf.predict(x)
    result=pd.merge(result,shop_in_mall,on=['shop_num'])
    result=result[['row_id','shop_id']]
    #result.to_csv('result.csv',index=None)
    return result


#给出距离测试数据最近的k个点的信息
def near_neighbors(train_data,test,mall_info,n_neighbors=15):

    X=train_data[['longitude','latitude','mall_num']]
#    Y=train_data[['shop_num']]
    nbrs = NearestNeighbors(n_neighbors)
    nbrs.fit(X)
    x=pd.merge(test,mall_info,how='left',on=['mall_id'])[['longitude','latitude','mall_num']]
    distances,indices=nbrs.kneighbors(x)

#    print(test.shape)
    test['neighbors']=list(indices)
    test.to_csv(r'C:\Users\lancer\cff\neighbors.csv',index=None)
    return 
'''--------------------方法------------------------'''
'''评分'''
def score(train_result,result):
    train_result.rename(columns={'shop_id':'train_shop_id'},inplace=True)
    train_result=pd.merge(train_result,result)
    #print(train_result,(train_result['shop_id']==train_result['train_shop_id']).sum(),train_result.shape[0])
    return (train_result['shop_id']==train_result['train_shop_id']).sum()/train_result.shape[0]

 
    
#all_info,on_test,mall_info=read_data_on_win()
#all_info.to_csv(r'C:\Users\lancer\cff\all_info.csv')
#near_neighbors(all_info,on_test,mall_info)
    

        

def wifi_in_shop():
    user_shop_behavior=pd.read_csv(r'C:\Users\lancer\cff\ccf_first_round_user_shop_behavior.csv')
    wifi_to_shops = defaultdict(lambda : defaultdict(lambda :0)) # 默认字典嵌套  wifi_to_shops[wifi][shop]为wifi与shop的关联个数
    for line in user_shop_behavior.values:
        wifi = sorted([wifi.split('|') for wifi in line[5].split(';')],key=lambda x:int(x[1]),reverse=True)[:1]
        for i,each_wifi in enumerate(wifi):
            wifi_to_shops[each_wifi[0]][line[1]] += 1
    return wifi_to_shops

'''-----------------训练-----------------------'''

'''knn调参数'''
def gogogo():
    n_neighbors=1
    score_list=[]
    while n_neighbors<=10:
        #score1=score(train_knn(train_1_20__25_30,test_data_20_25,n_neighbors),result_data_20_25)
        score2=score(train_knn(data_25,test_data_25,n_neighbors),result_data_25)
        #score3=(score1+score2)/2
        #print(n_neighbors,score1)
        score_list.append([n_neighbors,score2])
        n_neighbors=n_neighbors+3
        print
#print(score(train_knn(data_25,test_data_25),result_data_25))

'''用最强wifi做距离knn之后的规则选择'''
def second(wifi_to_shops):
    all_info=pd.read_csv(r'C:\Users\lancer\cff\all_info.csv')
    test_with_neighbors=pd.read_csv(r'C:\Users\lancer\cff\neighbors.csv')
    final_shop_list=[]
    i=0
    j=test_with_neighbors.shape[0]
    for index,line in test_with_neighbors.iterrows():
        wifi_id,wifi_strong=wifi_string_to_list(line['wifi_infos'])
        wifi_max_id=wifi_id[wifi_strong.index(max(wifi_strong))]
        neighbor_string=line['neighbors']
        neighbor=neighbor_string[1:-1].split()
        for node in neighbor:
            shop_name=all_info.at[int(node),'shop_id']
            wifi_infos=all_info.at[int(node),'wifi_infos']
            wifi_node_id,wifi_node_strong=wifi_string_to_list(wifi_infos)
            counter = defaultdict(lambda : 0)
            for k,v in wifi_to_shops[wifi_max_id].items():
                    counter[k] += v
            try:
                final_shop_id = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
            except:
                final_shop_id = None
        final_shop_list.append(final_shop_id)
        i=i+1;print(i,j)
    test_with_neighbors['shop_id']=final_shop_list
    test_with_neighbors[['row_id','shop_id']].to_csv(r'C:\Users\lancer\cff\result.csv',index=None)
#second(wifi_in_shop())
'''用wifi做knn'''
def wifi_knn(train_data,test_data,shop_in_mall,n_neighbors=10):
#    all_info,on_test,mall_info,shop_in_mall=read_data_on_mac()
    X=list(map(the_wifi,train_data['wifi_infos']))
    clf= neighbors.KNeighborsClassifier(n_neighbors, weights='uniform')

    Y=train_data[['shop_num']]
    clf.fit(X,Y)

    x=list(map(the_wifi,test_data['wifi_infos']))
    result=test_data[['row_id']]
    result['shop_num']=clf.predict(x)
    result=pd.merge(result,shop_in_mall,on=['shop_num'])
    result=result[['row_id','shop_id']]
#    result.to_csv('result.csv',index=None)
    return result
#wifi_knn()
    
'''    latitude  longitude mall_id  mall_num  row_id  shop_id  shop_num  wifi_infos hour  wifi_name wifi_strong'''
def do_lgb():
    all_info,on_test,shop_in_mall=read_data_on_win()
    all_info=pd.concat([all_info,on_test]).reset_index(drop=True)
    wifi_feature_pd=wifi_feature(all_info)
    all_info_with_hour=pd.concat([timestamp_to_hour(all_info),wifi_feature_pd],axis=1)
    mall_list=list(set(list(shop_in_mall.mall_id)))
    result=pd.DataFrame()
    i=0
    for mall in mall_list:
        train1=all_info_with_hour[all_info_with_hour.mall_id==mall].reset_index(drop=True)
        shop=shop_in_mall[shop_in_mall.mall_id==mall]
        shop['shop_num']=list(range(shop.shape[0]))
        train1=pd.merge(train1,shop,how='left',on='shop_id')
        df_train=train1[train1.shop_id.notnull()]
        df_train['shop_num']=df_train['shop_num'].astype(int)
        df_test=train1[train1.shop_id.isnull()]
        X_train = df_train[['longitude','latitude','wifi_name','wifi_strong','hour']]
        y_train =df_train['shop_num']  # Wrong type(ndarray) for label, should be list or numpy array
#        print(y_train)
        X_test = df_test[['longitude','latitude','wifi_name','wifi_strong','hour']]
        
        lgb_train = lgb.Dataset(X_train, y_train)
        num_class=y_train.max()+1
        params = {                            #'task': 'train',
                'boosting_type': 'gbdt',
                'num_class': num_class,
                'objective': 'multiclass',
                'metric': {'multi_logloss'},#'multi_error'
                'num_leaves': 200,
                'learning_rate': 0.05,
                'feature_fraction': 1,
                'bagging_fraction':1,
                'bagging_freq': 0,
                'verbose': 0
                }

        print('Start training...')
        # train
        gbm = lgb.train(params,
                        lgb_train,
                        num_boost_round=20)
        print('Start predicting...')
        df_test['shop_num'] = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        print(X_test.shape[0])
        result1=pd.merge(df_test[['row_id','shop_num']],shop,how='left',on='shop_num')[['row_id','shop_id']]
        result=pd.concat([result,result1])
        print(result1.head())
        i=i+1;print(i)   
        if i==1: break
    result['row_id']=result['row_id'].astype('int')
    result.to_csv('lgbresult.csv',index=False)
#do_lgb()

'''------------------测试打分-------------------------'''

'''测试函数，读入修改过后的数据地址dz '''
def test_data(all_info,dz,shop_in_mall):
    train_data=pd.read_csv(dz)
    train_data_25=pd.merge(train_data,shop_in_mall,on=['shop_id'])[all_info.time_stamp<'2017-08-25 23:51']
    test_data_25=all_info[all_info.time_stamp>'2017-08-25 23:51']
    test_data_25['row_id']=list(range(test_data_25.shape[0]))
    result_data_25=test_data_25[['row_id','shop_id','mall_num']]    
    return score(wifi_knn(train_data_25,test_data_25,shop_in_mall),result_data_25)
def test_more_data():
    dz=['/Users/wtt/ccf/ccf_first_round_user_shop_behavior.csv',
        '/Users/wtt/Desktop/归档/drop_duplicate_user.csv',
        '/Users/wtt/Desktop/归档/train_drop_shop10.csv',
        '/Users/wtt/Desktop/归档/without_hour_usertest.csv']
    all_info,on_test,mall_info,shop_in_mall=read_data_on_mac()
    for i in dz:
        print(i,test_data(all_info,i,shop_in_mall))
#test_more_data()
        
all_info,on_test,shop_in_mall=read_data_on_win()
all_info=pd.concat([all_info,on_test]).reset_index(drop=True)
wifi_feature_pd=wifi_feature(all_info)
all_info_with_hour=pd.concat([timestamp_to_hour(all_info),wifi_feature_pd],axis=1)
mall_list=list(set(list(shop_in_mall.mall_id)))
result=pd.DataFrame()
i=0
for mall in mall_list:
    train1=all_info_with_hour[all_info_with_hour.mall_id==mall].reset_index(drop=True)
    shop=shop_in_mall[shop_in_mall.mall_id==mall]
    shop['shop_num']=list(range(shop.shape[0]))
    train1=pd.merge(train1,shop,how='left',on='shop_id')
    df_train=train1[train1.shop_id.notnull()]
    df_train['shop_num']=df_train['shop_num'].astype(int)
    df_test=train1[train1.shop_id.isnull()]
    X_train = df_train[['longitude','latitude','wifi_name','wifi_strong','hour']]
    y_train =df_train['shop_num']  # Wrong type(ndarray) for label, should be list or numpy array
#        print(y_train)
    X_test = df_test[['longitude','latitude','wifi_name','wifi_strong','hour']]
        
    lgb_train = lgb.Dataset(X_train, y_train)
    num_class=y_train.max()+1
    params = {                            #'task': 'train',
            'boosting_type': 'gbdt',
            'num_class': num_class,
            'objective': 'multiclass',
            'metric': {'multi_logloss'},#'multi_error'
            'num_leaves': 200,
            'learning_rate': 0.05,
            'feature_fraction': 1,
            'bagging_fraction':1,
            'bagging_freq': 0,
            'verbose': 0
            }

    print('Start training...')
    # train
    gbm = lgb.train(params,
                    lgb_train,
                    num_boost_round=20)
    print('Start predicting...')
    df_test['shop_num'] = np.array(gbm.predict(X_test, num_iteration=gbm.best_iteration)).argmax(axis=1)#, num_iteration=gbm.best_iteration
    print(X_test.shape[0])
    result1=pd.merge(df_test[['row_id','shop_num']],shop,how='left',on='shop_num')[['row_id','shop_id']]
    result=pd.concat([result,result1])
    print(result1.head())
    i=i+1;print(i)
result['row_id']=result['row_id'].astype('int')
#result.to_csv('lgbresult.csv',index=False)