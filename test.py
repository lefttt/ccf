# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from datetime import date
from sklearn import neighbors, datasets

##导入文件
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
    return all_info,on_test,mall_info

def read_data_on_win():
    shop_info=pd.read_csv(r'C:\Users\lancer\cff\ccf_first_round_shop_info.csv')
    user_shop_behavior=pd.read_csv(r'C:\Users\lancer\cff\ccf_first_round_user_shop_behavior.csv')
    on_test=pd.read_csv(r'C:\Users\lancer\cff\evaluation_public.csv')
    mall_info=shop_info[['mall_id']].drop_duplicates()
    mall_info['mall_num']=list(range(mall_info.shape[0]))

    shop_in_mall=shop_info[['shop_id','mall_id']]
    shop_in_mall['shop_num']=list(range(shop_in_mall.shape[0])) #店名列表 shop_num为序号 'shop_id','mall_id','shop_num'

    all_info=pd.merge(user_shop_behavior,shop_in_mall,on=['shop_id'])
    all_info=pd.merge(all_info,mall_info,on=['mall_id']).reset_index(drop=True)
    return all_info,on_test,mall_info

#wifi列表的构建
def wifi_name_list(all_info):
    wifi_list=[]
    for wifis in all_info['wifi_infos']:
        for wifi in wifis.split(';'):
            s=wifi.split('|')[0]
            wifi_list.append(s)        
            wifi_list=list(set(wifi_list))
    print(len(wifi_list))
    return(wifi_list)

def wifi_feature(wifi_info,wifi_list):
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

#wifi_feature(all_info,wifi_list))

def wifi_string_to_list(wifi_string):
    wifi_name=[];wifi_strong=[]
    for wifi in wifi_string.split(';'):
        wifi_one=wifi.split('|')
        wifi_name.append(wifi_one[0])
        wifi_strong.append(int(wifi_one[1]))
    return wifi_name,wifi_strong

import re

#将时间戳转化为小时
#结果返回原表加一列column='hour'
def timestamp_to_hour(df_user_shop_timestamp):
    regex=re.compile('[0-9]{2}')
    hour = []
    for timestamp in df_user_shop_timestamp['time_stamp']:
        hour.append(regex.findall(timestamp)[-2])
    df_user_shop_timestamp['hour'] = hour
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

from sklearn.neighbors import NearestNeighbors
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

def score(train_result,result):
    train_result.rename(columns={'shop_id':'train_shop_id'},inplace=True)
    train_result=pd.merge(train_result,result)
    #print(train_result,(train_result['shop_id']==train_result['train_shop_id']).sum(),train_result.shape[0])
    return (train_result['shop_id']==train_result['train_shop_id']).sum()/train_result.shape[0]


    
#all_info,on_test,mall_info=read_data_on_win()
#all_info.to_csv(r'C:\Users\lancer\cff\all_info.csv')
#near_neighbors(all_info,on_test,mall_info)
    
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
        
def second():
    all_info=pd.read_csv(r'C:\Users\lancer\cff\all_info.csv')
    test_with_neighbors=pd.read_csv(r'C:\Users\lancer\cff\neighbors.csv')
    final_shop_list=[]
    i=0
    j=test_with_neighbors.shape[0]
    for index,line in test_with_neighbors.iterrows():
        wifi_id,wifi_strong=wifi_string_to_list(line['wifi_infos'])
        neighbor_string=line['neighbors']
        neighbor=neighbor_string[1:-1].split()
        wifi_dict={}
        for node in neighbor:
            shop_name=all_info.at[int(node),'shop_id']
            wifi_infos=all_info.at[int(node),'wifi_infos']
            wifi_node_id,wifi_node_strong=wifi_string_to_list(wifi_infos)
            same=[val for val in wifi_id if (val in wifi_node_id)&(wifi_strong[wifi_id.index(val)]>=-80)]
            same_num=len(same)
            if shop_name in wifi_dict:
                wifi_dict[shop_name].append(same_num)
            else:
                wifi_dict[shop_name]=[same_num]
        shop_same=[]
        shop_same=[sum(val)/len(val) for val in wifi_dict.values()]   # 为什么不能用？
        final_shop_list.append(list(wifi_dict.keys())[shop_same.index(max(shop_same))])
        i=i+1;print(i,j)
    test_with_neighbors['shop_id']=final_shop_list
    test_with_neighbors[['row_id','shop_id']].to_csv(r'C:\Users\lancer\cff\result.csv',index=None)

second()