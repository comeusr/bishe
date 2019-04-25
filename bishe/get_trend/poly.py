# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import time
import datetime
start=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
# path=r'E:\svm结构性断点论文\20180214\WTIData.xls'
path=r'..\Data\Brent-10-13.xlsx'
df=pd.read_excel(path,sheet_name='Sheet1',header=None)
# print(df)
# print(df.loc[:,2])
# print(df.loc[6926,])
# 选择拟合函数的形式
def func(x,a,b,c,d,e):
    #return a * np.exp(-b * x) + c  # 指数函数
    #return a*np.cos(b*x+c)+d # 正弦函数
    return a*np.power(x,4)+b*np.power(x,3)+c*np.power(x,2)+d*np.power(x,1)+e
# 获取多项式拟合的参数
def getPolyParams(a,b):
    x=np.array(df.loc[a:b,0].astype('float64'))
    y=np.array(df.loc[a:b,1].astype('float64'))
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(func,x,y)
    trend=list(trend)
    return trend
if __name__ == '__main__':
    step=22
    dataList=[]
    for i in range(1,df.shape[0]-step):
        for j in range(i,df.shape[0]-step,step):
        #if(i==10 and j==10):
        #    print('aaa')
            try:
                trend=getPolyParams(j,j+step-1)
                print('外循环', i, ',内循环', j, '标签已获取！')
                if j+step-1>df.shape[0]:
                    jEnd=df.shape[0]
                else:
                    jEnd=j+step-1
                data={
                    'iStart':i,
                    'jStart':j,
                    'jEnd':jEnd,
                    'trend':trend
                }
                dataList.append(data)
            except RuntimeError as e:
                print('外循环', i, ',内循环', j,'错误:',e)

    iStart=[]
    jStart=[]
    jEnd=[]
    trenda=[]
    trendb=[]
    trendc=[]
    trendd=[]
    trende=[]
    for i in range(len(dataList)):
        iStart.append(dataList[i].get('iStart'))
        jStart.append(dataList[i].get('jStart'))
        jEnd.append(dataList[i].get('jEnd'))
        trenda.append(dataList[i].get('trend')[0])
        trendb.append(dataList[i].get('trend')[1])
        trendc.append(dataList[i].get('trend')[2])
        trendd.append(dataList[i].get('trend')[3])
        trende.append(dataList[i].get('trend')[4])
        print('第', i + 1, '行已成功写入！')
    dataframe = pd.DataFrame({'iStart':iStart,'jStart':jStart,'jEnd':jEnd,'trenda':trenda,'trendb':trendb,'trendc':trendc,'trendd':trendd,'trende':trende})
    # dataframe.to_csv("E:\\svm结构性断点论文\\20180214\\WTI月度\\result-four"+str(step)+str(start)[0:10]+".csv")
    dataframe.to_csv("result-four"+str(step)+str(start)[0:10]+".csv")


    end=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
    print('本次程序运行时长:',end-start)