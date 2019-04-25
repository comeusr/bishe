# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import time
import datetime
from sklearn import linear_model
start=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
# path=r'E:\svm结构性断点论文\20180214\WTIData.xls'
path=r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\Data\Brent-11-15.xlsx'
resultsPath = r"C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\bishe\results\\"
df=pd.read_excel(path,sheet_name='Sheet1',header=None)

# 选择拟合函数的形式
# def Poly2func(x,a,b,c):
#     return a*np.power(x,2)+b*np.power(x,1)+c
#
def Poly3func(x,a,b,c,d):
    return d*np.power(x,3)+c*np.power(x,2)+b*np.power(x,1)+a
#
def Poly4func(x,a,b,c,d,e):
    return e*np.power(x,4)+d*np.power(x,3)+c*np.power(x,2)+b*np.power(x,1)+a
#
# def Poly5func(x,a,b,c,d,e,f):
#     return a*np.power(x,5)+b*np.power(x,4)+c*np.power(x,3)+d*np.power(x,2)+e*np.power(x,1)+f
#
# def Poly6func(x,a,b,c,d,e,f,g):
#     return a*np.power(x,6)+b*np.power(x,5)+c*np.power(x,4)+d*np.power(x,3)+e*np.power(x,2)\
#            +f*np.power(x,1)+g
#
# def Poly7func(x,a,b,c,d,e,f,g,h):
#     return a*np.power(x,7)+b*np.power(x,6)+c*np.power(x,5)+d*np.power(x,4)+e*np.power(x,3)\
#            +f*np.power(x,2)+g*np.power(x,1)+h
#
# def Poly8func(x,a,b,c,d,e,f,g,h,i):
#     return a*np.power(x,8)+b*np.power(x,7)+c*np.power(x,6)+d*np.power(x,5)+e*np.power(x,4)\
#            +f*np.power(x,3)+g*np.power(x,2)+h*np.power(x,1)+i


# 获取多项式拟合的参数
# def getPoly2Params(a,b):
#     # x=np.array(df.loc[a:b,0].astype('float64'))
#     # y=np.array(df.loc[a:b,1].astype('float64'))
#     x=np.array(df.loc[a:b,0])
#     y=np.array(df.loc[a:b,1])
#     #trend=np.polyfit(x,y,4)
#     trend,pocv=curve_fit(Poly2func,x,y)
#     trend=list(trend)
#     return trend
#
# def getPoly3Params(a,b):
#     # x=np.array(df.loc[a:b,0].astype('float64'))
#     # y=np.array(df.loc[a:b,1].astype('float64'))
#     x=np.array(df.loc[a:b,0])
#     y=np.array(df.loc[a:b,1])
#     #trend=np.polyfit(x,y,4)
#     trend,pocv=curve_fit(Poly3func,x,y)
#     trend=list(trend)
#     return trend
#
# def getPoly4Params(a,b):
#     # x=np.array(df.loc[a:b,0].astype('float64'))
#     # y=np.array(df.loc[a:b,1].astype('float64'))
#     x=np.array(df.loc[a:b,0])
#     y=np.array(df.loc[a:b,1])
#     #trend=np.polyfit(x,y,4)
#     trend,pocv=curve_fit(Poly4func,x,y)
#     trend=list(trend)
#     return trend
#
# def getPoly5Params(a,b):
#     # x=np.array(df.loc[a:b,0].astype('float64'))
#     # y=np.array(df.loc[a:b,1].astype('float64'))
#     x=np.array(df.loc[a:b,0])
#     y=np.array(df.loc[a:b,1])
#     #trend=np.polyfit(x,y,4)
#     trend,pocv=curve_fit(Poly5func,x,y)
#     trend=list(trend)
#     return trend
#
# def getPoly6Params(a,b):
#     # x=np.array(df.loc[a:b,0].astype('float64'))
#     # y=np.array(df.loc[a:b,1].astype('float64'))
#     x=np.array(df.loc[a:b,0])
#     y=np.array(df.loc[a:b,1])
#     #trend=np.polyfit(x,y,4)
#     trend,pocv=curve_fit(Poly6func,x,y)
#     trend=list(trend)
#     return trend
#
# def getPoly7Params(a,b):
#     # x=np.array(df.loc[a:b,0].astype('float64'))
#     # y=np.array(df.loc[a:b,1].astype('float64'))
#     x=np.array(df.loc[a:b,0])
#     y=np.array(df.loc[a:b,1])
#     #trend=np.polyfit(x,y,4)
#     trend,pocv=curve_fit(Poly7func,x,y)
#     trend=list(trend)
#     return trend
#
# def getPoly8Params(a,b):
#     # x=np.array(df.loc[a:b,0].astype('float64'))
#     # y=np.array(df.loc[a:b,1].astype('float64'))
#     x=np.array(df.loc[a:b,0])
#     y=np.array(df.loc[a:b,1])
#     #trend=np.polyfit(x,y,4)
#     trend,pocv=curve_fit(Poly8func,x,y)
#     trend=list(trend)
#     return trend

##############算各个模型的MSE######
# def getPoly2MSE(i,j,a,b,c):
#     x = np.array(df.loc[i:j,0])
#     y = np.array(df.loc[i:j,1])
#     y2 = [Poly2func(x,a,b,c) for i in x]
#     temp = np.power(y2-y,2)
#     n = np.sum(temp)
#     return n/len(temp)
#
def getPoly3MSE(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly3func(x,a,b,c,d) for i in x]
    temp = np.power(y2-y,2)
    n = np.sum(temp)
    return n/len(temp)
#
def getPoly4MSE(i,j,a,b,c,d,e):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly4func(x,a,b,c,d,e) for i in x]
    temp = np.power(y2-y,2)
    n = np.sum(temp)
    return n/len(temp)
#
#
# #########获取MAPE############
# def getPoly2MAPE(i,j,a,b,c):
#     x = np.array(df.loc[i:j,0])
#     y = np.array(df.loc[i:j,1])
#     y2 = [Poly2func(x,a,b,c) for i in x]
#     temp = np.power(y2-y,2)/y
#     return np.sum(temp)/len(temp)
#
def getPoly3MAPE(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly3func(x,a,b,c,d) for i in x]
    temp = np.power(y2-y,2)/y
    return np.sum(temp)/len(temp)
#
def getPoly4MAPE(i,j,a,b,c,d,e):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly4func(x,a,b,c,d,e) for i in x]
    temp = np.power(y2-y,2)/y
    return np.sum(temp)/len(temp)
#
#
# #######计算各个函数的MAE########
# def getPoly2MAD(i,j,a,b,c):
#     x = np.array(df.loc[i:j,0])
#     y = np.array(df.loc[i:j,1])
#     y2 = [Poly2func(x,a,b,c) for i in x]
#     temp = np.abs(y-np.abs(y2))
#     return np.sum(temp)/len(temp)
#
def getPoly3MAD(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly3func(x,a,b,c,d) for i in x]
    temp = np.abs(y-np.abs(y2))
    return np.sum(temp)/len(temp)
#
def getPoly4MAD(i,j,a,b,c,d,e):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly4func(x,a,b,c,d,e) for i in x]
    temp = np.abs(y-np.abs(y2))
    return np.sum(temp)/len(temp)
#
#
# ######预测函数#######
# def predictPoly2(i,j,a,b,c):
#     #i,j是数据起始点
#     #a,b,c是二次函数参数
#     x = np.array(df.loc[i:j,0])
#     result = [Poly2func(k,a,b,c) for k in x]
#     return result
#
# def predictPoly3(i,j,a,b,c,d):
#     #i,j是数据起始点
#     #a,b,c是二次函数参数
#     x = np.array(df.loc[i:j,0])
#     result = [Poly3func(k,a,b,c,d) for k in x]
#     return result
#
# def predictPoly4(i,j,a,b,c,d,e):
#     #i,j是数据起始点
#     #a,b,c是二次函数参数
#     x = np.array(df.loc[i:j,0])
#     result = [Poly4func(k,a,b,c,d,e) for k in x]
#     return result

######获取残差#######

def getResidual(a,b,y):
    #a,b是原始时间序列的起始点
    #y是预测结果
    x = np.array(df.loc[a:b,1])
    return x-y

#############定义Lasso模型###############
alpha = 0.01
clf = linear_model.Lasso()

############生成一个用于训练的时间向量########
def Time1(step):
    #output = [[1,1,1,1],[2,4,8,16],[3,9,27,81],[4,16,64,256],[5,25,125,625]]
    output = []
    for i in range(step):
        temp = []
        for j in range(1,step):
            k = np.power((i+1),j)
            temp.append(k)
        output.append(temp)
    return output

def Time2(step,a,b):
    #output  = [[2, 4, 8, 16],[3, 9, 27, 81],[4, 16, 64, 256],[5, 25, 125, 625],[6, 36, 216, 1296]]
    output  = []
    x = np.array(df.loc[a:b, 0])
    for k in x:
        temp = []
        for j in range(1,step):
            temp.append(np.power(k,j))
        output.append(temp)
    return output

if __name__ == '__main__':
    step = 5
    t  = Time1(5)
    dataList=[]
    ErrorCount = 0
    for i in range(1, df.shape[0] - step):
        MSE = 0
        MAPE  = 0
        MAD = 0
        Residual = []
        try:
            y = np.array(df.loc[i:i+step-1,1]).tolist()
            clf.fit(t,y)
            trend = clf.coef_.tolist()
            print('外循环', i, '标签已获取！')
            MSE = getPoly3MSE(i,i+step-1,trend[0],trend[1],trend[2],trend[3])
            MAPE = getPoly3MAPE(i,i+step-1,trend[0],trend[1],trend[2],trend[3])
            MAD = getPoly3MAD(i,i+step-1,trend[0],trend[1],trend[2],trend[3])

            # print(MSE[key])
            if i+step-1>df.shape[0]:
                iEnd = df.shape[0]
            else:
                iEnd = i+step-1
        except RuntimeError as e:
            ErrorCount += 1
            print('外循环',i,'错误',e)
        # print(MSE)
        data = {
            'istart': i,
            'iend': i+step - 1,
            'trend':trend,
            'MSE': MSE,
            'MAD': MAD,
            'MAPE': MAPE,
            # 'residual':Residual,
        }
        dataList.append(data)

    istart = []
    iend = []
    trenda = []
    trendb = []
    trendc = []
    trendd = []
    MSE = []
    MAD = []
    MAPE = []


    for i in range(len(dataList)):
        istart.append(dataList[i].get('istart'))
        iend.append(dataList[i].get('iend'))
        trenda.append(dataList[i].get('trend')[0])
        trendb.append(dataList[i].get('trend')[1])
        trendc.append(dataList[i].get('trend')[2])
        trendd.append(dataList[i].get('trend')[3])
        MSE.append(dataList[i].get('MSE'))
        MAD.append(dataList[i].get('MAD'))
        MAPE.append(dataList[i].get('MAPe'))



        print('第', i + 1, '行已成功写入！')

    #定义系数的输出
    dataframe_params = pd.DataFrame({
        'istart':istart ,
        'iend':iend,
        'trenda':trenda,
        'trendb':trendb,
        'trendc':trendc,
        'trendd':trendd,
        'MSE':MSE,
        'MAPE':MAPE,
        'MAD':MAD

    })
    dataframe_params.to_csv(resultsPath+'LassoPoly''-'+str(step)+'-Params'+'-'+str(start)[0:10]+".csv")

    #定义MSE的输出
    # dataframe_Evaluation = pd.DataFrame({
    #     'istart':istart,
    #     'iend':iend,
    #     'MSE_FinalFunc':MSEfinalFunc,
    #     'optimalMSE':optimalMSE,
    #     'MAD_FinalFunc':MADfinalFunc,
    #     'optimalMAD':optimalMAD,
    #     'MAPE_FinalFunc':MAPEfinalFunc,
    #     'optimalMAPE':optimalMAPE,
    #     'ErrorPoly2':ErrorPoly2,
    #     'ErrorPoly3':ErrorPoly3,
    #     'ErrorPoly4':ErrorPoly4
    # })
    # dataframe_Evaluation.to_csv(resultsPath+'Poly'+'-'+str(step)+'-Evaluation'+'-'+str(start)[0:10]+".csv")

    # dataframe_Residual = pd.DataFrame({
    #     'istart': istart,
    #     'iend': iend,
    #     'Poly2Residual0':Poly2Residual0,
    #     'Poly2Residual1':Poly2Residual1,
    #     'Poly2Residual2':Poly2Residual2,
    #     'Poly2Residual3':Poly2Residual3,
    #     'Poly2Residual4':Poly2Residual4,
    #     'Poly3Residual0':Poly3Residual0,
    #     'Poly3Residual1':Poly3Residual1,
    #     'Poly3Residual2':Poly3Residual2,
    #     'Poly3Residual3':Poly3Residual3,
    #     'Poly3Residual4':Poly3Residual4,
    #     'Poly4Residual0':Poly4Residual0,
    #     'Poly4Residual1':Poly4Residual1,
    #     'Poly4Residual2':Poly4Residual2,
    #     'Poly4Residual3':Poly4Residual3,
    #     'Poly4Residual4':Poly4Residual4
    # })
    # dataframe_Residual.to_csv(resultsPath+'Poly'+'-'+str(step)+'-Residual'+'-'+str(start)[0:10]+".csv")


end=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
print(ErrorCount)
print('本次程序运行时长:',end-start)