# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import time
import datetime
start=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
# path=r'E:\svm结构性断点论文\20180214\WTIData.xls'
path=r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\Data\Brent-11-15.xlsx'
resultsPath = r"C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\bishe\results\\"
df=pd.read_excel(path,sheet_name='Sheet1',header=None)

# 选择拟合函数的形式
def Poly2func(x,a,b,c):
    return a*np.power(x,2)+b*np.power(x,1)+c

def Poly3func(x,a,b,c,d):
    return a*np.power(x,3)+b*np.power(x,2)+c*np.power(x,1)+d

def Poly4func(x,a,b,c,d,e):
    return a*np.power(x,4)+b*np.power(x,3)+c*np.power(x,2)+d*np.power(x,1)+e

def Poly5func(x,a,b,c,d,e,f):
    return a*np.power(x,5)+b*np.power(x,4)+c*np.power(x,3)+d*np.power(x,2)+e*np.power(x,1)+f

def Poly6func(x,a,b,c,d,e,f,g):
    return a*np.power(x,6)+b*np.power(x,5)+c*np.power(x,4)+d*np.power(x,3)+e*np.power(x,2)\
           +f*np.power(x,1)+g

def Poly7func(x,a,b,c,d,e,f,g,h):
    return a*np.power(x,7)+b*np.power(x,6)+c*np.power(x,5)+d*np.power(x,4)+e*np.power(x,3)\
           +f*np.power(x,2)+g*np.power(x,1)+h

def Poly8func(x,a,b,c,d,e,f,g,h,i):
    return a*np.power(x,8)+b*np.power(x,7)+c*np.power(x,6)+d*np.power(x,5)+e*np.power(x,4)\
           +f*np.power(x,3)+g*np.power(x,2)+h*np.power(x,1)+i


# 获取多项式拟合的参数
def getPoly2Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x=np.array(df.loc[a:b,0])
    y=np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly2func,x,y)
    trend=list(trend)
    return trend

def getPoly3Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x=np.array(df.loc[a:b,0])
    y=np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly3func,x,y)
    trend=list(trend)
    return trend

def getPoly4Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x=np.array(df.loc[a:b,0])
    y=np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly4func,x,y)
    trend=list(trend)
    return trend

def getPoly5Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x=np.array(df.loc[a:b,0])
    y=np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly5func,x,y)
    trend=list(trend)
    return trend

def getPoly6Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x=np.array(df.loc[a:b,0])
    y=np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly6func,x,y)
    trend=list(trend)
    return trend

def getPoly7Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x=np.array(df.loc[a:b,0])
    y=np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly7func,x,y)
    trend=list(trend)
    return trend

def getPoly8Params(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x=np.array(df.loc[a:b,0])
    y=np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Poly8func,x,y)
    trend=list(trend)
    return trend

##############算各个模型的MSE######
def getPoly2MSE(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly2func(x,a,b,c) for i in x]
    temp = np.power(y2-y,2)
    n = np.sum(temp)
    return n/len(temp)

def getPoly3MSE(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly3func(x,a,b,c,d) for i in x]
    temp = np.power(y2-y,2)
    n = np.sum(temp)
    return n/len(temp)

def getPoly4MSE(i,j,a,b,c,d,e):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly4func(x,a,b,c,d,e) for i in x]
    temp = np.power(y2-y,2)
    n = np.sum(temp)
    return n/len(temp)


#########获取MAPE############
def getPoly2MAPE(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly2func(x,a,b,c) for i in x]
    temp = np.power(y2-y,2)/y
    return np.sum(temp)/len(temp)

def getPoly3MAPE(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly3func(x,a,b,c,d) for i in x]
    temp = np.power(y2-y,2)/y
    return np.sum(temp)/len(temp)

def getPoly4MAPE(i,j,a,b,c,d,e):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly4func(x,a,b,c,d,e) for i in x]
    temp = np.power(y2-y,2)/y
    return np.sum(temp)/len(temp)


#######计算各个函数的MAE########
def getPoly2MAD(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly2func(x,a,b,c) for i in x]
    temp = np.abs(y-np.abs(y2))
    return np.sum(temp)/len(temp)

def getPoly3MAD(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly3func(x,a,b,c,d) for i in x]
    temp = np.abs(y-np.abs(y2))
    return np.sum(temp)/len(temp)

def getPoly4MAD(i,j,a,b,c,d,e):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Poly4func(x,a,b,c,d,e) for i in x]
    temp = np.abs(y-np.abs(y2))
    return np.sum(temp)/len(temp)


######预测函数#######
def predictPoly2(i,j,a,b,c):
    #i,j是数据起始点
    #a,b,c是二次函数参数
    x = np.array(df.loc[i:j,0])
    result = [Poly2func(k,a,b,c) for k in x]
    return result

def predictPoly3(i,j,a,b,c,d):
    #i,j是数据起始点
    #a,b,c是二次函数参数
    x = np.array(df.loc[i:j,0])
    result = [Poly3func(k,a,b,c,d) for k in x]
    return result

def predictPoly4(i,j,a,b,c,d,e):
    #i,j是数据起始点
    #a,b,c是二次函数参数
    x = np.array(df.loc[i:j,0])
    result = [Poly4func(k,a,b,c,d,e) for k in x]
    return result

######获取残差#######
def getResidual(a,b,y):
    #a,b是原始时间序列的起始点
    #y是预测结果
    x = np.array(df.loc[a:b,1])
    return x-y


##########计算各个模型的BIC#########
#如果你的趋势向量没有提取出足够信息的话，每一个残差还是不是相互独立的就是个问题
#还能用极大似然吗？


if __name__ == '__main__':
    step = 5
    dataList=[]
    FuncDic = {'Poly2':getPoly2Params,'Poly3':getPoly3Params,'Poly4':getPoly4Params}
    MSEfunc = {'Poly2':getPoly2MSE,'Poly3':getPoly3MSE,'Poly4':getPoly4MSE}
    MAPEfunc = {'Poly2':getPoly2MAPE,'Poly3':getPoly3MAPE,'Poly4':getPoly4MAPE}
    MADfunc = {'Poly2':getPoly2MAD,'Poly3':getPoly3MAD,'Poly4':getPoly4MAD}
    ErrorCount = {'Poly2':0,'Poly3':0,'Poly4':0}
    for i in range(1, df.shape[0] - step):
        MSE = {'Poly2': 10000, 'Poly3': 10000, 'Poly4':10000}
        MAPE = {'Poly2': 10000, 'Poly3': 10000, 'Poly4':10000}
        MAD = {'Poly2': 10000, 'Poly3': 10000, 'Poly4':10000}
        Error = {'Poly2':0,'Poly3':0,'Poly4':0}
        Trend = {'Poly2':[],'Poly3':[],'Poly4':[]}
        Residual = {'Poly2':[],'Poly3':[],'Poly4':[]}
        for key in FuncDic:
            try:
                trend = FuncDic[key](i, i + step - 1)
                print('外循环', i, ',内循环', key, '标签已获取！')
                if key == 'Poly2':
                    MSE[key] = MSEfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    MAPE[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    MAD[key] = MADfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    Residual[key] = getResidual(i,i+step-1,predictPoly2(i,i+step-1,trend[0],trend[1],trend[2]))
                    trend = [0,0]+trend
                elif key == 'Poly3':
                    MSE[key] = MSEfunc[key](i,i+step-1,trend[0],trend[1],trend[2],trend[3])
                    MAPE[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1],trend[2],trend[3])
                    MAD[key] = MADfunc[key](i,i+step-1,trend[0],trend[1],trend[2],trend[3])
                    Residual[key] = getResidual(i,i+step-1,predictPoly3(i,i+step-1,trend[0],trend[1],trend[2],trend[3]))
                    trend = [0]+trend
                elif key == 'Poly4':
                    MSE[key] = MSEfunc[key](i,i+step-1,trend[0],trend[1],trend[2],trend[3],trend[4])
                    MAPE[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1],trend[2],trend[3],trend[4])
                    MAD[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1],trend[2],trend[3],trend[4])
                    Residual[key] = getResidual(i,i+step-1,predictPoly4(i,i+step-1,trend[0],trend[1],trend[2],trend[3],trend[4]))


                Trend[key] = trend
                # print(MSE[key])
                if i+step-1>df.shape[0]:
                    iEnd = df.shape[0]
                else:
                    iEnd = i+step-1
            except RuntimeError as e:
                Error[key] = 1
                ErrorCount[key] += 1
                print('外循环',i,'内循环',key,'错误',e)
        # print(MSE)
        data = {
            'istart': i,
            'iend': i+step - 1,
            'trend':Trend,
            'Error':Error,
            'MSE': MSE,
            'MAD': MAD,
            'MAPE': MAPE,
            'residual':Residual,
            'MSE_FinalFunc': min(MSE,key=MSE.get),
            'MSE_FinalTrend': Trend[min(MSE,key=MSE.get)],
            'MAD_FinalFunc': min(MAD,key=MAD.get),
            'MAD_FinalTrend': Trend[min(MAD, key =MAD.get)],
            'MAPE_FinalFunc': min(MAPE,key=MAPE.get),
            'MAPE_FinalTrend': Trend[min(MAPE, key=MAPE.get)]
        }
        dataList.append(data)

    istart = []
    iend = []
    MSEfinalFunc = []    #用来存经过选择之后最后的函数类型
    MADfinalFunc = []
    MAPEfinalFunc = []
    optimalMSE = []
    optimalMAD = []
    optimalMAPE = []
    MSEtrenda = []
    MSEtrendb = []
    MSEtrendc = []
    MSEtrendd = []
    MSEtrende = []
    MADtrenda = []
    MADtrendb = []
    MADtrendc = []
    MADtrendd = []
    MADtrende = []
    MAPEtrenda = []
    MAPEtrendb = []
    MAPEtrendc = []
    MAPEtrendd = []
    MAPEtrende = []
    ErrorPoly2 = []
    ErrorPoly3 = []
    ErrorPoly4 = []
    Poly2Residual0 = []
    Poly2Residual1 = []
    Poly2Residual2 = []
    Poly2Residual3 = []
    Poly2Residual4 = []
    Poly3Residual0 = []
    Poly3Residual1 = []
    Poly3Residual2 = []
    Poly3Residual3 = []
    Poly3Residual4 = []
    Poly4Residual0 = []
    Poly4Residual1 = []
    Poly4Residual2 = []
    Poly4Residual3 = []
    Poly4Residual4 = []


    for i in range(len(dataList)):
        istart.append(dataList[i].get('istart'))
        iend.append(dataList[i].get('iend'))
        MSEfinalFunc.append(dataList[i].get('MSE_FinalFunc'))
        MADfinalFunc.append(dataList[i].get('MAD_FinalFunc'))
        MAPEfinalFunc.append(dataList[i].get('MAPE_FinalFunc'))
        optimalMSE.append(dataList[i].get('MSE')[min(MSE,key=MSE.get)])
        optimalMAD.append(dataList[i].get('MAD')[min(MAD,key=MAD.get)])
        optimalMAPE.append(dataList[i].get('MAPE')[min(MAPE,key=MAPE.get)])
        ErrorPoly2.append(dataList[i].get('Error')['Poly2'])
        ErrorPoly3.append(dataList[i].get('Error')['Poly3'])
        ErrorPoly4.append(dataList[i].get('Error')['Poly4'])
        MSEtrenda.append(dataList[i].get('MSE_FinalTrend')[0])
        MSEtrendb.append(dataList[i].get('MSE_FinalTrend')[1])
        MSEtrendc.append(dataList[i].get('MSE_FinalTrend')[2])
        MSEtrendd.append(dataList[i].get('MSE_FinalTrend')[3])
        MSEtrende.append(dataList[i].get('MSE_FinalTrend')[4])
        MADtrenda.append(dataList[i].get('MAD_FinalTrend')[0])
        MADtrendb.append(dataList[i].get('MAD_FinalTrend')[1])
        MADtrendc.append(dataList[i].get('MAD_FinalTrend')[2])
        MADtrendd.append(dataList[i].get('MAD_FinalTrend')[3])
        MADtrende.append(dataList[i].get('MAD_FinalTrend')[4])
        MAPEtrenda.append(dataList[i].get('MAPE_FinalTrend')[0])
        MAPEtrendb.append(dataList[i].get('MAPE_FinalTrend')[1])
        MAPEtrendc.append(dataList[i].get('MAPE_FinalTrend')[2])
        MAPEtrendd.append(dataList[i].get('MAPE_FinalTrend')[3])
        MAPEtrende.append(dataList[i].get('MAPE_FinalTrend')[4])
        Poly2Residual0.append(dataList[i].get('residual')['Poly2'][0])
        Poly2Residual1.append(dataList[i].get('residual')['Poly2'][1])
        Poly2Residual2.append(dataList[i].get('residual')['Poly2'][2])
        Poly2Residual3.append(dataList[i].get('residual')['Poly2'][3])
        Poly2Residual4.append(dataList[i].get('residual')['Poly2'][4])
        Poly3Residual0.append(dataList[i].get('residual')['Poly3'][0])
        Poly3Residual1.append(dataList[i].get('residual')['Poly3'][1])
        Poly3Residual2.append(dataList[i].get('residual')['Poly3'][2])
        Poly3Residual3.append(dataList[i].get('residual')['Poly3'][3])
        Poly3Residual4.append(dataList[i].get('residual')['Poly3'][4])
        Poly4Residual0.append(dataList[i].get('residual')['Poly4'][0])
        Poly4Residual1.append(dataList[i].get('residual')['Poly4'][1])
        Poly4Residual2.append(dataList[i].get('residual')['Poly4'][2])
        Poly4Residual3.append(dataList[i].get('residual')['Poly4'][3])
        Poly4Residual4.append(dataList[i].get('residual')['Poly4'][4])



        print('第', i + 1, '行已成功写入！')

    #定义系数的输出
    dataframe_params = pd.DataFrame({
        'istart':istart ,
        'iend':iend,
        'MSE_FinalFunc':MSEfinalFunc,
        'MSEtrenda':MSEtrenda,
        'MSEtrendb':MSEtrendb,
        'MSEtrendc':MSEtrendc,
        'MSEtrendd':MSEtrendd,
        'MSEtrende':MSEtrende,
        'MAD_FinalFunc':MADfinalFunc,
        'MADtrenda': MADtrenda,
        'MADtrendb':MADtrendb,
        'MADtrendc':MADtrendc,
        'MADtrendd':MADtrendd,
        'MADtrende':MADtrende,
        'MAPE_FinalFunc':MAPEfinalFunc,
        'MAPEtrenda':MAPEtrenda,
        'MAPEtrendb':MAPEtrendb,
        'MAPEtrendc':MAPEtrendc,
        'MAPEtrendd':MAPEtrendd,
        'MAPEtrende':MAPEtrende
    })
    dataframe_params.to_csv(resultsPath+'Poly''-'+str(step)+'-Params'+'-'+str(start)[0:10]+".csv")

    #定义MSE的输出
    dataframe_Evaluation = pd.DataFrame({
        'istart':istart,
        'iend':iend,
        'MSE_FinalFunc':MSEfinalFunc,
        'optimalMSE':optimalMSE,
        'MAD_FinalFunc':MADfinalFunc,
        'optimalMAD':optimalMAD,
        'MAPE_FinalFunc':MAPEfinalFunc,
        'optimalMAPE':optimalMAPE,
        'ErrorPoly2':ErrorPoly2,
        'ErrorPoly3':ErrorPoly3,
        'ErrorPoly4':ErrorPoly4
    })
    dataframe_Evaluation.to_csv(resultsPath+'Poly'+'-'+str(step)+'-Evaluation'+'-'+str(start)[0:10]+".csv")

    dataframe_Residual = pd.DataFrame({
        'istart': istart,
        'iend': iend,
        'Poly2Residual0':Poly2Residual0,
        'Poly2Residual1':Poly2Residual1,
        'Poly2Residual2':Poly2Residual2,
        'Poly2Residual3':Poly2Residual3,
        'Poly2Residual4':Poly2Residual4,
        'Poly3Residual0':Poly3Residual0,
        'Poly3Residual1':Poly3Residual1,
        'Poly3Residual2':Poly3Residual2,
        'Poly3Residual3':Poly3Residual3,
        'Poly3Residual4':Poly3Residual4,
        'Poly4Residual0':Poly4Residual0,
        'Poly4Residual1':Poly4Residual1,
        'Poly4Residual2':Poly4Residual2,
        'Poly4Residual3':Poly4Residual3,
        'Poly4Residual4':Poly4Residual4
    })
    dataframe_Residual.to_csv(resultsPath+'Poly'+'-'+str(step)+'-Residual'+'-'+str(start)[0:10]+".csv")


end=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
print(ErrorCount)
print('本次程序运行时长:',end-start)