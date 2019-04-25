# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import time
import datetime
start=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
# path=r'E:\svm结构性断点论文\20180214\WTIData.xls'
path=r'..\Data\Brent-11-15.xlsx'
df=pd.read_excel(path,sheet_name='Sheet1',header=None)

# 选择拟合函数的形式
def Polyfunc(x,a,b,c):
    #return a * np.exp(-b * x) + c  # 指数函数
    #return a*np.cos(b*x+c)+d # 正弦函数
    # return a*np.power(x,4)+b*np.power(x,3)+c*np.power(x,2)+d*np.power(x,1)+e
    return a*np.power(x,2)+b*np.power(x,1)+c


def Exponentialfunc(x,a,b,c):
    return a*np.exp(-b*x)+c

def Lnfunc(x,a,b):
    return a*np.log(x)+b

def SinFunc(x,a,b,c,d):
    return a*np.sin(b*x+c)+d

# 获取多项式拟合的参数
def getPolyParams(a,b):
    # x=np.array(df.loc[a:b,0].astype('float64'))
    # y=np.array(df.loc[a:b,1].astype('float64'))
    x=np.array(df.loc[a:b,0])
    y=np.array(df.loc[a:b,1])
    #trend=np.polyfit(x,y,4)
    trend,pocv=curve_fit(Polyfunc,x,y)
    trend=list(trend)
    return trend

#获取指数函数拟合的参数
def getExponentialParams(a,b):
    x = np.array(df.loc[a:b,0])
    y = np.array(df.loc[a:b,1])
    trend,pocv=curve_fit(Exponentialfunc,x,y)
    trend=list(trend)
    return trend

#获取对数函数拟合的参数
def getLnParams(a,b):
    x = np.array(df.loc[a:b,0])
    y = np.array(df.loc[a:b,1])
    trend,pocv=curve_fit(Lnfunc,x,y)
    trend=list(trend)
    return trend

#获取三角函数拟合的参数
def getSinParams(a,b):
    x = np.array(df.loc[a:b,0])
    y = np.array(df.loc[a:b,1])
    trend,pocv=curve_fit(SinFunc,x,y)
    trend=list(trend)
    return trend


#算各个模型的MSE
def getPolyMSE(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Polyfunc(x,a,b,c) for i in x]
    temp = np.power(y2-y,2)
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)

def getExpoMSE(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Exponentialfunc(x,a,b,c) for i in x]
    temp = np.power(y2-y,2)
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)

def getLnMSE(i,j,a,b):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Lnfunc(x,a,b) for i in x]
    temp = np.power(y2 - y, 2)
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/ len(temp)

def getSinMSE(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [SinFunc(x,a,b,c,d) for i in x]
    temp = np.power(y2 - y, 2)
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/ len(temp)

#########获取MAPE############
def getPolyMAPE(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Polyfunc(x,a,b,c) for i in x]
    temp = np.power(y2-y,2)/y
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)

def getExpoMAPE(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Exponentialfunc(x,a,b,c) for i in x]
    temp = np.power(y2-y,2)/y
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)

def getLnMAPE(i,j,a,b):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Lnfunc(x,a,b) for i in x]
    temp = np.power(y2-y,2)/y
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)

def getSinMAPE(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [SinFunc(x,a,b,c,d) for i in x]
    temp = np.power(y2-y,2)/y
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)


#######计算各个函数的MAE########
def getPolyMAD(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Polyfunc(x,a,b,c) for i in x]
    temp = np.abs(y-np.abs(y2))
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)

def getExpoMAD(i,j,a,b,c):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Exponentialfunc(x,a,b,c) for i in x]
    temp = np.abs(y-np.abs(y2))
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)

def getLnMAD(i,j,a,b):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [Lnfunc(x,a,b) for i in x]
    temp = np.abs(y-np.abs(y2))
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)

def getSinMAD(i,j,a,b,c,d):
    x = np.array(df.loc[i:j,0])
    y = np.array(df.loc[i:j,1])
    y2 = [SinFunc(x,a,b,c,d) for i in x]
    temp = np.abs(y-np.abs(y2))
    temp = list(temp)
    # print(type(temp))
    return sum(sum(temp))/len(temp)


#####获得残差#####
def getResidual(x,y):
    return x-y

##########计算各个模型的BIC#########
#如果你的趋势向量没有提取出足够信息的话，每一个残差还是不是相互独立的就是个问题
#还能用极大似然吗？




if __name__ == '__main__':
    step = 5
    dataList=[]
    FuncDic = {'Poly':getPolyParams,'Expo':getExponentialParams,'Ln':getLnParams,'Sin':getSinParams}
    MSEfunc = {'Poly':getPolyMSE,'Expo':getExpoMSE,'Ln':getLnMSE,'Sin':getSinMSE}
    MAPEfunc = {'Poly':getPolyMAPE,'Expo':getExpoMAPE,'Ln':getLnMAPE,'Sin':getSinMAPE}
    MADfunc = {'Poly':getPolyMAD,'Expo':getExpoMAD,'Ln':getLnMAD,'Sin':getSinMAD}
    ErrorCount = {'Poly':0,'Expo':0,'Ln':0,'Sin':0}
    for i in range(1, df.shape[0] - step):
        MSE = {'Poly': 10000, 'Expo': 10000, 'Ln':10000,'Sin':10000}
        MAPE = {'Poly': 10000, 'Expo': 10000, 'Ln':10000,'Sin':10000}
        MAD = {'Poly': 10000, 'Expo': 10000, 'Ln':10000,'Sin':10000}
        Error = {'Poly':0,'Expo':0,'Ln':0,'Sin':0}
        Trend = {'Poly':[],'Expo':[],'Ln':[],'Sin':[]}
        for key in FuncDic:
            try:
                trend = FuncDic[key](i, i + step - 1)
                print('外循环', i, ',内循环', key, '标签已获取！')
                if key == 'Poly':
                    MSE[key] = MSEfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    MAPE[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    MAD[key] = MADfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    trend.append(0)
                elif key == 'Expo':
                    MSE[key] = MSEfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    MAPE[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    MAD[key] = MADfunc[key](i,i+step-1,trend[0],trend[1],trend[2])
                    trend.append(0)
                elif key == 'Ln':
                    MSE[key] = MSEfunc[key](i,i+step-1,trend[0],trend[1])
                    MAPE[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1])
                    MAD[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1])
                    trend.append(0)
                    trend.append(0)
                elif key == 'Sin':
                    MSE[key] = MSEfunc[key](i,i+step-1,trend[0], trend[1], trend[2],trend[3])
                    MAPE[key] = MAPEfunc[key](i,i+step-1,trend[0],trend[1],trend[2],trend[3])
                    MAD[key] = MADfunc[key](i,i+step-1,trend[0],trend[1],trend[2],trend[3])
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
    MADtrenda = []
    MADtrendb = []
    MADtrendc = []
    MADtrendd = []
    MAPEtrenda = []
    MAPEtrendb = []
    MAPEtrendc = []
    MAPEtrendd = []
    ErrorPoly = []
    ErrorExpo = []
    ErrorLn = []
    ErrorSin = []

    for i in range(len(dataList)):
        istart.append(dataList[i].get('istart'))
        iend.append(dataList[i].get('iend'))
        MSEfinalFunc.append(dataList[i].get('MSE_FinalFunc'))
        MADfinalFunc.append(dataList[i].get('MAD_FinalFunc'))
        MAPEfinalFunc.append(dataList[i].get('MAPE_FinalFunc'))
        optimalMSE.append(dataList[i].get('MSE')[min(MSE,key=MSE.get)])
        optimalMAD.append(dataList[i].get('MAD')[min(MAD,key=MAD.get)])
        optimalMAPE.append(dataList[i].get('MAPE')[min(MAPE,key=MAPE.get)])
        ErrorPoly.append(dataList[i].get('Error')['Poly'])
        ErrorExpo.append(dataList[i].get('Error')['Expo'])
        ErrorLn.append(dataList[i].get('Error')['Ln'])
        ErrorSin.append(dataList[i].get('Error')['Sin'])
        MSEtrenda.append(dataList[i].get('MSE_FinalTrend')[0])
        MSEtrendb.append(dataList[i].get('MSE_FinalTrend')[1])
        MSEtrendc.append(dataList[i].get('MSE_FinalTrend')[2])
        MSEtrendd.append(dataList[i].get('MSE_FinalTrend')[3])
        MADtrenda.append(dataList[i].get('MAD_FinalTrend')[0])
        MADtrendb.append(dataList[i].get('MAD_FinalTrend')[1])
        MADtrendc.append(dataList[i].get('MAD_FinalTrend')[2])
        MADtrendd.append(dataList[i].get('MAD_FinalTrend')[3])
        MAPEtrenda.append(dataList[i].get('MAPE_FinalTrend')[0])
        MAPEtrendb.append(dataList[i].get('MAPE_FinalTrend')[1])
        MAPEtrendc.append(dataList[i].get('MAPE_FinalTrend')[2])
        MAPEtrendd.append(dataList[i].get('MAPE_FinalTrend')[3])


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
        'MAD_FinalFunc':MADfinalFunc,
        'MADtrenda': MADtrenda,
        'MADtrendb':MADtrendb,
        'MADtrendc':MADtrendc,
        'MADtrendd':MADtrendd,
        'MAPE_FinalFunc':MAPEfinalFunc,
        'MAPEtrenda':MAPEtrenda,
        'MAPEtrendb':MAPEtrendb,
        'MAPEtrendc':MAPEtrendc,
        'MAPEtrendd':MAPEtrendd,
    })
    dataframe_params.to_csv("C:\\Users\\Ziyi Wang\\Desktop\\bishe\\realize wangyi\\bishe\\results\\revisedResult"+'-'+str(step)+'-Params'+'-'+str(start)[0:10]+".csv")

    #定义MSE的输出
    dataframe_MSE = pd.DataFrame({
        'istart':istart,
        'iend':iend,
        'MSE_FinalFunc':MSEfinalFunc,
        'optimalMSE':optimalMSE,
        'MAD_FinalFunc':MADfinalFunc,
        'optimalMAD':optimalMAD,
        'MAPE_FinalFunc':MAPEfinalFunc,
        'optimalMAPE':optimalMAPE,
        'ErrorPoly':ErrorPoly,
        'ErrorExpo':ErrorExpo,
        'ErrorLn':ErrorLn,
        'ErrorSin':ErrorSin
    })
    dataframe_MSE.to_csv("C:\\Users\\Ziyi Wang\\Desktop\\bishe\\realize wangyi\\bishe\\results\\revisedResults"+'-'+str(step)+'-Evaluation'+'-'+str(start)[0:10]+".csv")

end=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))
print(ErrorCount)
print('本次程序运行时长:',end-start)