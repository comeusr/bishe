# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as st

def getVARparams(df):
    a = np.array(df['trenda'])
    b = np.array(df['trendb'])
    X = []
    for i in range(len(a)):
        X.append([a[i], b[i]])
    X = np.array(X)
    result = st.VAR(X, dates=None, freq=None, missing='none')
    ols = result.fit(maxlags=5, method='ols', trend='nc')
    return ols.params

path=r'E:\svm结构性断点论文\20171220\季度\result-linear.xls'
df=pd.read_excel(path,sheet_name='Sheet')
path2=r'E:\svm结构性断点论文\20171208\dataWhole.xls'
df2=pd.read_excel(path2,sheet_name='Whole')
trenda = np.array(df.ix[:, 1].astype('float64'))
trendb = np.array(df.ix[:, 2].astype('float64'))
#trendc = np.array(df.ix[:, 3].astype('float64'))
#trendd= np.array(df.ix[:, 4].astype('float64'))
trenda1=[""]
trendb1=[""]
trendc1=[""]
trendd1=[""]
trenda2=["",""]
trendb2=["",""]
trendd2=["",""]
trenda3=["","",""]
trendb3=["","",""]
trenda4=["","","",""]
trenda5=["","","","",""]
trendb4=["","","",""]
trendb5=["","","","",""]
trendc4=["","","",""]
trendc5=["","","","",""]
trendc2=["",""]
for i in range(len(trenda)-1):
    trenda1.append(trenda[i])
    trendb1.append(trendb[i])
 #   trendc1.append(trendc[i])
 #   trendd1.append(trendd[i])
    if i<=len(trenda)-3:
        trenda2.append(trenda[i])
        trendb2.append(trendb[i])
  #      trendc2.append(trendc[i])
  #      trendd2.append(trendd[i])
    if i<=len(trenda)-4:
        trenda3.append(trenda[i])
        trendb3.append(trendb[i])
    if i<=len(trenda)-5:
        trenda4.append(trenda[i])
        trendb4.append(trendb[i])
    if i<=len(trenda)-6:
        trenda5.append(trenda[i])
        trendb5.append(trendb[i])
vix = list(df2.ix[:, 5].astype('float64'))
vix1 = list(df2.ix[:, 6].astype('float64'))
vix2 = list(df2.ix[:, 7].astype('float64'))
vix3 = list(df2.ix[:, 8].astype('float64'))
def getRegressResult(a,b,alpha):
    xa=[];xb=[];xc=[];xd=[];xv=[];ya=[];yb=[];yc=[];yv=[];yd=[];
    for i in range(a,b+1):
        xa.append([trenda1[i],trenda2[i]])#,trenda3[i],trenda4[i],trenda5[i]])
        xb.append([trendb1[i],trendb2[i]])#,trendb3[i],trendb4[i],trendb5[i]])
   #     xc.append([trendc1[i],trendc2[i]])
   #     xd.append([trendd1[i],trendd2[i]])
   #     xv.append([vix1[i],vix2[i]])
        ya.append(trenda[i])
        yb.append(trendb[i])
    #    yc.append(trendc[i])
    #    yd.append(trendd[i])
    #    yv.append(vix[i])
    xa=np.array(xa)
    xb = np.array(xb)
    #xc = np.array(xc)
    #xd=np.array(xd)
    #xv = np.array(xv)
    xa=sm.add_constant(xa)
    xb = sm.add_constant(xb)
    #xc = sm.add_constant(xc)
    #xd=sm.add_constant(xd)
    #xv = sm.add_constant(xv)
    resulta=sm.OLS(ya,xa).fit()
    resultb = sm.OLS(yb, xb).fit()
    #resultc = sm.OLS(yc, xc).fit()
    #resultd=sm.OLS(yd,xd).fit()
    #resultv = sm.OLS(yv, xv).fit()
 #   print(resulta.summary(),resultb.summary(),resultc.summary(),resultv.summary())
    '''
    for i in range(len(resulta.pvalues)):
        if(resulta.pvalues[i]>alpha):
            resulta.params[i]=0
    for i in range(len(resultb.pvalues)):
        if(resultb.pvalues[i]>alpha):
            resultb.params[i]=0

    for i in range(len(resultc.pvalues)):
        if(resultc.pvalues[i]>alpha):
            resultc.params[i]=0
    for i in range(len(resultv.pvalues)):
        if(resultv.pvalues[i]>alpha):
            resultv.params[i]=0
     '''
    ka=resulta.params
    kb=resultb.params
    #kc=resultc.params
    #kd=resultd.params
    #kv=resultv.params
    return ka,kb#,kc#,kd
# 设定显著性水平
alpha=0.1
# 窗口长度
step=22


def getResultForPredict(a,b):
 #   print(b,trenda[b],trendb[b])
    # 获取向量回归参数
    '''
    params=getVARparams(df)
    ya_ols=params[0][0]*trenda1[b]+params[2][0]*trenda2[b]+params[4][0]*trenda3[b]+params[6][0]*trenda4[b]+params[8][0]*trenda5[b]\
        +params[1][0]*trendb1[b]+params[3][0]*trendb2[b]+params[5][0]*trendb3[b]+params[7][0]*trendb4[b]+params[9][0]*trendb5[b]
    yb_ols=params[0][1]*trenda1[b]+params[2][1]*trenda2[b]+params[4][1]*trenda3[b]+params[6][1]*trenda4[b]+params[8][1]*trenda5[b]\
        +params[1][1]*trendb1[b]+params[3][1]*trendb2[b]+params[5][1]*trendb3[b]+params[7][1]*trendb4[b]+params[9][1]*trendb5[b]
    '''
    ka,kb=getRegressResult(a,b,alpha)
 #  print(ka,kb)
    ya_ols=ka[0]+ka[1]*trenda1[b]+ka[2]*trenda2[b]#+ka[3]*trenda3[b]+ka[4]*trenda4[b]+ka[5]*trenda5[b]
    yb_ols=kb[0]+kb[1]*trendb1[b]+kb[2]*trendb2[b]#+kb[3]*trenda3[b]+kb[4]*trendb4[b]+kb[5]*trenda5[b]
    #yc_ols=kc[0]+kc[1]*trendc1[b]+kc[2]*trendc2[b]
  #  yd_ols=kd[0]+kd[1]*trendd1[b]+kd[2]*trendd2[b]
    #yv_ols=kv[0]+kv[1]*vix1[b]+kv[2]*vix2[b]#+kv[3]*vix3[b]
    return ya_ols,yb_ols#,yc_ols#,yd_ols
# 设置回归区间
a=2;
#b=df.shape[0]-1-1
avError=0
mape=0
sse=0
for i in range(df.shape[0]-6,df.shape[0]-1):
    ya_ols, yb_ols=getResultForPredict(a,i)
    if i==df.shape[0]-6:
        i=df.shape[0]-7
    error=(0.5)*(np.square(ya_ols-trenda[i+1])+np.square(yb_ols-trendb[i+1]))#+np.square(yc_ols-trendc[i+1]))#+np.square(yd_ols-trendd[i+1]))
    dmape=np.average([np.abs((ya_ols - trenda[i + 1]) / trenda[i + 1]), np.abs((yb_ols - trendb[i + 1]) / trendb[i + 1])])#\
                         #,np.abs((yc_ols - trendc[i + 1]) / trendc[i + 1])])#,np.abs((yd_ols - trendd[i + 1]) / trendd[i + 1])])
    mape+=dmape
    avError+=error
    sse+=error*2
    print('预测第',i+1,'个点',',平方损失为:',error,',mape=',dmape,',sse=',error*2)
    #print('预测第',i+1,'个点,第一个参数：',ya_ols, ',真值:',trenda[i+1], '第二个参数:',yb_ols,',真值：',trendb[i+1],'\n')
    #print('预测第',i+1,'个点,第一个参数误差',trenda[i+1]-ya_ols,'，第二个参数误差:',trendb[i+1]-yb_ols,',平方损失:',np.square(trendb[i+1]-yb_ols)+np.square(trenda[i+1]-ya_ols),'\n')
print('loss=',avError/5,'mape=',mape/5,',sse=',sse/5)