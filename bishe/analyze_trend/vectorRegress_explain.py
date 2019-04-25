# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.tsa.api as st

'''
本程序主要基于不含外生变量的VAR模型，也叫计量联合方程模型
大概思想是：
假设区间(a,b)趋势向量是2维的(trenda,trendb) 利用trenda和trendb和他们自己的滞后一阶和两阶值分别进行回归，得到回归方程的系数数组ka,kb，数组中的元素依次是常数项，滞后一阶的系数
滞后二阶的系数，如：
tranda = k[0]+k[1]tranda(t-1)+k[2]tranda(t-2)
'''
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
trende1=[""]
trenda2=["",""]
trendb2=["",""]
trendc2=["",""]
trendd2=["",""]
trende2=["",""]
trenda3=["","",""]
trendb3=["","",""]
trendc3=["","",""]
trendd3=["","",""]
trende3=["","",""]
trenda4=["","","",""]
trendb4=["","","",""]
trendc4=["","","",""]
trendd4=["","","",""]
trende4=["","","",""]
trenda5=["","","","",""]
trendb5=["","","","",""]
trendc5=["","","","",""]
trendd5=["","","","",""]
trende5=["","","","",""]
'''
以下循环构造滞后各项的数组
'''
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

def getRegressResult(a,b,alpha):

    '''通过线性回归求趋势向量自回归系数，被注释的c和d是四维趋势向量用到的'''

    xa=[];xb=[];xc=[];xd=[];xv=[];ya=[];yb=[];yc=[];yv=[];yd=[];
    for i in range(a,b+1):
        xa.append([trenda1[i],trenda2[i]])#,trenda3[i],trenda4[i],trenda5[i]])
        xb.append([trendb1[i],trendb2[i]])#,trendb3[i],trendb4[i],trendb5[i]])
   #     xc.append([trendc1[i],trendc2[i]])
   #     xd.append([trendd1[i],trendd2[i]])
        ya.append(trenda[i])
        yb.append(trendb[i])
    #    yc.append(trendc[i])
    #    yd.append(trendd[i])
    xa=np.array(xa)
    xb = np.array(xb)
    #xc = np.array(xc)
    #xd=np.array(xd)
  
    xa=sm.add_constant(xa)
    xb = sm.add_constant(xb)
    #xc = sm.add_constant(xc)
    #xd=sm.add_constant(xd)
   
    resulta=sm.OLS(ya,xa).fit()
    resultb = sm.OLS(yb, xb).fit()
    #resultc = sm.OLS(yc, xc).fit()
    #resultd=sm.OLS(yd,xd).fit()
  
 #   print(resulta.summary(),resultb.summary(),resultc.summary())
    '''
	这一段本来想按照显著性水平筛掉一些结果，发现没什么用，可删掉
    for i in range(len(resulta.pvalues)):
        if(resulta.pvalues[i]>alpha):
            resulta.params[i]=0
    for i in range(len(resultb.pvalues)):
        if(resultb.pvalues[i]>alpha):
            resultb.params[i]=0

    for i in range(len(resultc.pvalues)):
        if(resultc.pvalues[i]>alpha):
            resultc.params[i]=0

     '''
	# 获取回归系数数组
    ka=resulta.params
    kb=resultb.params
    #kc=resultc.params
    #kd=resultd.params

    return ka,kb#,kc#,kd

# 设定显著性水平
alpha=0.1
# 窗口长度
step=22


def getResultForPredict(a,b):
    '''
	根据向量自回归方程获取在区间(b+1,b+1+step)上的趋势向量分量，所以后面数组无论trenda还是trendb都在b处取值，因为想根据已知区间的终点b趋势预测未知区间的起点b+1趋势
	'''
 #   print(b,trenda[b],trendb[b])
    # 获取向量回归参数
    ka,kb=getRegressResult(a,b,alpha)
 #  print(ka,kb)
    ya_ols=ka[0]+ka[1]*trenda1[b]+ka[2]*trenda2[b]#+ka[3]*trenda3[b]+ka[4]*trenda4[b]+ka[5]*trenda5[b]
    yb_ols=kb[0]+kb[1]*trendb1[b]+kb[2]*trendb2[b]#+kb[3]*trenda3[b]+kb[4]*trendb4[b]+kb[5]*trenda5[b]
    #yc_ols=kc[0]+kc[1]*trendc1[b]+kc[2]*trendc2[b]
  #  yd_ols=kd[0]+kd[1]*trendd1[b]+kd[2]*trendd2[b]
    return ya_ols,yb_ols#,yc_ols#,yd_ols
# 设置回归区间
a=2;
#b=df.shape[0]-1-1
avError=0
mape=0
sse=0
# 滚动向前一步预测，思路是每次向前一步，将结果取平均
# 比如 第一次用(a,b)预测(b+1,step+b+1)的趋势
# 第二次用(a,b+1)预测(b+2,step+b+2)的趋势，以此类推
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