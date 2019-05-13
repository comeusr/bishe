# coding:utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
import xlwt

# 获取bp数据
def getBPData(a,b):
    d1=list(df.iloc[a:b+1,2])
    d2=list(df.iloc[a:b+1,3])
    d3=list(df.iloc[a:b+1,4])
    d4=list(df.iloc[a:b+1,5])
    d5=list(df.iloc[a:b+1,6])
    x=[]
    y=[]
    for i in range(len(d1)):
        x.append([d2[i],d3[i],d4[i],d5[i]])
        y.append([d1[i]])
    return x,y
df=pd.read_excel(r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\Data\Brent-11-25.xlsx',sheet_name='Sheet1',header=None)

# 构建bp网络
def model_BP(x_data,y_data,x_test):
    x=tf.placeholder(tf.float32,shape=[None,4],name='x-input')
    y_=tf.placeholder(tf.float32,shape=[None,1],name='y-input')
    # 网络结构
    weights=tf.Variable(tf.truncated_normal([4,3],stddev=1,seed=1))
    weights2=tf.Variable(tf.truncated_normal([3,1],stddev=1,seed=1))
    bias=tf.Variable(tf.zeros([3]))
    bias2=tf.Variable(tf.zeros([1]))
    a=tf.nn.relu(tf.matmul(x,weights)+bias)
    y=tf.matmul(a,weights2)+bias2
    # 损失设计
    #cross_entroy=-tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
    #train_step=tf.train.AdamOptimizer(0.1).minimize(cross_entroy)
    loss=tf.reduce_mean(tf.abs(y-y_)/y_)
    train_step=tf.train.AdamOptimizer(0.1).minimize(loss)
    # batch尺寸
    batch_size=128
    with tf.Session() as sess:
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        for i in range(5000):
            start=(i*batch_size)%len(x_data)
            end=min(start+batch_size,len(x_data))
            sess.run(train_step,feed_dict={x:x_data[start:end],y_:y_data[start:end]})
            if i%100 == 0:
                total_loss=sess.run(loss,feed_dict={x:x_data,y_:y_data})
                #print(total_y)
               # print("step:%d,cross_entropy:%g"%(i,total_loss))
        y_pre=sess.run(y,feed_dict={x:x_test})
        return list(y_pre)

def getCurveResult(path):
    wb=xlwt.Workbook()
    ws=wb.add_sheet('Sheet1')
    x_data, y_data = getBPData(5, 7861)
    x_test,y_test = getBPData(7862,df.shape[0])
    y_pre=model_BP(x_data,y_data,x_test)
    for i in range(len(y_pre)):
        ws.write(i,0,i+5)
        ws.write(i,1,float(y_pre[i][0]))
        print(i,'完成')
    wb.save(path)
getCurveResult(r'C:\Users\123\Desktop\result.xls')

def getPolyParams(x,y):
    x1=[]
    y1=[]
    for i in range(len(x)):
        x1.append(x[i])
        y1.append(y[i])
    trend=np.polyfit(x1,y1,1)
    trend=list(trend)
    return trend

def getMultiResult(a,b):
    mape = 0;
    avError = 0;
    sse = 0;
    path2 = r'E:\svm结构性断点论文\20171220\月度\趋势向量-月度\result-linar-monthly.xls'
    df2 = pd.read_excel(path2, sheet_name='Sheet')
    trenda = np.array(df2.ix[:, 1].astype('float64'))
    trendb = np.array(df2.ix[:, 2].astype('float64'))
    step=22
    x_data, y_data = getBPData(5, df.shape[0] - 1)
    for i in range(a,b):
        pre=model_BP(x_data[i-step+1:i],y_data[i-step+1:i],x_data[i:i+1])
        y=float(pre[0])
        data_train = list(df.iloc[i - step + 1:i + 1, 2])
        data_train.append(y)
        x = [k for k in range(i - step + 1, i + 2)]
        trend = getPolyParams(x, data_train)
        error = (0.5) * (np.square(trend[0] - trenda[i]) + np.square(trend[1] - trendb[i]))
        mape += np.average([np.abs((trend[0] - trenda[i]) / trenda[i]), np.abs((trend[1] - trendb[i]) / trendb[i])])
        avError += error
        sse += error * 2
        print('预测第', i + 1, '个点', ',平方损失为:', error, ',mape=', \
              100 * np.average(
                  [np.abs((trend[0] - trenda[i]) / trenda[i]), np.abs((trend[1] - trendb[i]) / trendb[i])]), '%' \
              , ',sse=', error * 2)
    print('平均下来:loss=', avError / 7, 'mape=', mape / 7, ',sse=', sse / 7)
step=22
#getMultiResult(df.shape[0]-1-step-7,df.shape[0]-1-step)


if __name__ == '__main__':
    pass