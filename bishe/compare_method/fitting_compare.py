from sklearn.svm import SVR
import pandas as pd
import numpy as np
import tensorflow as tf
import statsmodels.api as sm


price_df = pd.read_excel(r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\Data\Brent-11-25.xlsx', sheet_name='Sheet1',
                         header=None)
TRAIN_SIZE = 3000
COMPARE_PATH = r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\bishe\data\compare'


def reshape_data(n):
    # 从price_df中提取出训练和目标数据
    # 整理成None*n的二维数组
    X= []
    y = []
    for i in range(n, price_df.shape[0]):
        X.append(list(price_df.loc[i-n:i-1, 1]))
        y.append(price_df.loc[i])
    return X,y


# #####SVR######
def svr_train(n=22, train_size=TRAIN_SIZE):
    train_x = []
    train_y = []

    # 准备训练对象
    for i in range(n, price_df.shape[0]-1):
        train_x.append(list(price_df.loc[i-n+1:i, 1]))
        train_y.append(price_df.loc[i+1, 1])
    # svr初始化
    svr = SVR(gamma=0.01, C=100, kernel='rbf')
    # 训练svr
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    svr.fit(train_x[:train_size], train_y[:train_size])
    return svr


def fitting_svr(svr, end, start=22, n=22):
    train = []
    for i in range(start, end+1):
        train.append(list(price_df.loc[i-n+1:i,1]))
    target = svr.predict(train)
    return target


# BP神经网络
def fetch_batch(epoch, batch_index, batch_size):
    n_batches = int(np.ceil(TRAIN_SIZE/batch_size))
    np.random.seed(epoch*n_batches+batch_index)
    indices = np.random.randint(TRAIN_SIZE-1, size=batch_size)
    return indices


def bp_train(fitting_start, fitting_end, n=22, train_size=TRAIN_SIZE):
    # 获取数据
    X_data, y_data = reshape_data(n=4)
    X_data = np.array(X_data)
    y_data = np.array(y_data).reshape(-1, 1)

    # 定义模型
    x = tf.placeholder(tf.float32, shape=[None, 4], name='X')
    y_target = tf.placeholder(tf.float32, shape=[None, 1], name='y')
    weights1 = tf.Variable(tf.truncated_normal([4, 3], stddev=1, seed=1), name='theta1')
    weights2 = tf.Variable(tf.truncated_normal([3, 1], stddev=1, seed=1), name='theta2')
    bias1 = tf.Variable(tf.zeros([3]))
    bias2 = tf.Variable(tf.zeros([1]))
    a = tf.nn.relu(tf.matmul(x, weights1)+bias1)
    y = tf.matmul(a, weights2)+bias2

    # 设计损失函数
    loss = tf.reduce_mean(tf.abs(y_target-y)/y_target, name='loss')
    train_step = tf.train.AdamOptimizer(0.1).minimize(loss)

    # 定义batch
    batch_size = 128
    n_bathes = int(np.ceil(train_size/batch_size))

    init = tf.global_variables_initializer()

    n_epochs = 5000
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(n_epochs):
            for batch_index in range(n_bathes):
                indices = fetch_batch(epoch, batch_index, batch_size)
                X_batch = X_data[indices]
                y_batch = y_data[indices]
                sess.run(train_step, feed_dict={x: X_batch, y_target: y_batch})
        y_test = sess.run(y, feed_dict={x: X_data[fitting_start:fitting_end+1]})
    y_test = y_test.reshape(1, -1)[0]
    return y_test


def arima(y, start, end, train_size=TRAIN_SIZE):
    mod = sm.tsa.ARIMA(list(y[:train_size]), order=(1, 1, 1))
    arima = mod.fit(disp=-1)
    res = arima.predict(start, end, dynamic=False)
    if end <= train_size:
        result = np.array(price_df.loc[start+1:end+1, 1])+np.array(res)
        result = list(result)
    else:       # 样本外
        result = []
        for i in range(len(res)):
            if i == 0:
                result.append(price_df.loc[train_size, 1]+res[i])
            else:
                result.append(result[-1]+res[i])
        result = result[-1]
    return result


def fitting():
    # 拟合2000到3000点
    fitting_start = 2000
    fitting_end = 3000

    # svr
    svr = svr_train()
    svr_result = fitting_svr(svr=svr, start=fitting_start, end=fitting_end)
    print('svr长度', len(svr_result))

    # BP
    bp_result = bp_train(fitting_start, fitting_end)
    print('bp长度', len(bp_result))

    # ARIMA
    arima_result = arima(list(price_df.loc[:, 1]), fitting_start, fitting_end)
    print('arima长度', len(arima_result))


    compare_df = pd.DataFrame({
        'SVR': svr_result,
        'BP': bp_result,
        'ARIMA': arima_result
    })

    compare_df.to_excel(COMPARE_PATH+r'\price_compare_SVR_BP_ARIMA'+str(fitting_start)+'-'+str(fitting_end)+'.xlsx')
    pass


if __name__ == '__main__':

    fitting()
    # fitting_start = 2000
    # fitting_end = 3000
    # arima_result = arima(list(price_df.loc[:, 1]), fitting_start, fitting_end)
    # print(arima_result)

    pass