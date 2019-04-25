import numpy as np
import pandas as pd
import time
import datetime
start=datetime.datetime.fromtimestamp(time.mktime(time.strptime(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())),"%Y-%m-%d %H:%M:%S")))

path = r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\bishe\results\Poly-6-Params-2019-04-16.csv'
oil_price_path = r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\Data\Brent-11-25.xlsm'
trend_df = pd.read_csv(path,header=0)
price_df = pd.read_excel(oil_price_path,header=None,sheet_name='Sheet1')

trenda = np.array(trend_df.loc[:, 'MSEtrenda'].astype('float64'))
trendb = np.array(trend_df.loc[:, 'MSEtrendb'].astype('float64'))
trendc = np.array(trend_df.loc[:, 'MSEtrendc'].astype('float64'))
trendd = np.array(trend_df.loc[:, 'MSEtrendd'].astype('float64'))
trende = np.array(trend_df.loc[:, 'MSEtrende'].astype('float64'))




def trend2oil(t, step=6):
    oil_price = []
    x = price_df.loc[t, 0]
    if t <= step:
        for time in range(t+1):
            temp_result = trenda[t-time]*np.power(x,4)+trendb[t-time]*np.power(x,3)+trendc[t-time]*np.power(x,2)+trendd[t-time]*x+trende[t-time]
            oil_price.append(temp_result)
    else:
        for time in range(step):
            temp_result = trenda[t-time]*np.power(x,4)+trendb[t-time]*np.power(x,3)+trendc[t-time]*np.power(x,2)+trendd[t-time]*x+trende[t-time]
            oil_price.append(temp_result)
            pass

    return oil_price


def window_based_trend2oil(t, step):
    oil_price = []
    if t <= step:
        pass
    else:
        pass


def select_trend(t, step=6):
    '''
    选择第t个的最优趋势
    :param t: 第t个
    :param step:
    :return: 最优趋势的index，同时返回最优油价
    '''
    true_price = price_df.loc[t, 1]
    regression_price = trend2oil(t,step=step)
    evaluation = []
    for price in regression_price:
        evaluation.append(abs(true_price-price))
    best_index = t-evaluation.index(min(evaluation))
    best_price = regression_price[evaluation.index(min(evaluation))]
    return best_index, best_price


def main():
    best_trenda = np.zeros(len(trenda))
    best_trendb = np.zeros(len(trendb))
    best_trendc = np.zeros(len(trendc))
    best_trendd = np.zeros(len(trendd))
    best_trende = np.zeros(len(trende))
    for i in range(len(trenda)):
        x = price_df.loc[i, 0]
        best_index, best_price = select_trend(i)
        best_trenda[i] = trenda[best_index]
        best_trendb[i] = trendb[best_index]
        best_trendc[i] = trendc[best_index]
        best_trendd[i] = trendd[best_index]
        best_trende[i] = trende[best_index]
        compute_best_price = best_trenda[i]*np.power(x, 4)+best_trendb[i]*np.power(x, 3)+best_trendc[i]*np.power(x, 2)\
                             + best_trendd[i]*x+best_trende[i]
        # print(compute_best_price, len(compute_best_price))
        if compute_best_price != best_price:
            print('index: '+str(i)+'计算油价出错')
        if i == 216:
            print(i)
            print(trend2oil(i))
            print(select_trend(i))
            print(compute_best_price)
        # if i <= 5:
        #     print(true_price)
        #     print(best_lag)
        #     print(oil_price_list)
        #     print(evaluation)
    result = pd.DataFrame({
        'trenda': best_trenda,
        'trendb': best_trendb,
        'trendc': best_trendc,
        'trendd': best_trendd,
        'trende': best_trende
    })
    return result


if __name__ == '__main__':
    test_df = main()
    selected_path = r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\bishe\data\best_trend'
    test_df.to_csv(selected_path+r'\best_trend-'+str(start)[0:10]+'(3).csv')
    oil_price = []
    for i in range(test_df.shape[0]):
        # if i <= 5:
        #     print(trend2oil(i))
        x = price_df.loc[i, 0]
        best_price = test_df['trenda'][i]*np.power(x, 4)+test_df['trendb'][i]*np.power(x, 3)\
                     + test_df['trendc'][i]*np.power(x, 2)+test_df['trendd'][i]*x+test_df['trende'][i]
        oil_price.append(best_price)
    price_comparison = pd.DataFrame({
        'Ture Price': price_df.loc[0:5530, 1],
        'Predict Price': oil_price
    })
    price_comparison.to_csv(r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\bishe\data\priceComparison'+str(start)[0:10]+'(3).csv')
