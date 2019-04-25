"""
本文将石油价格，转为收益率，并储存
"""

import pandas

#读取价格数据
read_path = r'C:\Users\Ziyi Wang\Desktop\bishe\realize wangyi\Data\Brent-11-15.xlsx'
df = pandas.read_excel(read_path,sheet_name='Sheet1',header=None)

#Profit存一阶差分的结果
profit = [0]

for i in range(1,df.shape[0]):
    profit.append(df.loc[i,1]-df.loc[i-1,1])

dataframe_profit = pandas.DataFrame(profit)

#输出路径
result_path = r'..\Data'
dataframe_profit.to_csv(result_path+r'\profit.csv')