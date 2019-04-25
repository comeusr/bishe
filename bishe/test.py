import numpy as np

def time(step):
    #output = [[1,1,1,1],[2,4,8,16],[3,9,27,81],[4,16,64,256],[5,25,125,625]]
    output = []
    for i in range(step):
        temp = []
        for j in range(1,step):
            k = np.power((i+1),j)
            temp.append(k)
        output.append(temp)
    return output

print(time(5))

'''
通过线性回归求趋势向量自回归系数，被注释的c和d是四维趋势向量用到的
'''