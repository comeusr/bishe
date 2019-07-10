# bishe
垃圾毕设，干！

## 内容
1. Data是所有训练好了的趋势向量
2. bishe里面代码，同时包含筛选后的趋势向量

## 运行流程
1. 首先通过Matlab检测断点，将是否为断点设定为0/1的虚拟变量
2. 通过get_trend/windowed_PolyStep22.py（或者其他文件， 最后个那个22数字代表窗口长度）训练窗口趋势向量。
3. 通过analyze_trend/windowed_select_trend.py对窗口趋势向量进行筛选，选定瞬时趋势向量
4. 通过analyze_trend/windowed_MSEvectorRegress.py对瞬时趋势向量进行建模，同时完成预测。
5. 和其他方法对比的程序再compare_method
