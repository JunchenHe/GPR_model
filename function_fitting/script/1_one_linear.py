import sys
sys.path.append(r'../../')
from get_data import Data
import err
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


input_set=[(0,),(1,),(2,)]
train_num_set=[5,10,15,20,25,30,35,40,45,52]

print("方程 = a * x + b")
print("训练数据量\t输入维度\ttrain_rmse\ttest_rmse\ttest_平均误差率\t模型参数")

for input_dim in input_set:
    for train_num in train_num_set:

        #input_dim=(0,)
        #train_num=5
        data=Data("../.././data/data.txt",input_dim,train_num)

        #训练指数模型
        def func(x, a, b):
            return a * x + b


        xdata=data.train.X[:,0]
        ydata=data.train.Y[:,0]

        popt, pcov = curve_fit(func, xdata, ydata,maxfev=500000)
        #popt数组中，三个值分别是待求参数a,b,c
        #y2 = [func(i, popt[0],popt[1],popt[2]) for i in xdata]


        pre_y_train=[]
        for x in data.train.X:
            pre_y_train.append(func(x,popt[0],popt[1]))
        pre_y_train=np.array(pre_y_train)


        pre_y_test=[]
        for x in data.test.X:
            pre_y_test.append(func(x,popt[0],popt[1]))
        pre_y_test=np.array(pre_y_test)

        
        print(train_num,end='\t\t')
        for a in input_dim:
            if(a==0):
                print("冠幅 cw",end='\t\t')
            elif(a==1):
                print("面积 CA",end='\t\t')
            elif(a==2):
                print("树高 CH",end='\t\t') 
                    
        #print(err.calc_rmse(data.train.Y,data.train.predict_Y),end='\t')
        print(err.calc_rmse(data.train.Y,pre_y_train),end='\t')

        print(err.calc_rmse(data.test.Y,pre_y_test),end='\t')   

        print(err.calc_err_rate(data.test.Y,pre_y_test),end='\t')
        print(popt,end="")


        print()
