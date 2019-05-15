import sys
sys.path.append(r'../../')
from get_data import Data
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
import err


input_set=[(0,1,),(0,2,),(1,2,)]
train_num_set=[5,10,15,20,25,30,35,40,45,52]

print("方程 = a * x +b*y+ c")
print("训练数据量\t输入维度\ttrain_rmse\ttest_rmse\ttest_平均误差率\t\t模型参数")

for input_dim in input_set:
    for train_num in train_num_set:
        data=Data("../.././data/data.txt",input_dim,train_num)

        #训练指数模型

        xdata=data.train.X
        ydata=data.train.Y[:,0]

        # 训练数据
        regr = linear_model.LinearRegression()
        regr.fit(xdata,ydata)
        #print('coefficients(b1,b2...):',regr.coef_)
        #print('intercept(b0):',regr.intercept_)


        def func(x, a , b,c):
            return a*x[0]+b*x[1]+c

        pre_y_train=[]
        for x in data.train.X:
            pre_y_train.append([func(x,regr.coef_[0],regr.coef_[1],regr.intercept_)])
        pre_y_train=np.array(pre_y_train)


        pre_y_test=[]
        for x in data.test.X:
            pre_y_test.append([func(x,regr.coef_[0],regr.coef_[1],regr.intercept_)])
        pre_y_test=np.array(pre_y_test)


        print(train_num,end='\t\t')
        for a in input_dim:
            if(a==0):
                print("冠幅cw ",end='')
            elif(a==1):
                print("面积CA ",end='')
            elif(a==2):
                print("树高CH ",end='') 
        print('\t',end="")            

        #print(err.calc_rmse(data.train.Y,data.train.predict_Y),end='\t')
        print(err.calc_rmse(data.train.Y,pre_y_train),end='\t')

        print(err.calc_rmse(data.test.Y,pre_y_test),end='\t')   

        print(err.calc_err_rate(data.test.Y,pre_y_test),end='\t\t')
        #print(,end="")


        print([regr.coef_,regr.intercept_],end="")


        print()