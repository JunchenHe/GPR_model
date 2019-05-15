import sys
sys.path.append(r'../../')
from get_data import Data
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
import err

def func_change(origin_x):
    a=origin_x[:,0]
    b=origin_x[:,1]

    xdata=[]

    tmp=[]
    for i in range(a.shape[0]):
        tmp.append(a[i]**2)
    xdata.append(np.array(tmp))

    tmp=[]
    for i in range(a.shape[0]):
        tmp.append(a[i])
    xdata.append(np.array(tmp))

    tmp=[]
    for i in range(a.shape[0]):
        tmp.append(b[i])
    xdata.append(np.array(tmp))

    tmp=[]
    for i in range(a.shape[0]):
        tmp.append(b[i]*a[i])
    xdata.append(np.array(tmp))

    return np.array(xdata).T

input_set=[(0,1,),(1,0),(0,2,),(2,0),(1,2,),(2,1)]
train_num_set=[5,10,15,20,25,30,35,40,45,52]

print("方程 = ax^2 + bx + cy + dxy + e ")
print("数据量\t输入维度\ttrain_rmse\ttest_rmse\ttest_平均误差率\t\t模型参数")

for input_dim in input_set:
    for train_num in train_num_set:
        data=Data("../.././data/data.txt",input_dim,train_num)

        #训练指数模型

        xdata=func_change(data.train.X)
        ydata=data.train.Y[:,0]

        # 训练数据
        regr = linear_model.LinearRegression()
        regr.fit(xdata,ydata)
        #print('coefficients(b1,b2...):',regr.coef_)
        #print('intercept(b0):',regr.intercept_)


        def func(x, a,b,c,d,e):
            return a*x[0]+b*x[1]+c*x[2]+d*x[3]+e

        pre_y_train=[]
        for x in func_change(data.train.X):
            pre_y_train.append([
                                    func(x,regr.coef_[0],
                                    regr.coef_[1],
                                    regr.coef_[2],
                                    regr.coef_[3],
                                    regr.intercept_)])
        pre_y_train=np.array(pre_y_train)


        pre_y_test=[]
        for x in func_change(data.test.X):
            pre_y_test.append([
                                    func(x,regr.coef_[0],
                                    regr.coef_[1],
                                    regr.coef_[2],
                                    regr.coef_[3],
                                    regr.intercept_)])
        pre_y_test=np.array(pre_y_test)


        print(train_num,end='\t')
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