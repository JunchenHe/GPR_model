
import numpy as np


class Train(object):
    # X => (52,input_dim)
    # Y => (52,1) 亦是二维的
    def __init__(self,x,y):
        self.X=x
        self.Y=y

    def mean_and_std(self):
        #减去均值除以标准差
        #将归一化后的X 和 Y 放在 
        # self.mean_and_std_X 
        # self.mean_and_std_Y 

        mean_x=[]
        std_x=[]
        for a in range(self.X.shape[1]):
            mean_x.append(np.mean(self.X[:,a],dtype=np.float64))
            std_x.append(np.std(self.X[:,a],dtype=np.float64))
        mean_y=[np.mean(self.Y[:,0],dtype=np.float64)]
        std_y=[np.std(self.Y[:,0],dtype=np.float64)]

        self.mean_x=mean_x.copy()
        self.std_x=std_x.copy()

        self.mean_y=mean_y.copy()
        self.std_y=std_y.copy()


        self.mean_and_std_X=self.X.copy()
        self.mean_and_std_Y=self.Y.copy()
        #深复制

        for i in range(self.mean_and_std_X.shape[0]):
            for j in range(self.mean_and_std_X.shape[1]):
                self.mean_and_std_X[i][j]=(self.X[i][j]-mean_x[j])/std_x[j]
        #print(self.mean_and_std_X)

        for i in range(self.Y.shape[0]):
            self.mean_and_std_Y[i][0]=(self.Y[i][0]-mean_y)/std_y[0]
        #print(self.mean_and_std_Y)

    def re_mean_and_std(self,mean):
        #输入模型预测的y值，反归一化

        self.predict_Y=mean.copy()
        for i in range(mean.shape[0]):
            self.predict_Y[i][0]=mean[i][0]* self.std_y[0] + self.mean_y[0]

    def re_min_and_max(self,mean):
        self.predict_Y=mean.copy()
        for i in range(mean.shape[0]):
            self.predict_Y[i][0]=mean[i][0] *(self.max_y[0]-self.min_y[0]) + self.min_y[0]


    def min_and_max(self):
        #最大最小归一化
        #将归一化后的X 和 Y 放在
        # self.min_and_max_X
        # self.min_and_max_Y

        min_x=[]
        max_x=[]
        for a in range(self.X.shape[1]):
            min_x.append(min(self.X[:,a]))
            max_x.append(max(self.X[:,a]))
        min_y=[min(self.Y[:,0])]
        max_y=[max(self.Y[:,0])]

        
        self.min_x=min_x.copy()
        self.max_x=max_x.copy()
        self.min_y=min_y.copy()
        self.max_y=max_y.copy()

        self.min_and_max_X=self.X.copy()
        self.min_and_max_Y=self.Y.copy()
        #深复制

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.min_and_max_X[i][j]=(self.X[i][j]-min_x[j])/(max_x[j]-min_x[j])


        for i in range(self.Y.shape[0]):
            self.min_and_max_Y[i][0]=(self.Y[i][0]-min_y[0])/(max_y[0]-min_y[0])


class Test(object):
    def __init__(self,x,y):
        self.X=x
        self.Y=y

    def mean_and_std(self,mean_x,mean_y,std_x,std_y):
        #只需要将x归一化就好
        # 用预测得的数据进行反归一化
        self.mean_and_std_X=self.X.copy()
        self.mean_and_std_Y=self.Y.copy()

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.mean_and_std_X[i][j]=(self.X[i][j]-mean_x[j])/std_x[j]

        self.mean_y=mean_y
        self.std_y=std_y
        """
        for i in range(self.Y.shape[0]):
            self.mean_and_std_Y[i][0]=(self.Y[i][0]-mean_y)/std_y[0]
        """
    def re_mean_and_std(self,mean):
        #输入模型预测的y值，反归一化

        self.predict_Y=mean.copy()
        for i in range(mean.shape[0]):
            self.predict_Y[i][0]=mean[i][0]* self.std_y[0] + self.mean_y[0]



    def min_and_max(self,min_x,min_y,max_x,max_y):
        #只需要将x归一化就好
        # 用预测得的数据进行反归一化
        self.min_and_max_X=self.X.copy()
        self.min_and_max_Y=self.Y.copy()

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                self.min_and_max_X[i][j]=(self.X[i][j]-min_x[j])/(max_x[j]-min_x[j])
        
        self.min_y=min_y
        self.max_y=max_y
        """
        for i in range(self.Y.shape[0]):
            self.min_and_max_Y[i][0]=(self.Y[i][0]-min_y[0])/(max_y[0]-min_y[0])
        """

    def re_min_and_max(self,mean):
        self.predict_Y=mean.copy()
        for i in range(mean.shape[0]):
            self.predict_Y[i][0]=mean[i][0] *(self.max_y[0]-self.min_y[0]) + self.min_y[0]


class Data(object):
    def __init__(self,file,input_dim,train_num):
        #file 指定输入的文件
        #input_dim 指定输入的哪一维度的数据,(0,1,2),矩阵切片 
        x=[]
        y=[]
        with open(file) as f:
            for line in f:
                tmp=line.split()
                tmp_x=[]

                if(tmp[0] =='#'):
                    continue
                    #忽略首行
                i=0
                for a in tmp:
                    if(i!=3):#0,1,2 输入数据
                        tmp_x.append(float(a))
                    else:#3 输出数据
                        y.append(float(a))
                    i+=1
                x.append(tmp_x)
        self.X=np.array(x).T
        self.Y=np.array(y)[:,None]
        self.input_dim=input_dim

        self.restruct_X=self.X[self.input_dim,:].T
        #指定输入的维度以重构输入空间(71,input_dim)

        self.train_num=train_num
        #指定选练数据的多少，( 5,10,15,20,25,30,35,40,45,52 )


        self.train=Train(self.restruct_X[0:self.train_num,:],self.Y[0:self.train_num,:])
        self.test=Test(self.restruct_X[52:72,:],self.Y[52:72,:])

"""

#调用(0,)(0,1)(0,1,2),不能直接(0)
data=Data("./data.txt",(0,1,2))

#调用函数计算 减去均值除以标准差 之后的归一化
#data.train.mean_and_std()
#print(data.train.mean_and_std_X)

data.train.min_and_max() 
#print(data.train.min_and_max_X)
#print(data.train.min_and_max_Y)
data.train.mean_and_std()
#print(data.train.mean_x)
#print(data.train.min_x)

data.test.mean_and_std(data.train.mean_x,data.train.mean_y,
                        data.train.std_x,data.train.std_y)

data.test.min_and_max(data.train.min_x,data.train.min_y,
                    data.train.max_x,data.train.max_y)

print(data.test.min_and_max_X)
print(data.test.min_and_max_Y)

print(data.test.mean_and_std_X)
print(data.test.mean_and_std_Y)
"""
