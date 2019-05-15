#!/home/he/anaconda3/bin/python
import matplotlib.pyplot as plt
import numpy as np

def calc_err_rate(true_value,predict_value):
    sum=np.float(0.0) 
    for i in range(true_value.shape[0]):
        sum+= abs(true_value[i][0]-predict_value[i][0])/(true_value[i][0])
    return sum/true_value.shape[0]

def calc_rmse(true_value,predict_value):
    sum=np.float64(0.0)
    for i in range(true_value.shape[0]):
        sum+= ( abs(true_value[i][0]-predict_value[i][0]) ** 2)
    sum=sum/(true_value.shape[0]-1)
    return np.sqrt(sum)
        

def show_err(size,file_path,
              predict_value,true_value,var,std,plt_name,
              predict_value2,true_value2,var2,std2,plt_name2 ):

    font = {
            'weight' : 'normal',
            'size'   : 23,
        }
    font2 = {
            'size'   : 20,
        }
    font3={
            'weight' : 'normal',
            'size'   : 23,
            'color': 'red'
    }


    plt.figure(figsize=size)

    ax1 = plt.subplot(211)
    
    for i in range(true_value.shape[0]):
        plt.errorbar(i+1, predict_value[i][0], yerr=pow(std,2)*var[i][0], fmt='-x',color='blue')
    x=np.linspace(0+1,true_value.shape[0],true_value.shape[0])
    plt.scatter(x,true_value,color='red',label='true_value') 
   
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlim([0,true_value.shape[0]+1])
    new_ticks=np.linspace(0+1,true_value.shape[0],true_value.shape[0])
    plt.xticks(new_ticks)

    plt.ylim(0,25)
    new_ticks_y=np.linspace(0,25,26)
    plt.yticks(new_ticks_y)

    plt.legend(loc='best')

    #plt.title(plt_name)
    ax1.set_title(plt_name,font3)

    plt.xlabel("index of train sample",font2)
    plt.ylabel("Diameter /cm",font)



    #plt.savefig(file_path)
    #plt.show()
   
      
    
    ax2 = plt.subplot(212)
    for i in range(true_value2.shape[0]):
        plt.errorbar(i+1, predict_value2[i][0], yerr=pow(std2,2)*var2[i][0], fmt='-x',color='blue')
    x=np.linspace(0+1,true_value2.shape[0],true_value2.shape[0])
    plt.scatter(x,true_value2,color='red',label='true_value') 
   
    ax=plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    plt.xlim([0,true_value2.shape[0]+1])
    new_ticks=np.linspace(0+1,true_value2.shape[0],true_value2.shape[0])
    plt.xticks(new_ticks)

    plt.ylim(0,25)
    new_ticks_y=np.linspace(0,25,26)
    plt.yticks(new_ticks_y)

    plt.legend(loc='best')

    #plt.title(plt_name2)
    ax2.set_title(plt_name2,font3)

    plt.xlabel("index of test sample",font2)
    plt.ylabel("Diameter /cm",font)

    plt.savefig(file_path)
    #plt.show()



