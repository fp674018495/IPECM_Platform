'''
Created on 2023年8月10日

@author: Gao Pan
'''
import os
import sys
import math
import random
import pickle
import numpy as np
import pandas as pd
from matplotlib import cm
from PyQt5.QtWidgets import *
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn import model_selection, svm
from scipy.optimize.minpack import fsolve 
from PyQt5 import QtCore, QtGui, QtWidgets
from sklearn.ensemble import RandomForestRegressor
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FCV

class F():
    def __init__(self,data,mini=0.1,maxi=0.9):#data为训练数据，a为归一化最小值，b为归一化最大值
        self.data=data 
        self.mi=data.min()
        self.ma=data.max()
        self.a=mini
        self.b=maxi
    def f(self,X):#将原始数据X归一化
        return (X-self.mi)*(self.b-self.a)/(self.ma-self.mi)+self.a
    def f_b(self,x):#将归一化数据x反归一化
        return (x-self.a)*(self.ma-self.mi)/(self.b-self.a)+self.mi
    
def readpkl(road):#存储的数据读取函数
    file_pkl=open(road,'rb')
    readed=pickle.load(file_pkl)
    file_pkl.close()
    return readed

def savepkl(road,data_pkl):#pkl文件存储函数
    file_pkl=open(road,'wb')
    pickle.dump(data_pkl,file_pkl)
    file_pkl.close()
    return

def MAE(x,y):
    '''平均绝对误差'''
    a=np.abs(x-y)
    return np.mean(a)

def Mjue(x,y):
    '''最大绝对误差'''
    num=len(x)
    er=0
    for i in range(num):
        if abs(y[i]-x[i])>er:
            er=abs(y[i]-x[i])
    return er

def Mxiang(x,y):
    '''平均相对误差'''
    num=len(x)
    er=[]
    for i in range(num):
        er.append(np.abs(y[i]-x[i]/x[i]))
    return np.mean(er)

def cov(x,y):
    '''协方差'''
    num=len(x)
    sum=0
    sum_x=0
    sum_y=0
    for i in range(num):
        sum+=(x[i]*y[i])
        sum_x+=x[i]
        sum_y+=y[i]
    mean=sum/num
    mean_x=sum_x/num
    mean_y=sum_y/num
    return(mean-mean_x*mean_y)

def Rou(x,y):
    '''相关系数'''
    a=cov(x,y)
    b1=math.sqrt(cov(x,x))
    b2=math.sqrt(cov(y,y))
    return a/(b1*b2)

def CanE(x,y):
    '''残差平方和'''
    num=len(x)
    sum=0
    for i in range(num):
        sum+=math.pow((x[i]-y[i]),2)
    return sum

def R2(x,y):
    '''决定系数'''
    M=np.mean(x)
    sse=CanE(x,y)
    num=len(x)
    ssr=0
    for i in range(num):
        ssr+=math.pow(y[i]-M,2)
    sst=ssr+sse
    R2=ssr/sst
    return R2

def Thigma(x,y):
    '''误差的标准差'''
    e=x-y 
    me=np.mean(e)
    s2=0
    for i in range(len(x)):
        s2 += (e[i]-me)**2
    return s2/(len(x)-1)

def NiXi(x,y):
    '''误差与真值的拟合系数与截距'''
    a=np.polyfit(x,y,1)
    return a[0],a[1]

class Qga():
    def __init__(self,pop_size=10,chr_num=2,chr_len=[12,12],MIN=[0.01,0.01],MAX=[50,5],gen=50,maxgen=10,px=0.8,pc=0.1):
        '''初始参数
        pop_size:种群大小int
        chr_num:染色体数，对应需要寻优的参数个数int
        chr_len：染色体基因个数列表，对应染色体量子位长度，如[len_c,len_g]
        MIN:各染色体数据的索引最小值列表，如[min_pfd,min_co,min_t]
        MAX:各染色体数据的索引最大值列表，如[max_pfd,max_co,max_t]
        gen:遗传代数或迭代次数int
        msxgen:设置局部最优不灾变的最大代数
        px:交叉概率
        pc:变异概率
        '''
        self.popsize=pop_size
        self.chrnum=chr_num
        self.chrlen=chr_len
        self.min=MIN
        self.max=MAX
        self.gen=gen
        self.maxgen=maxgen
        self.px=px
        self.pc=pc
        
    def initial(self):
        '''产生初始种群
        input:None
        output:[[[个体1的染色体1],[个体1的染色体2]..],[[个体2的染色体1],[个体2的染色体2]..],..,[[个体chr_len的染色体1],..,[个体chr_len的染色体chr_num]]]
        [个体1的染色体1]是一个基因个数为chr_len[0]的染色体，每个基因可以用角度thita表示，则其量子表示中[a|0,b|0]的a=sin(thita),b=cos (thita)
        [个体1的染色体1]=[thita111,thita112,...]其中thita11i表示第一个个体的第一个染色体的第i个基因的量子位角度
        '''
        pop=[]#初始种群
        for i in range(self.popsize):#遍历种群规模，挨个取随机初始个体
            pop_tem1=[]#当前个体的列表
            for j in range(self.chrnum):#遍历当前个体的染色体条数
                pop_tem2=[]#当前个体的当前染色体列表
                for k in range(self.chrlen[j]):#遍历当前染色体的基因个数
                    pop_tem2.append(random.random()*2*np.pi)#随机取当前基因的量子位角度[0,2pi]，并加入当前染色体列表
                pop_tem1.append(pop_tem2)#当前染色体加入当前个体列表
            pop.append(pop_tem1)#当前个体加入种群列表
#             print(pop)
        return pop#返回种群
    
    def decode(self,pop):
        '''解码操作input:pop;output:depopR,depop
        pop:[[[个体1的染色体1],[个体1的染色体2]..],[[个体2的染色体1],[个体2的染色体2]..],..,[[个体chr_len的染色体1],..,[个体chr_len的染色体chr_num]]]
        depopR:[[个体1的染色体1的构造二进制,个体1的染色体2的构造二进制..],..,[个体chr_len的染色体1的构造二进制,..,个体chr_len的染色体chr_num的构造二进制]]
        depop:[[个体1的染色体1的十进制值,个体1的染色体2的十进制值..],..,[个体chr_len的染色体1的十进制值,..,个体chr_len的染色体chr_num的十进制值]]
        '''
        depopR=[]#解码中构造的二进制序列种群
        depop=[]#解码后的十进制数种群列表
        for i in range(self.popsize):#遍历种群所有个体
            depop_tem1=[]#当前个体解码列表
            depopR_tem=[]#当前个体构造二进制字符串列表
            for j in range(self.chrnum):#遍历当前个体的所有染色体
                depop_tem2=[]#当前染色体解码列表
                for k in range(self.chrlen[j]):#遍历当前染色体所有基因
                    r=random.random()#设置随机比较值
                    alpha2=(np.cos(pop[i][j][k]))**2#当前基因量子位(a|0+b|1)的0的概率幅的范数平方，即量子位角度余弦值的平方a**2
                    if alpha2 > r:#量子位余弦平方若大于随机比较值
                        depop_tem2.append('0')#当前量子位取0，即当前基因为0，将字符‘0’加入当前染色体列表
                    else:#量子位余弦平方若小于随机比较值
                        depop_tem2.append('1')#当前量子位取1，即当前基因为1，将字符‘1’加入当前染色体列表
                depopR_temnow=''.join(depop_tem2)#将当前染色体列表变为二进制字符串
                depop_tem=int(depopR_temnow,2)#将当前染色体二进制字符串变为当前染色体十进制数
                depopR_tem.append(depopR_temnow)
                depop_tem1.append(self.min[j]+depop_tem*(self.max[j]-self.min[j])/(2**self.chrlen[j]-1))
                #将前染色体十进制数解码为解码后的十进制数，并正式加入当前个体列表
            depopR.append(depopR_tem)#将当前染色体二进制字符串加入二进制种群列表中
            depop.append(depop_tem1)#将当前个体列表加入种群列表中
        return depopR,depop#返回解码后的种群列表
    
    def fitness(self,pop,train,test):
        '''计算当前代的适应度，输入：pop,train,test；输出：depopR,depop,fitnow
        pop:当前种群量子位个体列表[[[gen1染色体1],[gen1染色体2],...],[[gen2染色体1]，[gen2染色体2],...],...]
        train:训练数据[x_tr,y_tr]
        test:测试数据[x_te,y_te]
        depopR:[[个体1的染色体1的构造二进制,个体1的染色体2的构造二进制..],..,[个体chr_len的染色体1的构造二进制,..,个体chr_len的染色体chr_num的构造二进制]]
        depop:[[个体1的染色体1的十进制值,个体1的染色体2的十进制值..],..,[个体chr_len的染色体1的十进制值,..,个体chr_len的染色体chr_num的十进制值]]
        fitnow:当前种群适应度列表[pop1fit,pop2fit,...,poppopsizefit]
        '''
        depopR,depop=self.decode(pop.copy())#种群解码后二进制序列列表和种群解码后十进制列表
        fitnow=[]#当前种群适应度列表[pop1fit,pop2fit,...,poppopsizefit]
        for i in range(self.popsize):#遍历当前种群所有个体
            rgs=svm.SVR(C=depop[i][0],kernel='rbf',gamma=depop[i][1],epsilon = 0.04)#根据当前个体建立SVR模型
            rgs.fit(train[0],train[1])#使用训练集数据训练模型
            yte_calcu=rgs.predict(test[0]) #计算测试集的预测值
            rmse=self.RMSE(test[1], yte_calcu)#比较测试集的预测值和真实值，计算模型均方根误差，以误差倒数作为适应度
            fitnow.append(1.0/rmse)#将当前个体建立的模型均方根误差作为适应度，加入当前种群的适应度列表
        return depopR,depop,fitnow
    
    def RMSE(self,x,y):
        num=len(x)
        sum=0
        for i in range(num):
            sum+=math.pow((x[i]-y[i]),2)
        sum=sum/num
        JFG=math.sqrt(sum)
        return JFG
    
    def rot(self,pop,depopR,fitnow,bestdepopRall,bestfitall):
        '''旋转操作input:pop,depopR,fitnow,bestdepopRall;output:rotpop
        pop:待旋转的种群[[[个体1的染色体1],[个体1的染色体2]..],..,[[个体chr_len的染色体1],..,[个体chr_len的染色体chr_num]]]
        depopR:待旋转种群的构造二进制序列列表[[个体1的染色体1的构造二进制,个体1的染色体2的构造二进制..],..]
        fitnow:当前代的种群适应度列表[pop1fit,pop2fit,...,poppopsizefit]
        bestdepopRall:是所有代中最佳适应度的构造二进制序列个体
        bestfitall:历史最大适应度
        rotpop:旋转后的种群[[[个体1的染色体1],[个体1的染色体2]..],..,[[个体chr_len的染色体1],..,[个体chr_len的染色体chr_num]]]
        '''
        for i in range(self.popsize):#遍历种群所有个体
            boolrb=(fitnow[i]>bestfitall)
            for j in range(self.chrnum):#遍历个体的所有染色体
                for k in range(self.chrlen[j]):#遍历染色体的所有基因
                    r=int(depopR[i][j][k])#当前基因的构造量子态二进制位，0或1
                    b=int(bestdepopRall[j][k])#历史最好个体当前基因的构造量子态二进制位，0或1
                    thita=pop[i][j][k]#当前基因量子态角度
                    thitai=self.gate(r,b,boolrb,thita)#当前基因量子态旋转角度
                    pop[i][j][k]=thita+thitai#当前基因量子态旋转后的角度
        rotpop=pop.copy()#旋转后的种群
        return rotpop
    
    def gate(self,r,b,boolrb,thita):
        '''旋转门，input:r,b,boolrb;output:thitai
        r:当前量子位
        b:最佳当前量子位
        boolrb:r所在个体是否优于b所在个体，True/False
        thita:当前量子态角度
        thitai:当前量子态角度的转动值
        '''
        alpha=np.cos(thita)#量子态0的振幅
        beita=np.sin(thita)#量子态1的振幅
        if r==0 and b==0:
            thitai=0
        elif r==0 and b==1 and (not boolrb):
            thitai=0
        elif r==0 and b==1 and boolrb and alpha*beita>0:
            thitai=-0.05*np.pi 
        elif r==0 and b==1 and boolrb and (alpha*beita<0 or alpha==0):
            thitai=0.05*np.pi 
        elif r==0 and b==1 and boolrb and beita==0:
            thitai=0
        elif r==1 and b==0 and (not boolrb) and alpha*beita>0:
            thitai=-0.01*np.pi
        elif r==1 and b==0 and (not boolrb) and (alpha*beita<0 or alpha==0):
            thitai=0.01*np.pi
        elif r==1 and b==0 and (not boolrb) and beita==0:
            thitai=0
        elif r==1 and b==0 and boolrb and alpha*beita>0:
            thitai=0.025*np.pi
        elif r==1 and b==0 and boolrb and (alpha*beita<0 or beita==0):
            thitai=-0.025*np.pi
        elif r==1 and b==0 and boolrb and alpha==0:
            thitai=0
        elif r==1 and b==1 and (not boolrb) and alpha*beita>0:
            thitai=0.05*np.pi 
        elif r==1 and b==1 and (not boolrb) and (alpha*beita<0 or beita==0):
            thitai=-0.05*np.pi
        elif r==1 and b==1 and alpha==0:
            thitai=0
        elif r==1 and b==1 and boolrb and alpha*beita>0:
            thitai=0.025*np.pi 
        else:
            thitai=-0.025*np.pi
        return thitai
    
    def change(self,pop):
        '''交叉操作，input:pop;output:children
        pop:待交叉种群[[[个体1的染色体1],[个体1的染色体2]..],..,[[个体chr_len的染色体1],..,[个体chr_len的染色体chr_num]]]
        children:交叉后的种群[[[个体1的染色体1],[个体1的染色体2]..],..,[[个体chr_len的染色体1],..,[个体chr_len的染色体chr_num]]]
        '''
        random.shuffle(pop)#将种群打乱
        half=int(self.popsize/2)#计算种群数量的一半为多少，便于将一半分为父本，一半分为母本
        father=pop[0:half]#前一半记为父本
        mother=pop[half:]#后一半记为母本
        children=[]#建立交叉后的子代列表
        for i in range(half):
            son=[]
            daughter=[]
            for j in range(len(pop[0])):
                if np.random.uniform(0,1)<=self.px:
                    copint = np.random.randint(0,int(len(father[i][j])/2))  #交叉点选择,将交叉基因数限制在总基因数一半以下 
                    son.append(father[i][j][:copint]+mother[i][j][copint:])         #子代1
                    daughter.append(mother[i][j][:copint]+father[i][j][copint:])    #子代2
                else:
                    son.append(father[i][j])#未交叉子代1的直接遗传父代
                    daughter.append(mother[i][j])#为交叉的子代2直接遗传母代
            children.append(son)#将交叉好的个体放入子代列表
            children.append(daughter)#将交叉好的个体放入子代列表
        if len(father)!=len(mother):#当种群为基数，母代的最后没有进行交叉，也没进入子代，子种群少一个个体
            children.append(mother[-1])#将母代最后一个未交叉的个体移入子代
        return children
    
    def variation(self,pop):#变异操作
        for i in range(self.popsize):
            for j in range(len(pop[i])):
                if np.random.uniform(0,1)<=self.pc:
                    position=np.random.randint(0,len(pop[i][j]))
                    pop[i][j][position]=1.5*np.pi-pop[i][j][position]
        return pop
    
    def run(self,train,test):
        pop=self.initial()#建立初始种群
        bestfit=[]#建立各代最优适应度的列表[bestfit1,bestfit2,...,bestfitgen]
        bestdepopR=[]#建立各代最优适应度对应的构造二进制序列个体列表[[gen1的染色体1的构造二进制,gen1的染色体2的构造二进制..],..]
        bestdepop=[]#建立各代最优适应度对应的最优十进制个体列表[[gen1染色体1十进制,gen1染色体2十进制,...],[gen2染色体1十进制，gen2染色体2十进制,...],...]
        allbestfit=[]#建立每次循环的所有代最佳适应度列表[第一代最佳适应度,前两代最佳适应度,...,前gen代最佳适应度]
        bestfitall=0#所有代中的最大适应度
        localgen=0#设置初始陷入局部最优的代数为0
        for gen_i in range(self.gen):#遍历所有进化代数
            depopR,depop,fitnow=self.fitness(pop, train, test)#当前代的种群解码后二进制列表、当前代的种群解码后列表和当前代的种群适应度列表
            fitnow_np=np.array(fitnow)#将列表转化为数组，方便对其中最大值和最小值操作
            bestfitnow=fitnow_np.max()#将当前代种群的最大适应度找出
            bestfitnow_i=fitnow_np.argmax()#将当前代种群的最大适应度位置找出
            if bestfitnow>bestfitall:
                bestfitall=bestfitnow #bestfitall是所有代最佳适应度bestdepopall
                bestdepopRall=depopR[bestfitnow_i]#bestdepopRall是所有代中最佳适应度的构造二进制序列个体[chr1的二进制序列,chr2的二进制序列]
                bestdepopall=depop[bestfitnow_i]#bestdepopall是所有代最佳适应度对应的十进制个体列表[chr1的十进制,chr2的十进制]
                bestpopall=pop[bestfitnow_i]#所有代最佳适应度对应的个体列表[[chr1的量子态角度列表],[chr2的量子态角度列表]]
                localgen=0#最佳适应度改变，则将局部最优代数设为0
            else:
                localgen+=1#最佳适应度不改变，则将局部最优代数加1
            allbestfit.append(bestfitall)#将所有代个体适应度里最大的那个加入每次循环的所有代最佳适应度列表
            bestfit.append(bestfitnow)#将当前代种群的最大适应度加入各代最优适应度列表
            bestdepopR.append(depopR[bestfitnow_i])#将当前代最大适应度对应的个体构造二进制序列加入各代最优适应度对应的构造二进制序列个体列表
            bestdepop.append(depop[bestfitnow_i])#将当前代种群的最大适应度对应的十进制个体找出并加入各代最优适应度个体列表
            if localgen>self.maxgen:#灾变操作
                pop=self.initial()#种群重构
                localgen=0#局部平缓代数归零
                pop.pop()#将种群中最后一个个体弹出，这一步是随机弹出哪个都行，为了让出一个空位给历史最佳适应度个体
                pop.append(bestpopall)#将历史最优个体加入该重构列表
                continue
            pop=self.rot(pop.copy(), depopR, fitnow, bestdepopRall, bestfitall)#量子门旋转
            pop=self.change(pop.copy())#交叉
            pop=self.variation(pop.copy())#变异
        print('将所有代个体适应度里最大的那个加入每次循环的所有代最佳适应度列表:\n',allbestfit)
        print('将当前代种群的最大适应度加入各代最优适应度列表:\n',bestfit)
        print('将当前代最大适应度对应的个体构造二进制序列加入各代最优适应度对应的构造二进制序列个体列表:\n',bestdepopR)
        print('将当前代种群的最大适应度对应的十进制个体找出并加入各代最优适应度个体列表:\n',bestdepop)
        print('所有代中最佳适应度的构造二进制序列个体:\n',bestdepopRall)
        print('所有代最佳适应度对应的十进制个体列表:\n',bestdepopall)
        print('所有代最佳适应度对应的个体列表[[chr1的量子态角度列表],[chr2的量子态角度列表]]:\n',bestpopall)
        return [allbestfit,bestfit,bestdepopR,bestdepop,bestdepopRall,bestdepopall,bestpopall]

class MyFigure(FCV):
    def __init__ (self,chang=100,kuan=100,dpi=100):
        width=chang/dpi 
        height=kuan/dpi
        self.fig=Figure(figsize=(width,height),dpi=dpi)
        super().__init__(self.fig)
        self.axes=self.fig.add_subplot(1,1,1)
        
class MyFigure3D(FCV):
    def __init__ (self, chang=100,kuan=100,dpi=100):
        width=chang/dpi 
        height=kuan/dpi
        self.fig=Figure(figsize=(width, height),dpi=dpi)
        super().__init__(self.fig)
        self.axes=Axes3D(self.fig)
        return
    
class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(590, 467)
        self.tableWidget = QtWidgets.QTableWidget(Form)
        self.tableWidget.setGeometry(QtCore.QRect(10, 10, 571, 411))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.tableWidget.setFont(font)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setRowCount(10)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(5, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(6, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(7, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(8, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(9, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setGeometry(QtCore.QRect(10, 430, 571, 28))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.retranslateUi(Form)
        self.pushButton.clicked.connect(Form.close)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Model performances"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("Form", "root mean square error"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("Form", "mean absolute error"))
        item = self.tableWidget.verticalHeaderItem(2)
        item.setText(_translate("Form", "maximum absolute error"))
        item = self.tableWidget.verticalHeaderItem(3)
        item.setText(_translate("Form", "average relative error"))
        item = self.tableWidget.verticalHeaderItem(4)
        item.setText(_translate("Form", "coefficient of determination"))
        item = self.tableWidget.verticalHeaderItem(5)
        item.setText(_translate("Form", "revised coefficient of determination"))
        item = self.tableWidget.verticalHeaderItem(6)
        item.setText(_translate("Form", "fitting coefficient"))
        item = self.tableWidget.verticalHeaderItem(7)
        item.setText(_translate("Form", "fitting intercept"))
        item = self.tableWidget.verticalHeaderItem(8)
        item.setText(_translate("Form", "correlation coefficient"))
        item = self.tableWidget.verticalHeaderItem(9)
        item.setText(_translate("Form", "Error standard deviation"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("Form", "Training Set"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("Form", "Test Set"))
        self.pushButton.setText(_translate("Form", "OK"))
        
class Ui_Form2(object):
    def setupUi(self, ModelSelect, lujing):
        self.ModelSelect=ModelSelect
        ModelSelect.setObjectName("ModelSelect")
        ModelSelect.resize(628, 480)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        ModelSelect.setFont(font)
        self.gridLayout = QtWidgets.QGridLayout(ModelSelect)
        self.gridLayout.setObjectName("gridLayout")
        self.listWidget = QtWidgets.QListWidget(ModelSelect)
        self.listWidget.setObjectName("listWidget")
        self.gridLayout.addWidget(self.listWidget, 0, 0, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(ModelSelect)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout.addWidget(self.textEdit, 0, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.pushButton_3 = QtWidgets.QPushButton(ModelSelect)
        self.pushButton_3.setObjectName("pushButton_3")
        self.horizontalLayout.addWidget(self.pushButton_3)
        self.pushButton_2 = QtWidgets.QPushButton(ModelSelect)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 2)
        self.pushButton_2.clicked.connect(self.OK)
        self.pushButton_3.clicked.connect(self.View)
        self.retranslateUi(ModelSelect)
        QtCore.QMetaObject.connectSlotsByName(ModelSelect)
        self.lujing=lujing

    def retranslateUi(self, ModelSelect):
        _translate = QtCore.QCoreApplication.translate
        ModelSelect.setWindowTitle(_translate("ModelSelect", "Model Selecting"))
        self.pushButton_3.setText(_translate("ModelSelect", "View"))
        self.pushButton_2.setText(_translate("ModelSelect", "Ok"))
        
    def OK(self):
        self.model=self.lujing+'\\'+self.listWidget.selectedItems()[0].text()
        self.ModelSelect.close()
        return
        
    def View(self):
        sa=self.listWidget.selectedItems()[0].text()
        sb='Parameters'+sa[5:]
        sc=self.lujing+'\\'+sb
        self.sd=readpkl(sc)
        self.textEdit.setPlainText(self.sd)
        return

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(821, 640)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        spacerItem = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem, 2, 0, 1, 1)
        spacerItem1 = QtWidgets.QSpacerItem(41, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem1, 2, 11, 1, 1)
        spacerItem2 = QtWidgets.QSpacerItem(41, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem2, 2, 4, 1, 1)
        spacerItem3 = QtWidgets.QSpacerItem(41, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem3, 2, 13, 1, 1)
        spacerItem4 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem4, 2, 5, 1, 1)
        spacerItem5 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem5, 2, 12, 1, 1)
        spacerItem6 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem6, 2, 9, 1, 1)
        spacerItem7 = QtWidgets.QSpacerItem(41, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem7, 2, 8, 1, 1)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 0, 1, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.centralwidget)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout.addWidget(self.line_3, 1, 3, 3, 1)
        self.line_4 = QtWidgets.QFrame(self.centralwidget)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout.addWidget(self.line_4, 1, 0, 3, 1)
        self.stackedWidget = QtWidgets.QStackedWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.stackedWidget.setFont(font)
        self.stackedWidget.setObjectName("stackedWidget")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.page)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(20)
        font.setBold(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.verticalLayout_2.addWidget(self.label)
        self.label_15 = QtWidgets.QLabel(self.page)
        self.label_15.setText("")
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.verticalLayout_2.addWidget(self.label_15)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_9 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_9.setFont(font)
        self.label_9.setFrameShape(QtWidgets.QFrame.Box)
        self.label_9.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 5, 2, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_3.addWidget(self.label_10, 5, 3, 1, 2)
        self.label_7 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_7.setFont(font)
        self.label_7.setFrameShape(QtWidgets.QFrame.Box)
        self.label_7.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_7.setAlignment(QtCore.Qt.AlignCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 4, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_5.setFont(font)
        self.label_5.setFrameShape(QtWidgets.QFrame.Box)
        self.label_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_13.setFont(font)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 3, 2, 1, 3)
        self.pushButton_4 = QtWidgets.QPushButton(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_4.setFont(font)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_3.addWidget(self.pushButton_4, 5, 1, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_12.setFont(font)
        self.label_12.setObjectName("label_12")
        self.gridLayout_3.addWidget(self.label_12, 4, 2, 1, 3)
        self.label_14 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_14.setFont(font)
        self.label_14.setObjectName("label_14")
        self.gridLayout_3.addWidget(self.label_14, 2, 2, 1, 3)
        self.label_8 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_8.setFont(font)
        self.label_8.setFrameShape(QtWidgets.QFrame.Box)
        self.label_8.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_8.setAlignment(QtCore.Qt.AlignCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 5, 0, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_6.setFont(font)
        self.label_6.setFrameShape(QtWidgets.QFrame.Box)
        self.label_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout_3.addWidget(self.label_6, 3, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 0, 2, 1, 3)
        self.label_11 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_11.setFont(font)
        self.label_11.setAlignment(QtCore.Qt.AlignCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout_3.addWidget(self.label_11, 1, 1, 1, 4)
        self.label_2 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setFrameShape(QtWidgets.QFrame.Box)
        self.label_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 0, 0, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.page)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_4.setFont(font)
        self.label_4.setFrameShape(QtWidgets.QFrame.Box)
        self.label_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout_3.addWidget(self.label_4, 1, 0, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout_3)
        self.stackedWidget.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_16 = QtWidgets.QLabel(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_16.setFont(font)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout.addWidget(self.label_16)
        self.lineEdit = QtWidgets.QLineEdit(self.page_2)
        self.lineEdit.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit.setFont(font)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout.addWidget(self.lineEdit)
        self.label_17 = QtWidgets.QLabel(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_17.setFont(font)
        self.label_17.setObjectName("label_17")
        self.horizontalLayout.addWidget(self.label_17)
        self.dateTimeEdit = QtWidgets.QDateTimeEdit(self.page_2)
        self.dateTimeEdit.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit.setFont(font)
        self.dateTimeEdit.setObjectName("dateTimeEdit")
        self.horizontalLayout.addWidget(self.dateTimeEdit)
        self.gridLayout_6.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.gridLayout_4 = QtWidgets.QGridLayout()
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_2.setFont(font)
        self.lineEdit_2.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout_4.addWidget(self.lineEdit_2, 1, 2, 1, 1)
        self.comboBox_2 = QtWidgets.QComboBox(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.comboBox_2.setFont(font)
        self.comboBox_2.setObjectName("comboBox_2")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.comboBox_2.addItem("")
        self.gridLayout_4.addWidget(self.comboBox_2, 0, 2, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_19.setFont(font)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.gridLayout_4.addWidget(self.label_19, 1, 1, 1, 1)
        self.label_18 = QtWidgets.QLabel(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_18.setFont(font)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.gridLayout_4.addWidget(self.label_18, 0, 1, 1, 1)
        self.graphicsView = QtWidgets.QGraphicsView(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.graphicsView.sizePolicy().hasHeightForWidth())
        self.graphicsView.setSizePolicy(sizePolicy)
        self.graphicsView.setMinimumSize(QtCore.QSize(220, 125))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.graphicsView.setFont(font)
        self.graphicsView.setObjectName("graphicsView")
        self.gridLayout_4.addWidget(self.graphicsView, 0, 3, 2, 1)
        self.gridLayout_6.addLayout(self.gridLayout_4, 1, 0, 1, 1)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tableWidget = QtWidgets.QTableWidget(self.page_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tableWidget.sizePolicy().hasHeightForWidth())
        self.tableWidget.setSizePolicy(sizePolicy)
        self.tableWidget.setMinimumSize(QtCore.QSize(0, 140))
        self.tableWidget.setMaximumSize(QtCore.QSize(16777215, 140))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.tableWidget.setFont(font)
        self.tableWidget.setLineWidth(3)
        self.tableWidget.setMidLineWidth(3)
        self.tableWidget.setRowCount(2)
        self.tableWidget.setColumnCount(4)
        self.tableWidget.setObjectName("tableWidget")
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget.setHorizontalHeaderItem(3, item)
        self.gridLayout_5.addWidget(self.tableWidget, 1, 0, 1, 3)
        self.pushButton_7 = QtWidgets.QPushButton(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_7.setFont(font)
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout_5.addWidget(self.pushButton_7, 4, 1, 1, 2)
        self.label_21 = QtWidgets.QLabel(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_21.setFont(font)
        self.label_21.setObjectName("label_21")
        self.gridLayout_5.addWidget(self.label_21, 2, 0, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_20.setFont(font)
        self.label_20.setObjectName("label_20")
        self.gridLayout_5.addWidget(self.label_20, 0, 0, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_22.setFont(font)
        self.label_22.setObjectName("label_22")
        self.gridLayout_5.addWidget(self.label_22, 3, 0, 1, 1)
        self.spinBox = QtWidgets.QSpinBox(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.spinBox.setFont(font)
        self.spinBox.setProperty("value", 4)
        self.spinBox.setObjectName("spinBox")
        self.gridLayout_5.addWidget(self.spinBox, 0, 1, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_3.setFont(font)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout_5.addWidget(self.lineEdit_3, 2, 1, 1, 2)
        self.spinBox_2 = QtWidgets.QSpinBox(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.spinBox_2.setFont(font)
        self.spinBox_2.setObjectName("spinBox_2")
        self.gridLayout_5.addWidget(self.spinBox_2, 3, 1, 1, 2)
        self.pushButton_ok=QtWidgets.QPushButton("OK")
        font.setPointSize(12)
        self.pushButton_ok.setFont(font)
        self.gridLayout_5.addWidget(self.pushButton_ok, 0, 2, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 2, 0, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.pushButton_5 = QtWidgets.QPushButton(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_5.setFont(font)
        self.pushButton_5.setObjectName("pushButton_5")
        self.horizontalLayout_2.addWidget(self.pushButton_5)
        spacerItem8 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_2.addItem(spacerItem8)
        self.pushButton_6 = QtWidgets.QPushButton(self.page_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_6.setFont(font)
        self.pushButton_6.setObjectName("pushButton_6")
        self.horizontalLayout_2.addWidget(self.pushButton_6)
        self.gridLayout_6.addLayout(self.horizontalLayout_2, 3, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_2)
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.pushButton_8 = QtWidgets.QPushButton(self.page_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_8.setFont(font)
        self.pushButton_8.setObjectName("pushButton_8")
        self.horizontalLayout_3.addWidget(self.pushButton_8)
        spacerItem9 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_3.addItem(spacerItem9)
        self.pushButton_9 = QtWidgets.QPushButton(self.page_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_9.setFont(font)
        self.pushButton_9.setObjectName("pushButton_9")
        self.horizontalLayout_3.addWidget(self.pushButton_9)
        self.gridLayout_7.addLayout(self.horizontalLayout_3, 5, 0, 1, 3)
        self.comboBox = QtWidgets.QComboBox(self.page_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.gridLayout_7.addWidget(self.comboBox, 1, 0, 1, 1)
        self.listWidget_3 = QtWidgets.QListWidget(self.page_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget_3.sizePolicy().hasHeightForWidth())
        self.listWidget_3.setSizePolicy(sizePolicy)
        self.listWidget_3.setMinimumSize(QtCore.QSize(150, 0))
        self.listWidget_3.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.listWidget_3.setFont(font)
        self.listWidget_3.setObjectName("listWidget_3")
        item = QtWidgets.QListWidgetItem()
        self.listWidget_3.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_3.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_3.addItem(item)
        self.gridLayout_7.addWidget(self.listWidget_3, 2, 0, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.page_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_25.sizePolicy().hasHeightForWidth())
        self.label_25.setSizePolicy(sizePolicy)
        self.label_25.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_25.setFont(font)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.gridLayout_7.addWidget(self.label_25, 4, 0, 1, 1)
        self.pushButton_10 = QtWidgets.QPushButton(self.page_3)
        self.pushButton_10.setMinimumSize(QtCore.QSize(150, 0))
        self.pushButton_10.setMaximumSize(QtCore.QSize(150, 16777215))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_10.setFont(font)
        self.pushButton_10.setObjectName("pushButton_10")
        self.gridLayout_7.addWidget(self.pushButton_10, 3, 0, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_23 = QtWidgets.QLabel(self.page_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_23.setFont(font)
        self.label_23.setObjectName("label_23")
        self.horizontalLayout_4.addWidget(self.label_23)
        self.lineEdit_4 = QtWidgets.QLineEdit(self.page_3)
        self.lineEdit_4.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_4.setFont(font)
        self.lineEdit_4.setObjectName("lineEdit_4")
        self.horizontalLayout_4.addWidget(self.lineEdit_4)
        self.label_24 = QtWidgets.QLabel(self.page_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_24.setFont(font)
        self.label_24.setObjectName("label_24")
        self.horizontalLayout_4.addWidget(self.label_24)
        self.dateTimeEdit_2 = QtWidgets.QDateTimeEdit(self.page_3)
        self.dateTimeEdit_2.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_2.setFont(font)
        self.dateTimeEdit_2.setObjectName("dateTimeEdit_2")
        self.horizontalLayout_4.addWidget(self.dateTimeEdit_2)
        self.gridLayout_7.addLayout(self.horizontalLayout_4, 0, 0, 1, 3)
        self.graphicsView_2 = QtWidgets.QGraphicsView(self.page_3)
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.gridLayout_7.addWidget(self.graphicsView_2, 1, 1, 2, 2)
        self.tableWidget_2 = QtWidgets.QTableWidget(self.page_3)
        self.tableWidget_2.setObjectName("tableWidget_2")
        self.tableWidget_2.setColumnCount(3)
        self.tableWidget_2.setRowCount(5)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setVerticalHeaderItem(4, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_2.setHorizontalHeaderItem(2, item)
        self.gridLayout_7.addWidget(self.tableWidget_2, 3, 1, 2, 2)
        self.stackedWidget.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.gridLayout_11 = QtWidgets.QGridLayout(self.page_4)
        self.gridLayout_11.setObjectName("gridLayout_11")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_26 = QtWidgets.QLabel(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_26.setFont(font)
        self.label_26.setObjectName("label_26")
        self.horizontalLayout_5.addWidget(self.label_26)
        self.lineEdit_5 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_5.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_5.setFont(font)
        self.lineEdit_5.setObjectName("lineEdit_5")
        self.horizontalLayout_5.addWidget(self.lineEdit_5)
        self.label_27 = QtWidgets.QLabel(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_27.setFont(font)
        self.label_27.setObjectName("label_27")
        self.horizontalLayout_5.addWidget(self.label_27)
        self.dateTimeEdit_3 = QtWidgets.QDateTimeEdit(self.page_4)
        self.dateTimeEdit_3.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_3.setFont(font)
        self.dateTimeEdit_3.setObjectName("dateTimeEdit_3")
        self.horizontalLayout_5.addWidget(self.dateTimeEdit_3)
        self.gridLayout_11.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.pushButton_14 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_14.setFont(font)
        self.pushButton_14.setObjectName("pushButton_14")
        self.horizontalLayout_7.addWidget(self.pushButton_14)
        self.pushButton_13 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_13.setFont(font)
        self.pushButton_13.setObjectName("pushButton_13")
        self.horizontalLayout_7.addWidget(self.pushButton_13)
        self.pushButton_AVShow = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_AVShow.setFont(font)
        self.pushButton_AVShow.setObjectName("pushButton_AVShow")
        self.horizontalLayout_7.addWidget(self.pushButton_AVShow)
        self.gridLayout_11.addLayout(self.horizontalLayout_7, 1, 0, 1, 1)
        self.gridLayout_8 = QtWidgets.QGridLayout()
        self.gridLayout_8.setObjectName("gridLayout_8")
        self.pushButton_18 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_18.setFont(font)
        self.pushButton_18.setObjectName("pushButton_18")
        self.gridLayout_8.addWidget(self.pushButton_18, 1, 3, 1, 1)
        self.pushButton_15 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_15.setFont(font)
        self.pushButton_15.setObjectName("pushButton_15")
        self.gridLayout_8.addWidget(self.pushButton_15, 0, 0, 1, 1)
        self.graphicsView_3 = QtWidgets.QGraphicsView(self.page_4)
        self.graphicsView_3.setMinimumSize(QtCore.QSize(250, 200))
        self.graphicsView_3.setMaximumSize(QtCore.QSize(11111111, 200))
        self.graphicsView_3.setObjectName("graphicsView_3")
        self.gridLayout_8.addWidget(self.graphicsView_3, 0, 1, 2, 2)
        self.pushButton_16 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_16.setFont(font)
        self.pushButton_16.setObjectName("pushButton_16")
        self.gridLayout_8.addWidget(self.pushButton_16, 0, 3, 1, 1)
        self.pushButton_17 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_17.setFont(font)
        self.pushButton_17.setObjectName("pushButton_17")
        self.gridLayout_8.addWidget(self.pushButton_17, 1, 0, 1, 1)
        self.gridLayout_11.addLayout(self.gridLayout_8, 2, 0, 1, 1)
        self.gridLayout_9 = QtWidgets.QGridLayout()
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.label_29 = QtWidgets.QLabel(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_29.sizePolicy().hasHeightForWidth())
        self.label_29.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.gridLayout_9.addWidget(self.label_29, 0, 3, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.page_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_28.sizePolicy().hasHeightForWidth())
        self.label_28.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.gridLayout_9.addWidget(self.label_28, 0, 1, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_6.setMaximumSize(QtCore.QSize(107, 16777215))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_6.setFont(font)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_9.addWidget(self.lineEdit_6, 0, 2, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_7.setMaximumSize(QtCore.QSize(107, 16777215))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_7.setFont(font)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout_9.addWidget(self.lineEdit_7, 0, 4, 1, 1)
        self.pushButton_20 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_20.setFont(font)
        self.pushButton_20.setObjectName("pushButton_20")
        self.gridLayout_9.addWidget(self.pushButton_20, 1, 1, 1, 4)
        self.pushButton_19 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_19.setFont(font)
        self.pushButton_19.setObjectName("pushButton_19")
        self.gridLayout_9.addWidget(self.pushButton_19, 0, 0, 2, 1)
        self.gridLayout_11.addLayout(self.gridLayout_9, 3, 0, 1, 1)
        self.gridLayout_10 = QtWidgets.QGridLayout()
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.label_88 = QtWidgets.QLabel(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_88.setFont(font)
        self.label_88.setObjectName("label_88")
        self.gridLayout_10.addWidget(self.label_88, 2, 1, 1, 1)
        self.label_87 = QtWidgets.QLabel(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_87.setFont(font)
        self.label_87.setObjectName("label_87")
        self.gridLayout_10.addWidget(self.label_87, 0, 0, 1, 2)
        self.label_89 = QtWidgets.QLabel(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_89.setFont(font)
        self.label_89.setObjectName("label_89")
        self.gridLayout_10.addWidget(self.label_89, 0, 2, 1, 1)
        self.lineEdit_34 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_34.setObjectName("lineEdit_34")
        self.gridLayout_10.addWidget(self.lineEdit_34, 0, 3, 1, 1)
        self.label_90 = QtWidgets.QLabel(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_90.setFont(font)
        self.label_90.setObjectName("label_90")
        self.gridLayout_10.addWidget(self.label_90, 2, 2, 1, 1)
        self.lineEdit_35 = QtWidgets.QLineEdit(self.page_4)
        self.lineEdit_35.setObjectName("lineEdit_35")
        self.gridLayout_10.addWidget(self.lineEdit_35, 2, 3, 1, 1)
        self.horizontalSlider_8 = QtWidgets.QSlider(self.page_4)
        self.horizontalSlider_8.setProperty("value", 0)
        self.horizontalSlider_8.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider_8.setObjectName("horizontalSlider_8")
        self.gridLayout_10.addWidget(self.horizontalSlider_8, 2, 0, 1, 1)
        self.gridLayout_11.addLayout(self.gridLayout_10, 4, 0, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.pushButton_11 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_11.setFont(font)
        self.pushButton_11.setObjectName("pushButton_11")
        self.horizontalLayout_6.addWidget(self.pushButton_11)
        self.pushButton_cancel = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_cancel.setFont(font)
        self.pushButton_cancel.setObjectName("self.pushButton_cancel")
        self.horizontalLayout_6.addWidget(self.pushButton_cancel)
        spacerItem10 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_6.addItem(spacerItem10)
        self.pushButton_12 = QtWidgets.QPushButton(self.page_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_12.setFont(font)
        self.pushButton_12.setObjectName("pushButton_12")
        self.horizontalLayout_6.addWidget(self.pushButton_12)
        self.gridLayout_11.addLayout(self.horizontalLayout_6, 5, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_4)
        self.page_5 = QtWidgets.QWidget()
        self.page_5.setObjectName("page_5")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.page_5)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label_30 = QtWidgets.QLabel(self.page_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        self.horizontalLayout_8.addWidget(self.label_30)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.page_5)
        self.lineEdit_8.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_8.setFont(font)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.horizontalLayout_8.addWidget(self.lineEdit_8)
        self.label_31 = QtWidgets.QLabel(self.page_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_31.setFont(font)
        self.label_31.setObjectName("label_31")
        self.horizontalLayout_8.addWidget(self.label_31)
        self.dateTimeEdit_4 = QtWidgets.QDateTimeEdit(self.page_5)
        self.dateTimeEdit_4.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_4.setFont(font)
        self.dateTimeEdit_4.setObjectName("dateTimeEdit_4")
        self.horizontalLayout_8.addWidget(self.dateTimeEdit_4)
        self.verticalLayout_3.addLayout(self.horizontalLayout_8)
        self.tabWidget = QtWidgets.QTabWidget(self.page_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.tabWidget.setFont(font)
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.gridLayout_12 = QtWidgets.QGridLayout(self.tab)
        self.gridLayout_12.setObjectName("gridLayout_12")
        self.graphicsView_5 = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView_5.setObjectName("graphicsView_5")
        self.gridLayout_12.addWidget(self.graphicsView_5, 2, 0, 1, 2)
        self.graphicsView_4 = QtWidgets.QGraphicsView(self.tab)
        self.graphicsView_4.setObjectName("graphicsView_4")
        self.gridLayout_12.addWidget(self.graphicsView_4, 2, 2, 1, 2)
        self.label_32 = QtWidgets.QLabel(self.tab)
        self.label_32.setObjectName("label_32")
        self.gridLayout_12.addWidget(self.label_32, 3, 0, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.tab)
        self.label_34.setObjectName("label_34")
        self.gridLayout_12.addWidget(self.label_34, 1, 0, 1, 1)
        self.pushButton_23 = QtWidgets.QPushButton(self.tab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_23.setFont(font)
        self.pushButton_23.setObjectName("pushButton_23")
        self.gridLayout_12.addWidget(self.pushButton_23, 0, 3, 1, 1)
        self.progressBar = QtWidgets.QProgressBar(self.tab)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.progressBar.setFont(font)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setObjectName("progressBar")
        self.gridLayout_12.addWidget(self.progressBar, 0, 0, 1, 3)
        self.pushButton_25 = QtWidgets.QPushButton(self.tab)
        self.pushButton_25.setObjectName("pushButton_25")
        self.gridLayout_12.addWidget(self.pushButton_25, 3, 3, 1, 1)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.tab)
        self.lineEdit_9.setEnabled(False)
        self.lineEdit_9.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout_12.addWidget(self.lineEdit_9, 4, 0, 1, 4)
        self.label_35 = QtWidgets.QLabel(self.tab)
        self.label_35.setObjectName("label_35")
        self.gridLayout_12.addWidget(self.label_35, 1, 3, 1, 1)
        self.tabWidget.addTab(self.tab, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_13 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_13.setObjectName("gridLayout_13")
        self.graphicsView_7 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_7.setObjectName("graphicsView_7")
        self.gridLayout_13.addWidget(self.graphicsView_7, 2, 0, 1, 2)
        self.label_33 = QtWidgets.QLabel(self.tab_2)
        self.label_33.setObjectName("label_33")
        self.gridLayout_13.addWidget(self.label_33, 3, 0, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.tab_2)
        self.label_36.setObjectName("label_36")
        self.gridLayout_13.addWidget(self.label_36, 1, 0, 1, 1)
        self.graphicsView_6 = QtWidgets.QGraphicsView(self.tab_2)
        self.graphicsView_6.setObjectName("graphicsView_6")
        self.gridLayout_13.addWidget(self.graphicsView_6, 2, 2, 1, 2)
        self.label_37 = QtWidgets.QLabel(self.tab_2)
        self.label_37.setObjectName("label_37")
        self.gridLayout_13.addWidget(self.label_37, 1, 3, 1, 1)
        self.pushButton_24 = QtWidgets.QPushButton(self.tab_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_24.setFont(font)
        self.pushButton_24.setObjectName("pushButton_24")
        self.gridLayout_13.addWidget(self.pushButton_24, 0, 3, 1, 1)
        self.progressBar_2 = QtWidgets.QProgressBar(self.tab_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.progressBar_2.setFont(font)
        self.progressBar_2.setProperty("value", 0)
        self.progressBar_2.setObjectName("progressBar_2")
        self.gridLayout_13.addWidget(self.progressBar_2, 0, 0, 1, 3)
        self.pushButton_26 = QtWidgets.QPushButton(self.tab_2)
        self.pushButton_26.setObjectName("pushButton_26")
        self.gridLayout_13.addWidget(self.pushButton_26, 3, 3, 1, 1)
        self.lineEdit_10 = QtWidgets.QLineEdit(self.tab_2)
        self.lineEdit_10.setEnabled(False)
        self.lineEdit_10.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_10.setObjectName("lineEdit_10")
        self.gridLayout_13.addWidget(self.lineEdit_10, 4, 0, 1, 4)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_14 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_14.setObjectName("gridLayout_14")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_38 = QtWidgets.QLabel(self.tab_3)
        self.label_38.setObjectName("label_38")
        self.horizontalLayout_10.addWidget(self.label_38)
        self.comboBox_3 = QtWidgets.QComboBox(self.tab_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.comboBox_3.setFont(font)
        self.comboBox_3.setObjectName("comboBox_3")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.comboBox_3.addItem("")
        self.horizontalLayout_10.addWidget(self.comboBox_3)
        self.label_39 = QtWidgets.QLabel(self.tab_3)
        self.label_39.setObjectName("label_39")
        self.horizontalLayout_10.addWidget(self.label_39)
        self.lineEdit_11 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_11.setObjectName("lineEdit_11")
        self.horizontalLayout_10.addWidget(self.lineEdit_11)
        self.label_40 = QtWidgets.QLabel(self.tab_3)
        self.label_40.setObjectName("label_40")
        self.horizontalLayout_10.addWidget(self.label_40)
        self.lineEdit_12 = QtWidgets.QLineEdit(self.tab_3)
        self.lineEdit_12.setObjectName("lineEdit_12")
        self.horizontalLayout_10.addWidget(self.lineEdit_12)
        self.pushButton_29 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_29.setObjectName("pushButton_29")
        self.horizontalLayout_10.addWidget(self.pushButton_29)
        self.gridLayout_14.addLayout(self.horizontalLayout_10, 0, 0, 1, 3)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.pushButton_31 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_31.setObjectName("pushButton_31")
        self.horizontalLayout_11.addWidget(self.pushButton_31)
        self.progressBar_3 = QtWidgets.QProgressBar(self.tab_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.progressBar_3.setFont(font)
        self.progressBar_3.setProperty("value", 0)
        self.progressBar_3.setObjectName("progressBar_3")
        self.horizontalLayout_11.addWidget(self.progressBar_3)
        self.pushButton_30 = QtWidgets.QPushButton(self.tab_3)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_30.setFont(font)
        self.pushButton_30.setObjectName("pushButton_30")
        self.horizontalLayout_11.addWidget(self.pushButton_30)
        self.gridLayout_14.addLayout(self.horizontalLayout_11, 1, 0, 1, 3)
        self.label_41 = QtWidgets.QLabel(self.tab_3)
        self.label_41.setObjectName("label_41")
        self.gridLayout_14.addWidget(self.label_41, 2, 2, 1, 1)
        self.graphicsView_9 = QtWidgets.QGraphicsView(self.tab_3)
        self.graphicsView_9.setObjectName("graphicsView_9")
        self.gridLayout_14.addWidget(self.graphicsView_9, 3, 0, 1, 1)
        self.graphicsView_8 = QtWidgets.QGraphicsView(self.tab_3)
        self.graphicsView_8.setObjectName("graphicsView_8")
        self.gridLayout_14.addWidget(self.graphicsView_8, 3, 2, 1, 1)
        self.pushButton_32 = QtWidgets.QPushButton(self.tab_3)
        self.pushButton_32.setObjectName("pushButton_32")
        self.gridLayout_14.addWidget(self.pushButton_32, 4, 2, 1, 1)
        self.label_42 = QtWidgets.QLabel(self.tab_3)
        self.label_42.setObjectName("label_42")
        self.gridLayout_14.addWidget(self.label_42, 2, 0, 1, 1)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_15 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_15.setObjectName("gridLayout_15")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_44 = QtWidgets.QLabel(self.tab_4)
        self.label_44.setObjectName("label_44")
        self.horizontalLayout_12.addWidget(self.label_44)
        self.lineEdit_13 = QtWidgets.QLineEdit(self.tab_4)
        self.lineEdit_13.setObjectName("lineEdit_13")
        self.horizontalLayout_12.addWidget(self.lineEdit_13)
        self.label_45 = QtWidgets.QLabel(self.tab_4)
        self.label_45.setObjectName("label_45")
        self.horizontalLayout_12.addWidget(self.label_45)
        self.lineEdit_14 = QtWidgets.QLineEdit(self.tab_4)
        self.lineEdit_14.setObjectName("lineEdit_14")
        self.horizontalLayout_12.addWidget(self.lineEdit_14)
        self.pushButton_33 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_33.setObjectName("pushButton_33")
        self.horizontalLayout_12.addWidget(self.pushButton_33)
        self.gridLayout_15.addLayout(self.horizontalLayout_12, 0, 0, 1, 2)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.pushButton_34 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_34.setObjectName("pushButton_34")
        self.horizontalLayout_13.addWidget(self.pushButton_34)
        self.progressBar_4 = QtWidgets.QProgressBar(self.tab_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.progressBar_4.setFont(font)
        self.progressBar_4.setProperty("value", 0)
        self.progressBar_4.setObjectName("progressBar_4")
        self.horizontalLayout_13.addWidget(self.progressBar_4)
        self.pushButton_35 = QtWidgets.QPushButton(self.tab_4)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_35.setFont(font)
        self.pushButton_35.setObjectName("pushButton_35")
        self.horizontalLayout_13.addWidget(self.pushButton_35)
        self.gridLayout_15.addLayout(self.horizontalLayout_13, 1, 0, 1, 2)
        self.label_46 = QtWidgets.QLabel(self.tab_4)
        self.label_46.setObjectName("label_46")
        self.gridLayout_15.addWidget(self.label_46, 2, 0, 1, 1)
        self.label_43 = QtWidgets.QLabel(self.tab_4)
        self.label_43.setObjectName("label_43")
        self.gridLayout_15.addWidget(self.label_43, 2, 1, 1, 1)
        self.graphicsView_10 = QtWidgets.QGraphicsView(self.tab_4)
        self.graphicsView_10.setObjectName("graphicsView_10")
        self.gridLayout_15.addWidget(self.graphicsView_10, 3, 0, 1, 1)
        self.graphicsView_11 = QtWidgets.QGraphicsView(self.tab_4)
        self.graphicsView_11.setObjectName("graphicsView_11")
        self.gridLayout_15.addWidget(self.graphicsView_11, 3, 1, 1, 1)
        self.pushButton_36 = QtWidgets.QPushButton(self.tab_4)
        self.pushButton_36.setObjectName("pushButton_36")
        self.gridLayout_15.addWidget(self.pushButton_36, 4, 1, 1, 1)
        self.tabWidget.addTab(self.tab_4, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.gridLayout_16 = QtWidgets.QGridLayout(self.tab_5)
        self.gridLayout_16.setObjectName("gridLayout_16")
        self.label_48 = QtWidgets.QLabel(self.tab_5)
        self.label_48.setObjectName("label_48")
        self.gridLayout_16.addWidget(self.label_48, 0, 0, 1, 1)
        self.lineEdit_17 = QtWidgets.QLineEdit(self.tab_5)
        self.lineEdit_17.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_17.setObjectName("lineEdit_17")
        self.gridLayout_16.addWidget(self.lineEdit_17, 0, 1, 1, 3)
        self.progressBar_5 = QtWidgets.QProgressBar(self.tab_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.progressBar_5.setFont(font)
        self.progressBar_5.setProperty("value", 0)
        self.progressBar_5.setObjectName("progressBar_5")
        self.gridLayout_16.addWidget(self.progressBar_5, 1, 0, 1, 3)
        self.pushButton_37 = QtWidgets.QPushButton(self.tab_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_37.setFont(font)
        self.pushButton_37.setObjectName("pushButton_37")
        self.gridLayout_16.addWidget(self.pushButton_37, 1, 3, 1, 1)
        self.label_51 = QtWidgets.QLabel(self.tab_5)
        self.label_51.setObjectName("label_51")
        self.gridLayout_16.addWidget(self.label_51, 2, 0, 2, 3)
        self.label_50 = QtWidgets.QLabel(self.tab_5)
        self.label_50.setObjectName("label_50")
        self.gridLayout_16.addWidget(self.label_50, 3, 2, 1, 2)
        self.graphicsView_13 = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView_13.setObjectName("graphicsView_13")
        self.gridLayout_16.addWidget(self.graphicsView_13, 4, 0, 1, 2)
        self.graphicsView_12 = QtWidgets.QGraphicsView(self.tab_5)
        self.graphicsView_12.setObjectName("graphicsView_12")
        self.gridLayout_16.addWidget(self.graphicsView_12, 4, 2, 1, 2)
        self.pushButton_38 = QtWidgets.QPushButton(self.tab_5)
        self.pushButton_38.setObjectName("pushButton_38")
        self.gridLayout_16.addWidget(self.pushButton_38, 5, 2, 1, 2)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.gridLayout_17 = QtWidgets.QGridLayout(self.tab_6)
        self.gridLayout_17.setObjectName("gridLayout_17")
        self.label_47 = QtWidgets.QLabel(self.tab_6)
        self.label_47.setObjectName("label_47")
        self.gridLayout_17.addWidget(self.label_47, 4, 0, 1, 1)
        self.label_53 = QtWidgets.QLabel(self.tab_6)
        self.label_53.setObjectName("label_53")
        self.gridLayout_17.addWidget(self.label_53, 1, 0, 2, 2)
        self.graphicsView_15 = QtWidgets.QGraphicsView(self.tab_6)
        self.graphicsView_15.setObjectName("graphicsView_15")
        self.gridLayout_17.addWidget(self.graphicsView_15, 3, 0, 1, 2)
        self.graphicsView_14 = QtWidgets.QGraphicsView(self.tab_6)
        self.graphicsView_14.setObjectName("graphicsView_14")
        self.gridLayout_17.addWidget(self.graphicsView_14, 3, 2, 1, 2)
        self.label_52 = QtWidgets.QLabel(self.tab_6)
        self.label_52.setObjectName("label_52")
        self.gridLayout_17.addWidget(self.label_52, 1, 3, 1, 1)
        self.pushButton_40 = QtWidgets.QPushButton(self.tab_6)
        self.pushButton_40.setObjectName("pushButton_40")
        self.gridLayout_17.addWidget(self.pushButton_40, 4, 3, 1, 1)
        self.lineEdit_15 = QtWidgets.QLineEdit(self.tab_6)
        self.lineEdit_15.setEnabled(False)
        self.lineEdit_15.setAlignment(QtCore.Qt.AlignCenter)
        self.lineEdit_15.setObjectName("lineEdit_15")
        self.gridLayout_17.addWidget(self.lineEdit_15, 5, 0, 1, 4)
        self.pushButton_39 = QtWidgets.QPushButton(self.tab_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_39.setFont(font)
        self.pushButton_39.setObjectName("pushButton_39")
        self.gridLayout_17.addWidget(self.pushButton_39, 0, 3, 1, 1)
        self.progressBar_6 = QtWidgets.QProgressBar(self.tab_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.progressBar_6.setFont(font)
        self.progressBar_6.setProperty("value", 0)
        self.progressBar_6.setObjectName("progressBar_6")
        self.gridLayout_17.addWidget(self.progressBar_6, 0, 0, 1, 3)
        self.tabWidget.addTab(self.tab_6, "")
        self.verticalLayout_3.addWidget(self.tabWidget)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.pushButton_21 = QtWidgets.QPushButton(self.page_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_21.setFont(font)
        self.pushButton_21.setObjectName("pushButton_21")
        self.horizontalLayout_9.addWidget(self.pushButton_21)
        spacerItem11 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_9.addItem(spacerItem11)
        self.pushButton_22 = QtWidgets.QPushButton(self.page_5)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_22.setFont(font)
        self.pushButton_22.setObjectName("pushButton_22")
        self.horizontalLayout_9.addWidget(self.pushButton_22)
        self.verticalLayout_3.addLayout(self.horizontalLayout_9)
        self.stackedWidget.addWidget(self.page_5)
        self.page_6 = QtWidgets.QWidget()
        self.page_6.setObjectName("page_6")
        self.gridLayout_18 = QtWidgets.QGridLayout(self.page_6)
        self.gridLayout_18.setObjectName("gridLayout_18")
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_55 = QtWidgets.QLabel(self.page_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_55.setFont(font)
        self.label_55.setObjectName("label_55")
        self.horizontalLayout_14.addWidget(self.label_55)
        self.lineEdit_16 = QtWidgets.QLineEdit(self.page_6)
        self.lineEdit_16.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_16.setFont(font)
        self.lineEdit_16.setObjectName("lineEdit_16")
        self.horizontalLayout_14.addWidget(self.lineEdit_16)
        self.label_56 = QtWidgets.QLabel(self.page_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_56.setFont(font)
        self.label_56.setObjectName("label_56")
        self.horizontalLayout_14.addWidget(self.label_56)
        self.dateTimeEdit_5 = QtWidgets.QDateTimeEdit(self.page_6)
        self.dateTimeEdit_5.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_5.setFont(font)
        self.dateTimeEdit_5.setObjectName("dateTimeEdit_5")
        self.horizontalLayout_14.addWidget(self.dateTimeEdit_5)
        self.gridLayout_18.addLayout(self.horizontalLayout_14, 0, 0, 1, 7)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label_57 = QtWidgets.QLabel(self.page_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_57.setFont(font)
        self.label_57.setObjectName("label_57")
        self.horizontalLayout_15.addWidget(self.label_57)
        self.lineEdit_18 = QtWidgets.QLineEdit(self.page_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_18.setFont(font)
        self.lineEdit_18.setObjectName("lineEdit_18")
        self.horizontalLayout_15.addWidget(self.lineEdit_18)
        self.pushButton_27 = QtWidgets.QPushButton(self.page_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_27.setFont(font)
        self.pushButton_27.setObjectName("pushButton_27")
        self.horizontalLayout_15.addWidget(self.pushButton_27)
        self.gridLayout_18.addLayout(self.horizontalLayout_15, 1, 0, 1, 7)
        self.label_49 = QtWidgets.QLabel(self.page_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_49.setFont(font)
        self.label_49.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_49.setObjectName("label_49")
        self.gridLayout_18.addWidget(self.label_49, 2, 0, 1, 2)
        self.label_54 = QtWidgets.QLabel(self.page_6)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_54.sizePolicy().hasHeightForWidth())
        self.label_54.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_54.setFont(font)
        self.label_54.setInputMethodHints(QtCore.Qt.ImhPreferLatin)
        self.label_54.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_54.setObjectName("label_54")
        self.gridLayout_18.addWidget(self.label_54, 2, 2, 1, 4)
        self.listWidget_6 = QtWidgets.QListWidget(self.page_6)
        self.listWidget_6.setObjectName("listWidget_6")
        self.gridLayout_18.addWidget(self.listWidget_6, 3, 0, 1, 2)
        self.pushButton_42 = QtWidgets.QPushButton(self.page_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_42.setFont(font)
        self.pushButton_42.setObjectName("pushButton_42")
        self.gridLayout_18.addWidget(self.pushButton_42, 4, 0, 1, 1)
        self.pushButton_43 = QtWidgets.QPushButton(self.page_6)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_43.setFont(font)
        self.pushButton_43.setObjectName("pushButton_43")
        self.gridLayout_18.addWidget(self.pushButton_43, 4, 1, 1, 1)
        spacerItem12 = QtWidgets.QSpacerItem(43, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_18.addItem(spacerItem12, 5, 0, 1, 1)
        spacerItem13 = QtWidgets.QSpacerItem(54, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_18.addItem(spacerItem13, 5, 1, 1, 1)
        spacerItem14 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_18.addItem(spacerItem14, 5, 2, 1, 1)
        spacerItem15 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_18.addItem(spacerItem15, 5, 3, 1, 1)
        spacerItem16 = QtWidgets.QSpacerItem(52, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_18.addItem(spacerItem16, 5, 4, 1, 1)
        spacerItem17 = QtWidgets.QSpacerItem(65, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_18.addItem(spacerItem17, 5, 5, 1, 1)
        spacerItem18 = QtWidgets.QSpacerItem(82, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_18.addItem(spacerItem18, 5, 6, 1, 1)
        self.textEdit = QtWidgets.QTextEdit(self.page_6)
        self.textEdit.setObjectName("textEdit")
        self.gridLayout_18.addWidget(self.textEdit, 3, 2, 2, 5)
        self.stackedWidget.addWidget(self.page_6)
        self.page_7 = QtWidgets.QWidget()
        self.page_7.setObjectName("page_7")
        self.gridLayout_23 = QtWidgets.QGridLayout(self.page_7)
        self.gridLayout_23.setObjectName("gridLayout_23")
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label_58 = QtWidgets.QLabel(self.page_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_58.setFont(font)
        self.label_58.setObjectName("label_58")
        self.horizontalLayout_16.addWidget(self.label_58)
        self.lineEdit_19 = QtWidgets.QLineEdit(self.page_7)
        self.lineEdit_19.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_19.setFont(font)
        self.lineEdit_19.setObjectName("lineEdit_19")
        self.horizontalLayout_16.addWidget(self.lineEdit_19)
        self.label_59 = QtWidgets.QLabel(self.page_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_59.setFont(font)
        self.label_59.setObjectName("label_59")
        self.horizontalLayout_16.addWidget(self.label_59)
        self.dateTimeEdit_6 = QtWidgets.QDateTimeEdit(self.page_7)
        self.dateTimeEdit_6.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_6.setFont(font)
        self.dateTimeEdit_6.setObjectName("dateTimeEdit_6")
        self.horizontalLayout_16.addWidget(self.dateTimeEdit_6)
        self.gridLayout_23.addLayout(self.horizontalLayout_16, 0, 0, 1, 1)
        self.groupBox_7 = QtWidgets.QGroupBox(self.page_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.groupBox_7.setFont(font)
        self.groupBox_7.setObjectName("groupBox_7")
        self.horizontalLayout_17 = QtWidgets.QHBoxLayout(self.groupBox_7)
        self.horizontalLayout_17.setObjectName("horizontalLayout_17")
        self.pushButton_41 = QtWidgets.QPushButton(self.groupBox_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_41.setFont(font)
        self.pushButton_41.setObjectName("pushButton_41")
        self.horizontalLayout_17.addWidget(self.pushButton_41)
        self.radioButton_3 = QtWidgets.QRadioButton(self.groupBox_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_3.setFont(font)
        self.radioButton_3.setObjectName("radioButton_3")
        self.horizontalLayout_17.addWidget(self.radioButton_3)
        self.radioButton_5 = QtWidgets.QRadioButton(self.groupBox_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_5.setFont(font)
        self.radioButton_5.setObjectName("radioButton_5")
        self.horizontalLayout_17.addWidget(self.radioButton_5)
        self.radioButton_4 = QtWidgets.QRadioButton(self.groupBox_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_4.setFont(font)
        self.radioButton_4.setObjectName("radioButton_4")
        self.horizontalLayout_17.addWidget(self.radioButton_4)
        self.radioButton_6 = QtWidgets.QRadioButton(self.groupBox_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_6.setFont(font)
        self.radioButton_6.setObjectName("radioButton_6")
        self.horizontalLayout_17.addWidget(self.radioButton_6)
        self.gridLayout_23.addWidget(self.groupBox_7, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.page_7)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_19 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_19.setObjectName("gridLayout_19")
        self.label_60 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_60.setFont(font)
        self.label_60.setObjectName("label_60")
        self.gridLayout_19.addWidget(self.label_60, 0, 0, 1, 1)
        self.label_62 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_62.setFont(font)
        self.label_62.setObjectName("label_62")
        self.gridLayout_19.addWidget(self.label_62, 3, 0, 1, 1)
        self.lineEdit_22 = QtWidgets.QLineEdit(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_22.setFont(font)
        self.lineEdit_22.setObjectName("lineEdit_22")
        self.gridLayout_19.addWidget(self.lineEdit_22, 3, 1, 1, 1)
        self.label_61 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_61.setFont(font)
        self.label_61.setObjectName("label_61")
        self.gridLayout_19.addWidget(self.label_61, 1, 0, 2, 1)
        self.lineEdit_23 = QtWidgets.QLineEdit(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_23.setFont(font)
        self.lineEdit_23.setObjectName("lineEdit_23")
        self.gridLayout_19.addWidget(self.lineEdit_23, 4, 1, 1, 1)
        self.label_63 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_63.setFont(font)
        self.label_63.setObjectName("label_63")
        self.gridLayout_19.addWidget(self.label_63, 4, 0, 1, 1)
        self.lineEdit_20 = QtWidgets.QLineEdit(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_20.setFont(font)
        self.lineEdit_20.setObjectName("lineEdit_20")
        self.gridLayout_19.addWidget(self.lineEdit_20, 0, 1, 1, 1)
        self.lineEdit_21 = QtWidgets.QLineEdit(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_21.setFont(font)
        self.lineEdit_21.setObjectName("lineEdit_21")
        self.gridLayout_19.addWidget(self.lineEdit_21, 1, 1, 2, 1)
        self.label_64 = QtWidgets.QLabel(self.groupBox_2)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_64.setFont(font)
        self.label_64.setObjectName("label_64")
        self.gridLayout_19.addWidget(self.label_64, 0, 3, 1, 1)
        self.progressBar_7 = QtWidgets.QProgressBar(self.groupBox_2)
        self.progressBar_7.setProperty("value", 0)
        self.progressBar_7.setObjectName("progressBar_7")
        self.gridLayout_19.addWidget(self.progressBar_7, 4, 2, 1, 2)
        self.pushButton_45 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_45.setObjectName("pushButton_45")
        self.gridLayout_19.addWidget(self.pushButton_45, 3, 3, 1, 1)
        self.pushButton_44 = QtWidgets.QPushButton(self.groupBox_2)
        self.pushButton_44.setObjectName("pushButton_44")
        self.gridLayout_19.addWidget(self.pushButton_44, 2, 3, 1, 1)
        spacerItem19 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_19.addItem(spacerItem19, 0, 2, 1, 1)
        self.gridLayout_23.addWidget(self.groupBox_2, 2, 0, 1, 1)
        self.tabWidget_2 = QtWidgets.QTabWidget(self.page_7)
        self.tabWidget_2.setObjectName("tabWidget_2")
        self.tab_7 = QtWidgets.QWidget()
        self.tab_7.setObjectName("tab_7")
        self.gridLayout_20 = QtWidgets.QGridLayout(self.tab_7)
        self.gridLayout_20.setObjectName("gridLayout_20")
        self.tableWidget_3 = QtWidgets.QTableWidget(self.tab_7)
        self.tableWidget_3.setObjectName("tableWidget_3")
        self.tableWidget_3.setColumnCount(5)
        self.tableWidget_3.setRowCount(4)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_3.setHorizontalHeaderItem(4, item)
        self.gridLayout_20.addWidget(self.tableWidget_3, 0, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_7, "")
        self.tab_8 = QtWidgets.QWidget()
        self.tab_8.setObjectName("tab_8")
        self.gridLayout_21 = QtWidgets.QGridLayout(self.tab_8)
        self.gridLayout_21.setObjectName("gridLayout_21")
        self.tableWidget_4 = QtWidgets.QTableWidget(self.tab_8)
        self.tableWidget_4.setObjectName("tableWidget_4")
        self.tableWidget_4.setColumnCount(5)
        self.tableWidget_4.setRowCount(4)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setVerticalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(0, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(1, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(2, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(3, item)
        item = QtWidgets.QTableWidgetItem()
        self.tableWidget_4.setHorizontalHeaderItem(4, item)
        self.gridLayout_21.addWidget(self.tableWidget_4, 0, 0, 1, 1)
        self.tabWidget_2.addTab(self.tab_8, "")
        self.tab_9 = QtWidgets.QWidget()
        self.tab_9.setObjectName("tab_9")
        self.gridLayout_22 = QtWidgets.QGridLayout(self.tab_9)
        self.gridLayout_22.setObjectName("gridLayout_22")
        self.formLayout_2 = QtWidgets.QFormLayout()
        self.formLayout_2.setVerticalSpacing(4)
        self.formLayout_2.setObjectName("formLayout_2")
        self.label_65 = QtWidgets.QLabel(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_65.setFont(font)
        self.label_65.setObjectName("label_65")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label_65)
        self.label_66 = QtWidgets.QLabel(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_66.setFont(font)
        self.label_66.setObjectName("label_66")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_66)
        self.lineEdit_24 = QtWidgets.QLineEdit(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.lineEdit_24.setFont(font)
        self.lineEdit_24.setObjectName("lineEdit_24")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.lineEdit_24)
        self.lineEdit_25 = QtWidgets.QLineEdit(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("AcadEref")
        font.setPointSize(9)
        self.lineEdit_25.setFont(font)
        self.lineEdit_25.setObjectName("lineEdit_25")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.lineEdit_25)
        self.label_67 = QtWidgets.QLabel(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_67.setFont(font)
        self.label_67.setObjectName("label_67")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_67)
        self.lineEdit_26 = QtWidgets.QLineEdit(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.lineEdit_26.setFont(font)
        self.lineEdit_26.setObjectName("lineEdit_26")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.lineEdit_26)
        self.label_68 = QtWidgets.QLabel(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.label_68.setFont(font)
        self.label_68.setObjectName("label_68")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_68)
        self.lineEdit_27 = QtWidgets.QLineEdit(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.lineEdit_27.setFont(font)
        self.lineEdit_27.setObjectName("lineEdit_27")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.lineEdit_27)
        self.pushButton_46 = QtWidgets.QPushButton(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.pushButton_46.setFont(font)
        self.pushButton_46.setObjectName("pushButton_46")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.SpanningRole, self.pushButton_46)
        self.label_69 = QtWidgets.QLabel(self.tab_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(10)
        self.label_69.setFont(font)
        self.label_69.setObjectName("label_69")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.SpanningRole, self.label_69)
        self.gridLayout_22.addLayout(self.formLayout_2, 0, 0, 1, 1)
        self.graphicsView_16 = QtWidgets.QGraphicsView(self.tab_9)
        self.graphicsView_16.setObjectName("graphicsView_16")
        self.gridLayout_22.addWidget(self.graphicsView_16, 0, 1, 1, 1)
        self.tabWidget_2.addTab(self.tab_9, "")
        self.gridLayout_23.addWidget(self.tabWidget_2, 3, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_7)
        self.page_8 = QtWidgets.QWidget()
        self.page_8.setObjectName("page_8")
        self.gridLayout_27 = QtWidgets.QGridLayout(self.page_8)
        self.gridLayout_27.setObjectName("gridLayout_27")
        self.horizontalLayout_18 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_18.setObjectName("horizontalLayout_18")
        self.label_70 = QtWidgets.QLabel(self.page_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_70.setFont(font)
        self.label_70.setObjectName("label_70")
        self.horizontalLayout_18.addWidget(self.label_70)
        self.lineEdit_28 = QtWidgets.QLineEdit(self.page_8)
        self.lineEdit_28.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_28.setFont(font)
        self.lineEdit_28.setObjectName("lineEdit_28")
        self.horizontalLayout_18.addWidget(self.lineEdit_28)
        self.label_71 = QtWidgets.QLabel(self.page_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_71.setFont(font)
        self.label_71.setObjectName("label_71")
        self.horizontalLayout_18.addWidget(self.label_71)
        self.dateTimeEdit_7 = QtWidgets.QDateTimeEdit(self.page_8)
        self.dateTimeEdit_7.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_7.setFont(font)
        self.dateTimeEdit_7.setObjectName("dateTimeEdit_7")
        self.horizontalLayout_18.addWidget(self.dateTimeEdit_7)
        self.gridLayout_27.addLayout(self.horizontalLayout_18, 0, 0, 1, 1)
        self.groupBox_8 = QtWidgets.QGroupBox(self.page_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.groupBox_8.setFont(font)
        self.groupBox_8.setObjectName("groupBox_8")
        self.horizontalLayout_19 = QtWidgets.QHBoxLayout(self.groupBox_8)
        self.horizontalLayout_19.setObjectName("horizontalLayout_19")
        self.pushButton_47 = QtWidgets.QPushButton(self.groupBox_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_47.setFont(font)
        self.pushButton_47.setObjectName("pushButton_47")
        self.horizontalLayout_19.addWidget(self.pushButton_47)
        self.radioButton_7 = QtWidgets.QRadioButton(self.groupBox_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_7.setFont(font)
        self.radioButton_7.setObjectName("radioButton_7")
        self.horizontalLayout_19.addWidget(self.radioButton_7)
        self.radioButton_8 = QtWidgets.QRadioButton(self.groupBox_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_8.setFont(font)
        self.radioButton_8.setObjectName("radioButton_8")
        self.horizontalLayout_19.addWidget(self.radioButton_8)
        self.radioButton_9 = QtWidgets.QRadioButton(self.groupBox_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_9.setFont(font)
        self.radioButton_9.setObjectName("radioButton_9")
        self.horizontalLayout_19.addWidget(self.radioButton_9)
        self.radioButton_10 = QtWidgets.QRadioButton(self.groupBox_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.radioButton_10.setFont(font)
        self.radioButton_10.setObjectName("radioButton_10")
        self.horizontalLayout_19.addWidget(self.radioButton_10)
        self.gridLayout_27.addWidget(self.groupBox_8, 1, 0, 1, 1)
        self.gridLayout_24 = QtWidgets.QGridLayout()
        self.gridLayout_24.setObjectName("gridLayout_24")
        self.label_72 = QtWidgets.QLabel(self.page_8)
        self.label_72.setObjectName("label_72")
        self.gridLayout_24.addWidget(self.label_72, 0, 0, 1, 1)
        self.progressBar_8 = QtWidgets.QProgressBar(self.page_8)
        self.progressBar_8.setProperty("value", 0)
        self.progressBar_8.setObjectName("progressBar_8")
        self.gridLayout_24.addWidget(self.progressBar_8, 0, 1, 1, 1)
        self.pushButton_48 = QtWidgets.QPushButton(self.page_8)
        self.pushButton_48.setObjectName("pushButton_48")
        self.gridLayout_24.addWidget(self.pushButton_48, 0, 2, 1, 1)
        self.label_73 = QtWidgets.QLabel(self.page_8)
        self.label_73.setObjectName("label_73")
        self.gridLayout_24.addWidget(self.label_73, 1, 0, 1, 1)
        self.progressBar_9 = QtWidgets.QProgressBar(self.page_8)
        self.progressBar_9.setProperty("value", 0)
        self.progressBar_9.setObjectName("progressBar_9")
        self.gridLayout_24.addWidget(self.progressBar_9, 1, 1, 1, 1)
        self.pushButton_49 = QtWidgets.QPushButton(self.page_8)
        self.pushButton_49.setObjectName("pushButton_49")
        self.gridLayout_24.addWidget(self.pushButton_49, 1, 2, 1, 1)
        self.gridLayout_27.addLayout(self.gridLayout_24, 2, 0, 1, 1)
        self.tabWidget_3 = QtWidgets.QTabWidget(self.page_8)
        self.tabWidget_3.setObjectName("tabWidget_3")
        self.tab_10 = QtWidgets.QWidget()
        self.tab_10.setObjectName("tab_10")
        self.gridLayout_25 = QtWidgets.QGridLayout(self.tab_10)
        self.gridLayout_25.setObjectName("gridLayout_25")
        self.graphicsView_17 = QtWidgets.QGraphicsView(self.tab_10)
        self.graphicsView_17.setObjectName("graphicsView_17")
        self.gridLayout_25.addWidget(self.graphicsView_17, 0, 0, 3, 1)
        self.pushButton_53 = QtWidgets.QPushButton(self.tab_10)
        self.pushButton_53.setObjectName("pushButton_53")
        self.gridLayout_25.addWidget(self.pushButton_53, 0, 1, 1, 1)
        self.pushButton_54 = QtWidgets.QPushButton(self.tab_10)
        self.pushButton_54.setObjectName("pushButton_54")
        self.gridLayout_25.addWidget(self.pushButton_54, 0, 2, 1, 1)
        self.label_74 = QtWidgets.QLabel(self.tab_10)
        self.label_74.setObjectName("label_74")
        self.gridLayout_25.addWidget(self.label_74, 1, 1, 1, 2)
        self.textEdit_sm = QtWidgets.QTextEdit(self.tab_10)
        self.textEdit_sm.setObjectName("textBrowser")
        self.gridLayout_25.addWidget(self.textEdit_sm, 2, 1, 1, 2)
        self.tabWidget_3.addTab(self.tab_10, "")
        self.tab_11 = QtWidgets.QWidget()
        self.tab_11.setObjectName("tab_11")
        self.gridLayout_26 = QtWidgets.QGridLayout(self.tab_11)
        self.gridLayout_26.setObjectName("gridLayout_26")
        self.graphicsView_18 = QtWidgets.QGraphicsView(self.tab_11)
        self.graphicsView_18.setObjectName("graphicsView_18")
        self.gridLayout_26.addWidget(self.graphicsView_18, 0, 0, 3, 1)
        self.pushButton_56 = QtWidgets.QPushButton(self.tab_11)
        self.pushButton_56.setObjectName("pushButton_56")
        self.gridLayout_26.addWidget(self.pushButton_56, 0, 1, 1, 1)
        self.pushButton_55 = QtWidgets.QPushButton(self.tab_11)
        self.pushButton_55.setObjectName("pushButton_55")
        self.gridLayout_26.addWidget(self.pushButton_55, 0, 2, 1, 1)
        self.label_75 = QtWidgets.QLabel(self.tab_11)
        self.label_75.setObjectName("label_75")
        self.gridLayout_26.addWidget(self.label_75, 1, 1, 1, 2)
        self.textEdit_em = QtWidgets.QTextEdit(self.tab_11)
        self.textEdit_em.setObjectName("textBrowser_2")
        self.gridLayout_26.addWidget(self.textEdit_em, 2, 1, 1, 2)
        self.tabWidget_3.addTab(self.tab_11, "")
        self.gridLayout_27.addWidget(self.tabWidget_3, 3, 0, 1, 1)
        self.horizontalLayout_20 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_20.setObjectName("horizontalLayout_20")
        self.pushButton_50 = QtWidgets.QPushButton(self.page_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_50.setFont(font)
        self.pushButton_50.setObjectName("pushButton_50")
        self.horizontalLayout_20.addWidget(self.pushButton_50)
        spacerItem20 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem20)
        self.pushButton_51 = QtWidgets.QPushButton(self.page_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_51.setFont(font)
        self.pushButton_51.setObjectName("pushButton_51")
        self.horizontalLayout_20.addWidget(self.pushButton_51)
        spacerItem21 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout_20.addItem(spacerItem21)
        self.pushButton_52 = QtWidgets.QPushButton(self.page_8)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(False)
        font.setWeight(50)
        self.pushButton_52.setFont(font)
        self.pushButton_52.setObjectName("pushButton_52")
        self.horizontalLayout_20.addWidget(self.pushButton_52)
        self.gridLayout_27.addLayout(self.horizontalLayout_20, 4, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_8)
        self.page_9 = QtWidgets.QWidget()
        self.page_9.setObjectName("page_9")
        self.gridLayout_28 = QtWidgets.QGridLayout(self.page_9)
        self.gridLayout_28.setObjectName("gridLayout_28")
        self.horizontalLayout_21 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_21.setObjectName("horizontalLayout_21")
        self.label_78 = QtWidgets.QLabel(self.page_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_78.setFont(font)
        self.label_78.setObjectName("label_78")
        self.horizontalLayout_21.addWidget(self.label_78)
        self.lineEdit_29 = QtWidgets.QLineEdit(self.page_9)
        self.lineEdit_29.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_29.setFont(font)
        self.lineEdit_29.setObjectName("lineEdit_29")
        self.horizontalLayout_21.addWidget(self.lineEdit_29)
        self.label_79 = QtWidgets.QLabel(self.page_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_79.setFont(font)
        self.label_79.setObjectName("label_79")
        self.horizontalLayout_21.addWidget(self.label_79)
        self.dateTimeEdit_8 = QtWidgets.QDateTimeEdit(self.page_9)
        self.dateTimeEdit_8.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_8.setFont(font)
        self.dateTimeEdit_8.setObjectName("dateTimeEdit_8")
        self.horizontalLayout_21.addWidget(self.dateTimeEdit_8)
        self.gridLayout_28.addLayout(self.horizontalLayout_21, 0, 0, 1, 3)
        self.label_76 = QtWidgets.QLabel(self.page_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_76.setFont(font)
        self.label_76.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_76.setObjectName("label_76")
        self.gridLayout_28.addWidget(self.label_76, 1, 0, 1, 2)
        self.label_77 = QtWidgets.QLabel(self.page_9)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_77.sizePolicy().hasHeightForWidth())
        self.label_77.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.label_77.setFont(font)
        self.label_77.setInputMethodHints(QtCore.Qt.ImhPreferLatin)
        self.label_77.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_77.setObjectName("label_77")
        self.gridLayout_28.addWidget(self.label_77, 1, 2, 1, 1)
        self.listWidget_7 = QtWidgets.QListWidget(self.page_9)
        self.listWidget_7.setObjectName("listWidget_7")
        self.gridLayout_28.addWidget(self.listWidget_7, 2, 0, 1, 2)
        self.pushButton_58 = QtWidgets.QPushButton(self.page_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_58.setFont(font)
        self.pushButton_58.setObjectName("pushButton_58")
        self.gridLayout_28.addWidget(self.pushButton_58, 3, 0, 1, 1)
        self.pushButton_57 = QtWidgets.QPushButton(self.page_9)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.pushButton_57.setFont(font)
        self.pushButton_57.setObjectName("pushButton_57")
        self.gridLayout_28.addWidget(self.pushButton_57, 3, 1, 1, 1)
        self.textEdit_2 = QtWidgets.QTextEdit(self.page_9)
        self.textEdit_2.setObjectName("textEdit_2")
        self.gridLayout_28.addWidget(self.textEdit_2, 2, 2, 2, 1)
        self.stackedWidget.addWidget(self.page_9)
        self.page_10 = QtWidgets.QWidget()
        self.page_10.setObjectName("page_10")
        self.gridLayout_30 = QtWidgets.QGridLayout(self.page_10)
        self.gridLayout_30.setObjectName("gridLayout_30")
        self.label_80 = QtWidgets.QLabel(self.page_10)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_80.setFont(font)
        self.label_80.setObjectName("label_80")
        self.gridLayout_30.addWidget(self.label_80, 0, 0, 1, 1)
        self.lineEdit_30 = QtWidgets.QLineEdit(self.page_10)
        self.lineEdit_30.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_30.setFont(font)
        self.lineEdit_30.setObjectName("lineEdit_30")
        self.gridLayout_30.addWidget(self.lineEdit_30, 0, 1, 1, 3)
        self.label_81 = QtWidgets.QLabel(self.page_10)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_81.setFont(font)
        self.label_81.setObjectName("label_81")
        self.gridLayout_30.addWidget(self.label_81, 0, 4, 1, 1)
        self.horizontalLayout_22 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_22.setObjectName("horizontalLayout_22")
        self.label_82 = QtWidgets.QLabel(self.page_10)
        self.label_82.setObjectName("label_82")
        self.horizontalLayout_22.addWidget(self.label_82)
        self.radioButton_2 = QtWidgets.QRadioButton(self.page_10)
        self.radioButton_2.setObjectName("radioButton_2")
        self.horizontalLayout_22.addWidget(self.radioButton_2)
        self.radioButton = QtWidgets.QRadioButton(self.page_10)
        self.radioButton.setObjectName("radioButton")
        self.horizontalLayout_22.addWidget(self.radioButton)
        self.pushButton_59 = QtWidgets.QPushButton(self.page_10)
        self.pushButton_59.setObjectName("pushButton_59")
        self.horizontalLayout_22.addWidget(self.pushButton_59)
        self.gridLayout_30.addLayout(self.horizontalLayout_22, 1, 0, 1, 9)
        spacerItem22 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_30.addItem(spacerItem22, 3, 0, 2, 1)
        spacerItem23 = QtWidgets.QSpacerItem(159, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_30.addItem(spacerItem23, 3, 1, 2, 1)
        self.label_83 = QtWidgets.QLabel(self.page_10)
        self.label_83.setAlignment(QtCore.Qt.AlignCenter)
        self.label_83.setObjectName("label_83")
        self.gridLayout_30.addWidget(self.label_83, 4, 3, 1, 3)
        spacerItem24 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_30.addItem(spacerItem24, 4, 7, 1, 1)
        spacerItem25 = QtWidgets.QSpacerItem(80, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_30.addItem(spacerItem25, 4, 8, 1, 1)
        self.gridLayout_29 = QtWidgets.QGridLayout()
        self.gridLayout_29.setObjectName("gridLayout_29")
        self.lineEdit_32 = QtWidgets.QLineEdit(self.page_10)
        self.lineEdit_32.setObjectName("lineEdit_32")
        self.gridLayout_29.addWidget(self.lineEdit_32, 1, 1, 1, 1)
        self.lineEdit_38 = QtWidgets.QLineEdit(self.page_10)
        self.lineEdit_38.setObjectName("lineEdit_38")
        self.gridLayout_29.addWidget(self.lineEdit_38, 1, 5, 1, 1)
        self.lineEdit_37 = QtWidgets.QLineEdit(self.page_10)
        self.lineEdit_37.setObjectName("lineEdit_37")
        self.gridLayout_29.addWidget(self.lineEdit_37, 0, 5, 1, 1)
        self.label_92 = QtWidgets.QLabel(self.page_10)
        self.label_92.setObjectName("label_92")
        self.gridLayout_29.addWidget(self.label_92, 0, 4, 1, 1)
        self.label_84 = QtWidgets.QLabel(self.page_10)
        self.label_84.setObjectName("label_84")
        self.gridLayout_29.addWidget(self.label_84, 0, 0, 1, 1)
        self.label_91 = QtWidgets.QLabel(self.page_10)
        self.label_91.setObjectName("label_91")
        self.gridLayout_29.addWidget(self.label_91, 1, 2, 1, 1)
        self.label_86 = QtWidgets.QLabel(self.page_10)
        self.label_86.setObjectName("label_86")
        self.gridLayout_29.addWidget(self.label_86, 0, 2, 1, 1)
        self.lineEdit_31 = QtWidgets.QLineEdit(self.page_10)
        self.lineEdit_31.setObjectName("lineEdit_31")
        self.gridLayout_29.addWidget(self.lineEdit_31, 0, 1, 1, 1)
        self.label_85 = QtWidgets.QLabel(self.page_10)
        self.label_85.setObjectName("label_85")
        self.gridLayout_29.addWidget(self.label_85, 1, 0, 1, 1)
        self.lineEdit_36 = QtWidgets.QLineEdit(self.page_10)
        self.lineEdit_36.setObjectName("lineEdit_36")
        self.gridLayout_29.addWidget(self.lineEdit_36, 1, 3, 1, 1)
        self.label_93 = QtWidgets.QLabel(self.page_10)
        self.label_93.setObjectName("label_93")
        self.gridLayout_29.addWidget(self.label_93, 1, 4, 1, 1)
        self.lineEdit_33 = QtWidgets.QLineEdit(self.page_10)
        self.lineEdit_33.setObjectName("lineEdit_33")
        self.gridLayout_29.addWidget(self.lineEdit_33, 0, 3, 1, 1)
        self.pushButton_60 = QtWidgets.QPushButton(self.page_10)
        self.pushButton_60.setObjectName("pushButton_60")
        self.gridLayout_29.addWidget(self.pushButton_60, 0, 6, 2, 1)
        self.gridLayout_30.addLayout(self.gridLayout_29, 5, 0, 1, 9)
        spacerItem26 = QtWidgets.QSpacerItem(80, 328, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_30.addItem(spacerItem26, 2, 0, 1, 1)
        spacerItem27 = QtWidgets.QSpacerItem(80, 328, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_30.addItem(spacerItem27, 2, 8, 1, 1)
        self.graphicsView_19 = QtWidgets.QGraphicsView(self.page_10)
        self.graphicsView_19.setObjectName("graphicsView_19")
        self.gridLayout_30.addWidget(self.graphicsView_19, 2, 1, 1, 7)
        self.dateTimeEdit_9 = QtWidgets.QDateTimeEdit(self.page_10)
        self.dateTimeEdit_9.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_9.setFont(font)
        self.dateTimeEdit_9.setObjectName("dateTimeEdit_9")
        self.gridLayout_30.addWidget(self.dateTimeEdit_9, 0, 5, 1, 4)
        self.stackedWidget.addWidget(self.page_10)
        self.page_11 = QtWidgets.QWidget()
        self.page_11.setObjectName("page_11")
        self.gridLayout_35 = QtWidgets.QGridLayout(self.page_11)
        self.gridLayout_35.setObjectName("gridLayout_35")
        self.horizontalLayout_23 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_23.setObjectName("horizontalLayout_23")
        self.label_94 = QtWidgets.QLabel(self.page_11)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_94.setFont(font)
        self.label_94.setObjectName("label_94")
        self.horizontalLayout_23.addWidget(self.label_94)
        self.lineEdit_39 = QtWidgets.QLineEdit(self.page_11)
        self.lineEdit_39.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.lineEdit_39.setFont(font)
        self.lineEdit_39.setObjectName("lineEdit_39")
        self.horizontalLayout_23.addWidget(self.lineEdit_39)
        self.label_95 = QtWidgets.QLabel(self.page_11)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.label_95.setFont(font)
        self.label_95.setObjectName("label_95")
        self.horizontalLayout_23.addWidget(self.label_95)
        self.dateTimeEdit_10 = QtWidgets.QDateTimeEdit(self.page_11)
        self.dateTimeEdit_10.setEnabled(False)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        self.dateTimeEdit_10.setFont(font)
        self.dateTimeEdit_10.setObjectName("dateTimeEdit_10")
        self.horizontalLayout_23.addWidget(self.dateTimeEdit_10)
        self.gridLayout_35.addLayout(self.horizontalLayout_23, 0, 0, 1, 1)
        self.toolBox = QtWidgets.QToolBox(self.page_11)
        self.toolBox.setObjectName("toolBox")
        self.page_12 = QtWidgets.QWidget()
        self.page_12.setGeometry(QtCore.QRect(0, 0, 572, 404))
        self.page_12.setObjectName("page_12")
        self.gridLayout_31 = QtWidgets.QGridLayout(self.page_12)
        self.gridLayout_31.setObjectName("gridLayout_31")
        self.gridLayout_33 = QtWidgets.QGridLayout()
        self.gridLayout_33.setObjectName("gridLayout_33")
        self.label_110 = QtWidgets.QLabel(self.page_12)
        self.label_110.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_110.setObjectName("label_110")
        self.gridLayout_33.addWidget(self.label_110, 0, 0, 1, 1)
        self.label_111 = QtWidgets.QLabel(self.page_12)
        self.label_111.setMinimumSize(QtCore.QSize(61, 0))
        self.label_111.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_111.setObjectName("label_111")
        self.gridLayout_33.addWidget(self.label_111, 1, 6, 1, 1)
        self.lineEdit_50 = QtWidgets.QLineEdit(self.page_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_50.sizePolicy().hasHeightForWidth())
        self.lineEdit_50.setSizePolicy(sizePolicy)
        self.lineEdit_50.setObjectName("lineEdit_50")
        self.gridLayout_33.addWidget(self.lineEdit_50, 1, 7, 1, 1)
        self.lineEdit_51 = QtWidgets.QLineEdit(self.page_12)
        self.lineEdit_51.setObjectName("lineEdit_51")
        self.gridLayout_33.addWidget(self.lineEdit_51, 0, 1, 1, 2)
        self.pushButton_63 = QtWidgets.QPushButton(self.page_12)
        self.pushButton_63.setObjectName("pushButton_63")
        self.gridLayout_33.addWidget(self.pushButton_63, 0, 6, 1, 2)
        self.label_112 = QtWidgets.QLabel(self.page_12)
        self.label_112.setMinimumSize(QtCore.QSize(61, 0))
        self.label_112.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_112.setObjectName("label_112")
        self.gridLayout_33.addWidget(self.label_112, 0, 3, 1, 1)
        self.lineEdit_52 = QtWidgets.QLineEdit(self.page_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_52.sizePolicy().hasHeightForWidth())
        self.lineEdit_52.setSizePolicy(sizePolicy)
        self.lineEdit_52.setObjectName("lineEdit_52")
        self.gridLayout_33.addWidget(self.lineEdit_52, 0, 4, 1, 1)
        self.label_113 = QtWidgets.QLabel(self.page_12)
        self.label_113.setMinimumSize(QtCore.QSize(61, 0))
        self.label_113.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_113.setObjectName("label_113")
        self.gridLayout_33.addWidget(self.label_113, 1, 0, 1, 1)
        self.lineEdit_53 = QtWidgets.QLineEdit(self.page_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_53.sizePolicy().hasHeightForWidth())
        self.lineEdit_53.setSizePolicy(sizePolicy)
        self.lineEdit_53.setObjectName("lineEdit_53")
        self.gridLayout_33.addWidget(self.lineEdit_53, 1, 1, 1, 1)
        self.label_114 = QtWidgets.QLabel(self.page_12)
        self.label_114.setMinimumSize(QtCore.QSize(61, 0))
        self.label_114.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_114.setObjectName("label_114")
        self.gridLayout_33.addWidget(self.label_114, 1, 3, 1, 1)
        self.lineEdit_54 = QtWidgets.QLineEdit(self.page_12)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_54.sizePolicy().hasHeightForWidth())
        self.lineEdit_54.setSizePolicy(sizePolicy)
        self.lineEdit_54.setObjectName("lineEdit_54")
        self.gridLayout_33.addWidget(self.lineEdit_54, 1, 4, 1, 1)
        self.gridLayout_31.addLayout(self.gridLayout_33, 0, 0, 1, 2)
        self.graphicsView_23 = QtWidgets.QGraphicsView(self.page_12)
        self.graphicsView_23.setObjectName("graphicsView_23")
        self.gridLayout_31.addWidget(self.graphicsView_23, 1, 0, 1, 1)
        self.graphicsView_22 = QtWidgets.QGraphicsView(self.page_12)
        self.graphicsView_22.setObjectName("graphicsView_22")
        self.gridLayout_31.addWidget(self.graphicsView_22, 1, 1, 1, 1)
        self.label_109 = QtWidgets.QLabel(self.page_12)
        self.label_109.setAlignment(QtCore.Qt.AlignCenter)
        self.label_109.setObjectName("label_109")
        self.gridLayout_31.addWidget(self.label_109, 2, 0, 1, 1)
        self.label_108 = QtWidgets.QLabel(self.page_12)
        self.label_108.setAlignment(QtCore.Qt.AlignCenter)
        self.label_108.setObjectName("label_108")
        self.gridLayout_31.addWidget(self.label_108, 2, 1, 1, 1)
        self.toolBox.addItem(self.page_12, "")
        self.page_13 = QtWidgets.QWidget()
        self.page_13.setGeometry(QtCore.QRect(0, 0, 572, 404))
        self.page_13.setObjectName("page_13")
        self.gridLayout_34 = QtWidgets.QGridLayout(self.page_13)
        self.gridLayout_34.setObjectName("gridLayout_34")
        self.gridLayout_32 = QtWidgets.QGridLayout()
        self.gridLayout_32.setObjectName("gridLayout_32")
        self.label_102 = QtWidgets.QLabel(self.page_13)
        self.label_102.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_102.setObjectName("label_102")
        self.gridLayout_32.addWidget(self.label_102, 0, 0, 1, 1)
        self.label_105 = QtWidgets.QLabel(self.page_13)
        self.label_105.setMinimumSize(QtCore.QSize(61, 0))
        self.label_105.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_105.setObjectName("label_105")
        self.gridLayout_32.addWidget(self.label_105, 1, 6, 1, 1)
        self.lineEdit_49 = QtWidgets.QLineEdit(self.page_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_49.sizePolicy().hasHeightForWidth())
        self.lineEdit_49.setSizePolicy(sizePolicy)
        self.lineEdit_49.setObjectName("lineEdit_49")
        self.gridLayout_32.addWidget(self.lineEdit_49, 1, 7, 1, 1)
        self.lineEdit_45 = QtWidgets.QLineEdit(self.page_13)
        self.lineEdit_45.setObjectName("lineEdit_45")
        self.gridLayout_32.addWidget(self.lineEdit_45, 0, 1, 1, 2)
        self.pushButton_62 = QtWidgets.QPushButton(self.page_13)
        self.pushButton_62.setObjectName("pushButton_62")
        self.gridLayout_32.addWidget(self.pushButton_62, 0, 6, 1, 2)
        self.label_104 = QtWidgets.QLabel(self.page_13)
        self.label_104.setMinimumSize(QtCore.QSize(61, 0))
        self.label_104.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_104.setObjectName("label_104")
        self.gridLayout_32.addWidget(self.label_104, 0, 3, 1, 1)
        self.lineEdit_48 = QtWidgets.QLineEdit(self.page_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_48.sizePolicy().hasHeightForWidth())
        self.lineEdit_48.setSizePolicy(sizePolicy)
        self.lineEdit_48.setObjectName("lineEdit_48")
        self.gridLayout_32.addWidget(self.lineEdit_48, 0, 4, 1, 1)
        self.label_103 = QtWidgets.QLabel(self.page_13)
        self.label_103.setMinimumSize(QtCore.QSize(61, 0))
        self.label_103.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_103.setObjectName("label_103")
        self.gridLayout_32.addWidget(self.label_103, 1, 0, 1, 1)
        self.lineEdit_47 = QtWidgets.QLineEdit(self.page_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_47.sizePolicy().hasHeightForWidth())
        self.lineEdit_47.setSizePolicy(sizePolicy)
        self.lineEdit_47.setObjectName("lineEdit_47")
        self.gridLayout_32.addWidget(self.lineEdit_47, 1, 1, 1, 1)
        self.label_101 = QtWidgets.QLabel(self.page_13)
        self.label_101.setMinimumSize(QtCore.QSize(61, 0))
        self.label_101.setMaximumSize(QtCore.QSize(61, 16777215))
        self.label_101.setObjectName("label_101")
        self.gridLayout_32.addWidget(self.label_101, 1, 3, 1, 1)
        self.lineEdit_46 = QtWidgets.QLineEdit(self.page_13)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.lineEdit_46.sizePolicy().hasHeightForWidth())
        self.lineEdit_46.setSizePolicy(sizePolicy)
        self.lineEdit_46.setObjectName("lineEdit_46")
        self.gridLayout_32.addWidget(self.lineEdit_46, 1, 4, 1, 1)
        self.gridLayout_34.addLayout(self.gridLayout_32, 0, 0, 1, 2)
        self.graphicsView_20 = QtWidgets.QGraphicsView(self.page_13)
        self.graphicsView_20.setObjectName("graphicsView_20")
        self.gridLayout_34.addWidget(self.graphicsView_20, 1, 0, 1, 1)
        self.graphicsView_21 = QtWidgets.QGraphicsView(self.page_13)
        self.graphicsView_21.setObjectName("graphicsView_21")
        self.gridLayout_34.addWidget(self.graphicsView_21, 1, 1, 1, 1)
        self.label_106 = QtWidgets.QLabel(self.page_13)
        self.label_106.setAlignment(QtCore.Qt.AlignCenter)
        self.label_106.setObjectName("label_106")
        self.gridLayout_34.addWidget(self.label_106, 2, 0, 1, 1)
        self.label_107 = QtWidgets.QLabel(self.page_13)
        self.label_107.setAlignment(QtCore.Qt.AlignCenter)
        self.label_107.setObjectName("label_107")
        self.gridLayout_34.addWidget(self.label_107, 2, 1, 1, 1)
        self.toolBox.addItem(self.page_13, "")
        self.gridLayout_35.addWidget(self.toolBox, 1, 0, 1, 1)
        self.stackedWidget.addWidget(self.page_11)
        self.gridLayout.addWidget(self.stackedWidget, 3, 1, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 3, 1, 12)
        spacerItem28 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem28, 2, 14, 1, 1)
        spacerItem29 = QtWidgets.QSpacerItem(43, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem29, 2, 1, 1, 1)
        spacerItem30 = QtWidgets.QSpacerItem(41, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem30, 2, 6, 1, 1)
        spacerItem31 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem31, 2, 3, 1, 1)
        spacerItem32 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem32, 2, 7, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton.setFont(font)
        self.pushButton.setObjectName("pushButton")
        self.verticalLayout.addWidget(self.pushButton)
        self.Label_NewPro_Pred = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.Label_NewPro_Pred.setFont(font)
        self.Label_NewPro_Pred.setFrameShape(QtWidgets.QFrame.Box)
        self.Label_NewPro_Pred.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Label_NewPro_Pred.setAlignment(QtCore.Qt.AlignCenter)
        self.Label_NewPro_Pred.setObjectName("Label_NewPro_Pred")
        self.verticalLayout.addWidget(self.Label_NewPro_Pred)
        self.listWidget = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget.sizePolicy().hasHeightForWidth())
        self.listWidget.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setBold(True)
        font.setWeight(75)
        self.listWidget.setFont(font)
        self.listWidget.setObjectName("listWidget")
        item = QtWidgets.QListWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.listWidget.addItem(item)
        item = QtWidgets.QListWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.listWidget.addItem(item)
        self.verticalLayout.addWidget(self.listWidget)
        item = QtWidgets.QListWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.listWidget.addItem(item)
        self.verticalLayout.addWidget(self.listWidget)
        self.pushButton_3 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_3.setFont(font)
        self.pushButton_3.setFrameShape(QtWidgets.QFrame.Box)
        self.pushButton_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.pushButton_3.setAlignment(QtCore.Qt.AlignCenter)
        self.pushButton_3.setObjectName("pushButton_3")
        self.verticalLayout.addWidget(self.pushButton_3)
        self.listWidget_2 = QtWidgets.QListWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.listWidget_2.sizePolicy().hasHeightForWidth())
        self.listWidget_2.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setBold(True)
        font.setWeight(75)
        self.listWidget_2.setFont(font)
        self.listWidget_2.setObjectName("listWidget_2")
        item = QtWidgets.QListWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.listWidget_2.addItem(item)
        item = QtWidgets.QListWidgetItem()
        font = QtGui.QFont()
        font.setPointSize(12)
        item.setFont(font)
        self.listWidget_2.addItem(item)
        self.verticalLayout.addWidget(self.listWidget_2)
        self.pushButton_28 = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.pushButton_28.setFont(font)
        self.pushButton_28.setFrameShape(QtWidgets.QFrame.Box)
        self.pushButton_28.setFrameShadow(QtWidgets.QFrame.Raised)
        self.pushButton_28.setAlignment(QtCore.Qt.AlignCenter)
        self.pushButton_28.setObjectName("pushButton_28")
        self.verticalLayout.addWidget(self.pushButton_28) 
        self.listWidget_5 = QtWidgets.QListWidget(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(12)
        font.setBold(True)
        font.setWeight(75)
        self.listWidget_5.setFont(font)
        self.listWidget_5.setObjectName("listWidget_5")
        item = QtWidgets.QListWidgetItem()
        self.listWidget_5.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_5.addItem(item)
        item = QtWidgets.QListWidgetItem()
        self.listWidget_5.addItem(item)
        self.verticalLayout.addWidget(self.listWidget_5)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 3)
        spacerItem33 = QtWidgets.QSpacerItem(20, 590, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.gridLayout_2.addItem(spacerItem33, 0, 15, 3, 1)
        spacerItem34 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem34, 2, 2, 1, 1)
        spacerItem35 = QtWidgets.QSpacerItem(42, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.gridLayout_2.addItem(spacerItem35, 2, 10, 1, 1)
        self.line_5 = QtWidgets.QFrame(self.centralwidget)
        self.line_5.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout_2.addWidget(self.line_5, 1, 3, 1, 12)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        
        #按键和对应的槽函数
        '''导航框和MainWin界面StackedWidget中的page'''
        #文件夹创建
        if not os.path.exists(r'.\Users'):
            os.makedirs(r'.\Users')
        if not os.path.exists(r'.\Datafile'):
            os.makedirs(r'.\Datafile')
        if not os.path.exists(r'.\Users\PredictiveProj'):
            os.makedirs(r'.\Users\PredictiveProj')
        if not os.path.exists(r'.\Users\RegulationProj'):
            os.makedirs(r'.\Users\RegulationProj')
        #时间显示
        timer=QtCore.QTimer(self)
        timer.start(1000)
        timer.timeout.connect(self.Timer)
        #对话框
        self.PerformanceUi=Ui_Form()
        self.ModelselectUi=Ui_Form2()
        #堆栈框
        self.listWidget.currentRowChanged.connect(self.NewPro_Pred_list)#NewPro_pred堆栈框数值改变函数
        self.listWidget_2.currentRowChanged.connect(self.NewPro_Con_list)#NewPro_Con堆栈框数值改变函数
        self.listWidget_5.currentRowChanged.connect(self.ConModLoading_list)#ConModLoading堆栈框数值改变函数
        #按键
        self.pushButton_4.clicked.connect(self.MainWin_InstructionsPB)
        self.pushButton.clicked.connect(self.MainWinPB)
#         self.pushButton_3.clicked.connect(self.NewPro_ConPB)
#         self.pushButton_28.clicked.connect(self.ConModLoadingPB)
#         self.pushButton_2.clicked.connect(self.NewPro_PredPB)
        #图片加载
        self.label_15.setPixmap(QtGui.QPixmap(r".\Picture\MainWinPic2.jpg"))
        #常数
        self.U=0.2
        '''NewProPred界面StackedWidget中的page_2'''
        #下拉框
        self.comboBox_2.currentIndexChanged.connect(self.NewPro_Pred_Box)
        #按键
        self.pushButton_5.clicked.connect(self.NewPro_Pred_Cancel)
        self.pushButton_6.clicked.connect(self.NewPro_Pred_Next)
        self.pushButton_7.clicked.connect(self.NewPro_Pred_OK)
        self.pushButton_ok.clicked.connect(self.NewPro_Pred_UpOK)
        '''DataIm界面StackedWidget中的page_3'''
        #按键
        self.pushButton_10.clicked.connect(self.DataIm_DataLoading)
        self.pushButton_9.clicked.connect(self.DataIm_Next)
        self.pushButton_8.clicked.connect(self.DataIm_Back)
        #下拉键
        self.comboBox.currentIndexChanged.connect(self.DataIm_Box)
        #列表框
        self.listWidget_3.currentRowChanged.connect(self.DataIm_list)         
        '''DataDealing界面StackedWidget中的page_4'''
        #按键
        self.pushButton_14.clicked.connect(self.DataDealing_EEE)#Experimental error elimination按钮
        self.pushButton_13.clicked.connect(self.DataDealing_Averaging)#Averaging按钮
        self.pushButton_15.clicked.connect(self.DataDealing_OR)#Outlier remove按钮
        self.pushButton_16.clicked.connect(self.DataDealing_DS)#Data smooth按钮
        self.pushButton_17.clicked.connect(self.DataDealing_RD)#Remove display按钮
        self.pushButton_18.clicked.connect(self.DataDealing_SD)#Smooth display按钮
        self.pushButton_20.clicked.connect(self.DataDealing_LN)#Linear normalization按钮
        self.pushButton_19.clicked.connect(self.DataDealing_CN)#Customm normalization按钮
        self.pushButton_11.clicked.connect(self.DataDealing_Back)#Back按钮
        self.pushButton_12.clicked.connect(self.DataDealing_Next)#Next按钮
        self.pushButton_cancel.clicked.connect(self.DataDealing_Cancel)#cancel按钮
        self.pushButton_AVShow.clicked.connect(self.DataDealing_AVShow)#Average display按钮
        #常数
        self.AverageButtom=0#平均按钮未按下
        self.ORButtom=0#Outlier remove按钮未按下
        self.DSButtom=0#Data Smoothing按钮未按下
        self.Nor=0#未完成归一化
        #滑动条
        self.horizontalSlider_8.setMinimum(1)
        self.horizontalSlider_8.valueChanged.connect(self.Datadealing_Silder)
        '''ModBuilding界面StackedWidget中的page_5'''
        #按键
        self.pushButton_21.clicked.connect(self.ModBulding_Back)
        self.pushButton_22.clicked.connect(self.ModBulding_Down)
        self.pushButton_24.clicked.connect(self.ModBulding_Multiple_Start)
        self.pushButton_26.clicked.connect(self.ModBulding_Multiple_ME)
        self.pushButton_23.clicked.connect(self.ModBulding_Linear_Start)
        self.pushButton_25.clicked.connect(self.ModBulding_Linear_ME)
        self.pushButton_32.clicked.connect(self.ModBulding_SVR_ME)
        self.pushButton_29.clicked.connect(self.ModBulding_SVR_CustomMod)
        self.pushButton_31.clicked.connect(self.ModBulding_SVR_SAP)
        self.pushButton_30.clicked.connect(self.ModBulding_SVR_Start)
        self.pushButton_36.clicked.connect(self.ModBulding_RF_ME)
        self.pushButton_33.clicked.connect(self.ModBulding_RF_CustomMod)
        self.pushButton_34.clicked.connect(self.ModBulding_RF_SAP)
        self.pushButton_35.clicked.connect(self.ModBulding_RF_Start)
        self.pushButton_38.clicked.connect(self.ModBulding_BPNN_ME)
        self.pushButton_37.clicked.connect(self.ModBulding_BPNN_Start)
        self.pushButton_40.clicked.connect(self.ModBulding_SA_ME)
        self.pushButton_39.clicked.connect(self.ModBulding_SA_Start)
        #常数
        self.modelbuliding=0
        '''NewPro_Con界面StackedWidget中的page_6'''
        #按键
        self.pushButton_27.clicked.connect(self.NewPro_Con_OK)
        self.pushButton_42.clicked.connect(self.NewPro_Con_Delete)
        self.pushButton_43.clicked.connect(self.NewPro_Con_Load)
        #常数
        self.NewPro_Con_OKmark=0
        '''ConTargets界面StackedWidget中的page_7'''
        #按钮
        self.pushButton_41.clicked.connect(self.ConTargets_OK)
        self.pushButton_44.clicked.connect(self.ConTargets_Start)
        self.pushButton_45.clicked.connect(self.ConTargets_Reset)
        self.pushButton_46.clicked.connect(self.ConTargets_Gd_OK)
        #常数
        self.ConTargets_model=-1
        '''ConModels界面StackedWidget中的page_8'''
        self.pushButton_47.clicked.connect(self.ConModels_OK)
        self.pushButton_50.clicked.connect(self.ConModels_Back)
        self.pushButton_51.clicked.connect(self.ConModels_Reset)
        self.pushButton_52.clicked.connect(self.ConModels_Down)
        self.pushButton_53.clicked.connect(self.ConModels_SM_3D)
        self.pushButton_54.clicked.connect(self.ConModels_SM_Prediction)
        self.pushButton_55.clicked.connect(self.ConModels_EM_Prediction)
        self.pushButton_56.clicked.connect(self.ConModels_EM_3D)
        self.pushButton_48.clicked.connect(self.ConModels_DA_Start)
        self.pushButton_49.clicked.connect(self.ConModels_MB_Start)
        #常数
        self.ConModels_model=-1
        '''ConModLoading界面StackedWidget中的page_9'''
        #按键
        self.pushButton_57.clicked.connect(self.ConModLoading_Load)
        self.pushButton_58.clicked.connect(self.ConModLoading_Delete)
        '''Display界面StackedWidget中的page_10'''
        #按键
        self.pushButton_59.clicked.connect(self.Display_Up_OK)
        self.pushButton_60.clicked.connect(self.Display_Down_OK)
        #常数
        self.Display_cm=-1
        '''Simulation界面StackedWidget中的page_11'''
        #按键
        self.pushButton_62.clicked.connect(self.Simulation_EMC_Record)
        self.pushButton_63.clicked.connect(self.Simulation_SMC_Record)
        
        '''界面显示的一些设置'''
        self.retranslateUi(MainWindow)
        self.stackedWidget.setCurrentIndex(0)
        self.tabWidget.setCurrentIndex(0)
        self.tabWidget_2.setCurrentIndex(0)
        self.tabWidget_3.setCurrentIndex(0)
        self.toolBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Development platform for protected environment control model"))
        self.setWindowIcon(QtGui.QIcon(r'.\Picture\keylabary.ico'))#给软件加图标
        self.label.setText(_translate("MainWindow", "Intelligent Protected Environmental\n"
" Control Model Platform"))
        self.label_9.setText(_translate("MainWindow", "Email"))
        self.label_10.setText(_translate("MainWindow", "jhudalab@163.com"))
        self.label_7.setText(_translate("MainWindow", "Date"))
        self.label_5.setText(_translate("MainWindow", "Tools"))
        self.label_13.setText(_translate("MainWindow", "Python"))
        self.pushButton_4.setText(_translate("MainWindow", "Instructions"))
        self.label_12.setText(_translate("MainWindow", "2023.6.24."))
        self.label_14.setText(_translate("MainWindow", "Qt Creater, Eclipse, Anaconda3"))
        self.label_8.setText(_translate("MainWindow", "Help"))
        self.label_6.setText(_translate("MainWindow", "Language"))
        self.label_3.setText(_translate("MainWindow", "V 3.0"))
        self.label_11.setText(_translate("MainWindow", "Key Laboratory of Agricultural Internet of Things,\n"
"Ministry of Agriculture and Rural Affairs"))
        self.label_2.setText(_translate("MainWindow", "Version"))
        self.label_4.setText(_translate("MainWindow", "Department"))
        self.label_16.setText(_translate("MainWindow", "Name"))
        self.label_17.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.comboBox_2.setItemText(0, _translate("MainWindow", "Cucumber"))
        self.comboBox_2.setItemText(1, _translate("MainWindow", "Tomato"))
        self.comboBox_2.setItemText(2, _translate("MainWindow", "Pepper"))
        self.comboBox_2.setItemText(3, _translate("MainWindow", "Eggplant"))
        self.comboBox_2.setItemText(4, _translate("MainWindow", "lettuce"))
        self.label_19.setText(_translate("MainWindow", "Project name"))
        self.label_18.setText(_translate("MainWindow", "Crop Type"))
        item = self.tableWidget.verticalHeaderItem(0)
        item.setText(_translate("MainWindow", "Name"))
        item = self.tableWidget.verticalHeaderItem(1)
        item.setText(_translate("MainWindow", "Repetitions"))
        item = self.tableWidget.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Factor 1"))
        item = self.tableWidget.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Factor 2"))
        item = self.tableWidget.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Factor 3"))
        item = self.tableWidget.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Factor 4"))
        self.pushButton_7.setText(_translate("MainWindow", "OK"))
        self.label_21.setText(_translate("MainWindow", "Output Data Name"))
        self.label_20.setText(_translate("MainWindow", "Numbers of input data/environmental factors"))
        self.label_22.setText(_translate("MainWindow", "Output Repetitions"))
        self.pushButton_5.setText(_translate("MainWindow", "Cancel"))
        self.pushButton_6.setText(_translate("MainWindow", "Next"))
        self.pushButton_8.setText(_translate("MainWindow", "Back"))
        self.pushButton_9.setText(_translate("MainWindow", "Next"))
        self.comboBox.setItemText(0, _translate("MainWindow", "Factor 1"))
        self.comboBox.setItemText(1, _translate("MainWindow", "Factor 2"))
        self.comboBox.setItemText(2, _translate("MainWindow", "Factor 3"))
        self.comboBox.setItemText(3, _translate("MainWindow", "Output"))
        __sortingEnabled = self.listWidget_3.isSortingEnabled()
        self.listWidget_3.setSortingEnabled(False)
        item = self.listWidget_3.item(0)
        item.setText(_translate("MainWindow", "Repetition 1"))
        item = self.listWidget_3.item(1)
        item.setText(_translate("MainWindow", "Repetition 2"))
        item = self.listWidget_3.item(2)
        item.setText(_translate("MainWindow", "Repetition 3"))
        self.listWidget_3.setSortingEnabled(__sortingEnabled)
        self.label_25.setText(_translate("MainWindow", ".\n"".\n"".\n."))
        self.pushButton_10.setText(_translate("MainWindow", "Data Loading"))
        self.label_23.setText(_translate("MainWindow", "Name"))
        self.label_24.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit_2.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        item = self.tableWidget_2.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Repetition1"))
        item = self.tableWidget_2.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Repetition2"))
        item = self.tableWidget_2.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Repetition3"))
        self.label_26.setText(_translate("MainWindow", "Name"))
        self.label_27.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit_3.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.pushButton_14.setText(_translate("MainWindow", "Experimental error elimination"))
        self.pushButton_13.setText(_translate("MainWindow", "Averaging"))
        self.pushButton_AVShow.setText(_translate("MainWindow", "Average display"))
        self.pushButton_18.setText(_translate("MainWindow", "Smooth display"))
        self.pushButton_15.setText(_translate("MainWindow", "Outlier remove"))
        self.pushButton_16.setText(_translate("MainWindow", "Data smoothing"))
        self.pushButton_17.setText(_translate("MainWindow", "Remove display"))
        self.label_29.setText(_translate("MainWindow", "Maximum"))
        self.label_28.setText(_translate("MainWindow", "Minimum"))
        self.pushButton_20.setText(_translate("MainWindow", "Linear normalization"))
        self.pushButton_19.setText(_translate("MainWindow", "Custom normalization"))
        self.label_88.setText(_translate("MainWindow", "0"))
        self.label_87.setText(_translate("MainWindow", "Training set proportion"))
        self.label_89.setText(_translate("MainWindow", "Number of Trs"))
        self.label_90.setText(_translate("MainWindow", "Number of Tes"))
        self.pushButton_11.setText(_translate("MainWindow", "Back"))
        self.pushButton_cancel.setText(_translate("MainWindow", "Cancel"))
        self.pushButton_12.setText(_translate("MainWindow", "Next"))
        self.label_30.setText(_translate("MainWindow", "Name"))
        self.label_31.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit_4.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.label_32.setText(_translate("MainWindow", ""))
        self.label_34.setText(_translate("MainWindow", "3-D display"))
        self.pushButton_23.setText(_translate("MainWindow", "Start"))
        self.pushButton_25.setText(_translate("MainWindow", "Model evaluation"))
        self.label_35.setText(_translate("MainWindow", "Test set prediction"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "Linear"))
        self.label_33.setText(_translate("MainWindow", ""))
        self.label_36.setText(_translate("MainWindow", "3-D display"))
        self.label_37.setText(_translate("MainWindow", "Test set prediction"))
        self.pushButton_24.setText(_translate("MainWindow", "Start"))
        self.pushButton_26.setText(_translate("MainWindow", "Model evaluation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "Multiple"))
        self.label_38.setText(_translate("MainWindow", "Kernel"))
        self.comboBox_3.setItemText(0, _translate("MainWindow", "rbf"))
        self.comboBox_3.setItemText(1, _translate("MainWindow", "sigmoid"))
        self.comboBox_3.setItemText(2, _translate("MainWindow", "linear"))
        self.comboBox_3.setItemText(3, _translate("MainWindow", "poly"))
        self.label_39.setText(_translate("MainWindow", "C"))
        self.label_40.setText(_translate("MainWindow", "gamma"))
        self.pushButton_29.setText(_translate("MainWindow", "CustomMod"))
        self.pushButton_31.setText(_translate("MainWindow", "SelfAdaptPara"))
        self.pushButton_30.setText(_translate("MainWindow", "Start"))
        self.label_41.setText(_translate("MainWindow", "Test set prediction"))
        self.pushButton_32.setText(_translate("MainWindow", "Model evaluation"))
        self.label_42.setText(_translate("MainWindow", "3-D display"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "SVR"))
        self.label_44.setText(_translate("MainWindow", "max_leaf"))
        self.label_45.setText(_translate("MainWindow", "n_estimators"))
        self.pushButton_33.setText(_translate("MainWindow", "CustomMod"))
        self.pushButton_34.setText(_translate("MainWindow", "SelfAdaptPara"))
        self.pushButton_35.setText(_translate("MainWindow", "Start"))
        self.label_46.setText(_translate("MainWindow", "3-D display"))
        self.label_43.setText(_translate("MainWindow", "Test set prediction"))
        self.pushButton_36.setText(_translate("MainWindow", "Model evaluation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "RF"))
        self.label_48.setText(_translate("MainWindow", "Hidden layer structure"))
        self.lineEdit_17.setText(_translate("MainWindow", "10-5"))
        self.pushButton_37.setText(_translate("MainWindow", "Start"))
        self.label_51.setText(_translate("MainWindow", "3-D display"))
        self.label_50.setText(_translate("MainWindow", "Test set prediction"))
        self.pushButton_38.setText(_translate("MainWindow", "Model evaluation"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "BPNN"))
        self.label_47.setText(_translate("MainWindow", "Model Parameters"))
        self.label_53.setText(_translate("MainWindow", "3-D display"))
        self.label_52.setText(_translate("MainWindow", "Test set prediction"))
        self.pushButton_40.setText(_translate("MainWindow", "Model evaluation"))
        self.pushButton_39.setText(_translate("MainWindow", "Start"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "Self-adaption"))
        self.pushButton_21.setText(_translate("MainWindow", "Back"))
        self.pushButton_22.setText(_translate("MainWindow", "Down"))
        self.label_55.setText(_translate("MainWindow", "Name"))
        self.label_56.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit_5.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.label_57.setText(_translate("MainWindow", "NewPro name"))
        self.pushButton_27.setText(_translate("MainWindow", "Ok"))
        self.label_49.setText(_translate("MainWindow", "Established predictive model"))
        self.label_54.setText(_translate("MainWindow", "Model infromation"))
        self.pushButton_42.setText(_translate("MainWindow", "Delete"))
        self.pushButton_43.setText(_translate("MainWindow", "Load"))
        self.label_58.setText(_translate("MainWindow", "Name"))
        self.label_59.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit_6.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.groupBox_7.setTitle(_translate("MainWindow", "The factor to be controlled"))
        self.pushButton_41.setText(_translate("MainWindow", "OK"))
        self.radioButton_3.setText(_translate("MainWindow", "Factor1"))
        self.radioButton_5.setText(_translate("MainWindow", "Factor2"))
        self.radioButton_4.setText(_translate("MainWindow", "Factor3"))
        self.radioButton_6.setText(_translate("MainWindow", "Factor4"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Environment gradient settings (separated by \",\")"))
        self.label_60.setText(_translate("MainWindow", "Factor 1"))
        self.label_62.setText(_translate("MainWindow", "Factor3"))
        self.label_61.setText(_translate("MainWindow", "Factor2"))
        self.label_63.setText(_translate("MainWindow", "Factor4"))
        self.label_64.setText(_translate("MainWindow", "Targets computing"))
        self.pushButton_45.setText(_translate("MainWindow", "Reset"))
        self.pushButton_44.setText(_translate("MainWindow", "Start"))
        item = self.tableWidget_3.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Factor1"))
        item = self.tableWidget_3.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Factor2"))
        item = self.tableWidget_3.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Factor3"))
        item = self.tableWidget_3.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Factor4"))
        item = self.tableWidget_3.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Output"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_7), _translate("MainWindow", "Saturation control"))
        item = self.tableWidget_4.horizontalHeaderItem(0)
        item.setText(_translate("MainWindow", "Factor1"))
        item = self.tableWidget_4.horizontalHeaderItem(1)
        item.setText(_translate("MainWindow", "Factor2"))
        item = self.tableWidget_4.horizontalHeaderItem(2)
        item.setText(_translate("MainWindow", "Factor3"))
        item = self.tableWidget_4.horizontalHeaderItem(3)
        item.setText(_translate("MainWindow", "Factor4"))
        item = self.tableWidget_4.horizontalHeaderItem(4)
        item.setText(_translate("MainWindow", "Output"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_8), _translate("MainWindow", "Efficiency control"))
        self.label_65.setText(_translate("MainWindow", "Factor1"))
        self.label_66.setText(_translate("MainWindow", "Factor2"))
        self.label_67.setText(_translate("MainWindow", "Factor3"))
        self.label_68.setText(_translate("MainWindow", "Factor4"))
        self.pushButton_46.setText(_translate("MainWindow", "Ok"))
        self.label_69.setText(_translate("MainWindow", "Environmental value"))
        self.tabWidget_2.setTabText(self.tabWidget_2.indexOf(self.tab_9), _translate("MainWindow", "Graphic display"))
        self.label_70.setText(_translate("MainWindow", "Name"))
        self.label_71.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit_7.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.groupBox_8.setTitle(_translate("MainWindow", "The factor to be controlled"))
        self.pushButton_47.setText(_translate("MainWindow", "OK"))
        self.radioButton_7.setText(_translate("MainWindow", "Factor1"))
        self.radioButton_8.setText(_translate("MainWindow", "Factor2"))
        self.radioButton_9.setText(_translate("MainWindow", "Factor3"))
        self.radioButton_10.setText(_translate("MainWindow", "Factor4"))
        self.label_72.setText(_translate("MainWindow", "Data Acquisition"))
        self.pushButton_48.setText(_translate("MainWindow", "Start"))
        self.label_73.setText(_translate("MainWindow", "Model Building"))
        self.pushButton_49.setText(_translate("MainWindow", "Start"))
        self.pushButton_53.setText(_translate("MainWindow", "3D display"))
        self.pushButton_54.setText(_translate("MainWindow", "Prediction"))
        self.label_74.setText(_translate("MainWindow", "Model performance:"))
        self.tabWidget_3.setTabText(self.tabWidget_3.indexOf(self.tab_10), _translate("MainWindow", "Saturation model"))
        self.pushButton_56.setText(_translate("MainWindow", "3D display"))
        self.pushButton_55.setText(_translate("MainWindow", "Prediction"))
        self.label_75.setText(_translate("MainWindow", "Model performance:"))
        self.tabWidget_3.setTabText(self.tabWidget_3.indexOf(self.tab_11), _translate("MainWindow", "Efficiency Model"))
        self.pushButton_50.setText(_translate("MainWindow", "Back"))
        self.pushButton_51.setText(_translate("MainWindow", "Reset"))
        self.pushButton_52.setText(_translate("MainWindow", "Down"))
        self.label_78.setText(_translate("MainWindow", "Name"))
        self.label_79.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit_8.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.label_76.setText(_translate("MainWindow", "Established predictive model"))
        self.label_77.setText(_translate("MainWindow", "Model infromation"))
        self.pushButton_58.setText(_translate("MainWindow", "Delete"))
        self.pushButton_57.setText(_translate("MainWindow", "Load"))
        self.label_80.setText(_translate("MainWindow", "Name"))
        self.label_81.setText(_translate("MainWindow", "Time"))
        self.label_82.setText(_translate("MainWindow", "Control mode"))
        self.radioButton_2.setText(_translate("MainWindow", "Satuation mode"))
        self.radioButton.setText(_translate("MainWindow", "Effective mode"))
        self.pushButton_59.setText(_translate("MainWindow", "Ok"))
        self.label_83.setText(_translate("MainWindow", "Model Display"))
        self.label_92.setText(_translate("MainWindow", "Target"))
        self.label_84.setText(_translate("MainWindow", "Factor1"))
        self.label_91.setText(_translate("MainWindow", "Factor4"))
        self.label_86.setText(_translate("MainWindow", "Factor3"))
        self.label_85.setText(_translate("MainWindow", "Factor2"))
        self.label_93.setText(_translate("MainWindow", "Output"))
        self.pushButton_60.setText(_translate("MainWindow", "Ok"))
        self.dateTimeEdit_9.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.label_94.setText(_translate("MainWindow", "Name"))
        self.label_95.setText(_translate("MainWindow", "Time"))
        self.dateTimeEdit_10.setDisplayFormat(_translate("MainWindow", "yyyy/M/d HH:mm:ss"))
        self.label_110.setText(_translate("MainWindow", "Time"))
        self.label_111.setText(_translate("MainWindow", "Factor4"))
        self.pushButton_63.setText(_translate("MainWindow", "Record"))
        self.label_112.setText(_translate("MainWindow", "Factor1"))
        self.label_113.setText(_translate("MainWindow", "Factor2"))
        self.label_114.setText(_translate("MainWindow", "Factor3"))
        self.label_109.setText(_translate("MainWindow", "Target value"))
        self.label_108.setText(_translate("MainWindow", "Target output"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_12), _translate("MainWindow", "Saturation model control"))
        self.label_102.setText(_translate("MainWindow", "Time"))
        self.label_105.setText(_translate("MainWindow", "Factor4"))
        self.pushButton_62.setText(_translate("MainWindow", "Record"))
        self.label_104.setText(_translate("MainWindow", "Factor1"))
        self.label_103.setText(_translate("MainWindow", "Factor2"))
        self.label_101.setText(_translate("MainWindow", "Factor3"))
        self.label_106.setText(_translate("MainWindow", "Target value"))
        self.label_107.setText(_translate("MainWindow", "Target output"))
        self.toolBox.setItemText(self.toolBox.indexOf(self.page_13), _translate("MainWindow", "Efficient Model Control"))
        self.pushButton.setText(_translate("MainWindow", "MainWin"))
        self.Label_NewPro_Pred.setText(_translate("MainWindow", "NewPro_Pred"))
        __sortingEnabled = self.listWidget.isSortingEnabled()
        self.listWidget.setSortingEnabled(False)
        item = self.listWidget.item(0)
        item.setText(_translate("MainWindow", "BPhysiProject"))
        item = self.listWidget.item(1)
        item.setText(_translate("MainWindow", "DataIm"))
        item = self.listWidget.item(2)
        item.setText(_translate("MainWindow", "DataDealing"))
        item = self.listWidget.item(3)
        item.setText(_translate("MainWindow", "ModBulding"))
        self.listWidget.setSortingEnabled(__sortingEnabled)
        self.pushButton_3.setText(_translate("MainWindow", "NewPro_Con"))
        __sortingEnabled = self.listWidget_2.isSortingEnabled()
        self.listWidget_2.setSortingEnabled(False)
        item = self.listWidget_2.item(0)
        item.setText(_translate("MainWindow", "BRegulProject"))
        item = self.listWidget_2.item(1)
        item.setText(_translate("MainWindow", "RegulModels"))
        self.listWidget_2.setSortingEnabled(__sortingEnabled)
        self.pushButton_28.setText(_translate("MainWindow", "ConModLoading"))
        __sortingEnabled = self.listWidget_5.isSortingEnabled()
        self.listWidget_5.setSortingEnabled(False)
        item = self.listWidget_5.item(0)
        item.setText(_translate("MainWindow", "Modeloading"))
        item = self.listWidget_5.item(1)
        item.setText(_translate("MainWindow", "Display"))
        item = self.listWidget_5.item(2)
        item.setText(_translate("MainWindow", "Simulation"))
        self.listWidget_5.setSortingEnabled(__sortingEnabled)
        
    def MainWin_InstructionsPB(self):
        print("帮助文档")
    
    def MainWinPB(self):
        self.listWidget.setCurrentRow(-1)#设置NewPro_Pred堆栈窗口未选择
        self.listWidget_2.setCurrentRow(-1)#设置NewPro_Con堆栈窗口未选择
        self.listWidget_5.setCurrentRow(-1)#设置ConModLoading_Pred堆栈窗口未选择
        self.stackedWidget.setCurrentIndex(0)
        return
    
    def NewPro_PredPB(self):
        self.listWidget.setCurrentRow(-1)#设置NewPro_Pred堆栈窗口未选择
        self.listWidget_2.setCurrentRow(-1)#设置NewPro_Con堆栈窗口未选择
        self.listWidget_5.setCurrentRow(-1)#设置ConModLoading_Pred堆栈窗口未选择
        self.stackedWidget.setCurrentIndex(1)
        return
    
    def NewPro_ConPB(self):
        self.listWidget.setCurrentRow(-1)#设置NewPro_Pred堆栈窗口未选择
        self.listWidget_2.setCurrentRow(-1)#设置NewPro_Con堆栈窗口未选择
        self.listWidget_5.setCurrentRow(-1)#设置ConModLoading_Pred堆栈窗口未选择
        self.stackedWidget.setCurrentIndex(5)
        a = list(os.walk(r'.\Users\PredictiveProj'))[0]
        self.listWidget_6.clear()
        self.listWidget_6.addItems(a[1])
        self.listWidget_6.setCurrentRow(0)
        return
    
    def ConModLoadingPB(self):
        self.listWidget.setCurrentRow(-1)#设置NewPro_Pred堆栈窗口未选择
        self.listWidget_2.setCurrentRow(-1)#设置NewPro_Con堆栈窗口未选择
        self.listWidget_5.setCurrentRow(-1)#设置ConModLoading_Pred堆栈窗口未选择
        self.stackedWidget.setCurrentIndex(8)
        a = list(os.walk(r'.\Users\RegulationProj'))[0]
        self.listWidget_7.clear()
        self.listWidget_7.addItems(a[1])
        self.listWidget_7.setCurrentRow(0)
        return
    
    def NewPro_Pred_OK(self):
        a=self.dateTimeEdit.text().split(' ')
        r=self.tableWidget.rowCount()
        c=self.tableWidget.columnCount()
        self.DataIm_VN=[]
        self.DataIm_VR=[]
        try:
            for i in range(c):
                self.DataIm_VN.append(self.tableWidget.item(0,i).text())
                self.DataIm_VR.append(int(self.tableWidget.item(1,i).text()))
            self.DataIm_VN.append(self.lineEdit_3.text())
            self.DataIm_VR.append(int(self.spinBox_2.value()))
        except:
            QMessageBox.critical(self,"Critical","Incomplete table")
            return
        if self.lineEdit_2.text():
            self.lineEdit.setText(self.comboBox_2.currentText()+'_'+self.lineEdit_2.text()+'_'+a[0]+'_'+a[1])
            self.lineEdit_4.setText(self.comboBox_2.currentText()+'_'+self.lineEdit_2.text()+'_'+a[0]+'_'+a[1])
            self.lineEdit_5.setText(self.comboBox_2.currentText()+'_'+self.lineEdit_2.text()+'_'+a[0]+'_'+a[1])
            self.lineEdit_8.setText(self.comboBox_2.currentText()+'_'+self.lineEdit_2.text()+'_'+a[0]+'_'+a[1])
        else:
            QMessageBox.critical(self,"Critical","No name")
            return
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.lineEdit_3.setEnabled(False)
        self.spinBox_2.setEnabled(False)
        self.lineEdit_2.setEnabled(False)
        self.pushButton_7.setEnabled(False)
        self.pushButton_ok.setEnabled(False)
        self.comboBox.clear()
        self.comboBox.addItems(self.DataIm_VN)
        return
    
    def NewPro_Pred_Cancel(self):
        self.lineEdit.setText('')
        self.lineEdit_2.setText('')
        self.lineEdit_4.setText('')
        self.lineEdit_5.setText('')
        self.lineEdit_8.setText('')
        self.lineEdit_3.setEnabled(True)
        self.spinBox_2.setEnabled(True)
        self.lineEdit_2.setEnabled(True)
        self.pushButton_7.setEnabled(True)
        self.pushButton_ok.setEnabled(True)
        self.spinBox.setEnabled(True)
        self.tableWidget.setEditTriggers(QAbstractItemView.CurrentChanged)
        return
        
    def NewPro_Pred_Next(self):
        self.listWidget.setCurrentRow(1)
        self.RowData=dict()#原始数据字典变量，{'变量.重复'：[该变量和该重复条件下的原始数据]}
        return
    
    def NewPro_Pred_list(self,j):
        self.listWidget_2.setCurrentRow(-1)#设置NewPro_Con堆栈窗口未选择
        self.listWidget_5.setCurrentRow(-1)#设置ConModLoading_Pred堆栈窗口未选择
        self.stackedWidget.setCurrentIndex(j+1)
        return
    
    def NewPro_Con_list(self,j):
        self.listWidget.setCurrentRow(-1)#设置NewPro_Pred堆栈窗口未选择
        self.listWidget_5.setCurrentRow(-1)#设置ConModLoading_Pred堆栈窗口未选择
        if j == 0:
            self.NewPro_ConPB()
        if j == 1:
            self.stackedWidget.setCurrentIndex(7)
        return
    
    def ConModLoading_list(self,j):
        self.listWidget.setCurrentRow(-1)#设置NewPro_Pred堆栈窗口未选择
        self.listWidget_2.setCurrentRow(-1)#设置NewPro_Con堆栈窗口未选择
        if j == 0:
            self.ConModLoadingPB()
        if j == 1:
            self.stackedWidget.setCurrentIndex(9)
        if j == 2:
            self.stackedWidget.setCurrentIndex(6)
        return
    
    def DataIm_DataLoading(self):
        fn=QFileDialog.getOpenFileName(self,"Open file",r"./Datafile/","*.csv")[0]#文件路径
        fo=pd.read_csv(fn,header=-1)
        d=list(fo[0])
        a=self.listWidget_3.currentRow()
        key=str(self.comboBox.currentIndex())+'.'+str(a)
        self.RowData[key]=d
        b=self.RowData.keys()
        b=list(b)
        b=list(map(float,b))
        self.tableWidget_2.setRowCount(len(d))
        for i in range(len(d)):
            self.tableWidget_2.setItem(i,a,QTableWidgetItem(str(d[i])))
        huatu=[]
        for i in range(len(b)):
            if int(b[i])==self.comboBox.currentIndex():
                huatu.append(self.RowData[str(b[i])])
        fig=MyFigure(chang=420, kuan=260, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        for i in range(len(huatu)):
            fig.axes.scatter([j for j in range(len(huatu[i]))],huatu[i],marker='.')
        self.graphicsView_2.setScene(scene)
        return
    
    def DataIm_Back(self):
        self.listWidget.setCurrentRow(0)
        return
    
    def DataIm_Next(self):
        self.listWidget.setCurrentRow(2)
        return
    
    def DataDealing_EEE(self):
        QMessageBox.information(self,"Information","Experimental error have been eliminated")
        return
        
    def DataDealing_Averaging(self):
        self.RowData_mean=[]#多次重复的均值
        for i in range(len(self.DataIm_VR)):
            meanvalue=0
            for j in range(self.DataIm_VR[i]):
                QApplication.processEvents()
                meanvalue=meanvalue+np.array(self.RowData[str(i)+'.'+str(j)])
            meanvalue=meanvalue/self.DataIm_VR[i]
            self.RowData_mean.append(meanvalue)
        self.RowData_mean=np.array(self.RowData_mean)
        self.Ff=[]
        self.RowData_mean1=[]#多次重复的均值归一化
        for i in range (len(self.RowData_mean)):
            Ff_tem=F(self.RowData_mean[i])
            self.Ff.append(Ff_tem)
            self.RowData_mean1.append(Ff_tem.f(self.RowData_mean[i]))
        self.RowData_mean1=np.array(self.RowData_mean1)
        QMessageBox.information(self,"Information","Experimental error have been averaged")
        self.AverageButtom=1#平均按钮已经按下
        return
    
    def DataDealing_AVShow(self):
        if self.AverageButtom==0:
            QMessageBox.critical(self,"Critical","Incomplete average data")
            return
        a=self.RowData_mean1.copy()
        a=a.T 
        pca = PCA(n_components=3)
        b= pca.fit_transform(a)
        b=b.T
        myfig=MyFigure3D(chang=274, kuan=190, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(myfig)
        myfig.axes.scatter(b[0],b[1],b[2],marker='.')
        self.graphicsView_3.setScene(scene)
        return
    
    def DataDealing_OR(self):
        datain=self.RowData_mean1[:-1]
        datain=datain.T
        dataout=self.RowData_mean1[-1]
        pca = PCA(n_components=1)
        X_pca = pca.fit_transform(datain)
        coefficients = np.linalg.lstsq(X_pca, dataout, rcond=None)[0]
        y_cal=np.dot(X_pca, coefficients)
        error=dataout-y_cal 
        errormean=np.mean(error)
        y_cal=np.dot(X_pca, coefficients)+errormean
        error=np.abs(dataout-y_cal)
        ind=np.array([i for i in range(len(error))])
        errorind=np.array([ind,error])
        errorind=errorind.T
        a=len(errorind)
        b=[]
        for i in range(a):
            b.append(tuple(errorind[i]))
        schema=[('ind',int),('error',float)]
        errorind=np.array(b,dtype=schema)
        errorind=np.sort(errorind,order='error')
        a=a-int(a*0.03)
        errorind=errorind[:a]
        ind=[]
        for i in range(a):
            ind.append(errorind[i][0])
        random.shuffle(ind)
        self.indata=[]#误差剔除后的归一化输入
        self.outdata=[]#误差剔除后的归一化输出
        for i in ind:
            self.indata.append(datain[i])
            self.outdata.append(dataout[i])
        self.indata=np.array(self.indata)
        self.outdata=np.array(self.outdata)
        self.ORButtom=1#Outlier remove按钮已经按下
        QMessageBox.information(self,"Information","Outliers have been removed")
        return 

    def DataDealing_DS(self):
        self.DSButtom=1#Data Smoothing按钮按下
        QMessageBox.information(self,"Information","Data have been smooth")
        return
        
    def DataDealing_RD(self):
        if self.ORButtom==0:
            QMessageBox.critical(self,"Critical","Outliers have not been removed")
            return
        data1=self.indata.T
        data2=self.outdata
        data=[]
        for i in range(len(data1)):
            data.append(data1[i])
        data.append(data2)
        data=np.array(data)
        data=data.T
        pca = PCA(n_components=3)
        b= pca.fit_transform(data)
        b=b.T
        myfig=MyFigure3D(chang=274, kuan=190, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(myfig)
        myfig.axes.scatter(b[0],b[1],b[2],marker='.',color='r')
        self.graphicsView_3.setScene(scene)
        return
        
    def DataDealing_SD(self):
        if self.DSButtom==0:
            QMessageBox.critical(self,"Critical","Data have not been smooth")
            return
        data1=self.indata.T
        data2=self.outdata
        data=[]
        for i in range(len(data1)):
            data.append(data1[i])
        data.append(data2)
        data=np.array(data)
        data=data.T
        pca = PCA(n_components=3)
        b= pca.fit_transform(data)
        b=b.T
        myfig=MyFigure3D(chang=274, kuan=190, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(myfig)
        myfig.axes.scatter(b[0],b[1],b[2],marker='.',color='g')
        self.graphicsView_3.setScene(scene)
        return
        
    def DataDealing_LN(self):
        minv=float(self.lineEdit_6.text())
        maxv=float(self.lineEdit_7.text())
        a=self.indata.T 
        b=self.outdata
        FF=[]
        a1=[]
        for i in range(len(a)):
            a2=self.Ff[i].f_b(a[i])
            FF.append(F(a2,mini=minv,maxi=maxv))
            a1.append(FF[i].f(a2))
        b2=self.Ff[-1].f_b(b)
        FF.append(F(b2,mini=minv,maxi=maxv))
        b1=FF[-1].f(b2)
        a1=np.array(a1)
        self.indata=a1.T 
        self.outdata=b1
        self.lineEdit_6.setEnabled(False)
        self.lineEdit_7.setEnabled(False)
        self.pushButton_20.setEnabled(False)
        self.pushButton_19.setEnabled(False)
        QMessageBox.information(self,"Information","Normalization has been done")
        self.Nor=1
        self.FF=FF
        return
        
    def DataDealing_CN(self):
        minv=0.1
        maxv=0.9
        a=self.indata.T 
        b=self.outdata
        FF=[]
        a1=[]
        for i in range(len(a)):
            a2=self.Ff[i].f_b(a[i])
            FF.append(F(a2,mini=minv,maxi=maxv))
            a1.append(FF[i].f(a2))
        b2=self.Ff[-1].f_b(b)
        FF.append(F(b2,mini=minv,maxi=maxv))
        b1=FF[-1].f(b2)
        a1=np.array(a1)
        self.indata=a1.T 
        self.outdata=b1
        self.lineEdit_6.setEnabled(False)
        self.lineEdit_7.setEnabled(False)
        self.pushButton_20.setEnabled(False)
        self.pushButton_19.setEnabled(False)
        QMessageBox.information(self,"Information","Normalization has been done")
        self.Nor=1
        self.FF=FF
        return
        
    def DataDealing_Back(self):
        self.listWidget.setCurrentRow(1)
        return 
    
    def DataDealing_Cancel(self):
        self.lineEdit_6.setEnabled(True)
        self.lineEdit_7.setEnabled(True)
        self.pushButton_20.setEnabled(True)
        self.pushButton_19.setEnabled(True)
        self.lineEdit_6.setText('')
        self.lineEdit_7.setText('')
        self.horizontalSlider_8.setValue(1)
        self.lineEdit_34.setText('')
        self.lineEdit_35.setText('')
        self.Nor=0
        return
        
    def Datadealing_Silder(self):
        if self.Nor==0:
            QMessageBox.critical(self,"Data have not been normalized")
            return
        slidevalue=self.horizontalSlider_8.value()
        self.label_88.setText('{:.2f}'.format(slidevalue*0.01))
        totlenum=len(self.outdata)
        self.lineEdit_34.setText('{}'.format(int(slidevalue*0.01*totlenum)))
        self.lineEdit_35.setText('{}'.format(totlenum-int(slidevalue*0.01*totlenum)))
        return
    
    def DataDealing_Next(self):
        radio=self.horizontalSlider_8.value()*0.01
        self.x_tr,self.x_te,self.y_tr,self.y_te=model_selection.train_test_split(self.indata,self.outdata,train_size=radio)
        self.listWidget.setCurrentRow(3)
        return
    
    def ModBulding_Back(self):
        self.listWidget.setCurrentRow(2)
        return
    
    def ModBulding_Down(self):
        if self.modelbuliding==0:
            QMessageBox.critical(self,"Model have not been build")
            return
        filename=self.lineEdit_8.text()
        filename=filename.split('/')
        filename='_'.join(filename)
        filename=filename.split(':')
        filename='_'.join(filename)
        road=r'.\Users\PredictiveProj\\'+filename
        if os.path.exists(road):
            reply = QMessageBox.information(self, "warning", "There is already a project with the same name here"+"\n"+
                                            "do you want to overwrite it?",
                                             QMessageBox.Yes | QMessageBox.No ,  QMessageBox.Yes )
            if reply==QMessageBox.Yes:
                a = list(os.walk(road))[0]
                for s in a[2]:
                    r=a[0]+'\\'+s 
                    os.remove(r)
                os.rmdir(road)
            elif reply==QMessageBox.No:
                return
        Paratext=[]
        os.makedirs(road)
        savepkl(road+'\\indata.pkl', self.indata)
        savepkl(road+'\\outdata.pkl', self.outdata)
        savepkl(road+'\\norm.pkl', self.FF)
        savepkl(road+'\\name.pkl', self.DataIm_VN)
        try:
            savepkl(road+'\\ModelofLinear.pkl', self.fl)
            Paral="The Linear model parameters are:\nroot mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.fl_train[0],self.fl_train[1],self.fl_train[2],self.fl_train[3],self.fl_train[4],
                       self.fl_train[5],self.fl_train[6],self.fl_train[7],self.fl_train[8],self.fl_train[9],
                       self.fl_test[0],self.fl_test[1],self.fl_test[2],self.fl_test[3],self.fl_test[4],
                       self.fl_test[5],self.fl_test[6],self.fl_test[7],self.fl_test[8],self.fl_test[9])
            savepkl(road+'\\ParametersofLinear.pkl', Paral)
        except:
            pass
        try:
            savepkl(road+'\\ModelofMultiple.pkl', self.fm)
            Param="The Multiple model parameters are:\nroot mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.fm_train[0],self.fm_train[1],self.fm_train[2],self.fm_train[3],self.fm_train[4],
                       self.fm_train[5],self.fm_train[6],self.fm_train[7],self.fm_train[8],self.fm_train[9],
                       self.fm_test[0],self.fm_test[1],self.fm_test[2],self.fm_test[3],self.fm_test[4],
                       self.fm_test[5],self.fm_test[6],self.fm_test[7],self.fm_test[8],self.fm_test[9])
            savepkl(road+'\\ParametersofMultiple.pkl', Param)
        except:
            pass
        try:
            savepkl(road+'\\ModelofSVR.pkl', self.fsvr)
            Parasvr="The SVR model parameters are:\nroot mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.fsvr_train[0],self.fsvr_train[1],self.fsvr_train[2],self.fsvr_train[3],
                                   self.fsvr_train[4],self.fsvr_train[5],self.fsvr_train[6],self.fsvr_train[7],
                                   self.fsvr_train[8],self.fsvr_train[9],self.fsvr_test[0],self.fsvr_test[1],
                                   self.fsvr_test[2],self.fsvr_test[3],self.fsvr_test[4],self.fsvr_test[5],
                                   self.fsvr_test[6],self.fsvr_test[7],self.fsvr_test[8],self.fsvr_test[9])
            savepkl(road+'\\ParametersofSVR.pkl', Parasvr)
        except:
            pass
        try:
            savepkl(road+'\\ModelofRF.pkl', self.frf)
            Pararf="The RF model parameters are:\nroot mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.frf_train[0],self.frf_train[1],self.frf_train[2],self.frf_train[3],
                       self.frf_train[4],self.frf_train[5],self.frf_train[6],self.frf_train[7],
                       self.frf_train[8],self.frf_train[9],self.frf_test[0],self.frf_test[1],
                       self.frf_test[2],self.frf_test[3],self.frf_test[4],self.frf_test[5],
                       self.frf_test[6],self.frf_test[7],self.frf_test[8],self.frf_test[9])
            savepkl(road+'\\ParametersofRF.pkl', Pararf)
        except:
            pass
        try:
            savepkl(road+'\\ModelofBPNN.pkl', self.fbp)
            Parabp="The BPNN model parameters are:\nroot mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test".format(self.fbp_train[0],self.fbp_train[1],self.fbp_train[2],self.fbp_train[3],
                       self.fbp_train[4],self.fbp_train[5],self.fbp_train[6],self.fbp_train[7],
                       self.fbp_train[8],self.fbp_train[9],self.fbp_test[0],self.fbp_test[1],
                       self.fbp_test[2],self.fbp_test[3],self.fbp_test[4],self.fbp_test[5],
                       self.fbp_test[6],self.fbp_test[7],self.fbp_test[8],self.fbp_test[9])
            savepkl(road+'\\ParametersofBPNN.pkl', Parabp)
        except:
            pass
        try:
            savepkl(road+'\\ModelofRgs.pkl', self.rgs)
            Parargs="The SelfAdaptionModel model parameters are:\nroot mean square error of {:.4f},\
 mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.rgs_train[0],self.rgs_train[1],self.rgs_train[2],self.rgs_train[3],
                       self.rgs_train[4],self.rgs_train[5],self.rgs_train[6],self.rgs_train[7],
                       self.rgs_train[8],self.rgs_train[9],self.rgs_test[0],self.rgs_test[1],
                       self.rgs_test[2],self.rgs_test[3],self.rgs_test[4],self.rgs_test[5],
                       self.rgs_test[6],self.rgs_test[7],self.rgs_test[8],self.rgs_test[9])
            savepkl(road+'\\ParametersofRgs.pkl', Parargs)
        except:
            pass
        QMessageBox.information(self,"Information","The photosynthetic rate prediction project\nhas been established")
        return
            
    def FunM(self,P,X):
        nv=len(X)#自变量个数
        pi=0
        s=0
        for i in range(nv):
            for j in range(i, nv):
                s+=P[pi]*X[i]*X[j]
                pi+=1
        s=s+P[pi]
        return s
    
    def ErrorM(self,P,X,Y):
        QApplication.processEvents()
        return (self.FunM(P,X)-Y)**2
        
    def ModBulding_Multiple_Start(self):
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        xtr,xte=xtr.T,xte.T
        self.progressBar_2.setValue(50)
        nv=len(xtr)#自变量个数
        lenP=int((nv*(nv-1))/2+nv+1)
        P0=[0.1 for i in range(lenP)]
        Para=leastsq(self.ErrorM,P0,args=(xtr,ytr)) #把error函数中除了p以外的参数打包到args中
        W=list(Para[0])
        self.fm=FM(W)
        self.progressBar_2.setValue(100)
        y_cal=self.fm.predict(xte.T)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        fig=MyFigure(chang=274, kuan=261, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        fig.axes.scatter(Y_te,Y_cal,marker='x',color='r')
        fig.axes.plot([Y_te.min(),Y_cal.max(),],[Y_te.min(),Y_cal.max()],color='b')
        self.graphicsView_6.setScene(scene)
        self.modelbuliding=1
        mi=self.FF[0].a
        ma=self.FF[0].b
        if nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.fm.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=255, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_7.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=255, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_7.setScene(scene)
            return
        return
        
    def ModBulding_Multiple_ME(self):
        try:
            self.fm 
        except:
            QMessageBox.critical(self,"Critical","No multiple model")
            return
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        ytr_=self.fm.predict(xtr)
        yte_=self.fm.predict(xte)
        ytr=np.ravel(ytr)
        yte=np.ravel(yte)
        ytr_=np.ravel(ytr_)
        yte_=np.ravel(yte_)
        Yte=self.FF[-1].f_b(yte)
        Yte_=self.FF[-1].f_b(yte_)
        Ytr=self.FF[-1].f_b(ytr)
        Ytr_=self.FF[-1].f_b(ytr_)
        a=self.RMSE(Yte, Yte_)
        b=MAE(Yte, Yte_)
        c=Mjue(Yte, Yte_)
        d=Mxiang(Yte, Yte_)
        e=R2(Yte, Yte_)
        f=self.R2g(Yte, Yte_)
        g,h=NiXi(Yte, Yte_)
        i=Rou(Yte, Yte_)
        j=Thigma(Yte, Yte_)
        self.fm_test=[a,b,c,d,e,f,g,h,i,j]
        a=self.RMSE(Ytr, Ytr_)
        b=MAE(Ytr, Ytr_)
        c=Mjue(Ytr, Ytr_)
        d=Mxiang(Ytr, Ytr_)
        e=R2(Ytr, Ytr_)
        f=self.R2g(Ytr, Ytr_)
        g,h=NiXi(Ytr, Ytr_)
        i=Rou(Ytr, Ytr_)
        j=Thigma(Ytr, Ytr_)
        self.fm_train=[a,b,c,d,e,f,g,h,i,j]
        dlg=QDialog()
        self.PerformanceUi.setupUi(dlg)
        for i in range(len(self.fm_train)):
            self.PerformanceUi.tableWidget.setItem(i,0,QTableWidgetItem('{:.4f}'.format(self.fm_train[i])))
        for i in range(len(self.fm_test)):
            self.PerformanceUi.tableWidget.setItem(i,1,QTableWidgetItem('{:.4f}'.format(self.fm_test[i])))
        dlg.exec_()
        return
                
    def ModBulding_Linear_Start(self):
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        xtr,xte=xtr.T,xte.T
        nv=len(xtr)#自变量个数
        self.progressBar.setValue(50)
        P0=[0.1 for i in range(nv+1)]
        Para=leastsq(self.ErrorL,P0,args=(xtr,ytr)) #把error函数中除了p以外的参数打包到args中
        W=list(Para[0])
        self.fl=FL(W)
        self.progressBar.setValue(100)
        y_cal=self.fl.predict(xte.T)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        fig=MyFigure(chang=274, kuan=261, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        fig.axes.scatter(Y_te,Y_cal,marker='x',color='r')
        fig.axes.plot([Y_te.min(),Y_cal.max(),],[Y_te.min(),Y_cal.max()],color='b')
        self.graphicsView_4.setScene(scene)
        self.modelbuliding=1
        mi=self.FF[0].a
        ma=self.FF[0].b
        if nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.fl.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=255, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_5.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=255, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fl.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fl.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fl.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_5.setScene(scene)
            return
        return
    
    def FunL(self,P,X):
        s=0
        for i in range(len(X)):
            s+=P[i]*X[i]
        s=s+P[-1]
        return s
    
    def ErrorL(self,P,X,Y):
        QApplication.processEvents()
        return (self.FunL(P,X)-Y)**2
        
    def ModBulding_Linear_ME(self):
        try:
            self.fl 
        except:
            QMessageBox.critical(self,"Critical","No multiple model")
            return
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        ytr_=self.fl.predict(xtr)
        yte_=self.fl.predict(xte)
        ytr=np.ravel(ytr)
        yte=np.ravel(yte)
        ytr_=np.ravel(ytr_)
        yte_=np.ravel(yte_)
        Yte=self.FF[-1].f_b(yte)
        Yte_=self.FF[-1].f_b(yte_)
        Ytr=self.FF[-1].f_b(ytr)
        Ytr_=self.FF[-1].f_b(ytr_)
        a=self.RMSE(Yte, Yte_)
        b=MAE(Yte, Yte_)
        c=Mjue(Yte, Yte_)
        d=Mxiang(Yte, Yte_)
        e=R2(Yte, Yte_)
        f=self.R2g(Yte, Yte_)
        g,h=NiXi(Yte, Yte_)
        i=Rou(Yte, Yte_)
        j=Thigma(Yte, Yte_)
        self.fl_test=[a,b,c,d,e,f,g,h,i,j]
        a=self.RMSE(Ytr, Ytr_)
        b=MAE(Ytr, Ytr_)
        c=Mjue(Ytr, Ytr_)
        d=Mxiang(Ytr, Ytr_)
        e=R2(Ytr, Ytr_)
        f=self.R2g(Ytr, Ytr_)
        g,h=NiXi(Ytr, Ytr_)
        i=Rou(Ytr, Ytr_)
        j=Thigma(Ytr, Ytr_)
        self.fl_train=[a,b,c,d,e,f,g,h,i,j]
        dlg=QDialog()
        self.PerformanceUi.setupUi(dlg)
        for i in range(len(self.fl_train)):
            self.PerformanceUi.tableWidget.setItem(i,0,QTableWidgetItem('{:.4f}'.format(self.fl_train[i])))
        for i in range(len(self.fl_test)):
            self.PerformanceUi.tableWidget.setItem(i,1,QTableWidgetItem('{:.4f}'.format(self.fl_test[i])))
        dlg.exec_()
        return
        
    def ModBulding_SVR_ME(self):
        try:
            self.fsvr 
        except:
            QMessageBox.critical(self,"Critical","No multiple model")
            return
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        ytr_=self.fsvr.predict(xtr)
        yte_=self.fsvr.predict(xte)
        ytr=np.ravel(ytr)
        yte=np.ravel(yte)
        ytr_=np.ravel(ytr_)
        yte_=np.ravel(yte_)
        Yte=self.FF[-1].f_b(yte)
        Yte_=self.FF[-1].f_b(yte_)
        Ytr=self.FF[-1].f_b(ytr)
        Ytr_=self.FF[-1].f_b(ytr_)
        a=self.RMSE(Yte, Yte_)
        b=MAE(Yte, Yte_)
        c=Mjue(Yte, Yte_)
        d=Mxiang(Yte, Yte_)
        e=R2(Yte, Yte_)
        f=self.R2g(Yte, Yte_)
        g,h=NiXi(Yte, Yte_)
        i=Rou(Yte, Yte_)
        j=Thigma(Yte, Yte_)
        self.fsvr_test=[a,b,c,d,e,f,g,h,i,j]
        a=self.RMSE(Ytr, Ytr_)
        b=MAE(Ytr, Ytr_)
        c=Mjue(Ytr, Ytr_)
        d=Mxiang(Ytr, Ytr_)
        e=R2(Ytr, Ytr_)
        f=self.R2g(Ytr, Ytr_)
        g,h=NiXi(Ytr, Ytr_)
        i=Rou(Ytr, Ytr_)
        j=Thigma(Ytr, Ytr_)
        self.fsvr_train=[a,b,c,d,e,f,g,h,i,j]
        dlg=QDialog()
        self.PerformanceUi.setupUi(dlg)
        for i in range(len(self.fsvr_train)):
            self.PerformanceUi.tableWidget.setItem(i,0,QTableWidgetItem('{:.4f}'.format(self.fsvr_train[i])))
        for i in range(len(self.fsvr_test)):
            self.PerformanceUi.tableWidget.setItem(i,1,QTableWidgetItem('{:.4f}'.format(self.fsvr_test[i])))
        dlg.exec_()
        return
        
    def ModBulding_SVR_CustomMod(self):
        try:
            eval(self.lineEdit_12.text())
            eval(self.lineEdit_11.text())
        except:
            QMessageBox.critical(self,"Critical","Incomplete parameters")
            return
        self.svrPara=1
        self.lineEdit_11.setEnabled(False)
        self.lineEdit_12.setEnabled(False)
        self.pushButton_29.setEnabled(False)
        self.pushButton_31.setEnabled(False)
        QMessageBox.information(self,"Information","User defined parameters")
        return
        
    def ModBulding_SVR_SAP(self):
        self.lineEdit_11.setText('')
        self.lineEdit_12.setText('')
        self.lineEdit_11.setEnabled(False)
        self.lineEdit_12.setEnabled(False)
        self.pushButton_29.setEnabled(False)
        self.pushButton_31.setEnabled(False)
        self.svrPara=0
        QMessageBox.information(self,"Information","Adaptive parameters")
        return
        
    def ModBulding_SVR_Start(self):
        try:
            self.svrPara
        except:
            QMessageBox.critical(self,"Critical","Incomplete parameters")
            return
        self.ker=self.comboBox_3.currentText()
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        if self.svrPara==1 :
            g=eval(self.lineEdit_12.text())
            c=eval(self.lineEdit_11.text())
            self.fsvr=svm.SVR(C=c,kernel=self.ker,gamma=g,epsilon = 0.04).fit(xtr,ytr)
        else:
            self.popsize=20
            self.chrnum=2
            self.chrlen=[12,7]
            self.min=[0.01,0.01]
            self.max=[100,10]
            self.gen=200
            self.maxgen=20
            self.px=0.8
            self.pc=0.1
            c,g=self.qgarun([xtr,ytr],[xte,yte])
            self.fsvr=svm.SVR(C=c,kernel=self.ker,gamma=g,epsilon = 0.04).fit(xtr,ytr)
        self.progressBar_3.setValue(100)
        y_cal=self.fsvr.predict(xte)
        y_cal=np.ravel(y_cal)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        self.modelbuliding=1
        fig=MyFigure(chang=261, kuan=254, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        fig.axes.scatter(Y_te,Y_cal,marker='x',color='r')
        fig.axes.plot([Y_te.min(),Y_cal.max(),],[Y_te.min(),Y_cal.max()],color='b')
        self.graphicsView_8.setScene(scene)
        nv=len(self.DataIm_VN)-1
        mi=self.FF[0].a
        ma=self.FF[0].b
        if nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.fsvr.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=255, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_9.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=261, kuan=254, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fsvr.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fsvr.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fsvr.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_9.setScene(scene)
            return
        return
        
    def ModBulding_RF_ME(self):
        try:
            self.frf 
        except:
            QMessageBox.critical(self,"Critical","No multiple model")
            return
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        ytr_=self.frf.predict(xtr)
        yte_=self.frf.predict(xte)
        ytr=np.ravel(ytr)
        yte=np.ravel(yte)
        ytr_=np.ravel(ytr_)
        yte_=np.ravel(yte_)
        Yte=self.FF[-1].f_b(yte)
        Yte_=self.FF[-1].f_b(yte_)
        Ytr=self.FF[-1].f_b(ytr)
        Ytr_=self.FF[-1].f_b(ytr_)
        a=self.RMSE(Yte, Yte_)
        b=MAE(Yte, Yte_)
        c=Mjue(Yte, Yte_)
        d=Mxiang(Yte, Yte_)
        e=R2(Yte, Yte_)
        f=self.R2g(Yte, Yte_)
        g,h=NiXi(Yte, Yte_)
        i=Rou(Yte, Yte_)
        j=Thigma(Yte, Yte_)
        self.frf_test=[a,b,c,d,e,f,g,h,i,j]
        a=self.RMSE(Ytr, Ytr_)
        b=MAE(Ytr, Ytr_)
        c=Mjue(Ytr, Ytr_)
        d=Mxiang(Ytr, Ytr_)
        e=R2(Ytr, Ytr_)
        f=self.R2g(Ytr, Ytr_)
        g,h=NiXi(Ytr, Ytr_)
        i=Rou(Ytr, Ytr_)
        j=Thigma(Ytr, Ytr_)
        self.frf_train=[a,b,c,d,e,f,g,h,i,j]
        dlg=QDialog()
        self.PerformanceUi.setupUi(dlg)
        for i in range(len(self.frf_train)):
            self.PerformanceUi.tableWidget.setItem(i,0,QTableWidgetItem('{:.4f}'.format(self.frf_train[i])))
        for i in range(len(self.frf_test)):
            self.PerformanceUi.tableWidget.setItem(i,1,QTableWidgetItem('{:.4f}'.format(self.frf_test[i])))
        dlg.exec_()
        return
        
    def ModBulding_RF_CustomMod(self):
        try:
            eval(self.lineEdit_13.text())
            eval(self.lineEdit_14.text())
        except:
            QMessageBox.critical(self,"Critical","Incomplete parameters")
            return
        self.rfPara=1
        self.lineEdit_13.setEnabled(False)
        self.lineEdit_14.setEnabled(False)
        self.pushButton_33.setEnabled(False)
        self.pushButton_34.setEnabled(False)
        QMessageBox.information(self,"Information","User defined parameters")
        return
        
    def ModBulding_RF_SAP(self):
        self.lineEdit_13.setText('')
        self.lineEdit_14.setText('')
        self.lineEdit_13.setEnabled(False)
        self.lineEdit_14.setEnabled(False)
        self.pushButton_33.setEnabled(False)
        self.pushButton_34.setEnabled(False)
        self.rfPara=0
        QMessageBox.information(self,"Information","Adaptive parameters")
        return
        
    def ModBulding_RF_Start(self):
        try:
            self.rfPara
        except:
            QMessageBox.critical(self,"Critical","Incomplete parameters")
            return
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        if self.rfPara==1:
            md=eval(self.lineEdit_13.text())
            ne=eval(self.lineEdit_14.text())
            self.frf=RandomForestRegressor(max_leaf_nodes=md,n_estimators=ne).fit(xtr,ytr)
        else:
            marks=100
            for i in range(10):
                md=5*i+5
                for j in range(10):
                    self.progressBar_4.setValue(i*10+j)
                    QApplication.processEvents()
                    ne=5*j+5
                    rgs=RandomForestRegressor(max_leaf_nodes=md,n_estimators=ne).fit(xtr,ytr)
                    Ycal=rgs.predict(xte)
                    Ycal=np.ravel(Ycal)
                    markstem=self.RMSE(Ycal,yte)
                    if markstem<marks:
                        marks=markstem 
                        nebest=md 
                        mdbest=ne
            self.frf=RandomForestRegressor(max_leaf_nodes=mdbest,n_estimators=nebest).fit(xtr,ytr)
        self.progressBar_4.setValue(100)
        y_cal=self.frf.predict(xte)
        y_cal=np.ravel(y_cal)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        fig=MyFigure(chang=261, kuan=254, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        fig.axes.scatter(Y_te,Y_cal,marker='x',color='r')
        fig.axes.plot([Y_te.min(),Y_cal.max(),],[Y_te.min(),Y_cal.max()],color='b')
        self.graphicsView_11.setScene(scene)
        self.modelbuliding=1
        nv=len(self.DataIm_VN)-1
        mi=self.FF[0].a
        ma=self.FF[0].b
        if nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.frf.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=255, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_10.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=265, kuan=254, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.frf.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.frf.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.frf.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_10.setScene(scene)
            return
        return
        
    def ModBulding_BPNN_ME(self):
        try:
            self.fbp 
        except:
            QMessageBox.critical(self,"Critical","No multiple model")
            return
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        ytr_=self.fbp.predict(xtr)
        yte_=self.fbp.predict(xte)
        ytr=np.ravel(ytr)
        yte=np.ravel(yte)
        ytr_=np.ravel(ytr_)
        yte_=np.ravel(yte_)
        Yte=self.FF[-1].f_b(yte)
        Yte_=self.FF[-1].f_b(yte_)
        Ytr=self.FF[-1].f_b(ytr)
        Ytr_=self.FF[-1].f_b(ytr_)
        a=self.RMSE(Yte, Yte_)
        b=MAE(Yte, Yte_)
        c=Mjue(Yte, Yte_)
        d=Mxiang(Yte, Yte_)
        e=R2(Yte, Yte_)
        f=self.R2g(Yte, Yte_)
        g,h=NiXi(Yte, Yte_)
        i=Rou(Yte, Yte_)
        j=Thigma(Yte, Yte_)
        self.fbp_test=[a,b,c,d,e,f,g,h,i,j]
        a=self.RMSE(Ytr, Ytr_)
        b=MAE(Ytr, Ytr_)
        c=Mjue(Ytr, Ytr_)
        d=Mxiang(Ytr, Ytr_)
        e=R2(Ytr, Ytr_)
        f=self.R2g(Ytr, Ytr_)
        g,h=NiXi(Ytr, Ytr_)
        i=Rou(Ytr, Ytr_)
        j=Thigma(Ytr, Ytr_)
        self.fbp_train=[a,b,c,d,e,f,g,h,i,j]
        dlg=QDialog()
        self.PerformanceUi.setupUi(dlg)
        for i in range(len(self.fbp_train)):
            self.PerformanceUi.tableWidget.setItem(i,0,QTableWidgetItem('{:.4f}'.format(self.fbp_train[i])))
        for i in range(len(self.fbp_test)):
            self.PerformanceUi.tableWidget.setItem(i,1,QTableWidgetItem('{:.4f}'.format(self.fbp_test[i])))
        dlg.exec_()
        return
        
    def ModBulding_BPNN_Start(self):
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  
        try:
            struction=[]
            hls=self.lineEdit_17.text()
            HLS=hls.split('-')
            struction.append(len(self.DataIm_VN)-1)
            for i in range(len(HLS)):
                struction.append(int(HLS[i]))
            struction.append(1)
        except:
            QMessageBox.critical(self,"Critical","Wrong structure")
            return
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        ytr=ytr.reshape(len(ytr),1)
        yte=yte.reshape(len(yte),1)
        data=[xtr,ytr,xte,yte]
        batch_count=20#子集数量
        learning_rate=0.02#学习率
        training_epochs=1000#训练步数
        X=tf.placeholder(tf.float32, [None,struction[0]], name="input")#占位符作为输入单元
        Y_=tf.placeholder(tf.float32,[None,struction[-1]])#输出单元
        canshuW=[]
        canshuB=[]
        for i in range(len(struction)-1):
            W=tf.Variable(tf.truncated_normal([struction[i],struction[i+1]],stddev=0.01))#输入层到隐层1的权值矩阵
            B=tf.Variable(tf.zeros([struction[i+1]]))#输入层到隐层1的阈值
            canshuW.append(W)
            canshuB.append(B)
        X1=X 
        for i in range(len(canshuW)-1):
            Y1=tf.nn.sigmoid(tf.matmul(X1,canshuW[i])+canshuB[i])
            X1=Y1
        Y=tf.nn.relu(tf.matmul(X1,canshuW[-1])+canshuB[-1],name='output')#神经网络输出
        E=0.5*(Y-Y_)**2
        E=tf.reduce_mean(E)*100#损失函数
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.035)#学习率0.1，精度0.03
        train_step = optimizer.minimize(E)#全局优化
        with tf.Session() as sess:#建立会话
            sess.run(tf.global_variables_initializer())#初始化变量
            for epoch in range(training_epochs):#对每一步训练
                self.progressBar_5.setValue(int(epoch/10))
                batch_size=int(len(data[0])/batch_count)#当前子集大小
                for i in range(batch_count):#对每个子集遍历训练
                    QApplication.processEvents()
                    batch_x,batch_y=self.nextBatch(data[0:2],batch_size,i)#当前子集的训练集特征和标签数据
                    sess.run([train_step],feed_dict={X:batch_x,Y_:batch_y})
            W=[]
            B=[]
            for i in range(len(struction)-1):
                W.append(sess.run(canshuW[i]))
                B.append(sess.run(canshuB[i]))
        self.fbp=FBP(W, B)
        self.progressBar_5.setValue(100)
        y_cal=self.fbp.predict(xte)
        y_cal=np.ravel(y_cal)
        yte=np.ravel(yte)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        fig=MyFigure(chang=246, kuan=254, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        fig.axes.scatter(Y_te,Y_cal,marker='x',color='r')
        fig.axes.plot([Y_te.min(),Y_cal.max(),],[Y_te.min(),Y_cal.max()],color='b')
        self.graphicsView_12.setScene(scene)
        self.modelbuliding=1
        nv=len(self.DataIm_VN)-1
        mi=self.FF[0].a
        ma=self.FF[0].b
        if nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.fbp.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=255, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_13.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=283, kuan=254, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fbp.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fbp.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.fbp.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_13.setScene(scene)
            return
        return
        
    def ModBulding_SA_ME(self):
        try:
            self.rgs 
        except:
            QMessageBox.critical(self,"Critical","No multiple model")
            return
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        ytr_=self.rgs.predict(xtr)
        yte_=self.rgs.predict(xte)
        ytr=np.ravel(ytr)
        yte=np.ravel(yte)
        ytr_=np.ravel(ytr_)
        yte_=np.ravel(yte_)
        Yte=self.FF[-1].f_b(yte)
        Yte_=self.FF[-1].f_b(yte_)
        Ytr=self.FF[-1].f_b(ytr)
        Ytr_=self.FF[-1].f_b(ytr_)
        a=self.RMSE(Yte, Yte_)
        b=MAE(Yte, Yte_)
        c=Mjue(Yte, Yte_)
        d=Mxiang(Yte, Yte_)
        e=R2(Yte, Yte_)
        f=self.R2g(Yte, Yte_)
        g,h=NiXi(Yte, Yte_)
        i=Rou(Yte, Yte_)
        j=Thigma(Yte, Yte_)
        self.rgs_test=[a,b,c,d,e,f,g,h,i,j]
        a=self.RMSE(Ytr, Ytr_)
        b=MAE(Ytr, Ytr_)
        c=Mjue(Ytr, Ytr_)
        d=Mxiang(Ytr, Ytr_)
        e=R2(Ytr, Ytr_)
        f=self.R2g(Ytr, Ytr_)
        g,h=NiXi(Ytr, Ytr_)
        i=Rou(Ytr, Ytr_)
        j=Thigma(Ytr, Ytr_)
        self.rgs_train=[a,b,c,d,e,f,g,h,i,j]
        dlg=QDialog()
        self.PerformanceUi.setupUi(dlg)
        for i in range(len(self.rgs_train)):
            self.PerformanceUi.tableWidget.setItem(i,0,QTableWidgetItem('{:.4f}'.format(self.rgs_train[i])))
        for i in range(len(self.rgs_test)):
            self.PerformanceUi.tableWidget.setItem(i,1,QTableWidgetItem('{:.4f}'.format(self.rgs_test[i])))
        dlg.exec_()
        return
        
    def ModBulding_SA_Start(self):
        rmse=100
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        xtr,xte=xtr.T,xte.T
        nv=len(xtr)#自变量个数
        self.progressBar_6.setValue(10)
        P0=[0.1 for i in range(nv+1)]
        Para=leastsq(self.ErrorL,P0,args=(xtr,ytr)) #把error函数中除了p以外的参数打包到args中
        W=list(Para[0])
        fl=FL(W)
        self.progressBar_6.setValue(20)
        y_cal=fl.predict(xte.T)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        rmsetem=self.RMSE(Y_te, Y_cal)
        if rmsetem<rmse:
            rmse=rmsetem 
            self.rgs=fl
            pmodel='Linear regression'
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        xtr,xte=xtr.T,xte.T
        self.progressBar_6.setValue(30)
        nv=len(xtr)#自变量个数
        lenP=int((nv*(nv-1))/2+nv+1)
        P0=[0.1 for i in range(lenP)]
        Para=leastsq(self.ErrorM,P0,args=(xtr,ytr)) #把error函数中除了p以外的参数打包到args中
        W=list(Para[0])
        fm=FM(W)
        self.progressBar_6.setValue(40)
        y_cal=fm.predict(xte.T)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        rmsetem=self.RMSE(Y_te, Y_cal)
        if rmsetem<rmse:
            rmse=rmsetem 
            self.rgs=fm 
            pmodel='Multiple regression'
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        self.popsize=20
        self.chrnum=2
        self.chrlen=[12,7]
        self.min=[0.01,0.01]
        self.max=[100,10]
        self.gen=100
        self.maxgen=20
        self.px=0.8
        self.pc=0.1
        c,g=self.qgarun2([xtr,ytr],[xte,yte])
        fsvr=svm.SVR(C=c,kernel='rbf',gamma=g,epsilon = 0.04).fit(xtr,ytr)
        self.progressBar_6.setValue(60)
        y_cal=fsvr.predict(xte)
        y_cal=np.ravel(y_cal)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        rmsetem=self.RMSE(Y_te, Y_cal)
        if rmsetem<rmse:
            rmse=rmsetem 
            self.rgs=fsvr 
            pmodel='SVR regression'
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        marks=100
        for i in range(10):
            md=5*i+5
            for j in range(10):
                self.progressBar_6.setValue(int((i*10+j)/5)+60)
                QApplication.processEvents()
                ne=5*j+5
                rgs=RandomForestRegressor(max_leaf_nodes=md,n_estimators=ne).fit(xtr,ytr)
                Ycal=rgs.predict(xte)
                Ycal=np.ravel(Ycal)
                markstem=self.RMSE(Ycal,yte)
                if markstem<marks:
                    marks=markstem 
                    nebest=md 
                    mdbest=ne
        frf=RandomForestRegressor(max_leaf_nodes=mdbest,n_estimators=nebest).fit(xtr,ytr)
        self.progressBar_6.setValue(80)
        y_cal=frf.predict(xte)
        y_cal=np.ravel(y_cal)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        rmsetem=self.RMSE(Y_te, Y_cal)
        if rmsetem<rmse:
            rmse=rmsetem 
#             self.rgs=frf 随机森岭函数不连续，不适合作为最终模型
            pmodel='Random forest regression'
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        os.environ['TF_CPP_MIN_LOG_LEVEL']='2'  
        struction=[]
        hls='10-5'
        HLS=hls.split('-')
        struction.append(len(self.DataIm_VN)-1)
        for i in range(len(HLS)):
            struction.append(int(HLS[i]))
        struction.append(1)
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        ytr=ytr.reshape(len(ytr),1)
        yte=yte.reshape(len(yte),1)
        data=[xtr,ytr,xte,yte]
        batch_count=20#子集数量
        learning_rate=0.02#学习率
        training_epochs=1000#训练步数
        X=tf.placeholder(tf.float32, [None,struction[0]], name="input")#占位符作为输入单元
        Y_=tf.placeholder(tf.float32,[None,struction[-1]])#输出单元
        canshuW=[]
        canshuB=[]
        for i in range(len(struction)-1):
            W=tf.Variable(tf.truncated_normal([struction[i],struction[i+1]],stddev=0.01))#输入层到隐层1的权值矩阵
            B=tf.Variable(tf.zeros([struction[i+1]]))#输入层到隐层1的阈值
            canshuW.append(W)
            canshuB.append(B)
        X1=X 
        for i in range(len(canshuW)-1):
            Y1=tf.nn.sigmoid(tf.matmul(X1,canshuW[i])+canshuB[i])
            X1=Y1
        Y=tf.nn.relu(tf.matmul(X1,canshuW[-1])+canshuB[-1],name='output')#神经网络输出
        E=0.5*(Y-Y_)**2
        E=tf.reduce_mean(E)*100#损失函数
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=0.035)#学习率0.1，精度0.03
        train_step = optimizer.minimize(E)#全局优化
        with tf.Session() as sess:#建立会话
            sess.run(tf.global_variables_initializer())#初始化变量
            for epoch in range(training_epochs):#对每一步训练
                self.progressBar_6.setValue(int(epoch/50)+80)
                batch_size=int(len(data[0])/batch_count)#当前子集大小
                for i in range(batch_count):#对每个子集遍历训练
                    QApplication.processEvents()
                    batch_x,batch_y=self.nextBatch(data[0:2],batch_size,i)#当前子集的训练集特征和标签数据
                    sess.run([train_step],feed_dict={X:batch_x,Y_:batch_y})
            W=[]
            B=[]
            for i in range(len(struction)-1):
                W.append(sess.run(canshuW[i]))
                B.append(sess.run(canshuB[i]))
        fbp=FBP(W, B)
        self.progressBar_6.setValue(100)
        y_cal=fbp.predict(xte)
        y_cal=np.ravel(y_cal)
        yte=np.ravel(yte)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        rmsetem=self.RMSE(Y_te, Y_cal)
        if rmsetem<rmse:
            rmse=rmsetem 
            self.rgs=fbp 
            pmodel='BPNN regression'
        xtr=self.x_tr.copy()
        ytr=self.y_tr.copy()
        xte=self.x_te.copy()
        yte=self.y_te.copy()
        y_cal=self.rgs.predict(xte)
        y_cal=np.ravel(y_cal)
        yte=np.ravel(yte)
        Y_te=self.FF[-1].f_b(yte)
        Y_cal=self.FF[-1].f_b(y_cal)
        self.modelbuliding=1
        fig=MyFigure(chang=266, kuan=261, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        fig.axes.scatter(Y_te,Y_cal,marker='x',color='r')
        fig.axes.plot([Y_te.min(),Y_cal.max(),],[Y_te.min(),Y_cal.max()],color='b')
        self.graphicsView_14.setScene(scene)
        self.lineEdit_15.setText(pmodel)
        mi=self.FF[0].a
        ma=self.FF[0].b
        if nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.rgs.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=255, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_15.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=263, kuan=261, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_15.setScene(scene)
        return
        
    def NewPro_Con_OK(self):
        a=self.dateTimeEdit_5.text().split(' ')
        if self.lineEdit_18.text():
            s=self.lineEdit_18.text()+'_'+a[0]+'_'+a[1]
            self.lineEdit_16.setText(s)
            self.lineEdit_19.setText(s)
            self.lineEdit_28.setText(s)
        else:
            QMessageBox.critical(self,"Critical","No name")
            return
        self.NewPro_Con_OKmark=1
        return
        
    def NewPro_Con_Delete(self):
        s=self.listWidget_6.selectedItems()[0].text()
        lujing=r'.\Users\PredictiveProj'+'\\'+s
        a = list(os.walk(lujing))[0][2]
        for s in a:
            r=lujing+'\\'+s 
            os.remove(r)
        os.rmdir(lujing)
        a = list(os.walk(r'.\Users\PredictiveProj'))[0]
        self.listWidget_6.clear()
        self.listWidget_6.addItems(a[1])
        self.listWidget_6.setCurrentRow(0)
        return
        
    def NewPro_Con_Load(self):
        if self.NewPro_Con_OKmark == 0:
            QMessageBox.critical(self,"Critical","No name")
            return
        s=self.listWidget_6.selectedItems()[0].text()
        self.lujing=r'.\Users\PredictiveProj'+'\\'+s
        a = list(os.walk(self.lujing))[0][2]
        b=[]
        for i in a:
            if i[0:5]=='Model':
                b.append(i)
        dlg=QDialog()
        self.ModelselectUi.setupUi(dlg,self.lujing)
        self.ModelselectUi.listWidget.addItems(b)
        dlg.exec_()
        try:
            self.rgs=readpkl(self.ModelselectUi.model)
            self.indata=readpkl(self.lujing+'\\'+'indata.pkl')
            self.outdata=readpkl(self.lujing+'\\'+'outdata.pkl')
            self.name=readpkl(self.lujing+'\\'+'name.pkl')
            self.FF=readpkl(self.lujing+'\\'+'norm.pkl')
            self.textEdit.setText(self.ModelselectUi.sd)
        except:
            QMessageBox.critical(self,"Critical","No Model")
            return
        return
        
    def ConTargets_OK(self):
        if self.radioButton_3.isChecked():
            self.ConTargets_model=0
        elif self.radioButton_5.isChecked():
            self.ConTargets_model=1
        elif self.radioButton_4.isChecked():
            self.ConTargets_model=2
        elif self.radioButton_6.isChecked():
            self.ConTargets_model=3
        if self.ConTargets_model>len(self.name)-2:
            QMessageBox.critical(self,"Critical","No this factor")
            self.ConTargets_model=-1
            return
        else:
            QMessageBox.information(self,"Information","The factor being controlled \
is "+self.name[self.ConTargets_model])
        self.radioButton_3.setEnabled(False)
        self.radioButton_4.setEnabled(False)
        self.radioButton_5.setEnabled(False)
        self.radioButton_6.setEnabled(False)
        self.lineEdit_20.setEnabled(False)
        self.lineEdit_21.setEnabled(False)
        self.lineEdit_22.setEnabled(False)
        self.lineEdit_23.setEnabled(False)
        self.lineEdit_20.setText('Unavailable')
        self.lineEdit_21.setText('Unavailable')
        self.lineEdit_22.setText('Unavailable')
        self.lineEdit_23.setText('Unavailable')
        self.lineEdit_24.setEnabled(False)
        self.lineEdit_25.setEnabled(False)
        self.lineEdit_26.setEnabled(False)
        self.lineEdit_27.setEnabled(False)
        self.lineEdit_24.setText('Unavailable')
        self.lineEdit_25.setText('Unavailable')
        self.lineEdit_26.setText('Unavailable')
        self.lineEdit_27.setText('Unavailable')
        for i in range(len(self.name)-1):
            if i == 0:
                if i == self.ConTargets_model:
                    self.label_60.setText(self.name[i])
                    self.label_65.setText(self.name[i])
                    self.lineEdit_20.setText('')
                    self.lineEdit_24.setText('')
                    continue
                self.lineEdit_20.setEnabled(True)
                self.label_60.setText(self.name[i])
                self.lineEdit_24.setEnabled(True)
                self.label_65.setText(self.name[i])
                self.lineEdit_20.setText('')
                self.lineEdit_24.setText('')
            elif i == 1:
                if i == self.ConTargets_model:
                    self.label_61.setText(self.name[i])
                    self.label_66.setText(self.name[i])
                    self.lineEdit_21.setText('')
                    self.lineEdit_25.setText('')
                    continue
                self.lineEdit_21.setEnabled(True)
                self.label_61.setText(self.name[i])
                self.lineEdit_25.setEnabled(True)
                self.label_66.setText(self.name[i])
                self.lineEdit_21.setText('')
                self.lineEdit_25.setText('')
            elif i==2:
                if i == self.ConTargets_model:
                    self.label_62.setText(self.name[i])
                    self.label_67.setText(self.name[i])
                    self.lineEdit_22.setText('')
                    self.lineEdit_26.setText('')
                    continue
                self.lineEdit_22.setEnabled(True)
                self.label_62.setText(self.name[i])
                self.lineEdit_26.setEnabled(True)
                self.label_67.setText(self.name[i])
                self.lineEdit_22.setText('')
                self.lineEdit_26.setText('')
            elif i==3:
                if i == self.ConTargets_model:
                    self.label_63.setText(self.name[i])
                    self.label_68.setText(self.name[i])
                    self.lineEdit_23.setText('')
                    self.lineEdit_27.setText('')
                    continue
                self.lineEdit_23.setEnabled(True)
                self.label_63.setText(self.name[i])
                self.lineEdit_27.setEnabled(True)
                self.label_68.setText(self.name[i])
                self.lineEdit_23.setText('')
                self.lineEdit_27.setText('')
        return
        
    def ConTargets_Start(self):
        if self.ConTargets_model == -1:
            QMessageBox.information(self,"Information","No Factor has been selected")
            return
        EnvData=[]#所设置的梯度信
        EnvNor=[]
        a=self.lineEdit_20.text()
        b=self.lineEdit_21.text()
        c=self.lineEdit_22.text()
        d=self.lineEdit_23.text()
        EnvGtem=[a,b,c,d]
        xin=self.indata.T
        for i in range(len(EnvGtem)):
            QApplication.processEvents()
            if EnvGtem[i]=='':
                if i==self.ConTargets_model:
                    ObjData=np.array([xin[i].min()+j*(xin[i].max()-xin[i].min())/1000 for j in range(1001)])
                    #调控目标的密集数据FF
                    OBJData=self.FF[i].f_b(ObjData)
                    continue
                else:
                    QMessageBox.warning(self,'warning','Not set',QMessageBox.Ok)
                    return
            elif EnvGtem[i]=='Unavailable':
                continue
            else:
                try:
                    EnvData.append(np.array(list(map(float,EnvGtem[i].split(',')))))
                    EnvNor.append(self.FF[i])
                except:
                    QMessageBox.warning(self,'warning','Wrong Form',QMessageBox.Ok)
                    return 
        EnvNorData=[]#归一化后的梯度数据
        TableData=[]#表格中的显示数据
        for i in range(len(EnvData)):
            EnvNorData.append(EnvNor[i].f(EnvData[i]))
        if len(EnvNorData)==1:
            ResponseDataofEnv=[]#未选中为目标的输入数据
            ResponseNorDataofEnv=[]#未选中为目标的归一化数据
            for i in range(len(EnvNorData[0])):
                ResponseNorDataofEnv.append([EnvNorData[0][i]])
                ResponseDataofEnv.append([EnvData[0][i]])
        elif len(EnvNorData)==2:
            ResponseDataofEnv=[]#未选中为目标的数据
            ResponseNorDataofEnv=[]#未选中为目标的归一化数据
            for i in range(len(EnvNorData[0])):
                for j in range(len(EnvNorData[1])):
                    ResponseNorDataofEnv.append([EnvNorData[0][i], EnvNorData[1][j]])
                    ResponseDataofEnv.append([EnvData[0][i], EnvData[1][j]])
        elif len(EnvNorData)==3:
            ResponseDataofEnv=[]#未选中为目标的数据
            ResponseNorDataofEnv=[]#未选中为目标的归一化数据
            for i in range(len(EnvNorData[0])):
                for j in range(len(EnvNorData[1])):
                    for k in range(len(EnvNorData[2])):
                        ResponseNorDataofEnv.append([EnvNorData[0][i], EnvNorData[1][j], EnvNorData[2][k]])
                        ResponseDataofEnv.append([EnvData[0][i], EnvData[1][j], EnvData[2][k]])
        ProgressPave=1
        for i in range(len(EnvData)):
            ProgressPave=ProgressPave*len(EnvData[i])  
        self.tableWidget_3.setRowCount(len(ResponseNorDataofEnv)) 
        self.tableWidget_4.setRowCount(len(ResponseNorDataofEnv)) 
        for i in range(len(ResponseNorDataofEnv)):
            QApplication.processEvents()
            ResponseData=[]#每一条响应曲线的输入点集
            for j in range(len(ObjData)):
                listtem=ResponseNorDataofEnv[i].copy()
                listtem.insert(self.ConTargets_model,ObjData[j])
                ResponseData.append(listtem)
            ResponseData=np.array(ResponseData)
            ResponseNorPn=self.rgs.predict(ResponseData)#每一条响应曲线的归一化输出
            ResponsePn=self.FF[-1].f_b(ResponseNorPn)#每一条响应曲线的原始输出
            MaxPni=ResponsePn.argmax()
            MaxPn=ResponsePn[MaxPni]#饱和输出值
            MaxObj=OBJData[MaxPni]#饱和输入值
            Cmax_i=self.runuchord(OBJData,ResponsePn)[2]
            MaxUObj=OBJData[Cmax_i]#曲率输入值
            MaxUPn=ResponsePn[Cmax_i]#曲率输出值
#             plt.plot(OBJData,ResponsePn)
#             plt.scatter(MaxObj,MaxPn)
#             plt.scatter(MaxUObj,MaxUPn)
#             plt.show()
            SC_inp=[]#饱和控制的输入数据和调控目标
            SC_oup=[]#饱和控制的输出数据
            EC_inp=[]#效益控制的输入数据和调控目标
            EC_oup=[]#效益控制的输出数据
            SC_inp=ResponseDataofEnv[i].copy()
            SC_inp.insert(self.ConTargets_model,MaxObj)
            SC_oup=str(MaxPn)
            EC_inp=ResponseDataofEnv[i].copy()
            EC_inp.insert(self.ConTargets_model,MaxUObj)
            EC_oup=str(MaxUPn)
            SC_inp=list(map(str,SC_inp))
            EC_inp=list(map(str,EC_inp))
            ProgressPaveNow=(i+1)*100/ProgressPave
            self.progressBar_7.setValue(ProgressPaveNow)
            for j in range(len(self.name)-1):
                newItem1=QTableWidgetItem(SC_inp[j])
                newItem2=QTableWidgetItem(EC_inp[j])
                if j == self.ConTargets_model:
                    newItem1.setForeground(QtGui.QBrush(QtGui.QColor(255,0,0)))
                    newItem2.setForeground(QtGui.QBrush(QtGui.QColor(255,0,0)))
                self.tableWidget_3.setItem(i,j,newItem1)
                self.tableWidget_4.setItem(i,j,newItem2)
            self.tableWidget_3.setItem(i,4,QTableWidgetItem(SC_oup))
            self.tableWidget_4.setItem(i,4,QTableWidgetItem(EC_oup))
        self.progressBar_7.setValue(100)
        return

    def ConTargets_Reset(self):
        self.radioButton_3.setEnabled(True)
        self.radioButton_4.setEnabled(True)
        self.radioButton_5.setEnabled(True)
        self.radioButton_6.setEnabled(True)
        self.lineEdit_20.setEnabled(True)
        self.lineEdit_21.setEnabled(True)
        self.lineEdit_22.setEnabled(True)
        self.lineEdit_23.setEnabled(True)
        self.lineEdit_20.setText('')
        self.lineEdit_21.setText('')
        self.lineEdit_22.setText('')
        self.lineEdit_23.setText('')
        self.lineEdit_24.setEnabled(True)
        self.lineEdit_25.setEnabled(True)
        self.lineEdit_26.setEnabled(True)
        self.lineEdit_27.setEnabled(True)
        self.lineEdit_24.setText('')
        self.lineEdit_25.setText('')
        self.lineEdit_26.setText('')
        self.lineEdit_27.setText('')
        self.ConTargets_model = -1
        return
        
    def ConTargets_Gd_OK(self):
        if self.ConTargets_model == -1:
            QMessageBox.information(self,"Information","No Factor has been selected")
            return
        EnvNow=[]
        a=self.lineEdit_24.text()
        b=self.lineEdit_25.text()
        c=self.lineEdit_26.text()
        d=self.lineEdit_27.text()
        EnvGtem=[a,b,c,d]
        xin=self.indata.T
        for i in range(len(EnvGtem)):
            if EnvGtem[i]=='':
                if i==self.ConTargets_model:
                    ObjData=np.array([xin[i].min()+j*(xin[i].max()-xin[i].min())/1000 for j in range(1001)])
                    OBJData=self.FF[i].f_b(ObjData)
                else:
                    QMessageBox.warning(self,'warning','Not set',QMessageBox.Ok)
                    return
            elif EnvGtem[i]=='Unavailable':
                continue
            else:
                try:
                    EnvNow.append(self.FF[i].f(eval(EnvGtem[i])))
                except:
                    QMessageBox.warning(self,'warning','Wrong Form',QMessageBox.Ok)
                    return
        ResponseInput=[]
        for i in range(len(ObjData)):
            a=EnvNow.copy()
            a.insert(self.ConTargets_model,ObjData[i])
            ResponseInput.append(a)
        ResponseInput=np.array(ResponseInput)
        ResponseOutput=self.rgs.predict(ResponseInput)
        ResponseOutput=self.FF[-1].f_b(ResponseOutput)
        MaxPni=ResponseOutput.argmax()
        MaxPn=ResponseOutput[MaxPni]#饱和输出值
        MaxObj=OBJData[MaxPni]#饱和输入值
        Cmax_i=self.runuchord(OBJData,ResponseOutput)[2]
        MaxUObj=OBJData[Cmax_i]#曲率输入值
        MaxUPn=ResponseOutput[Cmax_i]#曲率输出值
        myfig=MyFigure(chang=264, kuan=156, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(myfig)
        myfig.axes.plot(OBJData,ResponseOutput,color='g')
        myfig.axes.scatter(MaxObj,MaxPn,color='b',marker='o')
        myfig.axes.scatter(MaxUObj,MaxUPn,color='r',marker='*')
        self.graphicsView_16.setScene(scene)
        return
        
    def ConModels_OK(self):
        if self.radioButton_7.isChecked():
            self.ConModels_model=0
        elif self.radioButton_8.isChecked():
            self.ConModels_model=1
        elif self.radioButton_9.isChecked():
            self.ConModels_model=2
        elif self.radioButton_10.isChecked():
            self.ConModels_model=3
        if self.ConModels_model>len(self.name)-2:
            QMessageBox.critical(self,"Critical","No this factor")
            self.ConModels_model=-1
            return
        else:
            QMessageBox.information(self,"Information","The factor being controlled is "+self.name[self.ConModels_model])
        self.radioButton_7.setEnabled(False)
        self.radioButton_8.setEnabled(False)
        self.radioButton_9.setEnabled(False)
        self.radioButton_10.setEnabled(False)
        return 
        
    def ConModels_Back(self):
        self.listWidget_2.setCurrentRow(0)
        return
    
    def ConModels_Reset(self):
        self.radioButton_7.setEnabled(True)
        self.radioButton_8.setEnabled(True)
        self.radioButton_9.setEnabled(True)
        self.radioButton_10.setEnabled(True)
        self.ConModels_model=-1
        self.progressBar_8.setValue(0)
        self.progressBar_9.setValue(0)
        self.textEdit_sm.setText('')
        self.textEdit_em.setText('')
        scene= QGraphicsScene()
        self.graphicsView_17.setScene(scene)
        self.graphicsView_18.setScene(scene)
        return
        
    def ConModels_Down(self):
        try:
            self.rgs_em 
        except:
            QMessageBox.information(self,"Information","No Model has been established")
            return
        filename=self.lineEdit_28.text()
        filename=filename.split('/')
        filename='_'.join(filename)
        filename=filename.split(':')
        filename='_'.join(filename)
        road=r'.\Users\RegulationProj\\'+filename
        if os.path.exists(road):
            reply = QMessageBox.information(self, "warning", "There is already a project with the same name here"+"\n"+
                                            "do you want to overwrite it?",
                                             QMessageBox.Yes | QMessageBox.No ,  QMessageBox.Yes )
            if reply==QMessageBox.Yes:
                a = list(os.walk(road))[0]
                for s in a[2]:
                    r=a[0]+'\\'+s 
                    os.remove(r)
                os.rmdir(road)
            elif reply==QMessageBox.No:
                return
        os.makedirs(road)
        savepkl(road+'\\indata.pkl', self.indata)
        savepkl(road+'\\outdata.pkl', self.outdata)
        savepkl(road+'\\norm.pkl', self.FF)
        savepkl(road+'\\name.pkl', self.name)
        savepkl(road+'\\X_env.pkl', self.X_env)
        savepkl(road+'\\Y_sm.pkl', self.Y_sm)
        savepkl(road+'\\Y_em.pkl', self.Y_em)
        savepkl(road+'\\ConNorm.pkl', self.FFF)
        savepkl(road+'\\Conname.pkl', self.con_name)
        savepkl(road+'\\rgs.pkl', self.rgs)
        savepkl(road+'\\rgs_em.pkl', self.rgs_em)
        savepkl(road+'\\rgs_sm.pkl', self.rgs_sm)
        Paral_em="The effective control model parameters are:\nroot mean square error of {:.4f}, \
mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.em_train[0],self.em_train[1],self.em_train[2],self.em_train[3],self.em_train[4],
                       self.em_train[5],self.em_train[6],self.em_train[7],self.em_train[8],self.em_train[9],
                       self.em_test[0],self.em_test[1],self.em_test[2],self.em_test[3],self.em_test[4],
                       self.em_test[5],self.em_test[6],self.em_test[7],self.em_test[8],self.em_test[9])
        savepkl(road+'\\ParaEffectiveModel.pkl', Paral_em)
        Paral_sm="The saturation control model parameters are:\nroot mean square error of {:.4f}, \
mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.sm_train[0],self.sm_train[1],self.sm_train[2],self.sm_train[3],self.sm_train[4],
                       self.sm_train[5],self.sm_train[6],self.sm_train[7],self.sm_train[8],self.sm_train[9],
                       self.sm_test[0],self.sm_test[1],self.sm_test[2],self.sm_test[3],self.sm_test[4],
                       self.sm_test[5],self.sm_test[6],self.sm_test[7],self.sm_test[8],self.sm_test[9])
        savepkl(road+'\\ParaSaturationModel.pkl', Paral_sm)
        QMessageBox.information(self,"Information","The control model project\nhas been established")
        return
        
    def ConModels_SM_3D(self):
        nv=len(self.con_name)-1
        mi=self.FFF[0].a
        ma=self.FFF[0].b
        if nv == 1:
            inp=np.array([[mi+(ma-mi)*i/100 for i in range (101)]])
            inpdata=inp.T 
            oupdata=self.rgs_sm.predict(inpdata)
            oupdata=np.ravel(oupdata)
            inpdata=np.ravel(inpdata)
            fig=MyFigure(chang=283, kuan=239, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(fig)
            fig.axes.plot(inpdata,oupdata,color='b')
            self.graphicsView_17.setScene(scene)
            return
        elif nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.rgs_sm.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=283, kuan=239, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_17.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=283, kuan=239, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs_sm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs_sm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs_sm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_17.setScene(scene)
            return
        return
        
    def ConModels_SM_Prediction(self):
        X=self.X_env.copy()
        Y=self.Y_sm.copy()
        Y_=self.rgs_sm.predict(X)
        Y=np.ravel(Y)
        Y_=np.ravel(Y_)
        Y=self.FFF[-2].f_b(Y)
        Y_=self.FFF[-2].f_b(Y_)
        fig=MyFigure(chang=283, kuan=239, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        fig.axes.scatter(Y,Y_,marker='x',color='r')
        fig.axes.plot([Y.min(),Y.max(),],[Y.min(),Y.max()],color='b')
        self.graphicsView_17.setScene(scene)
        return
        
    def ConModels_EM_3D(self):
        nv=len(self.con_name)-1
        mi=self.FFF[0].a
        ma=self.FFF[0].b
        if nv == 1:
            inp=np.array([[mi+(ma-mi)*i/100 for i in range (101)]])
            inpdata=inp.T 
            oupdata=self.rgs_em.predict(inpdata)
            oupdata=np.ravel(oupdata)
            inpdata=np.ravel(inpdata)
            fig=MyFigure(chang=283, kuan=239, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(fig)
            fig.axes.plot(inpdata,oupdata,color='b')
            self.graphicsView_18.setScene(scene)
            return
        elif nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.rgs_em.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=283, kuan=239, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_18.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=283, kuan=239, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs_em.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs_em.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.rgs_em.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_18.setScene(scene)
            return
        return
        
    def ConModels_EM_Prediction(self):
        X=self.X_env.copy()
        Y=self.Y_em.copy()
        Y_=self.rgs_em.predict(X)
        Y=np.ravel(Y)
        Y_=np.ravel(Y_)
        Y=self.FFF[-1].f_b(Y)
        Y_=self.FFF[-1].f_b(Y_)
        fig=MyFigure(chang=283, kuan=239, dpi=100)
        scene= QGraphicsScene()
        scene.addWidget(fig)
        fig.axes.scatter(Y,Y_,marker='x',color='r')
        fig.axes.plot([Y.min(),Y.max(),],[Y.min(),Y.max()],color='b')
        self.graphicsView_18.setScene(scene)
        return
        
    def ConModels_DA_Start(self):
        if self.ConModels_model == -1:
            QMessageBox.information(self,"Information","No Factor has been selected")
            return
        xin=self.indata.T
        Xtrain=[]
        self.FFF=[]#装入调控模型的归一化实例类,最后一个是效率调控目标项的目标值归一化类，倒数第二个是饱和调控目标项的目标值归一化类，其他的都与训练模型类似
        self.con_name=[]
        for i in range(len(xin)):
            if i==self.ConModels_model:
                ObjData=np.array([xin[i].min()+j*(xin[i].max()-xin[i].min())/1000 for j in range(1001)])
                OBJData=self.FF[i].f_b(ObjData)
            else:
                temData=np.array([xin[i].min()+j*(xin[i].max()-xin[i].min())/15 for j in range(1,16)])#避免0值
                self.con_name.append(self.name[i])
                Xtrain.append(temData)
                self.FFF.append(self.FF[i])
        self.con_name.append(self.name[self.ConModels_model])
        if len(Xtrain)==1:
            self.X_env=np.array(Xtrain)
            self.X_env=self.X_env.reshape((len(Xtrain[0]),1))
        elif len(Xtrain)==2:
            self.X_env=np.array(np.meshgrid(Xtrain[0],Xtrain[1])).T
            self.X_env=self.X_env.reshape((len(Xtrain[0])*len(Xtrain[1]),2))
        elif len(Xtrain)==3:
            self.X_env=np.array(np.meshgrid(Xtrain[0],Xtrain[1],Xtrain[2])).T
            self.X_env=self.X_env.reshape((len(Xtrain[0])*len(Xtrain[1])*len(Xtrain[2]),3))
        y_sm=[]
        y_em=[]
        for i in range(len(self.X_env)):
            self.progressBar_8.setValue(100*i/len(self.X_env))
            QApplication.processEvents()
            Xtrain=[]
            for j in range(len(ObjData)):
                Xtem=list(self.X_env[i])
                Xtem.insert(self.ConModels_model,ObjData[j])
                Xtrain.append(Xtem)
            indata=np.array(Xtrain)
            oudata=self.rgs.predict(indata)
            OutData=self.FF[-1].f_b(oudata)
            MaxPni=OutData.argmax()
            y_sm.append(OBJData[MaxPni])
            Cmax_i=self.runuchord(OBJData,OutData)[2]
            y_em.append(OBJData[Cmax_i])
        self.progressBar_8.setValue(100)
        y_sm=np.array(y_sm)
        y_em=np.array(y_em)
        self.FFF.append(F(y_sm))
        self.FFF.append(F(y_em))
        self.Y_sm=self.FFF[-2].f(y_sm)
        self.Y_em=self.FFF[-1].f(y_em)
        return
        
    def ConModels_MB_Start(self):
        try:
            self.Y_em
        except:
            QMessageBox.information(self,"Information","No Training data")
            return
        Xin=self.X_env.copy()
        Y1=self.Y_sm.copy()
        Y2=self.Y_em.copy()
        xtrsm,xtesm,ytrsm,ytesm=model_selection.train_test_split(Xin,Y1,train_size=0.8)
        xtrem,xteem,ytrem,yteem=model_selection.train_test_split(Xin,Y2,train_size=0.8)
        Esm=100
        Eem=100
        for c in [0.1+i*5 for i in range(11)]:
            for g in [0.1+i*0.5 for i in range(11)]:
                self.progressBar_9.setValue(int((c*11+g)*100/121))
                clfsm=svm.SVR(C=c,kernel='rbf',gamma=g,epsilon = 0.04).fit(xtrsm,ytrsm)
                clfem=svm.SVR(C=c,kernel='rbf',gamma=g,epsilon = 0.04).fit(xtrem,ytrem)
                Ysm=clfsm.predict(xtesm)
                esm=self.RMSE(ytesm, Ysm)
                Yem=clfem.predict(xteem)
                eem=self.RMSE(yteem, Yem)
                if eem<Eem:
                    Eem=eem 
                    self.rgs_em=clfem 
                if esm<Esm:
                    Esm=esm 
                    self.rgs_sm=clfsm 
        xtr=xtrsm.copy()
        ytr=ytrsm.copy()
        xte=xtesm.copy()
        yte=ytesm.copy()
        ytr_=self.rgs_sm.predict(xtr)
        yte_=self.rgs_sm.predict(xte)
        ytr=np.ravel(ytr)
        yte=np.ravel(yte)
        ytr_=np.ravel(ytr_)
        yte_=np.ravel(yte_)
        Yte=self.FFF[-2].f_b(yte)
        Yte_=self.FFF[-2].f_b(yte_)
        Ytr=self.FFF[-2].f_b(ytr)
        Ytr_=self.FFF[-2].f_b(ytr_)
        a=self.RMSE(Yte, Yte_)
        b=MAE(Yte, Yte_)
        c=Mjue(Yte, Yte_)
        d=Mxiang(Yte, Yte_)
        e=R2(Yte, Yte_)
        f=self.R2g2(Yte, Yte_)
        g,h=NiXi(Yte, Yte_)
        i=Rou(Yte, Yte_)
        j=Thigma(Yte, Yte_)
        self.sm_test=[a,b,c,d,e,f,g,h,i,j]
        a=self.RMSE(Ytr, Ytr_)
        b=MAE(Ytr, Ytr_)
        c=Mjue(Ytr, Ytr_)
        d=Mxiang(Ytr, Ytr_)
        e=R2(Ytr, Ytr_)
        f=self.R2g2(Ytr, Ytr_)
        g,h=NiXi(Ytr, Ytr_)
        i=Rou(Ytr, Ytr_)
        j=Thigma(Ytr, Ytr_)
        self.sm_train=[a,b,c,d,e,f,g,h,i,j]
        xtr=xtrem.copy()
        ytr=ytrem.copy()
        xte=xteem.copy()
        yte=yteem.copy()
        ytr_=self.rgs_em.predict(xtr)
        yte_=self.rgs_em.predict(xte)
        ytr=np.ravel(ytr)
        yte=np.ravel(yte)
        ytr_=np.ravel(ytr_)
        yte_=np.ravel(yte_)
        Yte=self.FFF[-1].f_b(yte)
        Yte_=self.FFF[-1].f_b(yte_)
        Ytr=self.FFF[-1].f_b(ytr)
        Ytr_=self.FFF[-1].f_b(ytr_)
        a=self.RMSE(Yte, Yte_)
        b=MAE(Yte, Yte_)
        c=Mjue(Yte, Yte_)
        d=Mxiang(Yte, Yte_)
        e=R2(Yte, Yte_)
        f=self.R2g2(Yte, Yte_)
        g,h=NiXi(Yte, Yte_)
        i=Rou(Yte, Yte_)
        j=Thigma(Yte, Yte_)
        self.em_test=[a,b,c,d,e,f,g,h,i,j]
        a=self.RMSE(Ytr, Ytr_)
        b=MAE(Ytr, Ytr_)
        c=Mjue(Ytr, Ytr_)
        d=Mxiang(Ytr, Ytr_)
        e=R2(Ytr, Ytr_)
        f=self.R2g2(Ytr, Ytr_)
        g,h=NiXi(Ytr, Ytr_)
        i=Rou(Ytr, Ytr_)
        j=Thigma(Ytr, Ytr_)
        self.em_train=[a,b,c,d,e,f,g,h,i,j]
        self.progressBar_9.setValue(100)
        Paral_em="The effective control model parameters are:\nroot mean square error of {:.4f}, \
mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.em_train[0],self.em_train[1],self.em_train[2],self.em_train[3],self.em_train[4],
                       self.em_train[5],self.em_train[6],self.em_train[7],self.em_train[8],self.em_train[9],
                       self.em_test[0],self.em_test[1],self.em_test[2],self.em_test[3],self.em_test[4],
                       self.em_test[5],self.em_test[6],self.em_test[7],self.em_test[8],self.em_test[9])
        self.textEdit_em.setText(Paral_em)
        Paral_sm="The saturation control model parameters are:\nroot mean square error of {:.4f}, \
mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} for train test.\n\
root mean square error of {:.4f}, mean absolute error of {:.4f}, \
maximum absolute error of {:.4f}, average relative error of {:.4f}, coefficient of determination of {:.4f}, \
revised coefficient of determination of {:.4f}, fitting coefficient of {:.4f}, fitting interceptof {:.4f}, \
correlation coefficient of {:.4f}, Error standard deviation of {:.4f} \
for train test.".format(self.sm_train[0],self.sm_train[1],self.sm_train[2],self.sm_train[3],self.sm_train[4],
                       self.sm_train[5],self.sm_train[6],self.sm_train[7],self.sm_train[8],self.sm_train[9],
                       self.sm_test[0],self.sm_test[1],self.sm_test[2],self.sm_test[3],self.sm_test[4],
                       self.sm_test[5],self.sm_test[6],self.sm_test[7],self.sm_test[8],self.sm_test[9])
        self.textEdit_sm.setText(Paral_sm)
        return
        
    def ConModLoading_Delete(self):
        s=self.listWidget_7.selectedItems()[0].text()
        lujing=r'.\Users\RegulationProj'+'\\'+s
        a = list(os.walk(lujing))[0][2]
        for s in a:
            r=lujing+'\\'+s 
            os.remove(r)
        os.rmdir(lujing)
        a = list(os.walk(r'.\Users\RegulationProj'))[0]
        self.listWidget_7.clear()
        self.listWidget_7.addItems(a[1])
        self.listWidget_7.setCurrentRow(0)
        return
        
    def ConModLoading_Load(self):
        s=self.listWidget_7.selectedItems()[0].text()
        self.lujing=r'.\Users\RegulationProj'+'\\'+s
        self.lineEdit_29.setText(s)
        self.lineEdit_30.setText(s)
        self.lineEdit_39.setText(s)
        self.indata=readpkl(self.lujing+'\\indata.pkl')
        self.outdata=readpkl(self.lujing+'\\outdata.pkl')
        self.FF=readpkl(self.lujing+'\\norm.pkl')
        self.name=readpkl(self.lujing+'\\name.pkl')
        self.X_env=readpkl(self.lujing+'\\X_env.pkl')
        self.Y_sm =readpkl(self.lujing+'\\Y_sm.pkl')
        self.Y_em=readpkl(self.lujing+'\\Y_em.pkl')
        self.FFF=readpkl(self.lujing+'\\ConNorm.pkl')
        self.con_name=readpkl(self.lujing+'\\Conname.pkl')
        self.rgs=readpkl(self.lujing+'\\rgs.pkl')
        self.rgs_em=readpkl(self.lujing+'\\rgs_em.pkl')
        self.rgs_sm=readpkl(self.lujing+'\\rgs_sm.pkl')
        Paral_em=readpkl(self.lujing+'\\ParaEffectiveModel.pkl')
        Paral_sm=readpkl(self.lujing+'\\ParaSaturationModel.pkl')
        Paral=Paral_em+'\n'+'\n'+Paral_sm
        self.textEdit_2.setText(Paral)
        for i in range(len(self.name)-1):
            if self.name[i]==self.con_name[-1]:
                self.index=i
        a=[self.label_84,self.label_85,self.label_86,self.label_91]
        c1=[self.label_112,self.label_113,self.label_114,self.label_111]
        c2=[self.label_104,self.label_103,self.label_101,self.label_105]
        b=[self.lineEdit_31,self.lineEdit_32,self.lineEdit_33,self.lineEdit_36,self.lineEdit_37,self.lineEdit_38]
        d1=[self.lineEdit_52,self.lineEdit_53,self.lineEdit_54,self.lineEdit_50,self.lineEdit_51]
        d2=[self.lineEdit_48,self.lineEdit_47,self.lineEdit_46,self.lineEdit_49,self.lineEdit_45]
        for i in b:
            i.setText('')
            i.setEnabled(False)
        for i in d1:
            i.setText('')
            i.setEnabled(False)
        for i in d2:
            i.setText('')
            i.setEnabled(False)
        for i in range(len(self.con_name)-1):
            a[i].setText(self.con_name[i])
            b[i].setEnabled(True)
            c1[i].setText(self.con_name[i])
            c2[i].setText(self.con_name[i])
        self.label_92.setText(self.con_name[-1])
        self.label_93.setText(self.name[-1])
        self.label_109.setText(self.con_name[-1])
        self.label_108.setText(self.name[-1])
        self.label_106.setText(self.con_name[-1])
        self.label_107.setText(self.name[-1])
        return
        
    def Display_Up_OK(self):
        if self.radioButton_2.isChecked():
            self.Display_cm=self.rgs_sm
        elif self.radioButton.isChecked():
            self.Display_cm=self.rgs_em 
        else:
            QMessageBox.information(self,"Information","No Model")
            return
        nv=len(self.con_name)-1
        mi=self.FFF[0].a
        ma=self.FFF[0].b
        if nv == 1:
            inp=np.array([[mi+(ma-mi)*i/100 for i in range (101)]])
            inpdata=inp.T 
            oupdata=self.Display_cm.predict(inpdata)
            oupdata=np.ravel(oupdata)
            inpdata=np.ravel(inpdata)
            fig=MyFigure(chang=394, kuan=352, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(fig)
            fig.axes.plot(inpdata,oupdata,color='b')
            self.graphicsView_19.setScene(scene)
            return
        elif nv==2:
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            X=inpdata1.copy()
            Y=inpdata2.copy()
            X=np.ravel(X)
            Y=np.ravel(Y)
            inpdata=np.array([X,Y])
            inpdata=inpdata.T 
            oupdata=self.Display_cm.predict(inpdata)
            oupdata=np.ravel(oupdata)
            oupdata=oupdata.reshape(101,101)
            myfig=MyFigure3D(chang=394, kuan=352, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            myfig.axes.plot_surface(inpdata1,inpdata2,oupdata,cmap='rainbow')
            self.graphicsView_19.setScene(scene)
            return
        elif nv==3:
            jiange=(ma-mi)/5
            jiangelist=[mi+jiange/2+jiange*j for j in range(4)]
            x1=np.array(jiangelist)
            x2=np.array(jiangelist)
            x3=np.array(jiangelist)
            inp1=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inp2=np.array([mi+(ma-mi)*i/100 for i in range (101)])
            inpdata1=np.array([inp1 for i in range(101)])
            inpdata2=np.array([inp2 for i in range(101)])
            inpdata1=inpdata1.T 
            myfig=MyFigure3D(chang=394, kuan=352, dpi=100)
            scene= QGraphicsScene()
            scene.addWidget(myfig)
            for i in x1:
                QApplication.processEvents()
                inpdataX=np.array([i for j in range(101)])
                inpdataX=np.array([inpdataX for j in range(101)])
                X=inpdataX.copy()
                Y=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.Display_cm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdataX,inpdata1,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x2:
                QApplication.processEvents()
                inpdataY=np.array([i for j in range(101)])
                inpdataY=np.array([inpdataY for j in range(101)])
                Y=inpdataY.copy()
                X=inpdata1.copy()
                Z=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.Display_cm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdataY,inpdata2,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            for i in x3:
                QApplication.processEvents()
                inpdataZ=np.array([i for j in range(101)])
                inpdataZ=np.array([inpdataZ for j in range(101)])
                Z=inpdataZ.copy()
                X=inpdata1.copy()
                Y=inpdata2.copy()
                X=np.ravel(X)
                Y=np.ravel(Y)
                Z=np.ravel(Z)
                inpdata=np.array([X,Y,Z])
                inpdata=inpdata.T 
                oupdata=self.Display_cm.predict(inpdata)
                oupdata=np.ravel(oupdata)
                oupdata=oupdata.reshape(101,101)
                Norm=plt.Normalize(vmin=oupdata.min(),vmax=oupdata.max())
                myfig.axes.plot_surface(inpdata1,inpdata2,inpdataZ,facecolors=cm.jet(Norm(oupdata)),alpha=0.6,shade=True)
            self.graphicsView_19.setScene(scene)
            return
        return 
        
    def Display_Down_OK(self):
        if self.radioButton_2.isChecked():
            self.Display_cm=self.rgs_sm
            Fo=self.FFF[-2]
        elif self.radioButton.isChecked():
            self.Display_cm=self.rgs_em 
            Fo=self.FFF[-1]
        else:
            QMessageBox.information(self,"Information","No Model")
            return
        x1=self.lineEdit_31.text()
        x2=self.lineEdit_32.text()
        x3=self.lineEdit_33.text()
        x4=self.lineEdit_34.text()
        u=[x1,x2,x3,x4]
        inp=[]
        try:
            for i in range(len(u)):
                if u[i]!='':
                    a=float(u[i])
                    b=self.FFF[i].f(a)
                    inp.append(b)
        except:
            QMessageBox.critical(self,"Critical","Input data is wrong")
            return
        c=self.Display_cm.predict(np.array([inp]))[0]
        c=float(c)
        d=Fo.f_b(c)
        d=float(d)
        self.lineEdit_37.setText('{:.2f}'.format(d))
        inp.insert(self.index,c)
        e=self.rgs.predict(np.array([inp]))[0]
        e=float(e)
        f=self.FF[-1].f_b(e)
        f=float(f)
        self.lineEdit_38.setText('{:.2f}'.format(f))
        return
        
    def Simulation_SMC_Record(self):
        try:
            self.rgs_sm
        except:
            QMessageBox.information(self,"Information","Pleas Load a model by \"ConModLoading\"")
            return
        fn=QFileDialog.getOpenFileName(self,"Open file",r"./Datafile/","*.csv")[0]#文件路径
        fo=pd.read_csv(fn)
        self.time_sm=list(fo['Time'])
        datain=[]
        self.dataPrint_sm=[]
        for i in range(len(self.con_name)-1):
            datatem=np.array(list(fo[self.con_name[i]]))
            self.dataPrint_sm.append(datatem.copy())
            datatem=self.FFF[i].f(datatem)
            datain.append(datatem)
        self.dataPrint_sm=np.array(self.dataPrint_sm)
        datain1=datain.copy()
        datain1=np.array(datain1)
        datain1=datain1.T
        obj1=self.rgs_sm.predict(datain1)
        obj1=np.ravel(obj1)
        self.obj_sm=self.FFF[-2].f_b(obj1)
        datain.insert(self.index,obj1)
        datain=np.array(datain)
        datain=datain.T 
        dataout1=self.rgs.predict(datain)
        dataout1=np.ravel(dataout1)
        self.dataout_sm=self.FF[-1].f_b(dataout1)
        self.painer1=QtCore.QTimer(self)
        self.painer1.start(1000)
        self.painer1.timeout.connect(self.Painer1)
        self.painer1mark=0
        self.pushButton_63.setEnabled(False)
        return
        
    def Simulation_EMC_Record(self):
        try:
            self.rgs_em
        except:
            QMessageBox.information(self,"Information","Pleas Load a model by \"ConModLoading\"")
            return
        fn=QFileDialog.getOpenFileName(self,"Open file",r"./Datafile/","*.csv")[0]#文件路径
        fo=pd.read_csv(fn)
        self.time_em=list(fo['Time'])
        datain=[]
        self.dataPrint_em=[]
        for i in range(len(self.con_name)-1):
            datatem=np.array(list(fo[self.con_name[i]]))
            self.dataPrint_em.append(datatem.copy())
            datatem=self.FFF[i].f(datatem)
            datain.append(datatem)
        self.dataPrint_em=np.array(self.dataPrint_em)
        datain1=datain.copy()
        datain1=np.array(datain1)
        datain1=datain1.T
        obj1=self.rgs_em.predict(datain1)
        obj1=np.ravel(obj1)
        self.obj_em=self.FFF[-1].f_b(obj1)
        datain.insert(self.index,obj1)
        datain=np.array(datain)
        datain=datain.T 
        dataout1=self.rgs.predict(datain)
        dataout1=np.ravel(dataout1)
        self.dataout_em=self.FF[-1].f_b(dataout1)
        self.painer2=QtCore.QTimer(self)
        self.painer2.start(1000)
        self.painer2.timeout.connect(self.Painer2)
        self.painer2mark=0
        self.pushButton_62.setEnabled(False)
        return
        
    def Painer1(self):
        if self.painer1mark == len(self.time_sm):
            self.painer1.stop()
            self.pushButton_63.setEnabled(True)
            return
        self.painer1mark += 1
        x=self.time_sm[:self.painer1mark]
        y1=self.dataout_sm[:self.painer1mark]
        y2=self.obj_sm[:self.painer1mark]
        self.lineEdit_51.setText(str(x[-1]))
        a = [self.lineEdit_52,self.lineEdit_53,self.lineEdit_54,self.lineEdit_50]
        for i in range(len(self.dataPrint_sm)):
            a[i].setText(str(self.dataPrint_sm[i][self.painer1mark-1]))
        fig1=MyFigure(chang=268, kuan=298, dpi=100)
        scene1= QGraphicsScene()
        scene1.addWidget(fig1)
        fig1.axes.scatter(x,y1,marker='x',color='r')
        fig1.axes.plot(x,y1,color='b')
        self.graphicsView_22.setScene(scene1)
        fig2=MyFigure(chang=267, kuan=298, dpi=100)
        scene2= QGraphicsScene()
        scene2.addWidget(fig2)
        fig2.axes.scatter(x,y2,marker='x',color='r')
        fig2.axes.plot(x,y2,color='b')
        self.graphicsView_23.setScene(scene2)
        return
    
    def Painer2(self):
        if self.painer2mark == len(self.time_em):
            self.painer2.stop()
            self.pushButton_62.setEnabled(False)
            return
        self.painer2mark += 1
        x=self.time_em[:self.painer2mark]
        y1=self.dataout_em[:self.painer2mark]
        y2=self.obj_em[:self.painer2mark]
        self.lineEdit_45.setText(str(x[-1]))
        a = [self.lineEdit_48,self.lineEdit_47,self.lineEdit_46,self.lineEdit_49]
        for i in range(len(self.dataPrint_em)):
            a[i].setText(str(self.dataPrint_em[i][self.painer2mark-1]))
        fig1=MyFigure(chang=268, kuan=298, dpi=100)
        scene1= QGraphicsScene()
        scene1.addWidget(fig1)
        fig1.axes.scatter(x,y1,marker='x',color='r')
        fig1.axes.plot(x,y1,color='b')
        self.graphicsView_21.setScene(scene1)
        fig2=MyFigure(chang=267, kuan=298, dpi=100)
        scene2= QGraphicsScene()
        scene2.addWidget(fig2)
        fig2.axes.scatter(x,y2,marker='x',color='r')
        fig2.axes.plot(x,y2,color='b')
        self.graphicsView_20.setScene(scene2)
        return
        
    def Timer(self):
        self.dateTimeEdit.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_2.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_3.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_4.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_5.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_6.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_7.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_8.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_9.setDateTime(QtCore.QDateTime.currentDateTime())
        self.dateTimeEdit_10.setDateTime(QtCore.QDateTime.currentDateTime())
        return
    
    def NewPro_Pred_Box(self):
        if self.comboBox_2.currentIndex()==1:
            proad=r'.\picture\Tomato.jpg'
            stext='Tomato'
        elif self.comboBox_2.currentIndex()==2:
            proad=r'.\picture\Pepper.jpg'
            stext='Pepper'
        elif self.comboBox_2.currentIndex()==3:
            proad=r'.\picture\Eggplant.jpg'
            stext='Eggplant'
        elif self.comboBox_2.currentIndex()==4:
            proad=r'.\picture\Lettuce.png'
            stext='Lettuce'
        else:
            proad=r'.\picture\Cucumber.jpg'
            stext='Cucumber'
        pixmap=QtGui.QPixmap()
        scene=QGraphicsScene()
        pixmap.load(proad)
        item=QGraphicsPixmapItem(pixmap)
        scene.addItem(item)
        scene.addText(stext)
        self.graphicsView.setScene(scene)
        return
    
    def NewPro_Pred_UpOK(self):
        table_C=self.spinBox.value()
        self.spinBox.setEnabled(False)
        self.tableWidget.setColumnCount(table_C)
        table_name=[]
        for i in range(table_C):
            table_name.append('Factor'+str(i+1))
        self.tableWidget.setHorizontalHeaderLabels(table_name)
        return
    
    def DataIm_Box(self,i):
        self.listWidget_3.clear()
        a=['Repetition '+str(j+1) for j in range(self.DataIm_VR[i])]
        self.listWidget_3.addItems(a)
        self.listWidget_3.setCurrentRow(0)
        self.tableWidget_2.clearContents()
        self.tableWidget_2.setColumnCount(self.DataIm_VR[i])
        table_name=[]
        for i in range(self.DataIm_VR[i]):
            table_name.append('Repetition'+str(i+1))
        self.tableWidget_2.setHorizontalHeaderLabels(table_name)
        scene= QGraphicsScene()
        self.graphicsView_2.setScene(scene)
        return
        
    def DataIm_list(self,i):
        return
    
    def ModBulding_SVR_Box(self,i):
        if i==0:
            self.ker='rbf'
        elif i == 1:
            self.ker='sigmoid'
        elif i == 2:
            self.ker='linear'
        else:
            self.ker='poly'
        return
    
    def qgafit(self,pop,train,test):
        depopR,depop=self.decode(pop.copy())#种群解码后二进制序列列表和种群解码后十进制列表
        fitnow=[]#当前种群适应度列表[pop1fit,pop2fit,...,poppopsizefit]
        for i in range(self.popsize):#遍历当前种群所有个体
            QApplication.processEvents()
            rgs=svm.SVR(C=depop[i][0],kernel=self.ker,gamma=depop[i][1],epsilon = 0.04)#根据当前个体建立SVR模型
            rgs.fit(train[0],train[1])#使用训练集数据训练模型
            yte_calcu=rgs.predict(test[0]) #计算测试集的预测值
            rmse=self.RMSE(test[1], yte_calcu)#比较测试集的预测值和真实值，计算模型均方根误差，以误差倒数作为适应度
            fitnow.append(1.0/rmse)#将当前个体建立的模型均方根误差作为适应度，加入当前种群的适应度列表
        return depopR,depop,fitnow
    
    def qgarun(self,train,test):
        pop=self.initial()#建立初始种群
        bestfitall=0#所有代中的最大适应度
        localgen=0#设置初始陷入局部最优的代数为0
        for gen_i in range(self.gen):#遍历所有进化代数
            QApplication.processEvents()
            self.progressBar_3.setValue(int(100*gen_i/self.gen))
            depopR,depop,fitnow=self.qgafit(pop, train, test)#当前代的种群解码后二进制列表、当前代的种群解码后列表和当前代的种群适应度列表
            fitnow_np=np.array(fitnow)#将列表转化为数组，方便对其中最大值和最小值操作
            bestfitnow=fitnow_np.max()#将当前代种群的最大适应度找出
            bestfitnow_i=fitnow_np.argmax()#将当前代种群的最大适应度位置找出
            if bestfitnow>bestfitall:
                bestfitall=bestfitnow #bestfitall是所有代最佳适应度bestdepopall
                bestdepopRall=depopR[bestfitnow_i]#bestdepopRall是所有代中最佳适应度的构造二进制序列个体[chr1的二进制序列,chr2的二进制序列]
                bestdepopall=depop[bestfitnow_i]#bestdepopall是所有代最佳适应度对应的十进制个体列表[chr1的十进制,chr2的十进制]
                bestpopall=pop[bestfitnow_i]#所有代最佳适应度对应的个体列表[[chr1的量子态角度列表],[chr2的量子态角度列表]]
                localgen=0#最佳适应度改变，则将局部最优代数设为0
            else:
                localgen+=1#最佳适应度不改变，则将局部最优代数加1
            if localgen>self.maxgen:#灾变操作
                pop=self.initial()#种群重构
                localgen=0#局部平缓代数归零
                pop.pop()#将种群中最后一个个体弹出，这一步是随机弹出哪个都行，为了让出一个空位给历史最佳适应度个体
                pop.append(bestpopall)#将历史最优个体加入该重构列表
                continue
            pop=self.rot(pop.copy(), depopR, fitnow, bestdepopRall, bestfitall)#量子门旋转
            pop=self.change(pop.copy())#交叉
            pop=self.variation(pop.copy())#变异
        return bestdepopall
    
    def qgafit2(self,pop,train,test):
        depopR,depop=self.decode(pop.copy())#种群解码后二进制序列列表和种群解码后十进制列表
        fitnow=[]#当前种群适应度列表[pop1fit,pop2fit,...,poppopsizefit]
        for i in range(self.popsize):#遍历当前种群所有个体
            QApplication.processEvents()
            rgs=svm.SVR(C=depop[i][0],kernel='rbf',gamma=depop[i][1],epsilon = 0.04)#根据当前个体建立SVR模型
            rgs.fit(train[0],train[1])#使用训练集数据训练模型
            yte_calcu=rgs.predict(test[0]) #计算测试集的预测值
            rmse=self.RMSE(test[1], yte_calcu)#比较测试集的预测值和真实值，计算模型均方根误差，以误差倒数作为适应度
            fitnow.append(1.0/rmse)#将当前个体建立的模型均方根误差作为适应度，加入当前种群的适应度列表
        return depopR,depop,fitnow
    
    def qgarun2(self,train,test):
        pop=self.initial()#建立初始种群
        bestfitall=0#所有代中的最大适应度
        localgen=0#设置初始陷入局部最优的代数为0
        for gen_i in range(self.gen):#遍历所有进化代数
            QApplication.processEvents()
            self.progressBar_6.setValue(int(20*gen_i/self.gen)+40)
            depopR,depop,fitnow=self.qgafit2(pop, train, test)
            fitnow_np=np.array(fitnow)
            bestfitnow=fitnow_np.max()
            bestfitnow_i=fitnow_np.argmax()
            if bestfitnow>bestfitall:
                bestfitall=bestfitnow #bestfitall是所有代最佳适应度bestdepopall
                bestdepopRall=depopR[bestfitnow_i]#bestdepopRall是所有代中最佳适应度的构造二进制序列个体[chr1的二进制序列,chr2的二进制序列]
                bestdepopall=depop[bestfitnow_i]#bestdepopall是所有代最佳适应度对应的十进制个体列表[chr1的十进制,chr2的十进制]
                bestpopall=pop[bestfitnow_i]#所有代最佳适应度对应的个体列表[[chr1的量子态角度列表],[chr2的量子态角度列表]]
                localgen=0#最佳适应度改变，则将局部最优代数设为0
            else:
                localgen+=1#最佳适应度不改变，则将局部最优代数加1
            if localgen>self.maxgen:#灾变操作
                pop=self.initial()#种群重构
                localgen=0#局部平缓代数归零
                pop.pop()#将种群中最后一个个体弹出，这一步是随机弹出哪个都行，为了让出一个空位给历史最佳适应度个体
                pop.append(bestpopall)#将历史最优个体加入该重构列表
                continue
            pop=self.rot(pop.copy(), depopR, fitnow, bestdepopRall, bestfitall)#量子门旋转
            pop=self.change(pop.copy())#交叉
            pop=self.variation(pop.copy())#变异
        return bestdepopall
    
    def nextBatch(self,data,batch_size,n):#小批量梯度下降法获取当前数据集函数
        #data为全部数据[X,Y],batch_size为当前子数据集大小,n为第n个子数据集
        a=n*batch_size
        b=(n+1)*batch_size
        limdata=len(data[0])
        if b>limdata:
            b=limdata
            a=b-batch_size
        data_x=data[0][a:b]
        data_y=data[1][a:b]
        return data_x,data_y
    
    def RMSE(self,x,y):
        '''均方根误差'''
        num=len(x)
        sum=0
        for i in range(num):
            sum+=math.pow((x[i]-y[i]),2)
        sum=sum/num
        JFG=math.sqrt(sum)
        return JFG
    
    def R2g(self,x,y):
        r2=R2(x,y)
        n=len(x)
        P=len(self.DataIm_VN)-1
        return 1-(1-r2)*(n-1)/(n-P-1)
    
    def R2g2(self,x,y):
        r2=R2(x,y)
        n=len(x)
        P=len(self.con_name)-1
        return 1-(1-r2)*(n-1)/(n-P-1)
        
    def initialuchord(self,x,y):
        '''初始化函数，用于得到所需计算的数据(输入x和输出y)，以及进行归一化。input:x,y
        x为离散点的输入特征
        y为离散点的输出标签
        '''
        self.fx=F(x,mini=0,maxi=1)#x的归一化类实例
        self.fy=F(y,mini=0,maxi=1)#y的归一化类实例
        self.x=self.fx.f(x)#对离散点的输入特征进行归一化
        self.y=self.fy.f(y)#对离散点的输入标签进行归一化
        return
    
    def PU(self,P, Pl, Pr):#求与P相距为U的点，Pl为正好小于U的点，Pr为正好大于U的点
        a=fsolve(lambda x:(Pl[0]*x +(1-x)*Pr[0]-P[0])**2+(Pl[1]*x+(1-x)*Pr[1]-P[1])**2-self.U**2,[-1])
        return [float(Pl[0]*a+(1-a)*Pr[0]),float(Pl[1]*a+(1-a)*Pr[1])]
    
    def U_Curvature(self,Pl,Pr):#Pi点的U弦长曲率,夹角的余弦值
        Di= np.linalg.norm(np.array(Pl) - np.array(Pr))#左右两点的距离
        si = self.U**2+self.U**2-Di**2#余弦定理分子
        ci = si/self.U**2#Pi点的余弦值
        return ci
    
    def Hillmax(self,X,initial,last):#爬山法获得极大值点
        x=X.copy()
        for i in range(initial+1,last):
            cur=x[i]
            curl=x[i-1]
            curr=x[i+1]
            if cur>curr and cur>curl:
                return i
        else:
            return i
        return
    
    def runuchord(self,a,b):
        '''运算U弦长曲率函数，input:x,y;output:Cur_x,Cur,Cmax_i
        a为待求离散点的输入的特征[x1,x2,...,xn]
        b为待求离散点的输入标签[y1,y2,...,yn]
        Cur_x:所有曲率点坐标列表Cur_x=a=[x1,x2,...,xn]
        Cur:所有曲率点Cur_x对应的曲率值列表[C1,C2,...,Cn],其中无法计算曲率的点其曲率值用0表示
        Cmax_i:在所有点a中，最大曲率值点的索引值Cmax_i。a[Cmax_i]==cur_x[Cur.argmax()]==cur_x[Cmax_i]
        '''
        x=a.copy()
        y=b.copy()
        self.initialuchord(x,y)#数据初始化，得到属性self.x和self.y分别表示归一化后的输入特征和标签
        length=len(self.x)#一共有lenth个点需计算离散曲率
        '''在lenth个点里寻找最后一个可计算U弦长曲率的点'''
        P_last = np.array([self.x[-1], self.y[-1]])#最后一个离散点的坐标P_last
        for i in range(1, length):#从第倒数二个点到第一个点进行遍历，找到可进行U弦长曲率计算的最后一个点
            last = length - i#当前遍历点的索引，从第倒数二个点到第一个点进行遍历
            Pn = np.array([self.x[last], self.y[last]])#当前遍历点的坐标Pn
            distance = np.linalg.norm(Pn - P_last)#求Pn与P_last的距离
            if distance > self.U:#当前点和最后一个点的距离正好大于U则为可计算U弦长的最后一个点
                break#Pn为最后一个曲率点,last为其x、y的索引值
        '''在lenth个点里寻找第一个可计算U弦长曲率的点'''
        P0 = np.array([self.x[0], self.y[0]])##第一个离散点的坐标P0
        for initial in range(1, length):#从第二个点到最后一个点进行遍历，找到可进行U弦长曲率计算的第一个点，initial为当前遍历点的索引
            Pm = np.array([self.x[initial], self.y[initial]])#当前遍历点坐标Pm
            distance = np.linalg.norm(Pm - P0)#求Pm与P0的距离
            if distance > self.U:#当前点和第一个点的距离正好大于U则为可计算U弦长的第一个点
                break#Pm为第一个曲率点,initial为其x、y的索引值
        Cur_x = []#所有曲率点坐标
        Cur = []#所有曲率点对应的曲率值
        for i in range(length):#遍历所有点
            QApplication.processEvents()
            if i<initial or i>last:#判断当前点不在可计算曲率的点中
                Cur.append(0)#曲率值记为0并加入曲率列表Cur中
                Cur_x.append(self.x[i])#曲率点加入Cur_x列表中
                continue
            Pi= np.array([self.x[i], self.y[i]])#Pi为求曲率的待求点
            Cur_x.append(self.x[i])#曲率点加入Cur_x列表中,此处是可以计算曲率的点
            '''求Pi距离为U的左边的点Pl'''
            for j in range(1,i+1):#从第一个点遍历到Pi，其实程序是从Pi左边第一个点遍历到实际的第一个点去找的
                Pj=np.array([self.x[i-j],self.y[i-j]])#当前点坐标
                distance = np.linalg.norm(Pi - Pj)#Pi与Pj的距离
                if distance < self.U :
                    Pl_less=Pj #距离Pi左边小于U的最近的点
                else:
                    Pl_more=Pj #距离Pi左边大于U的最近的点
                    Pl=self.PU(Pi,Pl_less,Pl_more)#距离Pi距离正好等于U的左边的点
                    break
            '''求Pi距离为U的右边的点Pr'''
            for j in range(i+1,length):#从Pi开始遍历到最后一个点，其实程序是从Pi左边第一个点遍历到实际的第一个点去找的
                Pj=np.array([self.x[j],self.y[j]])#当前点坐标
                distance = np.linalg.norm(Pi-Pj)#Pi与Pj的距离
                if distance < self.U:
                    Pr_less= Pj#距离Pi右边小于U的最近的点
                else:
                    Pr_more = Pj#距离Pi右边大于U的最近的点
                    Pr=self.PU(Pi,Pr_less,Pr_more)#距离Pi距离正好等于U的右边的点
                    break
            ci=self.U_Curvature(Pl,Pr)#Pi点的U弦长曲率
            Cur.append(ci)#将该点曲率加入曲率列表Cur中
        Cmax_i=self.Hillmax(Cur, initial, last)#求曲率最大点的索引值
        return Cur_x,Cur,Cmax_i#归一化后的曲率最大值点横纵坐标与其索引值   
    
class FL():
    def __init__(self,W):
        self.W=W
    def predict(self,X):
        X=X.T
        s=0
        for i in range(len(X)):
            s+=self.W[i]*X[i]
        s=self.W[-1]+s
        return s
    
class FM():
    def __init__(self,W):
        self.W=W
    def predict(self,X):
        X=X.T
        nv=len(X)#自变量个数
        pi=0
        s=0
        for i in range(nv):
            for j in range(i, nv):
                s+=self.W[pi]*X[i]*X[j]
                pi+=1
        s=s+self.W[pi]
        return s

class FBP():
    def __init__(self,W,B):
        self.w=W 
        self.b=B 
        
    def sigmoid(self,x):
        return 1/(1 + np.exp(-x))
    
    def predict(self,X):
        X1=X.copy()
        for i in range(len(self.w)-1):
            Y1=self.sigmoid(np.matmul(X1,self.w[i])+self.b[i])
            X1=Y1
        Y=np.matmul(X1,self.w[-1])+self.b[-1]
        return Y

class MainWindowUi(QMainWindow,Ui_MainWindow,Qga):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
class PerformUi(Ui_Form,QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
class ModelSelectUi(Ui_Form2,QWidget):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

if __name__=='__main__':
    app=QApplication(sys.argv)
    mainwin=MainWindowUi()
#     mainwin=PerformUi()
#     mainwin=ModelSelectUi()
    mainwin.show()
    sys.exit(app.exec_())