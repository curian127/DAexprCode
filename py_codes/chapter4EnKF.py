#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorenz 63 模式，EnKF同化实验，方法包括：
@author: shenzheqi
"""
#%% 4-1 Lorenz63模式代码和孪生实验的观测模拟过程（同第三章）
import numpy as np           # 导入numpy工具包
def Lorenz63(state,*args):       # 此函数定义Lorenz63模式右端项
    sigma = args[0]
    beta = args[1]
    rho = args[2]              # 输入σ,β和ρ三个模式参数
    x, y, z = state              # 输入矢量的三个分量分别为方程式中的x,y,z
    f = np.zeros(3)             # f定义为右端
    f[0] = sigma * (y - x)        # （方程1）
    f[1] = x * (rho - z) - y       # （方程2）
    f[2] = x * y - beta * z        # （方程3）
    return f 
def RK4(rhs,state,dt,*args):      # 此函数提供Runge-Kutta积分格式
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state
# 以下代码构造孪生实验的观测真实解和观测数据          
sigma = 10.0; beta = 8.0/3.0; rho = 28.0         # 模式参数值   
dt = 0.01                                   # 模式积分步长
n = 3                                      # 状态维数
m = 3                                      # 观测数
tm = 10                                    # 同化实验窗口
nt = int(tm/dt)                               # 总积分步数
t = np.linspace(0,tm,nt+1)                     # 模式时间网格

x0True = np.array([1,1,1])                     # 真实值的初值
np.random.seed(seed=1)                     # 设置随机种子
sig_m= 0.15                                # 观测误差标准差
R = sig_m**2*np.eye(n)                      # 观测误差协方差矩阵

dt_m = 0.2                  # 观测之间的时间间隔（可见为20模式步）
tm_m = 10                  # 最大观测时间（可小于模式积分时间）
nt_m = int(tm_m/dt_m)       # 进行同化的总次数

ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)  
# 观测网格在时间网格中的指标
t_m = t[ind_m]              # 观测网格
def h(x):                   # 定义观测算子
    H = np.eye(n)          # 观测矩阵为单位阵
    yo = H@x             # 单位阵乘以状态变量
    return yo
def Dh(x):                  # 观测算子的线性观测矩阵
    n = len(x)
    D = np.eye(n)
    return D
xTrue = np.zeros([n,nt+1])     # 真值保存在xTrue变量中
xTrue[:,0] = x0True           # 初始化真值
km = 0                     # 观测计数
yo = np.zeros([3,nt_m])       # 观测保存在yo变量中
for k in range(nt):           # 按模式时间网格开展模式积分循环
    xTrue[:,k+1] = RK4(Lorenz63,xTrue[:,k],dt,sigma,beta,rho)   # 真实值积分
    if (km<nt_m) and (k+1==ind_m[km]):    # 用指标判断是否进行观测
        yo[:,km] = h(xTrue[:,k+1]) + np.random.normal(0,sig_m,[3,]) #采样造观测
        km = km+1                      # 观测计数
#%% 4-2 集合卡尔曼滤波器分析算法
def EnKF(xbi,yo,ObsOp,JObsOp,R):   
# 输入变量依次为预报集合、观测、观测算子、切线观测算子、观测误差协方差
    n,N = xbi.shape         # n-状态维数，N-集合成员数
    m = yo.shape[0]         # m-观测维数
    xb = np.mean(xbi,1)      # 计算预报集合平均 
    Dh = JObsOp(xb)        # 利用切线性观测算子得到观测矩阵H
    
    B = (1/(N-1)) * (xbi - xb.reshape(-1,1)) @ (xbi - xb.reshape(-1,1)).T  #背景误差协方差
    D = Dh@B@Dh.T + R  
    K = B @ Dh.T @ np.linalg.inv(D)   #计算卡尔曼增益矩阵
            
    yoi = np.zeros([m,N])      # 预分配空间，保存扰动后的观测集合 
    xai = np.zeros([n,N])       # 预分配空间，保存分析集合
    for i in range(N):
        yoi[:,i] = yo + np.random.multivariate_normal(np.zeros(m), R)   # 随机扰动观测
        xai[:,i] = xbi[:,i] + K @ (yoi[:,i]-ObsOp(xbi[:,i]))             # 更新每个成员
        
    return xai              
# 输出集合成员。不同于EKF，不需要输出分析误差协方差
#%% 4-4 集合卡尔曼滤波器高维格式的代码
def EnKFhe(xbi,yo,ObsOp,JObsOp,R):
    from numpy.linalg import svd
    n,N = xbi.shape          # n-状态维数，N-集合成员数
    m = yo.shape[0]         # m-观测维数
    xb = np.mean(xbi,1)      # 预报集合平均 
    Dh = JObsOp(xb)         # 切线性观测算子
    Y = np.zeros([m,N])       # 观测扰动量保存于Y中
    hxbi = np.zeros([m,N])    # 集合成员投影
    for i in range(N):
        Y[:,i] = np.random.multivariate_normal(np.zeros(m), R)    
        hxbi[:,i] = ObsOp(xbi[:,i])     #投影到观测空间
    Dp = Y+yo.reshape(-1,1)-hxbi    #更新量
    A = xbi - xb.reshape(-1,1)       # 集合异常

    U,Sig,V = svd(Dh@A+Y)             # 做SVD分解
    Lam_m1 = np.diag(1/Sig**2)         # 奇异值平方的倒数
    X1 = Lam_m1@U.T                 # 公式（4-2-37）
    X2 = X1@Dp                      # 公式（4-2-38）
    X3 = U@X2                       # 公式（4-2-39）
    X4 = (Dh@A).T@X3                # 公式（4-2-40）
    xai = xbi + A@X4                  # 公式（4-2-41）
    return xai
#%% 4-3 EnKF同化试验及结果
n = 3                       # 状态维数
m = 3                      # 观测数
x0b = np.array([2.0,3.0,4.0])    # 同化实验的初值
np.random.seed(seed=1)      # 初始化随机种子，便于重复结果

xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):            # xb得到的是不加同化的自由积分结果
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)    

sig_b= 0.5
B = sig_b**2*np.eye(n)           # 初始时刻背景误差协方差，设为对角阵
Q = 0.5*np.eye(n)               # 模式误差（若假设完美模式则取0）

N = 20                        # 设定集合成员数
xai = np.zeros([3,N])             # 设定集合，保存在xai中
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)   
# 通过对预报初值进行随机扰动构造初始集合

xa = np.zeros([n,nt+1]); xa[:,0] = x0b  #保存每步的集合均值作为分析场，存在xa
km = 0             # 对同化次数进行计数
for k in range(nt):     # 时间积分
    for i in range(N):  # 对每个集合成员积分
        xai[:,i] = RK4(Lorenz63,xai[:,i],dt,sigma,beta,rho) \
                 + np.random.multivariate_normal(np.zeros(n), Q)    
        # 积分每个集合成员得到预报集合
    if (km<nt_m) and (k+1==ind_m[km]):  # 当有观测的时刻，使用EnKF同化
        # inflation
        xa_m = np.mean(xai,axis=1)
        for i in range(N):
            xai[:,i] = xa_m+1.6*(xai[:,i]-xa_m)       
        xai = EnKF(xai,yo[:,km],h,Dh,R)      # 调用EnKF同化
        km = km+1
    xa[:,k+1] = np.mean(xai,1)    #非同化时刻使用预报平均，同化时刻分析平均
# EnKF结果画图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号
plt.figure(figsize=(10,8))
lbs = ['x','y','z']
for j in range(3):
    plt.subplot(4,1,j+1)
    plt.plot(t,xTrue[j],'b-',lw=2,label='真值')
    plt.plot(t,xb[j],'--',color='orange',lw=2,label='背景')
    plt.plot(t_m,yo[j],'go',ms=8,markerfacecolor='white',label='观测')
    plt.plot(t,xa[j],'-.',color='red',lw=2,label='分析')
    plt.ylabel(lbs[j],fontsize=15)
    plt.xticks(fontsize=16);plt.yticks(fontsize=16)
    if j==0:
        plt.legend(ncol=4, loc=9,fontsize=15)
        plt.title("EnKF同化实验",fontsize=16)
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
plt.subplot(4,1,4)
plt.plot(t,RMSEb,color='orange',label='背景均方根误差')
plt.plot(t,RMSEa,color='red',label='分析均方根误差')
plt.legend(ncol=2, loc=9,fontsize=15)
plt.text(1,9,'背景误差平均 = %0.2f' %np.mean(RMSEb),fontsize=14)
plt.text(1,4,'分析误差平均 = %0.2f' %np.mean(RMSEa),fontsize=14)
plt.ylabel('均方根误差',fontsize=16)
plt.xlabel('时间（TU）',fontsize=16)
plt.xticks(fontsize=16);plt.yticks(fontsize=16)
