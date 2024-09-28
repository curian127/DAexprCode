#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8. Lorenz63模式的粒子滤波器
@author: shenzheqi
"""
#%% 8-1 残量重采样方法
def SIR(weights):
    import numpy as np
    # 输入权重，输出重取样指标
    if np.sum(weights)!=1:
        weights = weights/np.sum(weights);  # 正规化        
    N = len(weights);
    outIndex = np.zeros(N,dtype=int)
    w = np.cumsum(weights);
    Nbins = np.arange(N)/N+0.5/N;
    idx = 0;
    for t in range(N):
        while Nbins[t] >= w[idx]:
            idx+=1
        outIndex[t] = idx;
    return outIndex         # 重取样指标

#%% 8-2 顺序重采样粒子滤波器（自举粒子滤波器）
def BootstrapPF(xbi,yo,ObsOp,JObsOp,R):
    n,N = xbi.shape 
    m = yo.shape[0] 
    weights = np.zeros(N)
    for i in range(N):                      # 权重公式
        weights[i] = 0.5*(yo-ObsOp(xbi[:,i])).T@np.linalg.inv(R)@(yo-ObsOp(xbi[:,i]))
    weights = np.exp(-weights)
    weights = weights/np.sum(weights)       # 正规化
    new_index= SIR(weights)                 # 重取样
    xai = xbi[:,new_index]                  # 重分配样本
    return xai,weights
#%% 6-1 集合平方根滤波器（直接法）
def EnSRF(xbi,yo,ObsOp,JObsOp,R):
    from scipy.linalg import sqrtm
    n,N = xbi.shape                      # n-状态维数，N-集合成员数
    m = yo.shape[0]                      # m-观测维数
    xb = np.mean(xbi,1)                   # 预报集合平均 
    Dh = JObsOp(xb)                     # 切线性观测算子
    B = (1/(N-1)) * (xbi - xb.reshape(-1,1)) @ (xbi - xb.reshape(-1,1)).T
    D = Dh@B@Dh.T + R
    K = B @ Dh.T @ np.linalg.inv(D)         # !!!以上与EnKF一致
    xa = xb + K @ (yo-ObsOp(xb))          # 用确定性格式更新集合平均
    A = xbi - xb.reshape(-1,1)              # 集合异常
    Z = A/np.sqrt(N-1)                    # 标准化集合异常值
    Y = np.linalg.inv(D)@Dh@Z           
    X = sqrtm(np.eye(N)-(Dh@Z).T@Y)      # 矩阵平方根
    X = np.real(X)                         # 保证矩阵平方根为实数
    Z = Z@X                             # 更新集合异常值
    A = Z*np.sqrt(N-1)
    xai = xa.reshape(-1,1)+A               # 用集合平均和集合异常计算集合成员
    return xai
#%% 8-4 加权集合卡尔曼滤波器WEnKF
def WEnKF(xbi,yo,ObsOp,JObsOp,R,Q):    # 相比于EnKF多输入模式误差Q 
    n,N = xbi.shape     # n-状态维数，N-集合成员数
    m = yo.shape[0]     # m-观测维数
    xb = np.mean(xbi,1) # 预报集合平均 
    ### 计算卡尔曼增益
    Dh = JObsOp(xb)     # 切线性观测算子
    B = (1/(N-1)) * (xbi - xb.reshape(-1,1)) @ (xbi - xb.reshape(-1,1)).T   # 样本协方差   
    D = Dh@B@Dh.T + R
    K = B @ Dh.T @ np.linalg.inv(D) # !!! 卡尔曼增益
    xai = np.zeros([n,N])
    ### 增加模式扰动量
    beta0 = np.zeros([n,N])
    for i in range(N):
        beta0[:,i] = np.random.multivariate_normal(np.zeros(n), Q)
        xbi[:,i] = xbi[:,i]+beta0[:,i]
    for i in range(N):
        xai[:,i] = xbi[:,i] + K @ (yo-ObsOp(xbi[:,i]))          
    ### 建议权重和似然权重
    Qhat = (np.eye(n)-K@Dh.T)@Q@(np.eye(n)-K@Dh.T).T+K@R@K.T
    beta = np.zeros([n,N])
    weights = np.zeros(N)       # 计算权重
    for i in range(N):       
        beta[:,i] = (np.eye(n)-K@Dh.T)@beta0[:,i]
        xai[:,i] = xai[:,i]+beta[:,i]
        weights[i] = 0.5*beta0[:,i]@np.linalg.inv(Q)@beta0[:,i].T
        weights[i] = weights[i]-0.5*beta[:,i]@np.linalg.inv(Qhat)@beta[:,i].T
        weights[i] = weights[i]+0.5*(yo-ObsOp(xbi[:,i])).T@np.linalg.inv(R)@(yo-ObsOp(xbi[:,i]))
    weights = np.exp(-weights)
    weights = weights/np.sum(weights)       # 正规化
    new_index= SIR(weights)                 # 重取样
    xai = xai[:,new_index]                  # 重分配样本
    return xai
#%% 8-5 集合卡尔曼粒子滤波器（EnKPF）
def EnKPF(xbi,yo,ObsOp,JObsOp,R,tau1,tau2):      # 多一个模式误差Q的输入 
    n,N = xbi.shape     # n-状态维数，N-集合成员数
    m = yo.shape[0]     # m-观测维数
    xb = np.mean(xbi,1) # 预报集合平均  
    Dh = JObsOp(xb)     # 切线性观测算子
    B = (1/(N-1)) * (xbi - xb.reshape(-1,1)) @ (xbi - xb.reshape(-1,1)).T   # 样本协方差   
    ### 迭代寻找最优gamma    
    gamma = 0.5;max_iter = 4;
    for k in range(max_iter):
        D = gamma*Dh@B@Dh.T + R
        K1 = gamma*B @ Dh.T @ np.linalg.inv(D)    # 公式（8-34）
        vi = np.zeros([n,N])
        for i in range(N):                          # 公式（8-35）
            vi[:,i] = xbi[:,i] + K1 @ (yo-ObsOp(xbi[:,i]))       
        Q = 1/gamma*K1*R*K1.T                   # 公式（8-36）    
        weights = np.zeros(N)
        R1 = R/(1-gamma)+Dh @ Q @ Dh.T
        for i in range(N):
            weights[i] = 0.5*(yo-ObsOp(xbi[:,i])).T@np.linalg.inv(R1)@(yo-ObsOp(xbi[:,i]))
        weights = np.exp(-weights)              # 公式（8-37）
        weights = weights/np.sum(weights)       # 标准化
        Neff = 1/np.sum(weights**2)
        tau = Neff/N
        if tau>tau2:
            gamma = gamma-0.5 / 2**(k+1)
        elif tau<tau1:
            gamma = gamma+0.5 / 2**(k+1)
        else:
            break        
    new_index= SIR(weights)                     # 重取样
    xui = np.zeros([n,N])
    for i in range(N):                          # 公式（8-38）
        xui[:,i] = vi[:,new_index[i]]+K1@ np.random.multivariate_normal(np.zeros(n), R)/np.sqrt(gamma)                 # 重分配样本    
    D = (1-gamma)*Dh@Q@Dh.T + R
    K2 = (1-gamma)*Q @ Dh.T @ np.linalg.inv(D)  # 公式（8-39）    
    xai = np.zeros([n,N])
    for i in range(N):                          # 公式（8-40）
        xai[:,i] = xui[:,i]+K2@(yo-Dh.T@xui[:,i]+np.random.multivariate_normal(np.zeros(n), R)/np.sqrt(1-gamma) )
    return xai

#%% 8-3 粒子滤波器的同化实验设置和结果
# 定义模式
import numpy as np
# 定义模式方程和积分格式
def Lorenz63(state,*args):                  # Lorenz63模式右端项
    sigma = args[0]
    beta = args[1]
    rho = args[2]
    x, y, z = state 
    f = np.zeros(3) 
    f[0] = sigma * (y - x)
    f[1] = x * (rho - z) - y
    f[2] = x * y - beta * z
    return f 
def RK4(rhs,state,dt,*args):                # Runge-Kutta积分格式
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state

sigma = 10.0; beta = 8.0/3.0; rho = 28.0    # 模式参数值   
dt = 0.02                                   # 模式积分步长
tm = 10                                     # 同化实验窗口
nt = int(tm/dt)
t = np.linspace(0,tm,nt+1)

def h(u):                   # 观测算子
    yo = u
    return yo

def Dh(u):                  # 观测的切线性算子
    n = len(u)
    D = np.eye(n)
    return D
# 实验参数
n = 3                       # 状态维数
m = 3                       # 观测数
x0True = np.array([1,1,1])  # 真实值的初值
np.random.seed(seed=1)     
sig_m= np.sqrt(3)           # 观测误差标准差
R = sig_m**2*np.eye(n)      # 观测误差协方差

dt_m = 0.5                  # 观测之间的时间间隔
tm_m = 10                   # 最大观测时间（可小于模式积分时间）
nt_m = int(tm_m/dt_m)       # 同化的次数

ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)
t_m = t[ind_m]              # 同化时间

xTrue = np.zeros([n,nt+1])
xTrue[:,0] = x0True
km = 0
yo = np.zeros([3,nt_m])
for k in range(nt):
    xTrue[:,k+1] = RK4(Lorenz63,xTrue[:,k],dt,sigma,beta,rho)         # 真实值积分
    if (km<nt_m) and (k+1==ind_m[km]):
        yo[:,km] = h(xTrue[:,k+1]) + np.random.normal(0,sig_m,[3,])   # 采样造观测
        km = km+1
# 同化实验
x0b = np.array([2.0,3.0,4.0])   # 同化实验的初值
np.random.seed(seed=0)

xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)    # 不加同化的自由积分结果

sig_b= 3
B = sig_b**2*np.eye(n)          # 初始时刻背景误差协方差
Q = 0.1**2*np.eye(n)               # 模式误差（若假设完美模式则取0）
# PF 同化
N = 256                    # 集合成员数
xai = np.zeros([3,N])
np.random.seed(0)
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)   # 随机扰动构造初始集合

xa = np.zeros([n,nt+1]); xa[:,0] = x0b
km = 0
np.random.seed(seed=0)
for k in range(nt):
    for i in range(N):
        xai[:,i] = RK4(Lorenz63,xai[:,i],dt,sigma,beta,rho) \
                  + np.random.multivariate_normal(np.zeros(n), Q)     # 公式（9）积分集合成员
    
    if (km<nt_m) and (k+1==ind_m[km]):
        # xai,weights = BootstrapPF(xai,yo[:,km],h,Dh,R)      # PF同化
        xai = WEnKF(xai,yo[:,km],h,Dh,R,Q)      # PF同化
        # xai = EnKPF(xai,yo[:,km],h,Dh,R,0.4,0.6)
        km = km+1
    xa[:,k+1] = np.mean(xai,1)        # 分析场平均       
# EnSRF同化
xai = np.zeros([3,N])
np.random.seed(0)
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)   # 随机扰动构造初始集合

xa1 = np.zeros([n,nt+1]); xa1[:,0] = x0b
km = 0
np.random.seed(seed=0)
for k in range(nt):
    for i in range(N):
        xai[:,i] = RK4(Lorenz63,xai[:,i],dt,sigma,beta,rho) \
                  + np.random.multivariate_normal(np.zeros(n), Q)     # 公式（9）积分集合成员
    if (km<nt_m) and (k+1==ind_m[km]):
        xai = EnSRF(xai,yo[:,km],h,Dh,R)      # 调用EnKF同化
        km = km+1
    xa1[:,k+1] = np.mean(xai,1)    #非同化时刻使用预报平均，同化时刻分析平均
#%% 结果画图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.figure(figsize=(10,8))
lbs = ['x','y','z']
for j in range(3):
    plt.subplot(4,1,j+1)
    plt.plot(t,xTrue[j],'b-',lw=2,label='真值')
    plt.plot(t,xb[j],'--',color='orange',lw=2,label='背景')
    plt.plot(t_m,yo[j],'go',ms=8,markerfacecolor='white',label='观测')
    plt.plot(t,xa[j],'-.',color='red',lw=2,label='WEnKF分析场')
    plt.plot(t,xa1[j],'-.',color='black',lw=2,label='EnSRF分析场')
    
    plt.ylabel(lbs[j],fontsize=16)
    if j==0:
        plt.legend(ncol=3, loc=9,fontsize=12)
        plt.title("WEnKF与EnSRF的同化效果对比",fontsize=16)
    plt.xticks(fontsize=16);plt.yticks(fontsize=16)
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
RMSEa1 = np.sqrt(np.mean((xa1-xTrue)**2,0))

plt.subplot(4,1,4)
plt.plot(t,RMSEb,color='orange',label='背景均方根误差')
plt.plot(t,RMSEa,color='red',label='WEnKF分析均方根误差')
plt.plot(t,RMSEa1,color='black',label='EnSRF分析均方根误差')
plt.ylim(0,20);
plt.text(.8,9,'N = %d' %N, fontsize=14)
plt.text(2,9,'背景误差平均值 = %0.2f' %np.mean(RMSEb[100::]),fontsize=14)
plt.text(2,6,'WEnKF分析误差平均值 = %0.2f' %np.mean(RMSEa[100::]),fontsize=14)
plt.text(2,3,'EnSRF分析误差平均值 = %0.2f' %np.mean(RMSEa1[100::]),fontsize=14)
plt.ylabel('均方根误差',fontsize=16)
plt.xlabel('时间（TU）',fontsize=16)
plt.xticks(fontsize=16);plt.yticks(fontsize=16)
plt.legend(ncol=3, loc=9,fontsize=12)