#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
同化结果 against N
@author: shenzheqi
"""

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
#%% 定义以初始集合为输入，同化结果为输出的同化实验
#>>>试验6.1 SIRPF和DEnKF使用256个粒子对Lorenz63模式的同化效果
x0b = np.array([2.0,3.0,4.0])   # 同化初值
np.random.seed(1)

sig_b= 3                        # 背景误差标准差
B = sig_b**2*np.eye(3)          # 背景误差协方差
Q = 0.2*np.eye(3)               # 模式误差协方差

#背景场
xb = np.zeros([3,nt+1])
xb[:,0] = x0b

n = 3 #状态维数
m = 3 #观测维数

# 没有同化的自由积分实验结果
for k in range(nt):
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho) 

def DA_exprEnKF(xai):
    n,N = xai.shape
    xa = np.zeros([3,nt+1])
    xa[:,0] = x0b
    np.random.seed(1)
    km = 0
    for k in range(nt):
        # EAKF 同化实验
        for i in range(N): # 预报
            xai[:,i] = RK4(Lorenz63,xai[:,i],dt,sigma,beta,rho) \
                     + np.random.multivariate_normal(np.zeros(n), Q)
        # 预报集合平均
        xa[:,k+1] = np.mean(xai,1)
        
        if (km<nt_m) and (k+1==ind_m[km]):
            xai = EnSRF(xai,yo[:,km],h,Dh,R)
            xa[:,k+1] = np.mean(xai,1)          
            km = km+1
    return xa

def DA_exprPF(xai):
    n,N = xai.shape
    xa = np.zeros([3,nt+1])
    xa[:,0] = x0b
    np.random.seed(1)
    Neff = np.zeros(nt_m) # Neff是有效集合成员数，输出可诊断粒子集合的有效性
    km = 0
    for k in range(nt):
        # PF同化
        for i in range(N): # 积分每个粒子
            xai[:,i] = RK4(Lorenz63,xai[:,i],dt,sigma,beta,rho) \
                     + np.random.multivariate_normal(np.zeros(n), Q)

        # 预报粒子都是等权重的（已经执行重取样）
        xa[:,k+1] = np.mean(xai,1)

        if (km<nt_m) and (k+1==ind_m[km]):
            # 分析步骤，输出重取样后的粒子以及重取样根据的权重
            xai,weights = BootstrapPF(xai,yo[:,km],h,Dh,R)
            xa[:,k+1] = np.mean(xai,1)
            Neff[km] = 1/np.sum(weights**2)
            km = km+1
    return xa
#%%
#>>>试验6.2 SIRPF和DEnKF同化结果的平均RMSE与集合成员数之间的关系。
sizes = np.array([4,8,16,32,64,128,256,512])
mRMSE_PF = np.zeros(8)
mRMSE_EnKF = np.zeros(8)
for j in range(8):
    N= sizes[j]
    np.random.seed(0)
    xai_int = np.zeros([3,N])
    for i in range(N):
        xai_int[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)    
    xai = xai_int.copy()
    xaP = DA_exprPF(xai)
    xai = xai_int.copy()
    xaE = DA_exprEnKF(xai)
    RMSEP = np.sqrt(np.mean((xaP-xTrue)**2,0))
    RMSEE = np.sqrt(np.mean((xaE-xTrue)**2,0))
    mRMSE_PF[j] = np.mean(RMSEP[100::])
    mRMSE_EnKF[j] = np.mean(RMSEE[100::])
#%%
mRMSE_PFp = mRMSE_PF.copy()
mRMSE_EnKFp = mRMSE_EnKF.copy()
#%%
mRMSE_PFp.sort()
mRMSE_PFp=np.abs(np.sort(-mRMSE_PFp))
mRMSE_EnKFp.sort()
mRMSE_EnKFp=np.abs(np.sort(-mRMSE_EnKFp))
import matplotlib.pyplot as plt    
plt.figure(figsize = (10,5))
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.plot(mRMSE_EnKFp,'-o',ms=8,label="EnSRF")
plt.plot(mRMSE_PFp,'-s',ms=8,label='PF')
plt.xticks(range(8),sizes)
plt.legend(fontsize=16)
plt.ylabel('平均均方根误差',fontsize=16)
plt.xlabel('集合成员数', fontsize=16)
plt.ylim(0.8,2)
plt.grid(axis='y')
plt.xticks(fontsize=16);plt.yticks(np.arange(1,2,0.2),fontsize=16)