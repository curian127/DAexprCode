#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorenz 96模式局地化EnKF同化实验
@author: shenzheqi
"""
#%% 5-1 L96模式的积分算子和观测算子
import numpy as np
def Lorenz96(state,*args):                      # 定义Lorenz 96 模式右端项
    x = state                                # 模式状态记为x
    F = args[0]                              # 输入外强迫
    n = len(x)                               # 状态空间维数
    f = np.zeros(n)                          
    f[0] = (x[1] - x[n-2]) * x[n-1] - x[0]      # 处理三个边界点: i=0,1,N-1
    f[1] = (x[2] - x[n-1]) * x[0] - x[1]        # 导入周期边界条件
    f[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]
    for i in range(2, n-1):                  
        f[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]   # 内部点符合方程（9）
    f = f + F                            # 加上外强迫
    return f

def RK4(rhs,state,dt,*args):                    # 使用Runge-Kutta方法求解（同L63）
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state

def h(x):                                         # 观测算子(假设只观测部分变量)
    n= x.shape[0]                                # 状态维数
    m= 9                                       # 总观测数
    H = np.zeros((m,n))                           # 设定观测算子
    di = int(n/m)                                 # 两个观测之间的空间距离
    for i in range(m):
        H[i,(i+1)*di-1] = 1                        # 通过设置观测位置给出观测矩阵
    z = H @ x                                   # 左乘观测矩阵得到观测算子
    return z
# 以下求出的线性化观测算子实际上就是输出观测矩阵。
def Dh(x):
    n= x.shape[0]
    m= 9
    H = np.zeros((m,n))    
    di = int(n/m) 
    for i in range(m):
        H[i,(i+1)*di-1] = 1
    return H
#%% 5-2 Lorenz 96模式的真值积分和观测模拟
n = 36                  # 状态空间维数
F = 8                   # 外强迫项
dt = 0.01               # 积分步长
# 1. spinup获取真实场初值: 从 t=-20 积分到 t = 0 以获取实验初值
x0 = F * np.ones(n)     # 初值
x0[19] = x0[19] + 0.01  # 在第20个变量上增加微小扰动
x0True = x0
nt1 = int(20/dt)
for k in range(nt1):
    x0True = RK4(Lorenz96,x0True,dt,F)   #从t=-20积分到t=0
# 2. 真值实验和观测的信息
tm = 20                   # 实验窗口长度
nt = int(tm/dt)             # 积分步数
t = np.linspace(0,tm,nt+1)
np.random.seed(seed=1)
m = 9                   # 观测变量数
dt_m = 0.2               # 两次观测之间的时间
tm_m = 20               # 最大观测时间
nt_m = int(tm_m/dt_m)    # 同化次数
ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)
t_m = t[ind_m]

sig_m= 0.1              # 观测误差标准差
R = sig_m**2*np.eye(m)   # 观测误差协方差
# 3. 造真值和观测
xTrue = np.zeros([n,nt+1])
xTrue[:,0] = x0True
km = 0
yo = np.zeros([m,nt_m])
for k in range(nt):
    xTrue[:,k+1] = RK4(Lorenz96,xTrue[:,k],dt,F)    # 真值
    if (km<nt_m) and (k+1==ind_m[km]):
        yo[:,km] = h(xTrue[:,k+1]) + np.random.normal(0,sig_m,[m,])  # 加噪声得到观测
        km = km+1
#%% 5-3 Gaspri-Cohn有理函数的代码和局地化矩阵
def comp_cov_factor(z_in,c):
    z=abs(z_in);             # 输入距离和局地化参数，输出局地化因子的数值
    if z<=c:                # 分段函数的各个条件
        r = z/c;
        cov_factor=((( -0.25*r +0.5)*r +0.625)*r -5.0/3.0)*r**2 + 1.0;
    elif z>=c*2.0:
        cov_factor=0.0;
    else:
        r = z / c;
        cov_factor = ((((r/12.0 -0.5)*r +0.625)*r +5.0/3.0)*r -5.0)*r + 4.0 - 2.0 / (3.0 * r);
    return cov_factor

def Rho(localP,size):
    from scipy.linalg import toeplitz
    rho0 = np.zeros(size)
    for j in range(size):
        rho0[j]=comp_cov_factor(j,localP)
    Loc = toeplitz(rho0,rho0) 
    return Loc
#%% 5-4 使用输入的局地化矩阵的EnKF同化方法
def EnKF(xbi,yo,ObsOp,JObsOp,R,RhoM):
    n,N = xbi.shape         # n-状态维数，N-集合成员数
    m = yo.shape[0]         # m-观测维数
    xb = np.mean(xbi,1)     # 预报集合平均 
    Dh = JObsOp(xb)         # 切线性观测算子    
    B = (1/(N-1)) * (xbi - xb.reshape(-1,1)) @ (xbi - xb.reshape(-1,1)).T  # 样本协方差   
    B = RhoM * B            # Shur积局地化
    D = Dh@B@Dh.T + R     
    K = B @ Dh.T @ np.linalg.inv(D)    # 求卡尔曼增益矩阵                                         
    yoi = np.zeros([m,N])
    xai = np.zeros([n,N])
    for i in range(N):
        yoi[:,i] = yo + np.random.multivariate_normal(np.zeros(m), R) # 扰动观测
        xai[:,i] = xbi[:,i] + K @ (yoi[:,i]-ObsOp(xbi[:,i]))     # 卡尔曼滤波更新          
    return xai
#%% 5-5 Lorenz96模式使用含有局地化的EnKF的同化实验
sig_b= 1
x0b = x0True + np.random.normal(0,sig_b,[n,])         # 初值
B = sig_b**2*np.eye(n)                              # 初始误差协方差
sig_p= 0.1
Q = sig_p**2*np.eye(n)                              # 模式误差

xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):
    xb[:,k+1] = RK4(Lorenz96,xb[:,k],dt,F)          # 控制实验

RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
N = 30                                       # 集合成员数
xai = np.zeros([n,N])
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)  # 初始集合 

def DA_EnKF_locP(localP,ens):
    # localP = 4;  # 参考
    rhom = Rho(localP ,n)            # !!!产生局地化矩阵，参数可调整
    
    xa = np.zeros([n,nt+1]); xa[:,0] = x0b
    km = 0
    for k in range(nt):
        for i in range(N):              # 集合预报
            ens[:,i] = RK4(Lorenz96,ens[:,i],dt,F) \
                     + np.random.multivariate_normal(np.zeros(n), Q)
        xa[:,k+1] = np.mean(ens,1)
        if (km<nt_m) and (k+1==ind_m[km]):  # 开始同化
            ens = EnKF(ens,yo[:,km],h,Dh,R,rhom)
            xa[:,k+1] = np.mean(ens,1)    
            km = km+1
    RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
    return RMSEa

localPs = np.array([2,4,8,1000])
RMSEs = np.zeros([4,2001])
for j in range(4):
    RMSEs[j] = DA_EnKF_locP(localPs[j],xai)

#%%画图相关代码
import matplotlib.pyplot as plt
plt.figure(figsize=(10,5))
plt.rcParams['font.sans-serif'] = ['Songti SC']
Colors = ['C2','C3','C4','C5'];
plt.plot(t,RMSEb,color='C1',label='背景')
plt.title("Lorenz96模式EnKF同化结果与局地化参数的关系",fontsize=16)
for j in range(4):
    plt.plot(t,RMSEs[j],color=Colors[j],label='c = '+str(localPs[j]))
plt.ylim(0,10)
plt.ylabel("均方根误差",fontsize=16)
plt.xlabel("时间(TU)",fontsize=16)
plt.xticks(fontsize=16);plt.yticks(fontsize=16) 
plt.legend(ncol=3,fontsize=15)   


