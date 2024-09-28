#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 09:06:23 2023

@author: shenzheqi
"""
#%% 3-1 定义Lorenz 63模式
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
dt = 0.01                                   # 模式积分步长
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

#%% 3-2 真值积分和模拟观测
n = 3                       # 状态维数
m = 3                       # 观测数

x0True = np.array([1,1,1])  # 真实值的初值
np.random.seed(seed=1)     
sig_m= 0.15                 # 观测误差标准差
R = sig_m**2*np.eye(n)      # 观测误差协方差

dt_m = 0.2                  # 观测之间的时间间隔
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
#%% 6-6 串行集合调整卡尔曼滤波器
def obs_increment_eakf(ensemble, observation, obs_error_var):       # 计算观测空间增量
    prior_mean = np.mean(ensemble);
    prior_var = np.var(ensemble);
    if prior_var >1e-6:         # 用于避免退化的先验集合造成错误更新
        post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_error_var);                       # 公式（2）
        post_mean = post_var * (prior_mean / prior_var + observation / obs_error_var);  # 公式（3）
    else:
        post_var = prior_var; post_mean = prior_mean;    
    updated_ensemble = ensemble - prior_mean + post_mean;
    var_ratio = post_var / prior_var;
    updated_ensemble = np.sqrt(var_ratio) * (updated_ensemble - post_mean) + post_mean; # 公式（4）
    obs_increments = updated_ensemble - ensemble;
    return obs_increments
def get_state_increments(state_ens, obs_ens, obs_incs):             # 将观测增量回归到状态增量
    covar = np.cov(state_ens, obs_ens);
    state_incs = obs_incs * covar[0,1]/covar[1,1];
    return state_incs
def sEAKF(xai,yo,ObsOp, R, RhoM):
    n,N = xai.shape;            # 状态维数
    m = yo.shape[0];            # 观测数
    Loc = ObsOp(RhoM)           # 观测空间局地化
    for i in range(m):          # 针对每个标量观测的循环
        hx = ObsOp(xai);        # 投影到观测空间 
        hxi = hx[i];            # 投影到对应的矢量观测，公式（1）
        obs_inc = obs_increment_eakf(hxi,yo[i],R[i,i]);
        for j in range(n):      # 针对状态变量的每个元素的循环
            state_inc = get_state_increments(xai[j], hxi,obs_inc)   # 获取状态增量
            cov_factor=Loc[i,j] # 使用局地化矩阵的相应元素
            if cov_factor>1e-6: # 在局地化范围内加增量
                xai[j]=xai[j]+cov_factor*state_inc;     # 公式（5）
    return xai
#%% 5-3 Gaspri-Cohn函数
def comp_cov_factor(z_in,c):
    z=abs(z_in);
    if z<=c:
        r = z/c;
        cov_factor=((( -0.25*r +0.5)*r +0.625)*r -5.0/3.0)*r**2 + 1.0;
    elif z>=c*2.0:
        cov_factor=0.0;
    else:
        r = z / c;
        cov_factor = ((((r/12.0 -0.5)*r +0.625)*r +5.0/3.0)*r -5.0)*r + 4.0 - 2.0 / (3.0 * r);
    return cov_factor
#%% 5-3 利用GC函数生成局地化矩阵
def Rho(localP,size):
    from scipy.linalg import toeplitz
    rho0 = np.zeros(size)
    for j in range(size):
        rho0[j]=comp_cov_factor(j,localP)
    Loc = toeplitz(rho0,rho0) 
    return Loc
#%% 9-1 实验5.2 参数有偏差的Lorenz63模式的EAKF状态估计实验
#### 使用有偏差的参数开展实验
sigma = 13     
beta = 3
rho = 30    
####
x0b = np.array([2.0,3.0,4.0])   # 同化实验的初值
np.random.seed(seed=1)

xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)    # 不加同化的自由积分结果

sig_b= 0.1
B = sig_b**2*np.eye(n)          # 初始时刻背景误差协方差
Q = 0.0*np.eye(n)               # 模式误差（若假设完美模式则取0）

N = 20 # 集合成员数
xai = np.zeros([3,N])
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)   # 随机扰动构造初始集合
LocM = np.ones([3,3])

xa = np.zeros([n,nt+1]); xa[:,0] = x0b
km = 0
for k in range(nt):
    for i in range(N):
        xai[:,i] = RK4(Lorenz63,xai[:,i],dt,sigma,beta,rho) \
                 + np.random.multivariate_normal(np.zeros(n), Q)     # 公式（9）积分集合成员
    xa[:,k+1] = np.mean(xai,1)
    if (km<nt_m) and (k+1==ind_m[km]):
        xai = sEAKF(xai,yo[:,km],h,R,LocM)      # EnKF同化
        xa[:,k+1] = np.mean(xai,1)        # 分析场平均
        km = km+1

#%% 状态估计画图
import matplotlib.pyplot as plt
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
mRMSEb = np.mean(RMSEb)
mRMSEa = np.mean(RMSEa)
plt.figure(figsize=(10,8))
lbs = ['x','y','z']
for j in range(3):
    plt.subplot(4,1,j+1)
    plt.plot(t,xTrue[j],'b-',lw=2,label='真值')
    plt.plot(t,xb[j],'--',color='orange',lw=2,label='背景')
    plt.plot(t_m,yo[j],'go',ms=8,markerfacecolor='white',label='观测')
    plt.plot(t,xa[j],'-.',color='red',lw=2,label='分析')
    plt.ylabel(lbs[j],fontsize=16)
    plt.xticks(fontsize=16);plt.yticks(fontsize=16)
    if j==0:
        plt.legend(ncol=4, loc=9,fontsize=16)
        plt.title("有偏参数模式的EAKF同化实验",fontsize=16)
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
plt.subplot(4,1,4)
plt.plot(t,RMSEb,color='orange',label='背景均方根误差')
plt.plot(t,RMSEa,color='red',label='分析均方根误差')
plt.legend(ncol=2, loc=9,fontsize=16)
plt.text(1,12,'背景误差平均 = %0.2f' %np.mean(RMSEb),fontsize=14)
plt.text(1,6,'分析误差平均 = %0.2f' %np.mean(RMSEa),fontsize=14)
plt.ylabel('均方根误差',fontsize=16)
plt.xlabel('时间（TU）',fontsize=16)
plt.xticks(fontsize=16);plt.yticks(fontsize=16)

#%% 附录9-3 实验5.3 参数有偏差的Lorenz63模式的EAKF参数估计实验
#### 使用有偏差的参数开展实验
sigma = 13     
beta = 3
rho = 30    
####
npara = 3           # 待估参数数目

def hp(x):          # 扩展观测算子
    ne= x.shape[0]  # 输入的x包括状态和参数：ne=n+ns  
    H = np.eye(ne)
    Hs = H[range(n),:]
    yo = Hs @ x     # 投影到状态变量
    return yo

x0b = np.array([2.0,3.0,4.0])   # 同化实验的初值
np.random.seed(seed=1)
xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)    # 不加同化的自由积分结果

sig_b= 0.1
B = sig_b**2*np.eye(n)          # 初始时刻背景误差协方差
Q = 0.0*np.eye(n)               # 模式误差（若假设完美模式则取0）
N = 20 # 集合成员数
xai = np.zeros([3,N])
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)   # 状态初始集合

p0b = np.array([sigma,beta,rho])                        # 参数向量
sig_p = np.array([4,4,4])
pB = np.diag(sig_p)                                     # 参数误差协方差
pai = np.zeros([npara,N])      # 参数集合
for i in range(N):
    pai[:,i] = p0b + np.random.multivariate_normal(np.zeros(npara), pB)
    
Rp = np.diag(np.concatenate([sig_m*np.ones(n),sig_p]))  # 扩展误差协方差

LocM = np.ones([n+npara,n+npara])          #!!! 不采用局地化，把局地化矩阵所有元素设置为1

xa = np.zeros([n,nt+1]); xa[:,0] = x0b   
pa = np.zeros([npara,nt+1]); pa[:,0] = p0b 
km = 0
for k in range(nt):
    for i in range(N):
        xai[:,i] = RK4(Lorenz63,xai[:,i],dt,pai[0,i],pai[1,i],pai[2,i]) \
                  + np.random.multivariate_normal(np.zeros(n), Q)

    xa[:,k+1] = np.mean(xai,1)    # 预报集合平均状态
    pa[:,k+1] = np.mean(pai,1)    # 预报集合平均参数
    if (km<nt_m) and (k+1==ind_m[km]):
        # 扩展向量并进行同化       
        xAi = np.concatenate([xai,pai],axis=0)        
        xAi= sEAKF(xAi,yo[:,km],hp,Rp,LocM)
        # 
        xai = xAi[0:3,:]
        pai = xAi[3:6,:]
        xa[:,k+1] = np.mean(xai,1)
        pa[:,k+1] = np.mean(pai,1)
        km = km+1
#%% 参数估计画图        
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)       
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
plt.plot(t,RMSEb,color='orange',label='背景')
plt.plot(t,RMSEa,color='red',label='分析')
plt.text(2,17,'背景的平均均方根误差 = %0.2f' %np.mean(RMSEb),fontsize=16)
plt.text(2,12,'分析的平均均方根误差 = %0.2f' %np.mean(RMSEa),fontsize=16)
plt.ylabel('均方根误差',fontsize=15)
plt.xlabel('时间',fontsize=15)
plt.legend(ncol=4, loc=9,fontsize=15)
plt.title("EAKF参数估计实验",fontsize=16)
plt.subplot(2,1,2)
plt.plot(t,pa[0,:],'--',label=r'$\sigma$')
plt.plot(t,pa[1,:],'--',label=r'$\beta$')
plt.plot(t,pa[2,:],'--',label=r'$\gamma$')
plt.ylabel('参数值',fontsize=15)
plt.xlabel('时间',fontsize=15)
plt.text(2,12,'最终参数值: %0.2f '%pa[0,-1]+', %0.2f'%pa[1,-1]+', %0.2f'%pa[2,-1],fontsize=14)
plt.yticks([8/3,10,28])
plt.grid(axis='y')
plt.legend(ncol=4, loc=9,fontsize=15)

#%% 实验5.4 参数有偏差的Lorenz63模式的EAKF参数估计实验（加入参数inflation）
x0b = np.array([2.0,3.0,4.0])   # 同化实验的初值
np.random.seed(seed=1)
xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)    # 不加同化的自由积分结果

sig_b= 0.1
B = sig_b**2*np.eye(n)          # 初始时刻背景误差协方差
Q = 0.0*np.eye(n)               # 模式误差（若假设完美模式则取0）
N = 20 # 集合成员数
xai = np.zeros([3,N])
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)   # 状态初始集合

p0b = np.array([sigma,beta,rho])                        # 参数向量
sig_p = np.array([4,4,4])
pB = np.diag(sig_p)                                     # 参数误差协方差
pai = np.zeros([npara,N])      # 参数集合
for i in range(N):
    pai[:,i] = p0b + np.random.multivariate_normal(np.zeros(npara), pB)
    
Rp = np.diag(np.concatenate([sig_m*np.ones(n),sig_p]))  # 扩展误差协方差

LocM = np.ones([n+npara,n+npara])          #!!! 不采用局地化，把局地化矩阵所有元素设置为1

xa = np.zeros([n,nt+1]); xa[:,0] = x0b   
pa = np.zeros([npara,nt+1]); pa[:,0] = p0b 
km = 0
for k in range(nt):
    for i in range(N):
        xai[:,i] = RK4(Lorenz63,xai[:,i],dt,pai[0,i],pai[1,i],pai[2,i]) \
                  + np.random.multivariate_normal(np.zeros(n), Q)

    xa[:,k+1] = np.mean(xai,1)    # 预报集合平均状态
    pa[:,k+1] = np.mean(pai,1)    # 预报集合平均参数
#%% 9-2 参数协方差膨胀部分代码    
    if (km<nt_m) and (k+1==ind_m[km]):
        for i in range(N):        # 参数协方差膨胀
            pai[:,i] = pa[:,k+1]+1.2*(pai[:,i]-pa[:,k+1])
        # 扩展向量并进行同化       
        xAi = np.concatenate([xai,pai],axis=0)        
        xAi= sEAKF(xAi,yo[:,km],hp,Rp,LocM)
        # 
        xai = xAi[0:3,:]
        pai = xAi[3:6,:]
        xa[:,k+1] = np.mean(xai,1)
        pa[:,k+1] = np.mean(pai,1)
        km = km+1
#%% 参数估计画图        
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)       
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
plt.plot(t,RMSEb,color='orange',label='background')
plt.plot(t,RMSEa,color='red',label='analysis')
plt.text(2,17,'背景的平均均方根误差 = %0.2f' %np.mean(RMSEb),fontsize=16)
plt.text(2,12,'分析的平均均方根误差 = %0.2f' %np.mean(RMSEa),fontsize=16)
plt.ylabel('RMSE',fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.legend(ncol=4, loc=9,fontsize=15)
plt.title("EAKF",fontsize=16)
plt.subplot(2,1,2)
plt.plot(t,pa[0,:],'--',label=r'$\sigma$')
plt.plot(t,pa[1,:],'--',label=r'$\beta$')
plt.plot(t,pa[2,:],'--',label=r'$\gamma$')
plt.ylabel('Parameters',fontsize=15)
plt.xlabel('Time',fontsize=15)
plt.text(2,12,'final values: %0.2f '%pa[0,-1]+', %0.2f'%pa[1,-1]+', %0.2f'%pa[2,-1],fontsize=14)
plt.yticks([8/3,10,28])
plt.grid(axis='y')
plt.legend(ncol=4, loc=9,fontsize=15)

#%% 实验5.5 Lorenz63模式有偏参数的参数估计(协方差膨胀和局地化修改)
x0b = np.array([2.0,3.0,4.0])   # 同化实验的初值
np.random.seed(seed=1)
xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)    # 不加同化的自由积分结果

sig_b= 0.1
B = sig_b**2*np.eye(n)          # 初始时刻背景误差协方差
Q = 0.0*np.eye(n)               # 模式误差（若假设完美模式则取0）
N = 20 # 集合成员数
xai = np.zeros([3,N])
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)   # 状态初始集合

p0b = np.array([sigma,beta,rho])                        # 参数向量
sig_p = np.array([4,4,4])
pB = np.diag(sig_p)                                     # 参数误差协方差
pai = np.zeros([npara,N])      # 参数集合
for i in range(N):
    pai[:,i] = p0b + np.random.multivariate_normal(np.zeros(npara), pB)
    
Rp = np.diag(np.concatenate([sig_m*np.ones(n),sig_p]))  # 扩展误差协方差

LocM = np.ones([n+npara,n+npara])          #!!! 不采用局地化，把局地化矩阵所有元素设置为1
LocM[3,2]=0; LocM[2,3]=0

xa = np.zeros([n,nt+1]); xa[:,0] = x0b   
pa = np.zeros([npara,nt+1]); pa[:,0] = p0b 
km = 0
for k in range(nt):
    for i in range(N):
        xai[:,i] = RK4(Lorenz63,xai[:,i],dt,pai[0,i],pai[1,i],pai[2,i]) \
                  + np.random.multivariate_normal(np.zeros(n), Q)

    xa[:,k+1] = np.mean(xai,1)    # 预报集合平均状态
    pa[:,k+1] = np.mean(pai,1)    # 预报集合平均参数
    if (km<nt_m) and (k+1==ind_m[km]):
        for i in range(N):        # 参数协方差膨胀
            pai[:,i] = pa[:,k+1]+1.2*(pai[:,i]-pa[:,k+1])
        # 扩展向量并进行同化       
        xAi = np.concatenate([xai,pai],axis=0)        
        xAi= sEAKF(xAi,yo[:,km],hp,Rp,LocM)
        # 
        xai = xAi[0:3,:]
        pai = xAi[3:6,:]
        xa[:,k+1] = np.mean(xai,1)
        pa[:,k+1] = np.mean(pai,1)
        km = km+1
#%% 参数估计画图        
plt.figure(figsize=(10,5))
plt.subplot(2,1,1)       
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
plt.plot(t,RMSEb,color='orange',label='背景')
plt.plot(t,RMSEa,color='red',label='分析')
plt.text(2,17,'背景的平均均方根误差 = %0.2f' %np.mean(RMSEb),fontsize=16)
plt.text(2,12,'分析的平均均方根误差 = %0.2f' %np.mean(RMSEa),fontsize=16)
plt.ylabel('均方根误差',fontsize=15)
plt.xlabel('时间',fontsize=15)
plt.legend(ncol=4, loc=9,fontsize=15)
plt.title("EAKF参数估计实验",fontsize=16)
plt.subplot(2,1,2)
plt.plot(t,pa[0,:],'--',label=r'$\sigma$')
plt.plot(t,pa[1,:],'--',label=r'$\beta$')
plt.plot(t,pa[2,:],'--',label=r'$\gamma$')
plt.ylabel('参数值',fontsize=15)
plt.xlabel('时间',fontsize=15)
plt.text(2,12,'最终参数值: %0.2f '%pa[0,-1]+', %0.2f'%pa[1,-1]+', %0.2f'%pa[2,-1],fontsize=14)
plt.yticks([8/3,10,28])
plt.grid(axis='y')
plt.legend(ncol=4, loc=9,fontsize=15)