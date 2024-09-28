#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:23:53 2023

@author: shenzheqi
"""
#%% 10-1 导入必要的库函数
import numpy as np
import scipy as sp
from random import gauss
from random import seed
from pandas import Series
from pandas.plotting import autocorrelation_plot
from matplotlib import pyplot as plt
from importlib import reload
from scipy import stats
import pickle
import warnings
#%%10-2  使用4阶Runge-Kutta方法积分耦合模式的代码和集合调整卡尔曼滤波EAKF的代码
def l63_5v(x, t, params): # 定义模式
    s, k, b, c1, c2, od, om, sm, ss, spd, g, c3, c4, c5, c6 = params
    dx = np.zeros_like(x)
    dx[0] = -s*x[0]+s*x[1]
    dx[1] = (1+c1*x[3])*k*x[0]-x[1]-x[0]*x[2]
    dx[2] = x[0]*x[1]-b*x[2]
    dx[3] = (c2*x[1]+c3*x[4]+c4*x[3]*x[4]-od*x[3]+sm+ss*np.cos(2*np.pi*t/spd))/om
    dx[4] = (c5*x[3]+c6*x[3]*x[4]-od*x[4])/g
    return dx

def l63_5v_rk4(x, t, params, dt):   # 用RK45解方程
    dx1 = l63_5v(x, t, params)
    Rx2 = x+.5*dt*dx1
    dx2 = l63_5v(Rx2, t, params)
    Rx3 = x+.5*dt*dx2
    dx3 = l63_5v(Rx3, t, params)
    Rx4 = x+dt*dx3
    dx4 = l63_5v(Rx4, t, params)
    return (dx1 + 2*dx2 + 2*dx3 + dx4)/6  # 返回增量的斜率

def eakf(obs, obs_var, prior_var, prior_mean, ens ): 
    # 计算后验量
    post_var = 1.0 / (1.0 / prior_var + 1.0 / obs_var)
    post_mean = post_var * (prior_mean / prior_var + obs / obs_var)
    var_ratio = post_var / (prior_var)
    a = np.sqrt(var_ratio)  # 放缩比例，需要排除prior_var为0导致的nan值
    if (type(a)==np.float64):
        obs_inc = a * (ens - post_mean) + post_mean - ens    # 增量
    else:
        obs_inc = np.zeros_like(ens)                         # nan值就不同化该观测
    return(obs_inc)         # eakf函数只输出观测增量
#%% 10-3 设置模式参数、积分“观测系统模拟试验”中的真实场和控制试验、并生成“观测”
sigma=9.95; kappa=28.; beta=8/3
c1=0.1;c2=1.
Od=1.;Om=10.
Sm=10.;Ss=1.;Spd=10.
Gamma=100.
c3=.01;c4=.01;c5=1.;c6=.001
params = [sigma, kappa, beta, c1, c2, Od, Om, Sm, Ss, Spd, Gamma, c3, c4, c5, c6]
dt = .01                          # 步长 （无量纲化时间单位）
Ntime = 4000.                     # 总时间（无量纲化时间单位）
nt = len(np.arange(0,Ntime,dt))   # 步数

x = np.nan*np.zeros((5, nt+1))    # x真值状态，预分配空间
x0 = [1, 1, 1, 0, 0]              # 初始状态
x[:,0] = x0                       # 初始化
time = 0
for ti in range(nt):
    time += dt                    # 时间步进
    dx = l63_5v_rk4(x[:,ti], time, params, dt)  # RK4返回斜率
    x[:,ti+1] = x[:,ti] + dx*dt   # 乘dt并加到前一状态中              

nt = 10000    # 在此我们将仅保留最后10000步的结果用于同化实验，前面的作为spinup
nens = 50     # 集合成员数
ens = np.zeros((5, nt, nens))    # 状态变量集合（变量数，时间，成员数）
for i in range(nens):
    ens[:,0,i] = x[:,-nt-i*2]    # 使用倒数nt步前最后的nens步的状态构成集合
# 以下是控制实验，即不进行同化的自由积分
for ensi in range(nens):
    time = 4000-nt*dt    # 模式积分需要输入时间，这是是对应真值的时间
    for ti in range(nt-1):
        time += dt
        dx = l63_5v_rk4(ens[:,ti,ensi], time, params, dt)
        ens[:,ti+1,ensi] = ens[:,ti,ensi] + dx*dt
# 自由积分，没有同化的结果
# 以下造观测数据用于同化
x = x[:,-nt:]     # 真实场仅保留最后nt步的结果，用于造观测进行实验
obs = np.copy(x)  # 造观测数据（变量，时间）
obs_std = [3., 3., 3., 1., .1]  # 观测误差
np.random.seed(0)
for i in range(5):
    obs[i,:] = x[i,:] + np.random.normal(0, obs_std[i], x.shape[1])
    # 假设全部观测，即观测算子为H=I
#%% 10-4 弱耦合同化
da_window = 50        # 这里我们统一取50的同化时间窗口，即每50步同化一次
time = Ntime-nt*dt    # 模式积分需要输入时间，这是是对应真值的时间     
da_uc = np.copy(ens)  # 弱耦合同化结果（变量数，时间，成员数）
for ti in range(nt-1): 
    if ti%da_window == 0:   # 整除判断 
        inf_factor = 1.02   # inflation factor
        # 只同化第二个变量，只inflate第二个变量
        prior_var = np.var(da_uc[1,ti,:])
        prior_mean = np.mean(da_uc[1,ti,:])
        da_uc[1,ti:] = prior_mean+inf_factor*(da_uc[1,ti,:]-prior_mean)     # 增大离散度，实现inflation
        obs_inc = eakf(obs[1,ti],obs_std[1]**2,prior_var,prior_mean,da_uc[1,ti,:])
        k0 = np.cov(da_uc[0,ti,:],da_uc[1,ti,:])[0,1]/np.cov(da_uc[0,ti,:],da_uc[1,ti,:])[1,1]
        k2 = np.cov(da_uc[2,ti,:],da_uc[1,ti,:])[0,1]/np.cov(da_uc[2,ti,:],da_uc[1,ti,:])[1,1]
        da_uc[0,ti,:] += k0*obs_inc
        da_uc[1,ti,:] += obs_inc
        da_uc[2,ti,:] += k2*obs_inc
    for n in range(nens):
        dx = l63_5v_rk4(da_uc[:,ti,n], time, params, dt)
        da_uc[:,ti+1,n] = da_uc[:,ti,n] + dx*dt
    time +=dt
#%%10-5 强耦合同化试验
da_window = 50 # 同上，这里我们统一取50的同化时间窗口，即每50步同化一次
time = 3900
da_sc = np.copy(ens)
for ti in range(nt-1):
    if ti%da_window == 0: 
        inf_factor = 1.05 # inflation
        prior_var = np.var(da_sc[1,ti,:])
        prior_mean = np.mean(da_sc[1,ti,:])
        da_sc[1,ti:] = prior_mean+inf_factor*(da_sc[1,ti,:]-prior_mean)
        k = np.cov(da_sc[3,ti,:],da_sc[1,ti,:])[0,1]/np.cov(da_sc[3,ti,:],da_sc[1,ti,:])[1,1]
        k0 = np.cov(da_sc[0,ti,:],da_sc[1,ti,:])[0,1]/np.cov(da_sc[0,ti,:],da_sc[1,ti,:])[1,1]
        k2 = np.cov(da_sc[2,ti,:],da_sc[1,ti,:])[0,1]/np.cov(da_sc[2,ti,:],da_sc[1,ti,:])[1,1]
        obs_inc = eakf(obs[1,ti],obs_std[1]**2,prior_var,prior_mean,da_sc[1,ti,:])
        da_sc[3,ti,:] += obs_inc*k
        da_sc[0,ti,:] += obs_inc*k0
        da_sc[2,ti,:] += obs_inc*k2
        da_sc[1,ti,:] += obs_inc    
    for n in range(nens):
        dx = l63_5v_rk4(da_sc[:,ti,n], time, params, dt)
        da_sc[:,ti+1,n] = da_sc[:,ti,n] + dx*dt
    time +=dt
#%% 10-6 图10-1的绘图部分代码
# 首先计算RMSE
RMSE_sc = np.nan*np.zeros((5,10000))
for i in range(5):
    for j in range(10000):
        RMSE_sc[i,j] = np.sqrt(np.mean((da_sc[i,j,:]-x[i, j])**2, 0))
RMSE_ens = np.nan*np.zeros((5,10000))
for i in range(5):
    for j in range(10000):
        RMSE_ens[i,j] = np.sqrt(np.mean((ens[i,j,:]-x[i, j])**2, 0))
RMSE_uc = np.nan*np.zeros((5,10000))
for i in range(5):
    for j in range(10000):
        RMSE_uc[i,j] = np.sqrt(np.mean((da_uc[i,j,:]-x[i, j])**2, 0))        

#%% 绘图
plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
plt.plot(np.nanmean(da_sc[3,9000:10000,:],-1),label='Strongly Coupled', lw=2, color='C0')
plt.plot(np.nanmean(da_uc[3,9000:10000,:],-1),label='Weakly Coupled', lw=2, color='C1')
plt.plot(np.nanmean(ens[3,9000:10000,:],-1),label='Control', lw=2, color='C2')
plt.plot(x[3,9000:10000],label='True state', lw=2, color='C3')
plt.gca().set(xticks=np.arange(0,1001,200), xticklabels=[])
plt.gca().set(xlabel='',ylabel='Omega')
plt.legend(loc='upper left')
plt.subplot(2,2,3)
plt.plot(np.nanmean(da_sc[1,9000:10000,:],-1),label='Strongly Coupled', lw=2, color='C0')
plt.plot(np.nanmean(da_uc[1,9000:10000,:],-1),label='Weakly Coupled', lw=2, color='C1')
plt.plot(np.nanmean(ens[1,9000:10000,:],-1),label='Control', lw=2, color='C2')
plt.plot(x[1,9000:10000],label='True state', lw=2, color='C3')
plt.gca().set(xticks=np.arange(0,1001,200), xticklabels=np.arange(9000,10001,200)*.01)
plt.gca().set(xlabel='t',ylabel='y')

plt.subplot(2,2,2)
plt.plot(RMSE_sc[3,9000:10000],label='Strongly Coupled', lw=2, color='C0')
plt.plot(RMSE_uc[3,9000:10000],label='Weakly Coupled', lw=2, color='C1')
plt.plot(RMSE_ens[3,9000:10000],label='Control', lw=2, color='C2')
plt.gca().set(xticks=np.arange(0,1001,200), xticklabels=[])
plt.gca().set(xlabel='',ylabel='RMSE')
plt.subplot(2,2,4)
plt.plot(RMSE_sc[1,9000:10000],label='Strongly Coupled', lw=2, color='C0')
plt.plot(RMSE_uc[1,9000:10000],label='Weakly Coupled', lw=2, color='C1')
plt.plot(RMSE_ens[1,9000:10000],label='Control', lw=2, color='C2')
plt.gca().set(xticks=np.arange(0,1001,200), xticklabels=np.arange(9000,10001,200)*.01)
plt.gca().set(xlabel='t',ylabel='RMSE')

# plt.savefig('fig1.pdf', bbox_inches='tight')
# plt.savefig('fig1.png', bbox_inches='tight', dpi=300)
#%% 10-8 LACC试验 
time = 3900
da_sc_avg = np.copy(ens)

for ti in range(1,nt-1):
  if ti%da_window == 0:   # da_window与前面相同，不做修改
    inf_factor = 1.05 # inflation
    if ti>=1000:  # 当ti大于1000时，对ti-1000至ti间的y变量取平均
      xx = np.mean(da_sc_avg[1,ti-1000:ti],0) #平均的模式变量
      yy = np.mean(obs[1,ti-1000:ti]) # 平均的观测
    else: # 否则只对0至ti间的y变量取平均
      xx = np.mean(da_sc_avg[1,:ti],0)
      yy = np.mean(obs[1,:ti])
    prior_var = np.var(da_sc_avg[1,ti,:])
    prior_mean = np.mean(da_sc_avg[1,ti,:])
    da_sc_avg[1,ti,:] = prior_mean+inf_factor*(da_sc_avg[1,ti,:]-prior_mean)
    k0 = np.cov(da_sc_avg[0,ti,:],da_sc_avg[1,ti,:])[0,1]/np.cov(da_sc_avg[0,ti,:],da_sc_avg[1,ti,:])[1,1]
    k2 = np.cov(da_sc_avg[2,ti,:],da_sc_avg[1,ti,:])[0,1]/np.cov(da_sc_avg[2,ti,:],da_sc_avg[1,ti,:])[1,1]
    obs_inc = eakf(obs[1,ti],obs_std[1]**2,prior_var,prior_mean,da_sc_avg[1,ti,:])
    da_sc_avg[0,ti,:] += obs_inc*k0
    da_sc_avg[2,ti,:] += obs_inc*k2
    
    k = np.cov(da_sc_avg[3,ti,:],xx)[0,1]/np.cov(da_sc_avg[3,ti,:],xx)[1,1] # 计算平均的y变量与o变量的协方差
    prior_var = np.var(xx) # 平均的y变量的先验variance
    prior_mean = np.mean(xx) # 平均的y变量的先验mean
    tmp = eakf(yy, 1**2, prior_var, prior_mean, xx) # # 平均的y变量的increment
    da_sc_avg[3,ti,:] += tmp*k
    
    da_sc_avg[1,ti,:] += obs_inc
  for n in range(nens):
    dx = l63_5v_rk4(da_sc_avg[:,ti,n], time, params, dt)
    da_sc_avg[:,ti+1,n] = da_sc_avg[:,ti,n] + dx*dt
  time +=dt

# 计算RMSE
RMSE_sc_avg = np.nan*np.zeros((5,nt))
for i in range(5):
  for j in range(nt):
    RMSE_sc_avg[i,j] = np.sqrt(np.mean((da_sc_avg[i,j,:]-x[i, j])**2, 0))

#%% 绘图
plt.figure(figsize=(8,4))
plt.subplot(2,2,1)
plt.plot(np.nanmean(da_sc_avg[3,-1000:,:],-1),label='LACC', lw=2, color='C0')
plt.plot(np.nanmean(da_sc[3,-1000:,:],-1),label='SC', lw=2, color='C1')
# plt.plot(np.nanmean(ens[3,-1000:,:],-1),label='Control', lw=2, color='C2')
plt.plot(x[3,-1000:],label='Truth', lw=2, color='C3')
plt.gca().set(xticks=np.arange(0,1001,200), xticklabels=[])
plt.gca().set(xlabel='',ylabel='Omega')
plt.legend(loc='upper left')
plt.subplot(2,2,3)
plt.plot(np.nanmean(da_sc_avg[1,-1000:,:],-1),label='LACC', lw=2, color='C0')
plt.plot(np.nanmean(da_sc[1,-1000:,:],-1),label='SC', lw=2, color='C1')
# plt.plot(np.nanmean(ens[1,-1000:,:],-1),label='Control', lw=2, color='C2')
plt.plot(x[1,-1000:],label='True state', lw=2, color='C3')
plt.gca().set(xticks=np.arange(0,1001,200), xticklabels=np.arange(900,1001,20)*.1)
plt.gca().set(xlabel='t',ylabel='y')

plt.subplot(2,2,2)
plt.plot(RMSE_sc_avg[3,-1000:],label='LACC', lw=2, color='C0')
plt.plot(RMSE_sc[3,-1000:],label='SC', lw=2, color='C1')
# plt.plot(RMSE_ens[3,-1000:],label='Control', lw=2, color='C2')
plt.gca().set(xticks=np.arange(0,1001,200), xticklabels=[])
plt.gca().set(xlabel='',ylabel='RMSE')
plt.subplot(2,2,4)
plt.plot(RMSE_sc_avg[1,-1000:],label='LACC', lw=2, color='C0')
plt.plot(RMSE_sc[1,-1000:],label='SC', lw=2, color='C1')
# plt.plot(RMSE_ens[1,-1000:],label='Control', lw=2, color='C2')
plt.gca().set(xticks=np.arange(0,1001,200), xticklabels=np.arange(900,1001,20)*.1)
plt.gca().set(xlabel='t',ylabel='RMSE')
