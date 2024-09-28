#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lorenz63 模式 EKF同化
@author: shenzheqi
"""
#%% 3-1 使用Runge-Kutta格式积分Lorenz63模式的代码
import numpy as np            # 导入numpy工具包
def Lorenz63(state,*args):       # 此函数定义Lorenz63模式右端项
    sigma = args[0]
    beta = args[1]
    rho = args[2]              # 输入σ,β和ρ三个模式参数
    x, y, z = state              # 输入矢量的三个分量分别为方程式中的x,y,z
    f = np.zeros(3)             # f定义为右端
    f[0] = sigma * (y - x)        # （57）
    f[1] = x * (rho - z) - y       # （58）
    f[2] = x * y - beta * z        # （59）
    return f 
def RK4(rhs,state,dt,*args):      # 此函数提供Runge-Kutta积分格式
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state          
# Runge-Kutta法参考余德浩等《微分方程数值解法》（科学出版社）

## 以下代码仅用于展示如何调用模式积分，并画图展示模式自由积分特性
sigma = 10.0; beta = 8.0/3.0; rho = 28.0         # 模式参数值   
dt = 0.01                                   # 模式积分步长
x0True = np.array([1,1,1])                     # 模式积分的初值
xTrue = np.zeros([3,5001])                    # 模式积分值
xTrue[:,0] = x0True                           # 设置积分初值
for k in range(5000):
    xTrue[:,k+1] = RK4(Lorenz63,xTrue[:,k],dt,sigma,beta,rho)  # 模式积分
import matplotlib.pyplot as plt                 # 调用画图包
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.plot3D(xTrue[0],xTrue[1],xTrue[2])            # 三维画图并设置坐标
ax.set_xlabel('x',fontsize=16)
ax.set_ylabel('y',fontsize=16)
ax.set_zlabel('z',fontsize=16)
plt.xticks(fontsize=16);plt.yticks(fontsize=16)
ax.set_zticks(np.arange(0,50,10));ax.set_zticklabels(np.arange(0,50,10),fontsize=16)
plt.show()

#%% 3-2 Lorenz63模式真值试验和观测构造
sigma = 10.0; beta = 8.0/3.0; rho = 28.0         # 模式参数值   
dt = 0.01                                   # 模式积分步长
n = 3                                      # 状态维数
m = 3                                      # 观测数
tm = 10                                    # 同化试验窗口
nt = int(tm/dt)                               # 总积分步数
t = np.linspace(0,tm,nt+1)                     # 模式时间网格
x0True = np.array([1,1,1])                     # 真实值的初值
np.random.seed(seed=1)                     # 设置随机种子
sig_m= 0.15                                # 观测误差标准差
R = sig_m**2*np.eye(n)                      # 观测误差协方差矩阵
dt_m = 0.2                             # 观测之间的时间间隔（可见为20模式步）
tm_m = 10                             # 最大观测时间（可小于模式积分时间）
nt_m = int(tm_m/dt_m)                  # 进行同化的总次数
ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)  
# 观测网格在时间网格中的指标
t_m = t[ind_m]              # 观测网格
def h(x):                    # 定义观测算子
    H = np.eye(n)           # 观测矩阵为单位阵
    yo = H@x              # 单位阵乘以状态变量
    return yo
xTrue = np.zeros([n,nt+1])     # 真值保存在xTrue变量中
xTrue[:,0] = x0True           # 初始化真值
km = 0                     # 观测计数
yo = np.zeros([3,nt_m])       # 观测保存在yo变量中
for k in range(nt):            # 按模式时间网格开展模式积分循环
    xTrue[:,k+1] = RK4(Lorenz63,xTrue[:,k],dt,sigma,beta,rho)   # 真实值积分
    if (km<nt_m) and (k+1==ind_m[km]):     # 用指标判断是否进行观测
        yo[:,km] = h(xTrue[:,k+1]) + np.random.normal(0,sig_m,[3,]) #采样造观测
        km = km+1                       # 观测计数
## 以下提供真值和观测画图的参考脚本
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.figure(figsize=(10,6))
lbs = ['x','y','z']
for j in range(3):
    plt.subplot(3,1,j+1)
    plt.plot(t,xTrue[j],'b-',lw=2,label='真值')
    plt.plot(t_m,yo[j],'go',ms=8,markerfacecolor='white',label='观测')
    plt.ylabel(lbs[j],fontsize=16)
    plt.xticks(fontsize=16);plt.yticks(fontsize=16)
    if j==0:
        plt.legend(ncol=4, loc=9,fontsize=16)
        plt.title('L63模式观测模拟',fontsize=16)
    if j==2:
        plt.xlabel('时间（TU）',fontsize=16)
plt.show()
#%% 3-3 Lorenz63模式的切线性模式
def JLorenz63(state,*args):         # Lorenz63 方程雅克比矩阵
    sigma = args[0]
    beta = args[1]
    rho = args[2]
    x, y, z = state
    
    df = np.zeros([3,3])           # 以下是切线性矩阵的9个元素配置
    df[0,0] = sigma * (-1)
    df[0,1] = sigma * (1)
    df[0,2] = sigma * (0)
    df[1,0] = 1 * (rho - z) 
    df[1,1] = -1
    df[1,2] = x * (-1)
    df[2,0] = 1 * y 
    df[2,1] = x * 1 
    df[2,2] = - beta
    return df 

def JRK4(rhs,Jrhs,state,dt,*args):   # 切线性模式的积分格式    
    n = len(state)
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    # 以下是对矩阵的Runge-Kutta格式
    dk1 = Jrhs(state,*args)
    dk2 = Jrhs(state+k1*dt/2,*args) @ (np.eye(n)+dk1*dt/2) 
    dk3 = Jrhs(state+k2*dt/2,*args) @ (np.eye(n)+dk2*dt/2) 
    dk4 = Jrhs(state+k3*dt,*args) @ (np.eye(n)+dk3*dt)  
    DM = np.eye(n) + (dt/6) * (dk1+2*dk2+2*dk3+dk4)
    return DM
#%% 3-4线性观测矩阵
def Dh(x):                  # 观测算子的线性观测矩阵
    n = len(x)
    H = np.eye(n)
    return H
#%% 3-5 扩展卡尔曼滤波器的分析算法
def EKF(xb,yo,ObsOp,JObsOp,R,B):         
# 输入的变量分别为：xb预报、yo观测、ObsOp观测算子、JObsOp切线性观测算子，R观测误差协方差，B背景误差协方差。
    n = xb.shape[0]     # 状态空间维数
    Dh = JObsOp(xb)    # 计算线性观测矩阵
    D = Dh@B@Dh.T + R 
    K = B @ Dh.T @ np.linalg.inv(D)      # 卡尔曼增益矩阵
    xa = xb + K @ (yo-ObsOp(xb))        # 更新状态
    P = (np.eye(n) - K@Dh) @ B          # 更新误差协方差矩阵
    return xa, P                         # 输出分析状态场和分析误差协方差矩阵
#%% 3-6 Lorenz63模式中的EKF同化试验
x0b = np.array([2.0,3.0,4.0])           # 同化试验的初值
np.random.seed(seed=1)             # 设置随机种子
xb = np.zeros([3,nt+1]); xb[:,0] = x0b   # 控制试验结果存在xb中
for k in range(nt):                    # 模式积分循环
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)   # 不加同化的自由积分结果
sig_b= 0.1                             # 设定初始的背景误差
B = sig_b**2*np.eye(3)                  # 初始背景误差协方差矩阵
Q = 0.0*np.eye(3)                    # 设置模式误差（若假设完美模式则取0）
xa = np.zeros([3,nt+1]); xa[:,0] = x0b     # 同化试验结果存在xa中
km = 0                              # 同化次数计数
for k in range(nt):                     # 模式积分循环
    xa[:,k+1] = RK4(Lorenz63,xa[:,k],dt,sigma,beta,rho)         # 用非线性模式积分
    DM = JRK4(Lorenz63,JLorenz63,xa[:,k],dt,sigma,beta,rho)  # 使用切线性模式积分
    B = DM @ B @ DM.T + Q                           # 积分过程协方差更新
    if (km<nt_m) and (k+1==ind_m[km]):   # 当有观测时，使用EKF同化
        xa[:,k+1],B = EKF(xa[:,k+1],yo[:,km],h,Dh,R,B)  #调用EKF，更新状态和协方差
        km = km+1
# EKF结果画图
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
        plt.title("EKF同化实验",fontsize=16)
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
plt.subplot(4,1,4)
plt.plot(t,RMSEb,color='orange',label='背景均方根误差')
plt.plot(t,RMSEa,color='red',label='分析均方根误差')
plt.legend(ncol=2, loc=9,fontsize=16)
plt.text(1,9,'背景误差平均 = %0.2f' %np.mean(RMSEb),fontsize=14)
plt.text(1,4,'分析误差平均 = %0.2f' %np.mean(RMSEa),fontsize=14)
plt.ylabel('均方根误差',fontsize=16)
plt.xlabel('时间（TU）',fontsize=16)
plt.xticks(fontsize=16);plt.yticks(fontsize=16)
plt.show()
