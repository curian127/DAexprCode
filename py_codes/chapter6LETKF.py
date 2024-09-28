#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6. LETKF
@author: gaoyanqiu
"""
#%% 6-5 局地集合转换卡尔曼滤波器（LETKF）算法应用代码
# 导入所需工具包
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import scipy

# Runge-Kutta格式求解Lorenz 63模式dX/dt = f(t,X)
def RK4(rhs,state,dt,*args):
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state   

# Lorenz 63 模式
def Lorenz63(state,*args): 
     # 三个模式参数
    sigma = args[0][0]
    beta = args[0][1]
    rho = args[0][2]
    q=args[1]
    x, y, z = state                     #状态变量分量
    f = np.zeros(3)                    #定义右端项
    # Lorenz 63模式方程
    f[0] = sigma * (y - x)
    f[1] = x * (rho - z) - y
    f[2] = x * y - beta * z
    f=f+q
    return f

def h(u):                            # 观测算子
    H=np.eye(3)
    w=H@u
    return w
# 生成真值
delta_t=0.01                          #积分步长
tm = 40                              #积分时间
nt = int(tm/delta_t)
T= np.linspace(0,tm,nt+1)
# Lorenz63参数设置
sigma = 10.0
beta = 8.0/3.0
rho = 28.0

param=np.zeros(3)
param[0]=sigma
param[1]=beta
param[2]=rho

q0=np.zeros(3)                        #真值
qb=np.random.randn(3)                 #模式误差
x0 = [1.508870,  -1.531271,  25.46091]   #初始条件

X = np.zeros((3,4001))
Xb = np.zeros((3,4001))
X[:,0]=x0
Xb[:,0]=x0
for j in range(np.size(X,1)-1):
    X[:,j+1] = RK4(Lorenz63, X[:,j] ,delta_t ,param,q0)     #模式积分
    Xb[:,j+1] = RK4(Lorenz63, Xb[:,j] ,delta_t ,param,qb)

# 生成观测
Tobs=T[np.arange(25,4025,25)]
Yobs = np.zeros((3,161))
q_obs = np.random.randn(1,161)
Yobs = X [:,0:4025:25]+q_obs

from numpy.matlib import repmat
from scipy.linalg import sqrtm
x = x0 + qb                     #初始值
N = 30                         #集合成员数

q_ensemble=np.sqrt(3)*np.random.randn(3,N)
E = repmat(x,N,1).T+q_ensemble #初始集合：以x为元素，堆叠成1×N的大矩阵
Xa=np.zeros(X.shape)
R = np.eye(3)   #观测误差方差
var_modelerr=0.01
Ide=np.eye(N)
q_pro=np.sqrt(var_modelerr)*np.random.randn(3)

# 循环积分模式
for k in range(4000):
    for j in range(N):
            E[:,j]=RK4(Lorenz63, E[:,j] ,delta_t ,param,q_pro)
    # 如果有观测，进行分析
    if k%25==0:
            y=Yobs[:,round(k/25)]
            xbb= np.nanmean(E,1).reshape(3,1)
            H1=h(E)
            H2=h(xbb)
            Hp=np.zeros((3,N))
            for j in range(N):
                    Hp[:,j]=H1[:,j]-H2.T
            P1=Hp.T @ np.linalg.inv(R) @ Hp + (N-1) * Ide;
            Pa=np.linalg.inv(P1)
            xbp=np.zeros((3,N))
            for j in range(N):
                    xbp[:,j]=E[:,j]-xbb[:,0]
            K=xbp @ Pa @ Hp.T @ np.linalg.inv(R)
            xab=xbb + (K @ (y-H2.T).T).reshape(3,1)
            xap=xbp @ sqrtm((N-1)*Pa)
            for j in range(N):
                    E[:,j]=xab[:,0]+xap[:,j]   #更新状态
    Xa[:,k]=np.nanmean(E,1)    
#计算集合平均，作为分析值：每一行N个计算平均，得到列向量

# 计算均方根误差
RMSEb = np.sqrt(np.mean((Xb-X)**2,0))
RMSEa = np.sqrt(np.mean((Xa-X)**2,0))
mRMSEb = np.mean(RMSEb)
mRMSEa = np.mean(RMSEa)
print('mRMSEb=%.5f'%mRMSEb)
print('mRMSEa=%.5f'%mRMSEa)

#%% 画图展示结果
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.rcParams['axes.unicode_minus']=False    # 用来正常显示负号

fig2 = plt.figure(figsize=(10,11))
ylabel=['x(t)','y(t)','z(t)']

for k in range(3):
    ax = plt.subplot(4,1,k+1)
    ax.plot(T[0:4000],X[k,0:4000], label='真值', linewidth = 3,color='k')
    ax.plot(T[0:4000],Xb[k,0:4000], ':', label='背景', linewidth = 3)
    ax.plot(Tobs,Yobs[k,0:160], fillstyle='none', \
                  label='观测', markersize = 8, markeredgewidth = 2,color='r')
    ax.plot(T,Xa[k,0:4001], label='分析', linewidth = 3,color='g')
    ax.set_ylabel(r'$'+ylabel[k]+'$', labelpad=10, fontsize=16)
    plt.xticks(fontsize=16);plt.yticks(fontsize=16)
    if k==0:
        ax.set_title('LETKF同化结果',fontsize=16)
        ax.legend(loc="center", bbox_to_anchor=(0.5,0.9),ncol =4,fontsize=16)

ax4 = plt.subplot(4,1,4)
ax4.plot(T,RMSEb,':',label='背景')
ax4.plot(T,RMSEa,label='分析',color='g')
ax4.legend(loc="upper right",fontsize=16)
ax4.text(6,35,'背景的平均均方根误差  = %.3f'%mRMSEb,fontsize=16)
ax4.text(6,30,'分析的平均均方根误差  = %.3f'%mRMSEa,fontsize=16)
ax4.set_xlabel('时间',fontsize=16)
ax4.set_ylabel('均方根误差',fontsize=16)
ax4.set_ylim(0,40)
plt.xticks(fontsize=16);plt.yticks(fontsize=16)
plt.tight_layout()

