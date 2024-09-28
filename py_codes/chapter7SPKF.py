#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
7 Sigma-Point Kalam filter
@author: Shenzheqi $ Xiaoyao
"""
#%% 7-1 Lorenz63模式代码和孪生试验的观测模拟过程（同第三章）
import numpy as np                           # 导入numpy工具包
def Lorenz63(state,*args):                   # 此函数定义Lorenz63模式右端项
    sigma = args[0]
    beta = args[1]
    rho = args[2]                                  # 输入σ,β和ρ三个模式参数
    x, y, z = state                                  # 输入矢量的三个分量分别为方程式中的x,y,z
    f = np.zeros(3)                                # f定义为右端
    f[0] = sigma * (y - x)                       # （57）
    f[1] = x * (rho - z) - y                      # （58）
    f[2] = x * y - beta * z                       # （59）
    return f 
def RK4(rhs,state,dt,*args):                  # 此函数提供Runge-Kutta积分格式
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state

# 以下代码构造孪生试验的观测真实解和观测数据          
sigma = 10.0; beta = 8.0/3.0; rho = 28.0         # 模式参数值   
dt = 0.01                                                         # 模式积分步长
n = 3                                                               # 状态维数
m = 3                                                              # 观测数
tm = 10                                                           # 同化试验窗口
nt = int(tm/dt)                                                 # 总积分步数
t = np.linspace(0,tm,nt+1)                               # 模式时间网格

x0True = np.array([1,1,1])                                # 真实值的初值
np.random.seed(seed=1)                                # 设置随机种子
sig_m= 0.15                                                    # 观测误差标准差
R = sig_m**2*np.eye(n)                                  # 观测误差协方差矩阵

dt_m = 0.2                                                      # 观测之间的时间间隔（可见为20模式步）
tm_m = 10                                                      # 最大观测时间（可小于模式积分时间）
nt_m = int(tm_m/dt_m)                                  # 进行同化的总次数

ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)  
                       # 观测网格在时间网格中的指标
t_m = t[ind_m]                                               # 观测网格
def h(x):                                                         # 定义观测算子
    H = np.eye(n)                                            # 观测矩阵为单位阵
    yo = H@x                                                 # 单位阵乘以状态变量
    return yo
def Dh(x):                                                      # 观测算子的线性观测矩阵
    n = len(x)
    D = np.eye(n)
    return D
xTrue = np.zeros([n,nt+1])                              # 真值保存在xTrue变量中
xTrue[:,0] = x0True                                          # 初始化真值
km = 0                                                            # 观测计数
yo = np.zeros([3,nt_m])                                   # 观测保存在yo变量中
for k in range(nt):                                            # 按模式时间网格开展模式积分循环
    xTrue[:,k+1] = RK4(Lorenz63,xTrue[:,k],dt,sigma,beta,rho)                      # 真实值积分
    if (km<nt_m) and (k+1==ind_m[km]):         # 用指标判断是否进行观测
        yo[:,km] = h(xTrue[:,k+1]) + np.random.normal(0,sig_m,[3,])              #采样生成观测
        km = km+1                                               # 观测计数

#%%  7-2 SP-UKF分析算法
def generate_SigmaP(xb,B,Q,R):                      # 生成sigma点，构建集合
    import scipy                                               #导入scipy工具包
    n = xb.shape[0]                                           # n-状态维数                
    m = R.shape[0]                                           # m-观测误差维数                 
    L = 2*n+m;                                                  # L-离散空间状态向量维数               
    kappa=0;alpha=1;beta0=2                          # 确定UKF的参数κ、α、β
    lam = alpha**2*(L+kappa)-L  
    wm = 0.5/(L+lam)*np.ones(2*L+1)               # 计算sigam点权重
    wm[0] = lam/(L+lam)
    wc = 0.5/(L+lam)*np.ones(2*L+1)                 # 计算sigam点权重
    wc[0] = lam/(L+lam)+(1-alpha**2+beta0)
    
    theta = np.concatenate([xb,np.zeros(n+m)])  # 扩充状态向量
    Pa = scipy.linalg.block_diag(B,Q,R)                 # 计算背景误差协方差
    sqP=np.linalg.cholesky(Pa)
    SigmaP = np.zeros([L,2*L+1])
    SigmaP[:,0] = theta                                          # 生成sigma点
    SigmaP[:,1:(L+1)] = theta.reshape(-1, 1) + np.sqrt(L+lam)*sqP
    SigmaP[:,(L+1):(2*L+1)] = theta.reshape(-1, 1) - np.sqrt(L+lam)*sqP   
    xbi = SigmaP[0:n,:]; vi = SigmaP[n:2*n,:]; ni = SigmaP[2*n::,:]
    return xbi,vi,ni,wm,wc

def update_SigmaP(xbi,wm,wc,yo,ObsOp,ni):
    n,N = xbi.shape                                                # n-状态维数，N-集合成员数
    m = yo.shape[0]                                                 # m-观测维数
    ybi = np.zeros([m,N])                                        # 预分配空间，保存扰动后的观测集合
    for i in range(N):                                               # 将状态集合投影道观测空间，构成观测集合
           ybi[:,i] = ObsOp(xbi[:,i])+ni[:,i]
    xbm = np.sum(xbi*wm,1)                                   # 利用sigma点权重计算集合平均
    ybm = np.sum(ybi*wm,1)
    Pxx = (xbi-xbm.reshape(-1,1))*wc@(xbi-xbm.reshape(-1,1)).T # 计算需要的协方差矩阵
    Pyy = (ybi-ybm.reshape(-1,1))*wc@(ybi-ybm.reshape(-1,1)).T
    Pxy = (xbi-xbm.reshape(-1,1))*wc@(ybi-ybm.reshape(-1,1)).T
        
    K = Pxy @ np.linalg.inv(Pyy)                               #计算卡尔曼增益矩阵
    xa = xbm + K @ (yo-ybm)                                 # 更新状态变量
    B = Pxx-K @ Pyy @K.T                                      # 计算下一个同化循环需要的背景误差协方差
    return xa,B
#%% 7-3 SP-UKF同化试验及结果
n = 3                                                                  # 状态维数
m = 3                                                                # 观测数	
x0b = np.array([2.0,3.0,4.0])                               # 同化试验的初值
np.random.seed(seed=1)                                  # 初始化随机种子，便于重复结果

xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):                                              # xb得到的是不加同化的自由积分结果
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)    

sig_b= 0.1
B = sig_b**2*np.eye(n)                                      # 初始时刻背景误差协方差，设为对角阵
Q = 0.1*np.eye(n)                                             # 模式误差（若假设完美模式则取0）

xa = np.zeros([n,nt+1]); xa[:,0] = x0b                #保存每步的集合均值作为分析场，存在xa
km = 0                                                              # 对同化次数进行计数

xbi,vi,ni,wm,wc = generate_SigmaP(xa[:,0], B, Q, R)     #根据初始条件生成sigma点构成集合
n,N = xbi.shape                                                            # N集合成员数
for k in range(nt):                                                         # 时间积分    
    for i in range(N):                                                      # 对每个集合成员积分
        xbi[:,i] = RK4(Lorenz63,xbi[:,i],dt,sigma,beta,rho) \
                 + np.random.multivariate_normal(np.zeros(n), Q)      # 积分每个集合成员得到预报集合
    xa[:,k] = np.sum(xbi*wm,1)                                   # 非同化时刻使用预报平均，同化时刻分析平均
    if (km<nt_m) and (k+1==ind_m[km]):                  # 当有观测时，使用SP-UKF进行更新
        xbi = xbi + vi                                                    # 在集合成员中加入背景误差
        xa[:,k+1],B = update_SigmaP(xbi, wm, wc, yo[:,km], h, ni)     # 调用SP-UKF同化
        xbi,vi,ni,wm,wc = generate_SigmaP(xa[:,k+1], B, Q, R)          # 为下一个同化循环生成集合成员
        km = km+1

# UKF结果画图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
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
        plt.title("SP-UKF同化实验",fontsize=16)
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

xa1 = xa

#%% 7-4 SP-CDKF分析算法
def time_generate_SigmaP(xb,B,Q):              # 生成时间积分步的sigma点
    import scipy                                 # 导入scipy工具包
    delta=np.sqrt(3)                            # 确定中心差分步长h
    n = xb.shape[0]                             # n-状态维数
    m = Q.shape[0]                             # m-背景误差维数
    Lx = n
    Lv = m
    L=Lx+Lv                                                     # L-离散状态空间向量维数
    wm = (1/(2*delta**2))*np.ones(2*L+1)       # 计算sigma点权重
    wm[0] = (delta**2-Lx-Lv)/delta**2
    wc1 = (1/(4*delta**2))*np.ones(2*L+1)
    wc2 = ((delta**2-1)/(4*delta**4))*np.ones(2*L+1)
    
    theta = np.concatenate([xb,np.zeros(m)])     # 扩充状态向量
    Pa = scipy.linalg.block_diag(B,Q)                  # 计算协方差矩阵
    sqP=np.linalg.cholesky(Pa)                 
    
    SigmaP = np.zeros([L,2*L+1])                      # 生成sigma点
    SigmaP[:,0] = theta
    SigmaP[:,1:(L+1)] = theta.reshape(-1, 1) + delta*sqP
    SigmaP[:,(L+1):(2*L+1)] = theta.reshape(-1, 1) - delta*sqP
    xbi = SigmaP[0:n,:]; vi = SigmaP[n:n+m,:];
    return xbi,vi,wm,wc1,wc2

def measurement_generate_SigmaP(xb,B,R):        #生成观测更新步的sigma点
    import scipy          
    delta=np.sqrt(3)                                              #确定中心差分步长h
    n = xb.shape[0]                                               #确定状态维数
    m = R.shape[0]
    Lx = n
    Lr = m
    L=Lx+Lr
    wm = (1/(2*delta**2))*np.ones(2*L+1)           #计算sigma点权重 
    wm[0] = (delta**2-Lx-Lr)/delta**2
    wc1 = (1/(4*delta**2))*np.ones(2*L+1)
    wc2 = ((delta**2-1)/(4*delta**4))*np.ones(2*L+1)
    theta = np.concatenate([xb,np.zeros(m)])        #扩充状态向量
    Pa = scipy.linalg.block_diag(B,R)                      #计算协方差矩阵
    sqP=np.linalg.cholesky(Pa)        
    SigmaP = np.zeros([L,2*L+1])                          #生成sigma点
    SigmaP[:,0] = theta
    SigmaP[:,1:(L+1)] = theta.reshape(-1, 1) + delta*sqP
    SigmaP[:,(L+1):(2*L+1)] = theta.reshape(-1, 1) - delta*sqP
    xbi = SigmaP[0:n,:] ; ni = SigmaP[n:n+m,:]
    return xbi,ni,wm,wc1,wc2

def time_update_SigmaP(xbi,wm,wc1,wc2):
    n,N = xbi.shape                             
    xbm = np.sum(xbi*wm,1)
    L=2*n
    Pxx = (xbi[:,1:L+1]-xbi[:,L+1:2*L+1])*wc1[1:L+1]@((xbi[:,1:L+1]-xbi[:,L+1:2*L+1])).T+\
          (xbi[:,1:L+1]+xbi[:,L+1:2*L+1]-2*xbi[:,0].reshape(-1,1))*wc2[1:L+1]@((xbi[:,1:L+1]+\
xbi[:,L+1:2*L+1]-2*xbi[:,0].reshape(-1,1))).T              #计算背景误差协方差
    return xbm, Pxx
        
def measurement_update_SigmaP(xbi,wm,wc1,wc2,yo,ObsOp,ni,xbm, Pxx ):
    n,N = xbi.shape
    m = yo.shape[0]
    L=n+m
    ybi = np.zeros([m,N])
    for i in range(N):
        ybi[:,i] = ObsOp(xbi[:,i])+ni[:,i]            # 将状态集合投影道观测空间，构成观测集合
    xbm = np.sum(xbi*wm,1)                      # 利用sigma点权重计算集合平均
    ybm = np.sum(ybi*wm,1)  
    Pyy = (ybi[:,1:L+1]-ybi[:,L+1:2*L+1])*wc1[1:L+1]@((ybi[:,1:L+1]-ybi[:,L+1:2*L+1])).T+\
          (ybi[:,1:L+1]+ybi[:,L+1:2*L+1]-2*ybi[:,0].reshape(-1,1))*wc2[1:L+1]@((ybi[:,1:L+1]+\
ybi[:,L+1:2*L+1]-2*ybi[:,0].reshape(-1,1))).T             # 计算观测误差协方差矩阵      
    
    AA=(xbi[:,1:L+1]-xbi[:,L+1:2*L+1])*(wc1[1:L+1])              #计算协方差矩阵Pxy
    BB=(xbi[:,1:L+1]+xbi[:,1+L:2*L+1]-2*xbi[:,0].reshape(-1,1))*(wc2[1:L+1])
    temp,Sxx= np.linalg.qr((AA+BB).T,mode='reduced')
    Sxx=Sxx.T
    CC=(ybi[:,1:L+1]-ybi[:,L+1:2*L+1])*(wc1[1:L+1])
    DD=(ybi[:,1:L+1]+ybi[:,1+L:2*L+1]-2*ybi[:,0].reshape(-1,1))*(wc2[1:L+1])
    temp,Syy= np.linalg.qr((CC+DD).T,mode='reduced')
    Syy=Syy.T
    Pxy=Sxx@CC[:,0:n].T
    
    K = Pxy @ np.linalg.inv(Syy@Syy.T)                               #计算卡尔曼增益矩阵
    xa = xbm + K @ (yo-h(xbm))                                        # 更新状态变量
    B = Pxx-K @ Pyy @K.T                       # 计算下一个同化循环需要的背景误差协方差
    return xa,B
#%%  7-5  SP-CDKF同化试验及结果
n = 3                                                     # 状态维数
m = 3                                                    # 观测数	
x0b = np.array([2.0,3.0,4.0])                                  # 同化试验的初值
np.random.seed(seed=1)                                    # 初始化随机种子，便于重复结果

xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):                                    # xb得到的是不加同化的自由积分结果
    xb[:,k+1] = RK4(Lorenz63,xb[:,k],dt,sigma,beta,rho)    

sig_b= 0.1
B = sig_b**2*np.eye(n)                              # 初始时刻背景误差协方差，设为对角阵
Q = 0.1*np.eye(n)                                  # 模式误差（若假设完美模式则取0）

xa = np.zeros([n,nt+1]); xa[:,0] = x0b                #保存每步的集合均值作为分析场，存在xa
km = 0                                         # 对同化次数进行计数

xbi,vi,wm,wc1,wc2 = time_generate_SigmaP(xa[:,0], B, Q)
 #根据初始条件生成时间积分步的sigma点构成集合
n,N = xbi.shape                                      # N集合成员数
for k in range(nt):                                    # 时间积分   
    for i in range(N):                                 # 对每个集合成员积分
        xbi[:,i] = RK4(Lorenz63,xbi[:,i],dt,sigma,beta,rho)   # 积分每个集合成员得到预报集合
    xa[:,k] = np.sum(xbi*wm,1)              # 非同化时刻使用预报平均，同化时刻分析平均
    if (km<nt_m) and (k+1==ind_m[km]):   # 当有观测时，使用SP-CDKF进行更新
        xbi = xbi +vi                                 # 在集合成员中加入背景误差
        xbm,Pxx = time_update_SigmaP(xbi,wm,wc1,wc2)  #计算背景误差协方差Pxx
        xbi,ni,wm,wc1,wc2 = measurement_generate_SigmaP(xa[:,k],Pxx,R) 
#生成观测更新步的sigma点
        xa[:,k+1],B =  measurement_update_SigmaP(xbi,wm,wc1,wc2,yo[:,km],h,ni,xbm, Pxx )
# 调用SP-CDKF同化
        xbi,vi,wm,wc1,wc2 = time_generate_SigmaP(xa[:,k+1], B, Q) 
                                      # 为下一个同化循环生成集合成员
        km = km+1



# CDKF结果画图
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Songti SC']
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
        plt.title("SP-CDKF同化实验",fontsize=16)
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

