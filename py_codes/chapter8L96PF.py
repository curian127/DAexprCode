#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 15:46:30 2023

@author: shenzheqi
"""
import numpy as np
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
#%% 8-6 重取样指标转移程序
def Indexswift(idx_in):
      idx_out = -1*np.ones_like(idx_in)
      for i in range(len(idx_out)):
          if len(np.argwhere(idx_in==i))!=0:
              idx_out[i] = i;
              dum = np.argwhere(idx_in==i);
              idx_in = np.delete(idx_in,dum[0],axis=0);
      nil_idx = np.argwhere(idx_out==-1);
      nil_idx = nil_idx.flatten();
      idx_out[nil_idx] = idx_in;
      return idx_out
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
#%% 8-7核密度估计，梯形公式，以及高斯核估计方法和核分布概率映射（KDDM）方法
def kernel_density(xm,w):
    N = xm.shape[0]
    x = np.linspace(np.min(xm)-2*1.0,np.max(xm)+2*1.0,200)
    kk = np.int(len(xm)/5)-1

    fx = np.zeros(200)
    dis = np.zeros(N)
    for i in range(N):
        dis = np.abs(xm[i]-xm)
        sig = np.max([dis[kk],0.1])

        fx = fx + w[i]*np.exp(-(x-xm[i])**2/2/(sig**2))/np.sqrt(2*np.pi)/sig
    return x,fx

def trapezoid(a, dx):
    z = ( np.cumsum(a) - a/2)*dx;
    z = z / max(z);
    return z

def kddm(x,xo,w):
    # for vectors input
    N = w.shape[0]
    xma = np.sum(w*xo)
    xva = np.sqrt(np.sum(w*(xo-xma)**2)*N/(N-1))
    
    x = (x-np.mean(x))/np.sqrt(np.var(x))
    xo = (xo-np.mean(xo))/np.sqrt(np.var(xo))
    
    xda,fxa = kernel_density(xo, w)
    xdf,fxf = kernel_density(x, np.ones(N)/N)
    
    dx = xdf[1]-xdf[0]
    cdfxf = trapezoid(fxf, dx)
    dx = xda[1]-xda[0]
    cdfxa = trapezoid(fxa, dx)
    
    cdfxf = cdfxf[fxf>1e-5];xdf = xdf[fxf>1e-5]
    cdfxa = cdfxa[fxa>1e-5];xda = xda[fxa>1e-5]
    from scipy import interpolate
    f1 = interpolate.interp1d(xdf,cdfxf,kind='cubic')
    p = f1(x)
    f2 = interpolate.interp1d(cdfxa, xda,kind='cubic')
    q = f2(p)
    
    q = (q-np.mean(q))*xva/np.sqrt(np.var(q))+xma
    return q

#%% 8-8 LPF16的实施算法
def LPF(xbi,yo,R,ObsOp,LOC_MAT,kddm_flag):      # 输入局地化矩阵，kddm_flag用于选择是否使用KDDM
    n,N = xbi.shape   # n维数，N集合成员数
    m = yo.shape[0]    # m观测数
    alpha = 0.99
    LocM = ObsOp(LOC_MAT)
    
    xbio = xbi.copy()   # 保存一份不循环更新的原始先验场
    
    wo = np.ones([n,N])
    w1 = np.zeros(N)

    for i in range(m):   # 观测循环 # 循环指标(i,j,k) --> (obs,state,ens) -->(m,n,N)
        hx = ObsOp(xbi)
        hxi = hx[i,:]   # 先验场投影到观测i
        hxo = ObsOp(xbio)
        hxoi = hxo[i,:]   # 原始投影到观测i
        
        r = R[i,i]
        loc = LocM[i,:]*alpha
        # 计算观测点标量权重和相应重取样指标
        for k in range(N):          
            d1 = (yo[i]-hxi[k])/np.sqrt(2*r)
            wn = np.exp(-d1*d1)/np.sqrt(2*np.pi)   #每个标量观测计算出来的权重   
            w1[k] = (wn-1)*alpha+1;                #微调，去掉极小值
        
            d2 = (yo[i]-hxoi[k])/np.sqrt(2*r)
            wn = np.exp(-d2*d2)/np.sqrt(2*np.pi)
            wo[:,k] = wo[:,k]*((wn-1)*loc+1)       # 矢量权重的迭代更新
        # 权重正规化
        w1sum = np.sum(w1)
        w1 = w1/w1sum
        
        wosum = np.sum(wo,axis=1)
        for j in range(n):
            wo[j,:] = wo[j,:]/wosum[j]
        
        # 用原始先验场和迭代后的矢量权重求后验均值和方差
        xb = np.zeros(n)
        for k in range(N):
            xb = xb + wo[:,k]*xbio[:,k]
        var_b = np.zeros(n) 
        for k in range(N):
            var_b = var_b + wo[:,k]*(xbio[:,k]-xb)**2*N/(N-1)
        # 重取样指标
        idx = SIR(w1)
        idx = Indexswift(idx)
        
        # 在局地化范围内更新
        n0 = np.sum(loc>0)
        c = N*(1-loc[loc>0])/loc[loc>0]/w1sum
        r1 = np.zeros(n0); r2 = np.zeros(n0);
        for k in range(N):
            r1 = r1 + (xbi[loc>0,idx[k]]-xb[loc>0]+c*(xbi[loc>0,k]-xb[loc>0]))**2
            r2 = r2 + ((xbi[loc>0,idx[k]]-xb[loc>0])/c+(xbi[loc>0,k]-xb[loc>0]))**2
        r1 = np.sqrt((N-1)*var_b[loc>0]/r1)
        r2 = np.sqrt((N-1)*var_b[loc>0]/r2)
        xai = xbi.copy()
        for k in range(N):           
            xai[loc>0,k] = xb[loc>0] + r1*(xbi[loc>0,idx[k]] - xb[loc>0]) + r2*(xbi[loc>0,k] - xb[loc>0]);
        # 一二阶矩的调整公式
        vs = np.zeros(n0); pfm = np.zeros(n0); var_p = np.zeros(n0); 
        vm = np.zeros(n0); pm = np.zeros(n0);
        for k in range(N):
            pfm = pfm + xai[loc>0,k]/N
            vm = vm + xbio[loc>0,k]/N
            pm = pm + xbi[loc>0,k]/N
        for k in range(N):
            var_p = var_p+ (xbio[loc>0,k]-vm)**2/(N-1)
            vs = vs + (xai[loc>0,k]-pfm)**2/(N-1)
        correction = np.sqrt(var_b[loc>0])/np.sqrt(vs)
        for k in range(N):
            xai[loc>0,k] = xb[loc>0]+(xai[loc>0,k]-pfm)*correction
        # 高阶矩的KDDM调整，只在最后一个观测元素同化之后做
        if kddm_flag:
            if i == m-1:
                for j in range(n):
                    x = xbi[j]
                    xo = xbio[j]
                    q = kddm(x, xo, wo[j])
                    xai[j]=q
        xbi = xai.copy()                
    return xai

#%% 8-9 Lorenz96模式中的LPF同化实验
import numpy as np
## 模式定义：
def Lorenz96(state,*args):                      # Lorenz 96 模式右端项
    x = state
    F = args[0]
    n = len(x)    
    f = np.zeros(n)
    f[0] = (x[1] - x[n-2]) * x[n-1] - x[0]      # 边界点: i=0,1,N-1
    f[1] = (x[2] - x[n-1]) * x[0] - x[1]
    f[n-1] = (x[0] - x[n-3]) * x[n-2] - x[n-1]
    for i in range(2, n-1):
        f[i] = (x[i+1] - x[i-2]) * x[i-1] - x[i]
    f = f + F                                   # 外强迫
    return f

def RK4(rhs,state,dt,*args):                    # RK积分算子
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state

def h(x):                                       # 观测算子
    n= x.shape[0]
    m= 36                                        # 总观测数
    H = np.zeros((m,n))
    di = int(n/m)                               # 两个观测之间的空间距离
    for i in range(m):
        H[i,(i+1)*di-1] = 1
    z = H @ x
    return z
# 线性化观测算子
def Dh(x):
    n= x.shape[0]
    m= 36
    H = np.zeros((m,n))    
    di = int(n/m) 
    for i in range(m):
        H[i,(i+1)*di-1] = 1
    return H
# Lorenz96模式的真值积分和观测模拟
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
tm = 20                 # 实验窗口长度
nt = int(tm/dt)         # 积分步数
t = np.linspace(0,tm,nt+1)
np.random.seed(seed=1)
m = 36                   # 观测变量数
dt_m = 0.2              # 两次观测之间的时间
tm_m = 20               # 最大观测时间
nt_m = int(tm_m/dt_m)   # 同化次数
ind_m = (np.linspace(int(dt_m/dt),int(tm_m/dt),nt_m)).astype(int)
t_m = t[ind_m]

sig_m= 0.1              # 观测误差标准差
R = sig_m**2*np.eye(m)  # 观测误差协方差
# 3. 造真值和观测
xTrue = np.zeros([n,nt+1])
xTrue[:,0] = x0True
km = 0
yo = np.zeros([m,nt_m])
for k in range(nt):
    xTrue[:,k+1] = RK4(Lorenz96,xTrue[:,k],dt,F)    # 真值
    if (km<nt_m) and (k+1==ind_m[km]):
        yo[:,km] = h(xTrue[:,k+1]) + np.random.normal(0,sig_m,[m,])     # 观测
        km = km+1
## 滤波器调用：
sig_b= 1
x0b = x0True + np.random.normal(0,sig_b,[n,])         # 初值
B = sig_b**2*np.eye(n)                              # 初始误差协方差
sig_p= 0.1
Q = sig_p**2*np.eye(n)                              # 模式误差

xb = np.zeros([n,nt+1]); xb[:,0] = x0b
for k in range(nt):
    xb[:,k+1] = RK4(Lorenz96,xb[:,k],dt,F)          # 控制实验

N = 30                                       # 集合成员数
xai = np.zeros([n,N])
for i in range(N):
    xai[:,i] = x0b + np.random.multivariate_normal(np.zeros(n), B)  # 初始集合 

np.random.seed(seed=1) 
localP = 3; rhom = Rho(localP ,n)            # !!!产生局地化矩阵，参数可调整

xa = np.zeros([n,nt+1]); xa[:,0] = x0b
km = 0
for k in range(nt):
    for i in range(N):              # 集合预报
        xai[:,i] = RK4(Lorenz96,xai[:,i],dt,F) \
                 + np.random.multivariate_normal(np.zeros(n), Q)
    xa[:,k+1] = np.mean(xai,1)
    if (km<nt_m) and (k+1==ind_m[km]):  # 开始同化
        # xai = EnKF(xai,yo[:,km],h,Dh,R,rhom)
        xai = LPF(xai,yo[:,km],R,h,rhom,0)
        xa[:,k+1] = np.mean(xai,1)    
        km = km+1
RMSEb = np.sqrt(np.mean((xb-xTrue)**2,0))
RMSEa = np.sqrt(np.mean((xa-xTrue)**2,0))
mRMSEb = np.mean(RMSEb)
mRMSEa = np.mean(RMSEa)
#%% 画图相关代码
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Simhei']
plt.figure(figsize=(10,7))
plt.subplot(4,1,1)
plt.plot(t,xTrue[8,:], label='真值', linewidth = 3, color='C0')
plt.plot(t,xb[8,:], ':', label='背景', linewidth = 3, color='C1')
plt.plot(t,xa[8,:], '--', label='分析', linewidth = 3, color='C3')
plt.plot(t[ind_m],yo[8,:], 'o', fillstyle='none', \
                label='观测', markersize = 8, markeredgewidth = 2, color='C2')
plt.ylabel(r'$X_{9}(t)$',labelpad=7,fontsize=16)
plt.xticks(range(0,20,5),[],fontsize=16);plt.yticks(fontsize=16)
plt.title("Lorenz96模式的局地化粒子滤波器同化",fontsize=16)
plt.legend(loc=9,ncol=4,fontsize=15)

plt.subplot(4,1,2)
plt.plot(t,xTrue[17,:], label='真值', linewidth = 3, color='C0')
plt.plot(t,xb[17,:], ':', label='背景', linewidth = 3, color='C1')
plt.plot(t,xa[17,:], '--', label='分析', linewidth = 3, color='C3')
plt.plot(t[ind_m],yo[17,:], 'o', fillstyle='none', \
                label='观测', markersize = 8, markeredgewidth = 2, color='C2')
plt.ylabel(r'$X_{18}(t)$', labelpad=7,fontsize=16)
plt.xticks(range(0,20,5),[],fontsize=16);plt.yticks(fontsize=16)

plt.subplot(4,1,3)
plt.plot(t,xTrue[35,:], label='真值', linewidth = 3, color='C0')
plt.plot(t,xb[35,:], ':', label='背景', linewidth = 3, color='C1')
plt.plot(t[ind_m],yo[35,:], 'o', fillstyle='none', \
                label='观测', markersize = 8, markeredgewidth = 2, color='C2')
plt.plot(t,xa[35,:], '--', label='分析', linewidth = 3, color='C3')
plt.ylabel(r'$X_{36}(t)$', labelpad=7,fontsize=16)
plt.xticks(range(0,20,5),[],fontsize=16);plt.yticks(fontsize=16)

plt.subplot(4,1,4)
plt.plot(t,RMSEb,color='C1',label='背景均方根误差')
plt.plot(t,RMSEa,color='C3',label='分析均方根误差')
plt.text(2,1.5,'集合尺寸 = %.1f'%N + ', 局地化参数 = %0.1f'%localP,fontsize=13)
plt.text(2,3.5,'背景误差平均值 = %.3f'%mRMSEb +', 分析误差平均值 = %.3f'%mRMSEa,fontsize=13)
plt.ylim(0,10)
plt.ylabel('均方根误差',labelpad=7,fontsize=16);
plt.xlabel('时间（TU）',fontsize=16)
plt.legend(loc=9,ncol=2,fontsize=15)
plt.xticks(range(0,20,5),fontsize=16);plt.yticks(fontsize=16)
plt.show()