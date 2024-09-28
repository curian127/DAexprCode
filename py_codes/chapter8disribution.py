#>>> 3-1使用Runge-Kutta积分Lorenz63模式
import numpy as np
import matplotlib.pyplot as plt

def Lorenz63(state,*args): #Lorenz63模式
    #参考参数值为 rho = 28.0     #sigma = 10.0     #beta = 8.0 / 3.0    
    sigma = args[0]
    beta = args[1]
    rho = args[2]
    x, y, z = state 
    f = np.zeros(3) 
    f[0] = sigma * (y - x)
    f[1] = x * (rho - z) - y
    f[2] = x * y - beta * z
    return f 
def RK4(rhs,state,dt,*args):    # RK4数值积分格式
    k1 = rhs(state,*args)
    k2 = rhs(state+k1*dt/2,*args)
    k3 = rhs(state+k2*dt/2,*args)
    k4 = rhs(state+k3*dt,*args)
    new_state = state + (dt/6)*(k1+2*k2+2*k3+k4)
    return new_state

sigma = 10.0     
beta = 8.0/3.0
rho = 28.0     
dt = 0.01

u0=np.random.randn(3,2000)
ua = np.zeros([3,2000,31])
ua[:,:,0] = u0
for t in range(30):
    for k in range(2000):
        ua[:,k,t+1]=RK4(Lorenz63,ua[:,k,t],dt,sigma,beta,rho)
#%%
plt.rcParams['font.sans-serif'] = ['Songti SC']
plt.figure(figsize=(8,8))
for n in range(1,5):
    plt.subplot(2,2,n)
    plt.hist(ua[0,:,10*(n-1)],bins=50)
    plt.ylim(0,120);plt.xlim(-20,20)
    plt.text(-15,100,'t='+str(10*(n-1)),fontsize=20)
    if n==3 or n==4:
        plt.xlabel("x的数值",fontsize=16);plt.xticks(fontsize=20);
    else:
        plt.xticks([]);
    if n==1 or n==3:
        plt.ylabel("概率分布",fontsize=16);plt.yticks(fontsize=20);
    else:
        plt.yticks([]);
        
#%%
#>>> 重取样演示
import numpy as np
weights = np.abs(np.random.randn(100));
weights = weights/np.sum(weights)
idx = SIR(weights)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(weights,color='maroon')
plt.yticks([0.01,0.02,0.03,0.04],fontsize=15);plt.xticks(fontsize=15)
plt.ylabel('weights',fontsize=15)
plt.xlabel('particles',fontsize=15)
plt.subplot(1,2,2)
plt.hist(idx,bins=100,color='maroon')
plt.yticks([0,1,2,3,4],fontsize=15);plt.xticks(fontsize=15)
plt.ylabel('Resampling',fontsize=15)
plt.xlabel('particles',fontsize=15)