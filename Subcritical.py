import numpy as np
import sciann as sn
import matplotlib.pyplot as plt

from sciann.utils.math import diff , abs
initial_lr = 1e-1
final_lr = initial_lr/10
EPOCHS=200
learning_rate_finall = {
    "scheduler": "ExponentialDecay", 
    "initial_learning_rate": initial_lr,
    "final_learning_rate": final_lr, 
    "decay_epochs": EPOCHS
}
g=9.80665
b=1.0
#L = 25.0
L = ((0.18**2)/g)**(1/3)
Vc=np.sqrt(g*L)
t0=L/Vc
td_f = 200.0/t0

xd_max = 25.0/L
td_0 = 0.0   
xd_min = 0.0 

#DTYPE = 'float32'
xd = sn.Variable('xd')
td = sn.Variable('td')

yd=sn.Functional("yd", [xd,td],6*[60], "tanh")
Vd=sn.Functional("Vd", [xd,td],6*[60], "tanh")

x = xd*L
t = td*t0
y = yd*L
V = Vd*Vc

S=(0.2)-((0.05)*(xd*L-10)**2)
S1=(1.0)-(0.1*xd*L)

dd=sn.step(xd,7.999/L) - sn.step(xd,12.0/L)
Z=yd+(dd*S)
q=Vd*yd
L1 = diff(yd,td) + diff(q,xd)
L2 = diff(Vd,td) + Vd*diff(Vd,xd) + diff(yd,xd) + dd*diff(S,xd)
 
ic = (td==td_0)*abs(Z-(0.5/L))
ic2 = (td==td_0)*abs(Vd)
bc_left = (xd==xd_min)*abs(Vd-(0.36/Vc))
bc_right = (xd==xd_max)*abs(Z-(0.5/L))

m = sn.SciModel([xd,td],[L1 , L2 , ic, ic2, bc_left, bc_right] ,"mse" ,optimizer= "Adam")
from sciann_datagenerator import DataGeneratorXT
dg = DataGeneratorXT(X=[xd_min, xd_max],T=[td_0, td_f],
    targets=["domain","domain", "ic","ic", "bc-left", "bc-right"],
    num_sample=3000,
    logT=False
)
input_data, target_data = dg.get_data()
dg.plot_data()

h = m.train(input_data, target_data, epochs=20000,learning_rate=0.001,
            batch_size=100,
            adaptive_weights={'method': 'NTK', 'freq':10, 'use_score':True}) 
"""
x_data,t_data = np.meshgrid(np.linspace(xd_min,xd_max, 100),
                            np.linspace(td_0,td_f, 400))
h = m.train([x_data,t_data], 5*['zero'], epochs=200,batch_size =100,
            learning_rate=0.001,
            adaptive_weights={'method': 'NTK', 'freq':10, 'use_score':True})
"""
x_test,t_test = np.meshgrid(np.linspace(xd_min,xd_max, 100),td_f)
y_pred = y.eval(m, [x_test,t_test])
V_pred = V.eval(m, [x_test,t_test])

"""
Z= -y+ (dd*(S))
Z_pred = Z.eval(m, [x_test,t_test])
print(Z_pred)
ZZ_pred = Z_pred.reshape(100, 1)
xx_test = x_test.reshape(100, 1)
plt.plot(xx_test, ZZ_pred)
"""
xt=x_test*L
tt=t_test*t0

A=b*y_pred
q=V_pred*A

VV_pred = q.reshape(100, 1)
xx_test = xt.reshape(100, 1)
yy_pred = y_pred.reshape(100, 1)

"""
plt.plot( xx_test, yy_pred,c="b" , label='PINN' )
plt.plot(x,h1, c='r' ,linestyle='--', label = 'Exact')
plt.plot(x,dd*Z ,marker='o' , c='black' )
plt.legend(loc='best')
plt.ylim(ymin=0.46)
plt.ylim(ymax=0.54)
plt.xlim(xmin=6)
plt.xlim(xmax=14)
plt.ylabel('Z')
plt.xlabel('x')

"""
from scipy.optimize import fsolve
g=9.81
y1=0.5
q=0.18
V1=q/y1

x= np.linspace(0,25,1000)
dd=-1*(np.heaviside(x<8,0) -  np.heaviside(x<12,0))
Z=dd*(0.2-(0.05*(x-10)**2))
#print (dd*Z)
#plt.plot(x,dd*Z)

def eq (y,E):
    return y+(q**2/(2*g*y**2)) - E
y_values=[]
for  z in Z:
    E = y1 + (V1**2 / (2*g)) - z
    result = fsolve(eq , y1 , args=(E,))
    y_values.append(result[0])
    #print (y_values)
h1= (y_values+Z)
#print(h)
h2=h1.reshape(1000,1)
mse = (1/1000) * np.sum(h2-yy_pred)**2
rmsd = np.sqrt(mse)
print (mse, rmsd)
fig =  plt.subplots(1,3,figsize=(20,4))
plt.subplot(1, 3, 1)
plt.plot( xx_test, yy_pred,c="b" , label='PINN' )
plt.plot(x,h1, c='r' ,linestyle='--', label = 'Exact')
plt.plot(x,dd*Z ,marker='o' , c='black' )
plt.legend(loc='best')
#plt.ylim(ymin=0.46)
plt.ylim(ymax=0.59)
#plt.xlim(xmin=13)
#plt.xlim(xmax=7)
plt.ylabel('y (m)')
plt.xlabel('x (m)')
#plt.title('PINN')
plt.savefig("sample113.jpg", dpi=600)
plt.subplot(1, 3, 2)
plt.plot( xx_test, VV_pred ,color="blue")
#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymin=0)
plt.ylim(ymax=0.2)
plt.ylabel('Q')
plt.title('PINN')

plt.subplot(1, 3, 3)
plt.plot( xx_test, yy_pred,c="b" , label='PINN' )
plt.plot(x,h1, c='r' ,linestyle='--', label = 'Exact')
plt.plot(x,dd*Z ,marker='o' , c='black' )
plt.legend(loc='best')
plt.ylim(ymin=0.46)
plt.ylim(ymax=0.54)
plt.xlim(xmin=6)
plt.xlim(xmax=14)
plt.ylabel('y (m)')
plt.xlabel('x (m)')
#plt.title('PINN')
plt.savefig("sample112.jpg", dpi=600)

"""
plt.semilogy(h.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True)
plt.show()


plt.semilogy(h.history['lr'])
plt.xlabel('epochs')
plt.ylabel('lr')
plt.grid(True)
plt.show()


from scipy.optimize import fsolve
g=9.81
y1=0.5
q=0.18
V1=q/y1

x= np.linspace(0,25,100)
dd=-1*(np.heaviside(x<8,0) -  np.heaviside(x<12,0))
Z=dd*(0.2-(0.05*(x-10)**2))
#print (dd*Z)
#plt.plot(x,dd*Z)

def eq (y,E):
    return y+(q**2/(2*g*y**2)) - E
y_values=[]
for  z in Z:
    E = y1 + (V1**2 / (2*g)) - z
    result = fsolve(eq , y1 , args=(E,))
    y_values.append(result[0])
    #print (y_values)
h1= (y_values+Z)
#print(h)
    
#plt.plot(x,h, c='r' ,linestyle='--')
#plt.ylim(ymin=0.45)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=0.28)
#plt.xlim(xmax=0.52)
plt.ylabel('y')
plt.title('PINN')

xx_test = xt.reshape(100, 1)
x_test,t_test = np.meshgrid(np.linspace(xd_min,xd_max, 100),1)
y_pred1 = y.eval(m, [x_test,t_test])
V_pred1 = V.eval(m, [x_test,t_test])
yy_pred1 = y_pred1.reshape(100, 1)
x_test,t_test = np.meshgrid(np.linspace(xd_min,xd_max, 100),10)
y_pred10 = y.eval(m, [x_test,t_test])
V_pred10 = V.eval(m, [x_test,t_test])
yy_pred10 = y_pred10.reshape(100, 1)
x_test,t_test = np.meshgrid(np.linspace(xd_min,xd_max, 100),20)
y_pred20 = y.eval(m, [x_test,t_test])
V_pred120 = V.eval(m, [x_test,t_test])
yy_pred20 = y_pred20.reshape(100, 1)
x_test,t_test = np.meshgrid(np.linspace(xd_min,xd_max, 100),30)
y_pred50 = y.eval(m, [x_test,t_test])
V_pred50 = V.eval(m, [x_test,t_test])
yy_pred50 = y_pred50.reshape(100, 1)
x_test,t_test = np.meshgrid(np.linspace(xd_min,xd_max, 100),80)
y_pred80 = y.eval(m, [x_test,t_test])
V_pred80 = V.eval(m, [x_test,t_test])
yy_pred80 = y_pred80.reshape(100, 1)
x_test,t_test = np.meshgrid(np.linspace(xd_min,xd_max, 100),100)
y_pred100 = y.eval(m, [x_test,t_test])
V_pred100 = V.eval(m, [x_test,t_test])
yy_pred100 = y_pred100.reshape(100, 1)

plt.plot( xx_test, yy_pred1,c="k" , label='t=1 s' )
plt.plot( xx_test, yy_pred10,c="y" , label='t=10 s' )
#plt.plot( xx_test, yy_pred20,c="c" , label='20 s' )
plt.plot( xx_test, yy_pred50,c="g" , label='t=30 s' )
#plt.plot( xx_test, yy_pred80,c="r" , label='80 s' )
plt.plot( xx_test, yy_pred100,c="c" , label='t=100 s' )

plt.legend(loc='best')
plt.ylim(ymin=0.46)
plt.ylim(ymax=0.54)
plt.xlim(xmin=6)
plt.xlim(xmax=14)
plt.ylabel('y (m)')
plt.xlabel('x (m)')
plt.savefig("sample116.jpg", dpi=600)


x1_test,t1_test = np.meshgrid(np.linspace(xd_min,xd_max, 100),np.linspace(td_0,td_f, 100))
y1_pred = y.eval(m, [x1_test,t1_test])
V1_pred = V.eval(m, [x1_test,t1_test])

xt1=x1_test*L
tt1=t1_test*t0

plt.pcolor(xt1,tt1,y1_pred, cmap='magma_r', shading='auto')
plt.colorbar()
plt.ylabel('t (s)')
plt.xlabel('x (m)')
plt.title('y (m)')
plt.xticks(range(8,13,1))
plt.xlim(xmin=8)
plt.xlim(xmax=12)
plt.savefig("sample610.jpg", dpi=600)
"""

