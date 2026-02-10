import numpy as np
import sciann as sn
import matplotlib.pyplot as plt
initial_lr = 1e-3
final_lr = initial_lr/100
EPOCHS=200
learning_rate_finall = {
    "scheduler": "ExponentialDecay", 
    "initial_learning_rate": initial_lr,
    "final_learning_rate": final_lr, 
    "decay_epochs": EPOCHS}
from sciann_datagenerator import DataGeneratorXT
x = sn.Variable("x")
t = sn.Variable('t')
y=sn.Functional("y", [x,t],6*[60], "tanh")
V=sn.Functional("V", [x,t],6*[60], "tanh")
#Q.set_trainable ( False )
from sciann.utils.math import diff , abs, exp
g=9.81
S=0.2-(0.05*(x-10.0)**2)
dd=sn.step(x,8) - sn.step(x,11.99)
Z= y+ (dd*(S))

a=V*diff(V,x)
b=diff(y,x)
j=1/(0.006*(dd*(abs(a)-a))+1)**2
#j=(20*(dd*(5.5*abs(a)-0.8*a))+1)
#j=1/(0.008*(dd*(abs(a)))+1)
#i=1/(0.01*(dd*(abs(b)-b))+1)
#r=exp(-0.02*(x-11.7)**2)
L1 = (diff(V*y,x) + diff(y,t))
L2 = j*((1/g)*(diff(V,t) + V*diff(V,x))+diff(y,x)+dd*(1-0.1*x))

ic = (t==0)*(abs(Z-0.33))
ic2 = (t==0)*(abs(V))
bc_left = (x==0)*(abs(V-0.545454))
bc_right = (x==25)*(abs(Z-0.33))

m = sn.SciModel([x,t],[L1 , L2  , ic,ic2  , bc_left, bc_right] ,"mse" ,optimizer= "adam")
#x_data,t_data = np.meshgrid(np.linspace(0,25,100),np.linspace(0,300,100))
#h1 = m.train([x_data,t_data], 6*['zero'], epochs=100,batch_size =100,learning_rate=learning_rate_finall,adaptive_weights={'method': 'NTK', 'freq':10, 'use_score':True})

dg = DataGeneratorXT(X=[0.0, 25.0],T=[0.0, 200.0],
    targets=["domain","domain","ic", "ic", "bc-left", "bc-right"],
    num_sample=10000,
    logT=False)
input_data, target_data = dg.get_data()
dg.plot_data()
h1 = m.train(input_data, target_data, epochs=5000,learning_rate=0.001,
            batch_size=200,
            adaptive_weights={'method': 'NTK', 'freq':50, 'use_score':True})


x_test,t_test = np.meshgrid(np.linspace( 0 , 25, 110), 200)
y_pred = y.eval(m, [x_test,t_test]) 
Q_pred = V.eval(m, [x_test,t_test])
QQ_pred = Q_pred.reshape(110, 1)
xx_test = x_test.reshape(110, 1)
tt_test = t_test.reshape(110, 1)
yy_pred = y_pred.reshape(110, 1)
"""a_pred = a.eval(m, [x_test,t_test]) 
aa_pred = a_pred.reshape(110, 1)
plt.pcolor (xx_test,tt_test, aa_pred,cmap=('inferno'), shading=('auto'), vmin=0, vmax=1.,)
"""
from scipy.optimize import fsolve
g=9.81
y1=0.33
q=0.18
V1=q/y1
yc = (q**2/g)**(1/3)
YC = np.full((41,1),yc)
Ec = 1.5*yc
Eup = Ec+0.2
def eq1 (yup,Eup):
    return yup+(q**2/(2*g*yup**2)) - Eup
y_up=[]
result = fsolve(eq1 , y1 , args=(Eup,))
y_up.append(result[0])
#print (y_up)
Y1 =np.full ((41,1), y_up)
#print(Y1)

x= np.linspace(0,25,126)
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
h= (y_values+Z)
#print(h)

Y1 =np.full ((40,1), y_up)
Y1.shape
ss=np.append([Y1],[y1,yc])
ss2=np.append([ss],h[58:126])
ss2    

xf=np.append(x[1:41],[x[50],x[58]])
xf2=np.append([xf], x[58:126])
xf2.shape
#plt.plot(xf2,ss2)

x= np.linspace(0,25,126)
dd=-1*(np.heaviside(x<8,0) -  np.heaviside(x<12,0))
Z=dd*(0.2-(0.05*(x-10)**2))
#print (dd*Z)
#plt.plot(x,dd*Z)
#plt.xticks(np.arange(0,25,2))
#plt.grid()

mse1 = np.sum((ss2[0:41]-yy_pred[0:41])**2)
mse2 = np.sum((ss2[58:126]-yy_pred[58:126])**2)
mse3 = np.sum((ss2[50]-yy_pred[50] )**2)
mse4 = np.sum((ss2[58]-yy_pred[58] )**2)
mse = (1/110) * np.sum((mse1,mse2,mse3,mse4))
rmsd = np.sqrt(mse)

fig =  plt.subplots(1,3,figsize=(20,4))
plt.subplot(1, 3, 1)
plt.plot( xx_test, yy_pred,color="blue", label='PINN-RWF')
plt.plot(xf2,ss2, c='r' ,linestyle='--', label = 'Exact')
plt.plot(x,dd*Z ,marker='o' , c='black',label = 'Bed' )
plt.legend(loc='best')
#plt.ylim(ymin=0.1)
plt.ylim(ymax=0.5)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.ylabel('y (m)')
plt.xlabel('x (m)')

plt.savefig("sampl481.jpg", dpi=600)
print (mse, rmsd)

plt.subplot(1, 3, 2)
plt.plot( xx_test, QQ_pred*yy_pred ,color="blue")
#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
#plt.ylim(ymax=0.3)
plt.ylabel('Q')
plt.xlabel('x')


plt.subplot(1, 3, 3)
plt.semilogy(h1.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True)
plt.show()
#plt.savefig("sampl412.jpg", dpi=600)