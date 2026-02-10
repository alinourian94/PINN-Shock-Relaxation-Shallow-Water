import numpy as np
import matplotlib.pyplot as plt
import sciann as sn

#from sciann_datagenerator import DataGeneratorXT

x = sn.Variable("x")
t = sn.Variable('t')
y=sn.Functional("y", [x,t],4*[40], "tanh")
Q=sn.Functional("Q", [x,t],4*[40], "tanh")

from sciann.utils.math import diff
g=9.81
b=1.0
L=1000.0
t0=20.0
A=b*y
L1 = diff(Q,x) + diff(y,t)
L2 = (1/(A*g))*diff(Q,t)+(1/(A*g))*diff((Q**2)/A,x) 

ic = (t==0.0)*(x<=500.0)*(y-10.0)
ic2 = (t==0.0)*(Q)
bc_left = (x==0.0)*(Q)
bc_right = (x==L)*(y)

m = sn.SciModel([x,t],[L1 , L2  , ic ,ic2, bc_left,bc_right] ,"mse" ,optimizer= "Adam")
"""
dg = DataGeneratorXT(X=[0, L],T=[0, t0],
    targets=["domain","domain", "ic","bc-left","bc-right"],
    num_sample=10000,
    logT=False
)
input_data, target_data = dg.get_data()
dg.plot_data()

h = m.train(input_data, target_data, epochs=2000,learning_rate=0.001,
            batch_size=100,
            adaptive_weights={'method': 'NTK', 'freq':10, 'use_score':True})
"""
x_data,t_data = np.meshgrid(np.linspace(0.,L, 10),np.linspace(0.,t0,10))
h = m.train([x_data,t_data], 6*['zero'], epochs=10,batch_size =100,learning_rate=0.001,adaptive_weights={'method': 'NTK', 'freq':20, 'use_score':True})

x_test,t_test = np.meshgrid(np.linspace(0,L, 100),t0)
y_pred = y.eval(m, [x_test,t_test])
Q_pred = Q.eval(m, [x_test,t_test])

QQ_pred = Q_pred.reshape(100, 1)
xx_test = x_test.reshape(100, 1)
yy_pred = y_pred.reshape(100, 1)

fig =  plt.subplots(1,3,figsize=(20,4))
plt.subplot(1, 3, 1)
plt.plot( xx_test, yy_pred,color="blue")
#plt.ylim(ymin=0.46)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.ylabel('y')
plt.title('PINN')
plt.subplot(1, 3, 2)
plt.plot( xx_test, QQ_pred ,color="blue")
plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymax=20)
plt.ylabel('V')
plt.title('PINN')

plt.subplot(1, 3, 3)
plt.semilogy(h.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.grid(True)
plt.show()


