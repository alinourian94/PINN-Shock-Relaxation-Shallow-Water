import numpy as np
import sciann as sn
import matplotlib.pyplot as plt
from sciann_datagenerator import DataGeneratorXT
g=9.80665
L = 10.0
#L = ((0.18**2)/g)**(1/3)
Vc=np.sqrt(g*L)
t0=L/Vc
td_f = 20.0/t0
xd_max = 500.0/L
td_0 = 0.0   
xd_min = 0.0 

DTYPE = 'float32'
xd = sn.Variable('xd', dtype=DTYPE)
td = sn.Variable('td', dtype=DTYPE)
yd=sn.Functional("yd", [xd,td],6*[60], "tanh")
Vd=sn.Functional("Vd", [xd,td],6*[60], "tanh")

x = xd*L
t = td*t0
y = yd*L
V = Vd*Vc

from sciann.utils.math import diff ,abs
qd=Vd*yd
a=diff(Vd,xd)
j=1/(abs(a)-a+1)
b=diff(yd,xd)
i=1/(abs(b)-b+1)
L1 = (diff(yd,td) + diff(qd,xd))
L2 = j*(diff(Vd,td) + Vd*diff(Vd,xd) + diff(yd,xd))

ic = (td==0.0)*(xd<=0.0)*abs(yd-(10.0/L))
ic2 = (td==0.0)*(xd>0.0)*abs(yd)
ic3 = (td==0.0)*abs(qd)
bc_left = (xd==-xd_max)*abs(qd)
bc_right = (xd==xd_max)*abs(yd)

m = sn.SciModel([xd,td],[L1,L2,ic,ic2,ic3,bc_left,bc_right],"mse" ,optimizer= "Adam")
dg = DataGeneratorXT(X=[-xd_max , xd_max],T=[0., td_f],
    targets=["domain","domain", "ic","ic","ic","bc-left","bc-right"],
    num_sample=4000,
    logT=False)
input_data, target_data = dg.get_data()
dg.plot_data()
h = m.train(input_data, target_data, epochs=20000,learning_rate=0.001,
            batch_size=100,
            adaptive_weights={'method': 'NTK', 'freq':10, 'use_score':True})
"""
x_data,t_data = np.meshgrid(np.linspace(-xd_max,xd_max, 100),np.linspace(0.,td_f,100))
h = m.train([x_data,t_data], 7*['zero'], epochs=220,batch_size =100,learning_rate=0.001,adaptive_weights={'method': 'NTK', 'freq':10, 'use_score':True})
"""
x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),td_f)
y_pred = y.eval(m, [x_test,t_test])
Q_pred = V.eval(m, [x_test,t_test])
q=Q_pred*y_pred
xt=x_test*L
tt=t_test*t0

print(y_pred)
print(q)

QQ_pred = q.reshape(1000, 1)
xx_test = xt.reshape(1000, 1)
yy_pred = y_pred.reshape(1000, 1)

y1 = 10
t=20
g=9.80665

x1= np.linspace(0,396,396)
z=  2*t*np.sqrt(g*y1)
z2 = 3*t*np.sqrt(9.81)
y11= ((x1-z)/-z2)**2
v1=2*np.sqrt(g*y1) - 2*np.sqrt(g*y11)
q1=v1*y11
#plt.plot(x1,q)

x2= np.linspace(-198,0,198)
z=  2*t*np.sqrt(g*y1)
z2 = 3*t*np.sqrt(g)
y12= ((x2-z)/-z2)**2
v2=2*np.sqrt(g*y1) - 2*np.sqrt(g*y12)
q2=v2*y12
#plt.plot(x2,q)

x3= np.linspace(-198,-500,302)
y2 = np.full ((302),10)
q3=np.full ((302),0)
#plt.plot(x3,q)

x4= np.linspace(396,500,104)
y22 = np.full ((104),0)
v=2*np.sqrt(g*y1) - 2*np.sqrt(g*y2)
q4= np.full ((104),0)
#plt.plot(x4,q)

y2_pred = y_pred.reshape(1000,)
mse3 = np.sum((y11-y2_pred[500:896])**2)
mse2 = np.sum((y12-y2_pred[302:500])**2)
mse1 = np.sum((y2-y2_pred[0:302] )**2)
mse4 = np.sum((y22-y2_pred[896:1000] )**2)
mse_y = (1/1000) * np.sum((mse1,mse2,mse3,mse4))
rmsd_y = np.sqrt(mse_y)
print (mse_y, rmsd_y)

q2_pred = QQ_pred.reshape(1000,)
mse7 = np.sum((q1-q2_pred[500:896])**2)
mse6 = np.sum((q2-q2_pred[302:500])**2)
mse5 = np.sum((q3-q2_pred[0:302] )**2)
mse8 = np.sum((q4-q2_pred[896:1000] )**2)
mse_q = (1/1000) * np.sum((mse5,mse6,mse7,mse8))
rmsd_q = np.sqrt(mse_q)
print (mse_q, rmsd_q)

fig =  plt.subplots(1,3,figsize=(20,4))
plt.subplot(1, 3, 1)
plt.plot( xx_test, yy_pred,color="blue", label='PINN')
plt.plot(x1,y11,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,y12,c='r' ,linestyle='--' )
plt.plot(x3,y2,c='r' ,linestyle='--' )
plt.plot(x4,y22,c='r' ,linestyle='--', )

#plt.ylim(ymin=0.46)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
#plt.grid()
plt.ylabel('y (m)')
plt.xlabel('x (m)')
plt.savefig("sample151.jpg", dpi=600)
plt.subplot(1, 3, 2)
plt.plot( xx_test, QQ_pred ,color="blue", label='PINN')
plt.plot(x1,q1,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,q2,c='r' ,linestyle='--' )
plt.plot(x3,q3,c='r' ,linestyle='--' )
plt.plot(x4,q4,c='r' ,linestyle='--', )
#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymax=35)
plt.legend(loc='best')
plt.ylabel('Q (m3/s)')
plt.xlabel('x (m)')
plt.savefig("sample152.jpg", dpi=600)

plt.subplot(1, 3, 3)
plt.semilogy(h.history['loss'])
plt.xlabel('epochs')
plt.ylabel('total loss')
#plt.grid(True)
plt.show()

#np.save('epochs20000.npy', h.history['loss'])
#data11 = np.load('data11.npy')
"""
x1_test,t1_test = np.meshgrid(np.linspace(-xd_max,xd_max, 100),np.linspace(td_0,td_f, 100))
y1_pred = y.eval(m, [x1_test,t1_test])
V1_pred = V.eval(m, [x1_test,t1_test])

xt1=x1_test*L
tt1=t1_test*t0

plt.pcolor(xt1,tt1,y1_pred, cmap='winter', shading='auto')
plt.colorbar()
plt.ylabel('t (s)')
plt.xlabel('x (m)')
plt.title('y (m)')
plt.xticks(range(-500,501,100))
plt.savefig("sample158.jpg", dpi=600)

plt.pcolor(xt1,tt1,V1_pred*y1_pred, cmap='gray', shading='auto')
plt.colorbar()
plt.ylabel('t (s)')
plt.xlabel('x (m)')
plt.title('Q (m3/s)')
plt.xticks(range(-500,501,100))
plt.savefig("sample165.jpg", dpi=600)


fig =  plt.subplots(1,3,figsize=(20,4))
plt.subplot(1, 3, 1)
plt.plot( xx_test, QQ_pred ,color="blue")
plt.plot(x1,q1,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,q2,c='r' ,linestyle='--' )
plt.plot(x3,q3,c='r' ,linestyle='--' )
plt.plot(x4,q4,c='r' ,linestyle='--', )
#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymax=35)
plt.legend(loc='best')
plt.ylabel('Q')
plt.title('PINN')

plt.subplot(1, 3, 2)
plt.plot( xx_test, QQ_pred ,color="blue")
plt.plot(x1,q1,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,q2,c='r' ,linestyle='--' )
plt.plot(x3,q3,c='r' ,linestyle='--' )
plt.plot(x4,q4,c='r' ,linestyle='--', )
plt.ylim(ymin=0)
plt.xlim(xmin=-300)
plt.xlim(xmax=-100)
plt.ylim(ymax=5)
plt.legend(loc='best')
plt.ylabel('Q')
plt.title('PINN')

plt.subplot(1, 3, 3)
plt.plot( xx_test, QQ_pred ,color="blue")
plt.plot(x1,q1,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,q2,c='r' ,linestyle='--' )
plt.plot(x3,q3,c='r' ,linestyle='--' )
plt.plot(x4,q4,c='r' ,linestyle='--', )
plt.ylim(ymin=0)
plt.xlim(xmin=250)
plt.xlim(xmax=500)
plt.ylim(ymax=5)
plt.legend(loc='best')
plt.ylabel('Q')
plt.title('PINN')


fig =  plt.subplots(1,3,figsize=(20,4))
plt.subplot(1, 3, 1)
plt.plot( xx_test, yy_pred,color="blue", label='PINN')
plt.plot(x1,y11,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,y12,c='r' ,linestyle='--' )
plt.plot(x3,y2,c='r' ,linestyle='--' )
plt.plot(x4,y22,c='r' ,linestyle='--', )

#plt.ylim(ymin=0.46)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.grid()
plt.ylabel('y')
plt.title('PINN')

plt.subplot(1, 3, 2)
plt.plot( xx_test, yy_pred,color="blue", label='PINN')
plt.plot(x1,y11,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,y12,c='r' ,linestyle='--' )
plt.plot(x3,y2,c='r' ,linestyle='--' )
plt.plot(x4,y22,c='r' ,linestyle='--', )

plt.ylim(ymin=8.5)
plt.ylim(ymax=10.1)
plt.xlim(xmin=-300)
plt.xlim(xmax=-100)
plt.legend(loc='best')
#plt.xticks(range(-500,501,100))
plt.grid()
plt.ylabel('y')
plt.title('PINN')

plt.subplot(1, 3, 3)
plt.plot( xx_test, yy_pred,color="blue", label='PINN')
plt.plot(x1,y11,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,y12,c='r' ,linestyle='--' )
plt.plot(x3,y2,c='r' ,linestyle='--' )
plt.plot(x4,y22,c='r' ,linestyle='--', )

plt.ylim(ymin=-0.03)
plt.ylim(ymax=0.4)
plt.xlim(xmin=100)
plt.xlim(xmax=500)
plt.legend(loc='best')
#plt.xticks(range(-500,501,100))
plt.grid()
plt.ylabel('y')
plt.title('PINN')



x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),2)
y2_pred = y.eval(m, [x_test,t_test])
Q2_pred = V.eval(m, [x_test,t_test])
q2=Q2_pred*y2_pred
x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),5)
y5_pred = y.eval(m, [x_test,t_test])
Q5_pred = V.eval(m, [x_test,t_test])
q5=Q5_pred*y5_pred
x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),10)
y10_pred = y.eval(m, [x_test,t_test])
Q10_pred = V.eval(m, [x_test,t_test])
q10=Q10_pred*y10_pred
x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),15)
y15_pred = y.eval(m, [x_test,t_test])
Q15_pred = V.eval(m, [x_test,t_test])
q15=Q15_pred*y15_pred
x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),20)
y20_pred = y.eval(m, [x_test,t_test])
Q20_pred = V.eval(m, [x_test,t_test])
q20=Q20_pred*y20_pred
xt=x_test*L
tt=t_test*t0

QQ2_pred = q2.reshape(1000, 1)
xx_test = xt.reshape(1000, 1)
yy2_pred = y2_pred.reshape(1000, 1)
plt.plot( xx_test, yy2_pred,color="c", label='t=2 s')
plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('y')
plt.xlabel('x')

QQ5_pred = q5.reshape(1000, 1)
yy5_pred = y5_pred.reshape(1000, 1)
plt.plot( xx_test, yy5_pred,color="r", label='t=5 s')
plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('y')
plt.xlabel('x')

QQ10_pred = q10.reshape(1000, 1)
yy10_pred = y10_pred.reshape(1000, 1)
plt.plot( xx_test, yy10_pred,color="k", label='t=10 s')
plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('y')
plt.xlabel('x')

QQ15_pred = q15.reshape(1000, 1)
yy15_pred = y15_pred.reshape(1000, 1)
plt.plot( xx_test, yy15_pred,color="y", label='t=15 s')
plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('y(m)')
plt.xlabel('x(m)')

QQ20_pred = q20.reshape(1000, 1)
yy20_pred = y20_pred.reshape(1000, 1)
plt.plot( xx_test, yy20_pred,color="b", label='t=20 s')
plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('y(m)')
plt.xlabel('x(m)')
plt.savefig("sample4.jpg", dpi=600)


x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),25)
y25_pred = y.eval(m, [x_test,t_test])
Q25_pred = V.eval(m, [x_test,t_test])
q25=Q25_pred*y25_pred
x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),30)
y30_pred = y.eval(m, [x_test,t_test])
Q30_pred = V.eval(m, [x_test,t_test])
q30=Q30_pred*y30_pred
xt=x_test*L
tt=t_test*t0

QQ25_pred = q25.reshape(1000, 1)
yy25_pred = y25_pred.reshape(1000, 1)
plt.plot( xx_test, yy25_pred,color="g", label='t=25 s')
plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('y')
plt.xlabel('x')

QQ30_pred = q30.reshape(1000, 1)
yy30_pred = y30_pred.reshape(1000, 1)
plt.plot( xx_test, yy30_pred,color="k", label='t=30 s')
plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('y(m)')
plt.xlabel('x(m)')
plt.savefig("sample15.jpg", dpi=600)




plt.plot( xx_test, QQ2_pred ,color="c", label='t=2 s')
plt.plot( xx_test, QQ5_pred ,color="r", label='t=5 s')
plt.plot( xx_test, QQ10_pred ,color="k", label='t=10 s')
plt.plot( xx_test, QQ15_pred ,color="y", label='t=15 s')
plt.plot( xx_test, QQ20_pred ,color="blue", label='t=20 s')

#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymax=35)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('Q (m3/s)')
plt.xlabel('x (m)')
plt.savefig("sample12.jpg", dpi=600)


x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),24)
y24_pred = y.eval(m, [x_test,t_test])
Q24_pred = V.eval(m, [x_test,t_test])
q24=Q24_pred*y24_pred
x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),27)
y28_pred = y.eval(m, [x_test,t_test])
Q28_pred = V.eval(m, [x_test,t_test])
q28=Q28_pred*y28_pred
QQ24_pred = q24.reshape(1000, 1)
QQ28_pred = q28.reshape(1000, 1)

plt.plot( xx_test, QQ24_pred ,color="g", label='t=24 s')
plt.plot( xx_test, QQ28_pred ,color="k", label='t=27 s')
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('Q (m3/s)')
plt.xlabel('x (m)')
plt.savefig("sample13.jpg", dpi=600)
"""