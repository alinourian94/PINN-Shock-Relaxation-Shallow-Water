import numpy as np
import sciann as sn
import matplotlib.pyplot as plt

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
j=1/(60*(abs(a)-a)+1)
b=diff(yd,xd)
i=1/(abs(b)-b+1)
L1 = (diff(yd,td) + diff(qd,xd))
L2 = j*(diff(Vd,td) + Vd*diff(Vd,xd) + diff(yd,xd))

ic = (td==0.0)*(xd<=0.0)*abs(yd-(10.0/L))
ic2 = (td==0.0)*(xd>0.0)*abs(yd-(2.0/L))
ic3 = (td==0.0)*abs(qd)
bc_left = (xd==-xd_max)*abs(qd)
bc_right = (xd==xd_max)*abs(yd-(2.0/L))

m = sn.SciModel([xd,td],[L1,L2,ic,ic2,ic3,bc_left,bc_right],"mse" ,optimizer= "Adam")

x_data,t_data = np.meshgrid(np.linspace(-xd_max,xd_max, 100),np.linspace(0.,td_f,100))
h = m.train([x_data,t_data], 7*['zero'], epochs=50,batch_size =100,learning_rate=0.001,adaptive_weights={'method': 'NTK', 'freq':10, 'use_score':True})

x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000),td_f)
y_pred = y.eval(m, [x_test,t_test])
Q_pred = V.eval(m, [x_test,t_test])
q=Q_pred*y_pred
xt=x_test*L
tt=t_test*t0

QQ_pred = q.reshape(1000, 1)
xx_test = xt.reshape(1000, 1)
yy_pred = y_pred.reshape(1000, 1)

y1 = 10
y3=5.078
t=20
x2= np.linspace(-27,198,225)
y2 = np.full ((224),y3)
y8=2
y22=np.append(y2,y8)
v2=2*np.sqrt(g*y1) - 2*np.sqrt(g*y2)
q2=v2*y2
q22=np.append(q2,0)
#plt.plot(x2,y2)

x1= np.linspace(-198,-27,171)
z=  2*t*np.sqrt(g*y1)
z2 = 3*t*np.sqrt(g)
y11= ((x1-z)/-z2)**2
v3=2*np.sqrt(g*y1) - 2*np.sqrt(g*y11)
q3=v3*y11

x3= np.linspace(-198,-500,302)
y3 = np.full ((302),10)
q4=np.full ((302),0)
#plt.plot(x3,y3)

x4= np.linspace(198,500,302)
y4 = np.full ((302),2)
q5=np.full ((302),0)

y2_pred = y_pred.reshape(1000,)
mse2 = np.sum((y11-y2_pred[302:473])**2)
mse3 = np.sum((y22-y2_pred[473:698])**2)
mse1 = np.sum((y3-y2_pred[0:302] )**2)
mse4 = np.sum((y4-y2_pred[698:1000] )**2)
mse_y = (1/1000) * np.sum((mse1,mse2,mse3,mse4))
rmsd_y = np.sqrt(mse_y)
print (mse_y, rmsd_y)

q2_pred = QQ_pred.reshape(1000,)
mse22 = np.sum((q3-q2_pred[302:473])**2)
mse33 = np.sum((q22-q2_pred[473:698])**2)
mse11 = np.sum((q4-q2_pred[0:302] )**2)
mse44 = np.sum((q5-q2_pred[698:1000] )**2)
mse_q = (1/1000) * np.sum((mse11,mse22,mse33,mse44))
rmsd_q = np.sqrt(mse_q)
print (mse_q, rmsd_q)
data11 = np.load('data1y.npy')
data12 = np.load('data1q.npy')
#np.save('data1y.npy', yy_pred)
#np.save('data1q.npy', QQ_pred)
fig =  plt.subplots(1,3,figsize=(20,4))
plt.subplot(1, 3, 1)
plt.plot( xx_test, yy_pred,linestyle='--',color="green", label='PINN')
plt.plot( xx_test, data11 ,color="blue", label='PINN-RWF')
plt.plot(x1,y11,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,y22,c='r' ,linestyle='--' )
plt.plot(x3,y3,c='r' ,linestyle='--' )
plt.plot(x4,y4,c='r' ,linestyle='--', )
plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('y (m)')
plt.xlabel('x (m)')
plt.savefig("sample30011.jpg", dpi=600)

plt.subplot(1, 3, 2)
plt.plot( xx_test, QQ_pred ,linestyle='--',color="green", label='PINN')
plt.plot( xx_test, data12 ,color="blue", label='PINN-RWF')
plt.plot(x1,q3,c='r' ,linestyle='--', label = 'Exact')
plt.plot(x2,q22,c='r' ,linestyle='--' )
plt.plot(x3,q4,c='r' ,linestyle='--' )
plt.plot(x4,q5,c='r' ,linestyle='--', )
#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymax=35)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
#plt.grid()
plt.ylabel('Q (m3/s)')
plt.xlabel('x (m)')
plt.savefig("sample303.jpg", dpi=600)

plt.subplot(1, 3, 3)
plt.semilogy(h.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

"""
x1_test,t1_test = np.meshgrid(np.linspace(-xd_max,xd_max, 100),np.linspace(td_0,td_f, 100))
y1_pred = y.eval(m, [x1_test,t1_test])
V1_pred = V.eval(m, [x1_test,t1_test])

xt1=x1_test*L
tt1=t1_test*t0

plt.pcolor(xt1,tt1,y1_pred, cmap='gray', shading='auto')
plt.colorbar()
plt.ylabel('t (s)')
plt.xlabel('x (m)')
plt.title('y (m)')
plt.xticks(range(-500,501,100))
plt.savefig("sample3006.jpg", dpi=600)

plt.pcolor(xt1,tt1,V1_pred*y1_pred, cmap='summer', shading='auto')
plt.colorbar()
plt.ylabel('t (s)')
plt.xlabel('x (m)')
plt.title('Q (m3/s)')
plt.xticks(range(-500,501,100))
plt.savefig("sample3010.jpg", dpi=600)


fig =  plt.subplots(1,2,figsize=(15,5))
plt.subplot(1, 2, 1)
plt.plot( xx_test, yy_pred,color="blue", label='PINN')

plt.ylim(ymin=0)
#plt.ylim(ymax=0.54)
#plt.xlim(xmin=7)
#plt.xlim(xmax=13)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.grid()
plt.ylabel('y')
plt.xlabel('x')

plt.subplot(1, 2, 2)
plt.plot( xx_test, QQ_pred ,color="blue", label='PINN')

#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymax=35)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.grid()
plt.ylabel('Q')
plt.xlabel('x')



plt.semilogy(h.history['lr'])
plt.xlabel('epochs')
plt.ylabel('lr')
plt.grid(True)
plt.show()

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
plt.savefig("sample14.jpg", dpi=600)




plt.plot( xx_test, QQ2_pred ,color="c", label='2 s')
plt.plot( xx_test, QQ5_pred ,color="r", label='5 s')
plt.plot( xx_test, QQ10_pred ,color="k", label='10 s')
plt.plot( xx_test, QQ15_pred ,color="y", label='15 s')
plt.plot( xx_test, QQ20_pred ,color="blue", label='20 s')

#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymax=35)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('Q (m3/s)')
plt.xlabel('x (m)')
plt.savefig("sample12.jpg", dpi=600)


plt.plot( xx_test, QQ25_pred ,color="g", label='25 s')
plt.plot( xx_test, QQ30_pred ,color="k", label='30 s')
"""