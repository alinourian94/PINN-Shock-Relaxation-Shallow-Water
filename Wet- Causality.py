import numpy as np
import sciann as sn
import matplotlib.pyplot as plt
from sciann.utils.math import diff, abs,exp

# Characteristic scales
g = 9.80665
L = 10.0
Vc = np.sqrt(g*L)
t0 = L/Vc
td_f = 20.0/t0
xd_max = 500.0/L

DTYPE = 'float32'
xd = sn.Variable('xd', dtype=DTYPE)
td = sn.Variable('td', dtype=DTYPE)
yd = sn.Functional("yd", [xd, td], 6*[60], "tanh")
Vd = sn.Functional("Vd", [xd, td], 6*[60], "tanh")

# Non-Dimention
x = xd*L
t = td*t0
y = yd*L
V = Vd*Vc
qd = Vd*yd
# Rational Weight Function
a=diff(Vd,xd)
j=1/(60*(abs(a)-a)+1)
b=diff(yd,xd)
i=1/(abs(b)-b+1)

L1 = i*(diff(yd, td) + diff(qd, xd))          # Continuity equation
L2 = i*(diff(Vd, td) + Vd*diff(Vd, xd) + diff(yd, xd))  # Momentum equation

# Time-marching parameters
N_intervals = 2
interval_length = td_f / N_intervals
alpha = 100  # Causal decay rate
cumulative_residual = 0.0

# Spatial discretization (adding points near expected shocks)
x_vals = np.concatenate([
    np.linspace(-xd_max, -xd_max/2, 300),
    np.linspace(-xd_max/2, xd_max/2, 1000),
    np.linspace(xd_max/2, xd_max, 300)
])

# Time-marching loop
previous_weights = None
y_prev = None
V_prev = None

for interval in range(N_intervals):
    # Current time window
    td_start = interval * interval_length
    td_end = (interval + 1) * interval_length
    print(f"Training interval {interval+1}/{N_intervals}: t_d in [{td_start:.2f}, {td_end:.2f}]")

    # Generate temporal collocation points
    t_vals = np.linspace(td_start, td_end, 500)
    x_grid, t_grid = np.meshgrid(x_vals, t_vals)
    x_flat = x_grid.flatten()
    t_flat = t_grid.flatten()

    # Causal weighting
    w_i = np.exp(-alpha * cumulative_residual)
#ic = (t==0)*(abs(Z-0.33))
    def sigmoid(td, beta=10):
        return 1 / (1 + exp(-beta * td))

#ic2 = sigmoid(t, beta=50) * abs(V)
#ic = sigmoid(t, beta=50) * abs(Z - 0.33)
    # Define conditions
    if interval == 0:  # Initial dam break conditions
        ic1 = (td == 0.0)*(xd <= 0.0)*abs(yd - 10.0/L)
        ic2 = (td == 0.0)*(xd > 0.0)*abs(yd - 2.0/L)
        ic3 = (td == 0.0)*abs(qd)
        conditions = [ic1, ic2, ic3]
    else:  # Continue from previous solution
        ic1 = (td == td_start)*(yd - y_prev)
        ic2 = (td == td_start)*(Vd - V_prev)
        conditions = [ic1, ic2]

    # Boundary conditions (constant through all intervals)
    bc_left = (xd == -xd_max)*abs(qd)
    bc_right = (xd == xd_max)*abs(yd - 2.0/L)
    conditions += [bc_left, bc_right]

    # Create model
    m = sn.SciModel([xd, td], 
                   [w_i*L1, w_i*L2] + conditions,
                   "mse", optimizer="adam")

    # Warm start from previous interval
    if previous_weights is not None:
        m.model.set_weights(previous_weights)

    # Train on current interval
    h = m.train(
        [x_flat, t_flat],
        (2 + len(conditions)) * ['zero'],
        epochs=5000,
        batch_size=500,
        learning_rate=0.001,
        adaptive_weights={'method': 'NTK', 'freq': 25}
    )

    # Update residual tracking
    res_L1 = L1.eval(m, [x_flat, t_flat])
    res_L2 = L2.eval(m, [x_flat, t_flat])
    cumulative_residual += np.mean(res_L1**2 + res_L2**2)

    # Save solution at interval end for next IC
    y_prev = yd.eval(m, [x_vals, np.full_like(x_vals, td_end)])
    V_prev = Vd.eval(m, [x_vals, np.full_like(x_vals, td_end)])
    previous_weights = m.model.get_weights()

x_test,t_test = np.meshgrid(np.linspace(-xd_max,xd_max, 1000), td_f)
y_pred = y.eval(m, [x_test,t_test])
Q_pred = V.eval(m, [x_test,t_test])
q=Q_pred*y_pred
xt=x_test*L
tt=t_test*t0

QQ_pred = q.reshape(1000, 1)
xx_test = xt.reshape(1000, 1)
yy_pred = y_pred.reshape(1000, 1)

#Exact Sulotion
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
#plt.plot( xx_test, data11 ,color="blue", label='PINN-RWF')
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
plt.savefig("sampple3013.jpg", dpi=600)

plt.subplot(1, 3, 2)
plt.plot( xx_test, QQ_pred ,linestyle='--',color="green", label='PINN')
#plt.plot( xx_test, data12 ,color="blue", label='PINN-RWF')
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
plt.savefig("sampple305.jpg", dpi=600)

plt.subplot(1, 3, 3)
plt.semilogy(h.history['loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig("sampple9311.jpg", dpi=600)
plt.show()


x1_test,t1_test = np.meshgrid(np.linspace(-xd_max,xd_max, 100),np.linspace(0,td_f, 100))
y1_pred = y.eval(m, [x1_test,t1_test])
V1_pred = V.eval(m, [x1_test,t1_test])

xt1=x1_test*L
tt1=t1_test*t0

fig =  plt.subplots(1,2,figsize=(15,4))
plt.subplot(1, 2, 1)
plt.pcolor(xt1,tt1,y1_pred, cmap='gray', shading='auto')
plt.colorbar()
plt.ylabel('t (s)')
plt.xlabel('x (m)')
plt.title('y (m)')
plt.xticks(range(-500,501,100))
plt.savefig("sample3006.jpg", dpi=600)
plt.subplot(1, 2, 2)
plt.pcolor(xt1,tt1,V1_pred*y1_pred, cmap='summer', shading='auto')
plt.colorbar()
plt.ylabel('t (s)')
plt.xlabel('x (m)')
plt.title('Q (m3/s)')
plt.xticks(range(-500,501,100))
plt.savefig("sample3010.jpg", dpi=600)

# Water Depth in Various times

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


fig =  plt.subplots(1,2,figsize=(15,4))
plt.subplot(1, 2, 1)
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


# Discharge in Various times
plt.subplot(1, 2, 2)
plt.plot( xx_test, QQ2_pred ,color="c", label='t=2 s')
plt.plot( xx_test, QQ5_pred ,color="r", label='t=5 s')
plt.plot( xx_test, QQ10_pred ,color="k", label='t=10 s')
plt.plot( xx_test, QQ15_pred ,color="y", label='t=15 s')
plt.plot( xx_test, QQ20_pred ,color="blue", label='t=20 s')
plt.plot( xx_test, QQ25_pred ,color="g", label='t=25 s')
plt.plot( xx_test, QQ30_pred ,color="k", label='t=30 s')
#plt.ylim(ymin=0)
#plt.xlim(xmin=0)
plt.ylim(ymax=35)
plt.legend(loc='best')
plt.xticks(range(-500,501,100))
plt.ylabel('Q (m3/s)')
plt.xlabel('x (m)')
plt.savefig("sample12.jpg", dpi=600)

