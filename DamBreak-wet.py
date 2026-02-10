import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

DTYPE = tf.float32

g = 9.80665
L = 10.0
Vc = np.sqrt(g * L)
t0 = L / Vc

td_f = 20.0 / t0
xd_max = 500.0 / L
xd_min = 0.0

class PINN(tf.keras.Model):
    def __init__(self, layers):
        super().__init__()
        self.hidden = []
        for width in layers[:-1]:
            self.hidden.append(
                tf.keras.layers.Dense(
                    width,
                    activation=tf.nn.tanh,
                    kernel_initializer="glorot_normal"
                )
            )
        self.out = tf.keras.layers.Dense(layers[-1])

    def call(self, X):
        Z = X
        for layer in self.hidden:
            Z = layer(Z)
        return self.out(Z)

layers = [40, 40, 40, 40, 1]
yd_net = PINN(layers)
Vd_net = PINN(layers)

def gradients(xd, td):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([xd, td])
        X = tf.concat([xd, td], axis=1)

        yd = yd_net(X)
        Vd = Vd_net(X)

        qd = Vd * yd

    yd_t = tape.gradient(yd, td)
    yd_x = tape.gradient(yd, xd)
    Vd_t = tape.gradient(Vd, td)
    Vd_x = tape.gradient(Vd, xd)

    del tape
    return yd, Vd, qd, yd_t, yd_x, Vd_t, Vd_x

# ========================
# Loss function
# ========================
def loss_fn(xd, td):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch([xd, td])

        X = tf.concat([xd, td], axis=1)

        yd = yd_net(X)
        Vd = Vd_net(X)
        qd = Vd * yd

    yd_t = tape.gradient(yd, td)
    yd_x = tape.gradient(yd, xd)
    Vd_t = tape.gradient(Vd, td)
    Vd_x = tape.gradient(Vd, xd)
    qd_x = tape.gradient(qd, xd)

    del tape
    i=1/(abs(yd_x)-yd_x+1)
    # PDE residuals
    L1 = (tf.reduce_mean(tf.square(yd_t + qd_x)*i))
    L2 = (tf.reduce_mean(tf.square(Vd_t + Vd * Vd_x + yd_x)*i))

    # Initial conditions
    ic1 = tf.reduce_mean(tf.square(
        tf.where((td == 0.0) & (xd <= 0.0), yd - (10.0/L), 0.0)
    ))

    ic2 = tf.reduce_mean(tf.square(
        tf.where((td == 0.0) & (xd > 0.0), yd - (2.0/L), 0.0)
    ))

    ic3 = tf.reduce_mean(tf.square(
        tf.where(td == 0.0, qd, 0.0)
    ))

    # Boundary conditions
    bc_left = tf.reduce_mean(tf.square(
        tf.where(xd == -xd_max, qd, 0.0)
    ))

    bc_right = tf.reduce_mean(tf.square(
        tf.where(xd == xd_max, yd - (2.0/L), 0.0)
    ))

    return L1 + L2 + ic1 + ic2 + ic3 + bc_left + bc_right

# ========================
# Training data
# ========================

x_data1= np.linspace(-xd_max,-xd_max/5,50)
x_data2= np.linspace(-xd_max/5,xd_max/3,50)
x_data3= np.linspace(xd_max/3,xd_max/2.2,100)
x_data4= np.linspace(xd_max/2.2,xd_max,50)
x_data5= np.concatenate([x_data1,x_data2,x_data3,x_data4])
x_data, t_data = np.meshgrid(
    x_data5,
    np.linspace(0.0, td_f, 200)
)

xd_train = tf.convert_to_tensor(x_data.reshape(-1,1), dtype=DTYPE)
td_train = tf.convert_to_tensor(t_data.reshape(-1,1), dtype=DTYPE)

# ========================
# Adam optimizer
# ========================
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

def train_step():
    with tf.GradientTape() as tape:
        loss = loss_fn(xd_train, td_train)

    vars = yd_net.trainable_variables + Vd_net.trainable_variables
    grads = tape.gradient(loss, vars)
    optimizer.apply_gradients(zip(grads, vars))

    return loss

# ========================
# Training loop
# ========================
loss_history = []

for epoch in range(4000):
    loss = train_step()
    loss_history.append(loss.numpy())
    batch_size=500
    if epoch % 40 == 0:
        print(f"Epoch {epoch}, Loss = {loss.numpy():.3e}")

# Save Adam-only model
adam_weights = [v.numpy().copy() for v in (yd_net.trainable_variables + Vd_net.trainable_variables)]
adam_loss_history = loss_history.copy()

# Prediction
x_test, t_test = np.meshgrid(
    np.linspace(-xd_max, xd_max, 1000),
    td_f
)

X_test = tf.convert_to_tensor(
    np.vstack([x_test.flatten(), t_test.flatten()]).T,
    dtype=DTYPE
)
# ========================
# Prediction using ADAM only
# ========================
# Restore Adam weights
vars_all = yd_net.trainable_variables + Vd_net.trainable_variables
for v, w in zip(vars_all, adam_weights):
    v.assign(w)

yd_pred_adam = yd_net(X_test).numpy().reshape(-1,1)
Vd_pred_adam = Vd_net(X_test).numpy().reshape(-1,1)

y_adam = yd_pred_adam * L
Q_adam = Vd_pred_adam * Vc
q_adam = Q_adam * y_adam

# ========================
# L-BFGS (SciPy) Optimizer
# ========================

import scipy.optimize as sopt

# Collect all trainable variables
vars = yd_net.trainable_variables + Vd_net.trainable_variables

# Helper: flatten weights
def get_weights():
    return tf.concat([tf.reshape(v, [-1]) for v in vars], axis=0).numpy()

# Helper: assign weights back to model
def set_weights(flat_weights):
    idx = 0
    for v in vars:
        shape = v.shape
        size = tf.size(v).numpy()
        new_val = flat_weights[idx:idx+size].reshape(shape)
        v.assign(new_val)
        idx += size

# Loss + Gradient for SciPy
def loss_and_grad(flat_weights):
    set_weights(flat_weights)

    with tf.GradientTape() as tape:
        loss = loss_fn(xd_train, td_train)

    grads = tape.gradient(loss, vars)
    grad_flat = tf.concat([tf.reshape(g, [-1]) for g in grads], axis=0)

    return loss.numpy().astype(np.float64), grad_flat.numpy().astype(np.float64)

print("\nStarting L-BFGS optimization...")

initial_weights = get_weights()

result = sopt.minimize(
    fun=loss_and_grad,
    x0=initial_weights,
    jac=True,
    method='L-BFGS-B',
    options={
        'maxiter': 1000,
        'maxfun': 1000,
        'maxcor': 500,
        'ftol': 1.0 * np.finfo(float).eps,
        'gtol': 1e-8,
        'iprint': 1
    }
)

# Set final optimized weights
set_weights(result.x)

print("\nL-BFGS finished.")
print("Final loss:", result.fun)

# ========================
# Prediction after L-BFGS
# ========================
yd_pred_lbfgs = yd_net(X_test).numpy().reshape(-1,1)
Vd_pred_lbfgs = Vd_net(X_test).numpy().reshape(-1,1)

y_lbfgs = yd_pred_lbfgs * L
Q_lbfgs = Vd_pred_lbfgs * Vc
q_lbfgs = Q_lbfgs * y_lbfgs


# Prediction
x_test, t_test = np.meshgrid(
    np.linspace(-xd_max, xd_max, 1000),
    td_f
)

X_test = tf.convert_to_tensor(
    np.vstack([x_test.flatten(), t_test.flatten()]).T,
    dtype=DTYPE
)

yd_pred = yd_net(X_test).numpy().reshape(-1,1)
Vd_pred = Vd_net(X_test).numpy().reshape(-1,1)

y_pred = yd_pred * L
Q_pred = Vd_pred * Vc
q = Q_pred * y_pred

xt = x_test * L
QQ_pred = q.reshape(1000,1)
xx_test = xt.reshape(1000,1)
yy_pred = y_pred.reshape(1000,1)

# Exact solution (unchanged)
y1 = 10
y3_val = 5.078
t_phys = 20

x2 = np.linspace(-27,198,225)
y2 = np.full((224), y3_val)
y8 = 2
y22 = np.append(y2, y8)

v2 = 2*np.sqrt(g*y1) - 2*np.sqrt(g*y2)
q2 = v2*y2
q22 = np.append(q2,0)

x1 = np.linspace(-198,-27,171)
z = 2*t_phys*np.sqrt(g*y1)
z2 = 3*t_phys*np.sqrt(g)
y11 = ((x1-z)/-z2)**2
v3 = 2*np.sqrt(g*y1) - 2*np.sqrt(g*y11)
q3 = v3*y11

x3 = np.linspace(-198,-500,302)
y3_exact = np.full((302),10)
q4 = np.full((302),0)

x4 = np.linspace(198,500,302)
y4 = np.full((302),2)
q5 = np.full((302),0)


# Error
y2_pred = yy_pred.reshape(1000,)
mse2 = np.sum((y11 - y2_pred[302:473])**2)
mse3 = np.sum((y22 - y2_pred[473:698])**2)
mse1 = np.sum((y3_exact - y2_pred[0:302])**2)
mse4 = np.sum((y4 - y2_pred[698:1000])**2)

mse_y = (1/1000) * np.sum((mse1, mse2, mse3, mse4))
rmsd_y = np.sqrt(mse_y)
print(mse_y, rmsd_y)

q2_pred = QQ_pred.reshape(1000,)
mse22 = np.sum((q3 - q2_pred[302:473])**2)
mse33 = np.sum((q22 - q2_pred[473:698])**2)
mse11 = np.sum((q4 - q2_pred[0:302])**2)
mse44 = np.sum((q5 - q2_pred[698:1000])**2)

mse_q = (1/1000) * np.sum((mse11, mse22, mse33, mse44))
rmsd_q = np.sqrt(mse_q)
print(mse_q, rmsd_q)


# Load comparison data

data11 = np.load('data1y.npy')
data12 = np.load('data1q.npy')

# Plot
plt.figure(figsize=(20,4))

# --- Water depth ---
plt.subplot(1,3,1)
plt.plot(xx_test, y_adam, '--', color='orange', label='Adam')
plt.plot(xx_test, y_lbfgs, '-', color='green', label='Adam + L-BFGS')
plt.plot(xx_test, data11, color="blue", label='PINN-RWF')
plt.plot(x1,y11,'r--',label='Exact')
plt.plot(x2,y22,'r--')
plt.plot(x3,y3_exact,'r--')
plt.plot(x4,y4,'r--')
plt.legend()
plt.ylabel('y (m)')
plt.xlabel('x (m)')
plt.title('Water Depth')

# --- Discharge ---
plt.subplot(1,3,2)
plt.plot(xx_test, q_adam, '--', color='orange', label='Adam')
plt.plot(xx_test, q_lbfgs, '-', color='green', label='Adam + L-BFGS')
plt.plot(xx_test, data12,color="blue",label='PINN-RWF')
plt.plot(x1,q3,'r--',label='Exact')
plt.plot(x2,q22,'r--')
plt.plot(x3,q4,'r--')
plt.plot(x4,q5,'r--')
plt.legend()
plt.ylabel('Q')
plt.xlabel('x')
plt.title('Discharge')

# --- Loss comparison ---
plt.subplot(1,3,3)
plt.semilogy(adam_loss_history, label='Adam')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss (Adam)')

plt.tight_layout()
plt.show()
