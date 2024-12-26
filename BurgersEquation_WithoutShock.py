import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Define Burgers' equation parameters
nu = 0.01 / np.pi  # Viscosity
L = 2.0            # Length of the domain
T = 1.0            # Simulation time

# Create training data (collocation points)
N_x = 256  # Spatial points
N_t = 100  # Temporal points
x = np.linspace(-L, L, N_x)
t = np.linspace(0, T, N_t)
X, T = np.meshgrid(x, t)
x_train = X.flatten()[:, None]
t_train = T.flatten()[:, None]
XT_train = np.hstack((x_train, t_train))

# Initial and boundary conditions
u_train = np.ones_like(x_train) * nu
u0 = -np.sin(np.pi * x_train)  # Initial condition
u_train[t_train[:, 0] == 0] = u0[:, 0]

# PINN model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)),
        tf.keras.layers.Dense(50, activation="tanh"),
        tf.keras.layers.Dense(50, activation="tanh"),
        tf.keras.layers.Dense(50, activation="tanh"),
        tf.keras.layers.Dense(1, activation=None)
    ])
    return model

model = create_model()

# Physics-Informed Loss function
def compute_loss(model, x, t):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch([x, t])
        xt = tf.concat([x, t], axis=1)
        u = model(xt)  # Predict solution
        u_x = tape.gradient(u, x)
        u_t = tape.gradient(u, t)
    
    u_xx = tape.gradient(u_x, x)
    
    # Burgers' equation residual
    f = u_t + u * u_x - nu * u_xx

    # Mean squared residuals
    mse_f = tf.reduce_mean(tf.square(f))

    # Initial condition loss
    u_pred = model(tf.concat([x[t[:, 0] == 0], t[t[:, 0] == 0]], axis=1))
    mse_ic = tf.reduce_mean(tf.square(u_pred - u0))

    return mse_f + mse_ic

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# Training loop
@tf.function
def train_step(x, t):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x, t)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

losses = []
epochs = 20000
for epoch in range(epochs):
    loss = train_step(tf.convert_to_tensor(x_train, dtype=tf.float32),
                      tf.convert_to_tensor(t_train, dtype=tf.float32))
    losses.append(loss.numpy())
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.numpy()}")

# Plot the loss curve
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

# Predict the solution
X_star = np.hstack((x_train, t_train))
u_pred = model.predict(X_star)

# Reshape predictions to plot
U_pred = u_pred.reshape(N_t, N_x)

plt.figure(figsize=(10, 6))
plt.contourf(X, T, U_pred, levels=100, cmap="jet")
plt.colorbar()
plt.xlabel("x")
plt.ylabel("t")
plt.title("PINN Solution to Burgers' Equation")
plt.show()
