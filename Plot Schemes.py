import matplotlib.pyplot as plt
from Explicit_Schemes import *
from Implicit_Schemes import *
from Useful_Functions import *

dt = 1/500
a = 0.005
x = 0.005
k = 1
sigma = np.sqrt(2 * a / 0.2)
dW = Wiener_inc(dt, N_MC=1)
timepoints = np.linspace(0, 1, 500)

# Truncation schemes
X_1, _ = Partial_Truncation(dW, x=x, N_MC=1, T=1, a=a, sigma=sigma, k=k)
X_2, _ = Full_Truncation(dW, x=x, N_MC=1, T=1, a=a, sigma=sigma, k=k)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
ax[0].plot(timepoints, X_1[0])
ax[1].plot(timepoints, X_2[0])

for i in range(2):
    ax[i].set_xlabel('t')
    ax[i].set_ylabel('X')
    ax[i].grid()
ax[0].set_title('Partial Truncation Scheme')
ax[1].set_title('Full Truncation Scheme')
plt.savefig('Truncation Schemes')

# reflection schemes
X_1, _ = Partial_Reflection(dW, x=x, N_MC=1, T=1, a=a, sigma=sigma, k=k)
X_2, _ = Reflection(dW, x=x, N_MC=1, T=1, a=a, sigma=sigma, k=k)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5), sharey=True)
ax[0].plot(timepoints, X_1[0])
ax[1].plot(timepoints, X_2[0])

for i in range(2):
    ax[i].set_xlabel('t')
    ax[i].set_ylabel('X')
    ax[i].grid()
ax[0].set_title('Partial Reflection Scheme')
ax[1].set_title('Reflection Scheme')
plt.savefig('Reflection Schemes')
