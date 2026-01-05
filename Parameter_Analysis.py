# Analysis of convergence of modified Milstein Scheme as Feller ratio decreases towards 1
from Useful_Functions import *
from Explicit_Schemes import *
from Implicit_Schemes import *

a = 0.02
k = 0.4
x = 0.03
T = 1
max_index = 14
N_MC = 20000
Feller_Ratios = [0.2, 0.3, 0.35, 0.4, 0.5]



schemes = [Truncated_Milstein, Adjusted_Modified_Milstein]
dW = Wiener_inc(2 ** (-max_index), N_MC=N_MC, T=T)

fig = plt.figure(figsize=(10, 10))
orders_1 = []
orders_2 = []

for i in range(len(Feller_Ratios)):
    sigma = np.sqrt(2 * a / Feller_Ratios[i])
    true_X1, true_X2, _ = Truncated_Milstein(dW, Antithetic=True, x=x, N_MC=N_MC, T=T, a=a, sigma=sigma, k=k)
    err, runtime, _, _, _ = error_analysis(dW, true_X1, true_X2, schemes[0], max_timestep=1/2, min_timestep_index=max_index - 3, T=T, N_MC=N_MC, x=x, a=a,
                                            sigma=sigma, k=k)

    slope, _ = line_of_best_fit(np.log(runtime), err)
    orders_1.append(-slope)

    err, runtime, _, _, _ = error_analysis(dW, true_X1, true_X2, schemes[1], max_timestep=1 / 2,
                                             min_timestep_index=max_index - 2, T=T, N_MC=N_MC, x=x, a=a,
                                             sigma=sigma, k=k)

    slope, _ = line_of_best_fit(np.log(runtime), err)
    orders_2.append(-slope)
    if i == 0:
        orders_1.append(0.38438783095239515)
        orders_2.append(0.34881427171012025)
    elif i == 3:
        orders_1.append(0.4359664686735729)
        orders_2.append(0.4446202845195378)

Feller_Ratios = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
_, line = line_of_best_fit(Feller_Ratios, orders_1)
plt.plot(Feller_Ratios, line(Feller_Ratios), '-.', color='palevioletred', label='Truncated Milstein')
plt.plot(Feller_Ratios, orders_1, 'o', color='palevioletred')

_, line = line_of_best_fit(Feller_Ratios, orders_2)
plt.plot(Feller_Ratios, line(Feller_Ratios), '-.', color='green', label='(Truncated) Modified Milstein')
plt.plot(Feller_Ratios, orders_2, 'o', color='green')

plt.title('Truncated Milstein Scheme vs. Modified Milstein Scheme Truncated at 0')
plt.xlabel('Feller Ratio')
plt.ylabel('Observed Order of Convergence')
plt.legend()
plt.grid()

plt.savefig('Order of Convergence, FR < ' + str(0.5) + '.png')

print(orders_1)
print(orders_2)

