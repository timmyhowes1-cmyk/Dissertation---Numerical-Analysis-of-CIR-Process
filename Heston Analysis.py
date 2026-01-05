from Useful_Functions import *
from Explicit_Schemes import *
from Implicit_Schemes import *

a = 0.08
k = 0.4
x = 0.17
ro = -0.9
S0 = 1
r = 0
K = 1.1
T = 1
#max_index = 10
N_MC = 450000
Feller_ratios = [0.25, 0.45, 0.75, 1.15]
my_list = [Full_Truncation, Reflection, Implicit_1, Implicit_2, Modified_Milstein, Truncated_Milstein]
colors = {Full_Truncation: 'blue', Reflection: 'red', Implicit_1: 'grey', Implicit_2: 'purple', Modified_Milstein: 'green', Adjusted_Modified_Milstein: 'green', Truncated_Milstein: 'palevioletred'}
label_list = {Full_Truncation: 'Full Truncation', Reflection: 'Reflection', Implicit_1: 'Implicit 1', Implicit_2: 'Implicit 2', Modified_Milstein: 'Modified Milstein', Adjusted_Modified_Milstein: '(Truncated) Modified Milstein', Truncated_Milstein: 'Truncated Milstein'}

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

for i in range(len(Feller_ratios)):
    if i == 0:
        dt = [1/10, 1/16, 1/32, 1/64, 1/120, 1/220]
    else:
        dt = [1/2, 1/3, 1/4, 1/6, 1/8, 1/16]

    sigma = np.sqrt(2 * a / Feller_ratios[i])
    true_price = Heston_call_option(ro=ro, r=r, S0=S0, K=K, v0=x, T=T,  a=a, sigma=sigma, k=k)
    print(true_price)

    if Feller_ratios[i] < 0.5:
        scheme_list = [Full_Truncation, Reflection, Adjusted_Modified_Milstein, Truncated_Milstein]

    elif 0.5 <= Feller_ratios[i] <= 1:
        scheme_list = [Full_Truncation, Reflection, Implicit_2, Modified_Milstein,
                       Truncated_Milstein]

    else:
        scheme_list = my_list

    MC_errors = []
    for j in range(len(scheme_list)):
        print(j)

        err, runtime, line, MC_err = Heston_analysis(true_price, scheme_list[j], dt, K=K, ro=ro, r=r, S0=S0, T=T, N_MC=N_MC, x=x, a=a, sigma=sigma, k=k)
        if i <= 1:
            ax[0, i].plot(np.log(runtime), line(np.log(runtime)), '-.', color=colors[scheme_list[j]], label=label_list[scheme_list[j]])
            ax[0, i].plot(np.log(runtime), np.log(err), 'o', ms=4, color=colors[scheme_list[j]])

        else:
            ax[1, i-2].plot(np.log(runtime), line(np.log(runtime)), '-.', color=colors[scheme_list[j]],
                          label=label_list[scheme_list[j]])
            ax[1, i-2].plot(np.log(runtime), np.log(err), 'o', ms=4, color=colors[scheme_list[j]])

        MC_errors.append(MC_err)
    average_MC = np.log(np.mean(MC_errors))

    if i <= 1:
        ax[0, i].set_xlabel('Runtime in Seconds (Log Scale)')
        ax[0, i].set_ylabel('Option Price Error (Log Scale)')
        ax[0, i].set_title(f'Feller Ratio = {Feller_ratios[i]}, Average Log MC Error = {np.round(average_MC, 2)}')
        ax[0, i].legend()
        ax[0, i].grid()

    else:
        ax[1, i - 2].set_xlabel('Runtime in Seconds (Log Scale)')
        ax[1, i - 2].set_ylabel('Option Price Error (Log Scale)')
        ax[1, i - 2].set_title(f'Feller Ratio = {Feller_ratios[i]}, Average Log MC Error = {np.round(average_MC, 2)}')
        ax[1, i - 2].legend()
        ax[1, i - 2].grid()

plt.savefig('Heston Analysis (1)')