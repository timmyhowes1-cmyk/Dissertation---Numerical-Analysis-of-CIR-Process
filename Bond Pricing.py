from Useful_Functions import *
from Implicit_Schemes import *
from Explicit_Schemes import *

a = 0.02
k = 0.4
x = 0.03
T = 1
N_MC = 500000
Feller_ratios = [0.25, 0.45, 0.75, 1.15]
my_list = [Partial_Truncation, Full_Truncation, Partial_Reflection, Reflection, Implicit_1, Implicit_2, Modified_Milstein, Truncated_Milstein]
colors = {Partial_Truncation: 'orange', Full_Truncation: 'blue', Partial_Reflection: 'black', Reflection: 'red', Implicit_1: 'grey', Implicit_2: 'purple', Modified_Milstein: 'green', Adjusted_Modified_Milstein: 'green', Truncated_Milstein: 'palevioletred'}
label_list = {Partial_Truncation: 'Partial Truncation', Full_Truncation: 'Full Truncation', Partial_Reflection: 'Partial Reflection', Reflection: 'Reflection', Implicit_1: 'Implicit 1',
                  Implicit_2: 'Implicit 2', Modified_Milstein: 'Modified Milstein',
                  Adjusted_Modified_Milstein: '(Truncated) Modified Milstein', Truncated_Milstein: 'Truncated Milstein'}

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))

for i in range(len(Feller_ratios)):
    print(i)
    if i < 2:
        dt = [1/2, 1/4, 1/8, 1/32, 1/64]

    else:
        dt = [1/8, 1/16, 1/32, 1/64, 1/128]

    sigma = np.sqrt(2 * a / Feller_ratios[i])
    true_price = True_Bond_price(T=T, sigma=sigma, a=a, k=k, x=x)

    if Feller_ratios[i] <= 0.5:
        scheme_list = [Partial_Truncation, Full_Truncation, Partial_Reflection, Reflection, Adjusted_Modified_Milstein,
                       Truncated_Milstein]

    elif 0.5 < Feller_ratios[i] <= 1:
        scheme_list = [Partial_Truncation, Full_Truncation, Partial_Reflection, Reflection, Implicit_2, Modified_Milstein,
                       Truncated_Milstein]

    else:
        scheme_list = my_list

    MC_errors = []
    for j in range(len(scheme_list)):
        err, runtime, MC_err = Bond_error(true_price, scheme_list[j], dt, T=T, N_MC=N_MC, x=x, a=a, sigma=sigma, k=k)
        _, line = line_of_best_fit(np.log(runtime), np.log(err), 1)
        if i <= 1:
            ax[0, i].plot(np.log(runtime), line(np.log(runtime)), '-.', color=colors[scheme_list[j]], label=label_list[scheme_list[j]])
            ax[0, i].plot(np.log(runtime), np.log(err), 'o', color=colors[scheme_list[j]],
                          ms=4)
        else:
            ax[1, i-2].plot(np.log(runtime), line(np.log(runtime)), '-.', color=colors[scheme_list[j]],
                          label=label_list[scheme_list[j]])
            ax[1, i-2].plot(np.log(runtime), np.log(err), 'o', color=colors[scheme_list[j]],
                            ms=4)

        MC_errors.append(MC_err)

    average_MC = np.log(np.mean(MC_errors))

    if i <= 1:
        ax[0, i].set_xlabel('Runtime in Seconds (Log Scale)')
        ax[0, i].set_ylabel('Bond Price Error (Log Scale)')
        ax[0, i].set_title(f'Feller Ratio = {Feller_ratios[i]}, Average Log MC Error = {np.round(average_MC, 2)}')
        ax[0, i].legend()
        ax[0, i].grid()
    else:
        ax[1, i-2].set_xlabel('Runtime in Seconds (Log Scale)')
        ax[1, i-2].set_ylabel('Bond Price Error (Log Scale)')
        ax[1, i-2].set_title(f'Feller Ratio = {Feller_ratios[i]}, Average Log MC Error = {np.round(average_MC, 2)}')
        ax[1, i-2].legend()
        ax[1, i-2].grid()

plt.savefig('Bond Pricing Comparison')





