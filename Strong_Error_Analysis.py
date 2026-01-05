# Comparison of Scheme Errors by using solution of strongly convergent scheme as true solution

from Useful_Functions import *
from Explicit_Schemes import *
from Implicit_Schemes import *
import matplotlib.pyplot as plt

x = 0.03
a = 0.02
FR = [0.25, 0.45, 0.75, 1.15]
k = 0.4
T = 1
max_index = 14
N_MC = 20000
colors = {Partial_Truncation: 'orange', Full_Truncation: 'blue', Partial_Reflection: 'black', Reflection: 'red', Implicit_1: 'grey', Implicit_2: 'purple',
                  Modified_Milstein: 'green', Adjusted_Modified_Milstein: 'green', Truncated_Milstein: 'palevioletred'}
label_list = {Partial_Truncation: 'Partial Truncation', Full_Truncation: 'Full Truncation', Partial_Reflection: 'Partial Reflection', Reflection: 'Reflection', Implicit_1: 'Implicit 1',
                      Implicit_2: 'Implicit 2', Modified_Milstein: 'Modified Milstein',
                      Adjusted_Modified_Milstein: '(Truncated) Modified Milstein', Truncated_Milstein: 'Truncated Milstein'}

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]
dW = Wiener_inc(2 ** (-max_index), N_MC=N_MC, T=T)

for i in range(len(FR)):
    sigma = np.sqrt(2 * a / FR[i])
    true_X1, true_X2,  _ = Truncated_Milstein(dW, Antithetic=True, x=x, N_MC=N_MC, T=T, a=a, sigma=sigma, k=k)

    # get true process
    if FR[i] >= 1:
        scheme_list = [Partial_Truncation, Full_Truncation, Partial_Reflection, Reflection, Implicit_1, Implicit_2,
                       Modified_Milstein, Truncated_Milstein]

    elif FR[i] >= 0.5:
        scheme_list = [Partial_Truncation, Full_Truncation, Partial_Reflection, Reflection, Implicit_2, Modified_Milstein,
                       Truncated_Milstein]

    else:
        scheme_list = [Partial_Truncation, Full_Truncation, Partial_Reflection, Reflection, Adjusted_Modified_Milstein,
         Truncated_Milstein]

    colors = {Partial_Truncation: 'orange', Full_Truncation: 'blue', Partial_Reflection: 'black', Reflection: 'red', Implicit_1: 'grey', Implicit_2: 'purple',
                  Modified_Milstein: 'green', Adjusted_Modified_Milstein: 'green', Truncated_Milstein: 'palevioletred'}
    label_list = {Partial_Truncation: 'Partial Truncation', Full_Truncation: 'Full Truncation', Partial_Reflection: 'Partial Reflection', Reflection: 'Reflection', Implicit_1: 'Implicit 1',
                      Implicit_2: 'Implicit 2', Modified_Milstein: 'Modified Milstein',
                      Adjusted_Modified_Milstein: '(Truncated) Modified Milstein', Truncated_Milstein: 'Truncated Milstein'}

    MC_errs = []
    runtimes = []

    for scheme in scheme_list:
        err, runtime, MC_err = error_analysis(dW, true_X1, true_X2, scheme, max_timestep=1/2, min_timestep_index=max_index-1, T=T, N_MC=N_MC, x=x, a=a, sigma=sigma, k=k)
        MC_errs.append(MC_err)
        runtimes.append(runtime)
        slope, line = line_of_best_fit(np.log(runtime), err)
        print(str(FR[i]) + scheme.__name__ + str(-slope))
        axes[i].plot(np.log(runtime),
                 line(np.log(runtime)), '-.', label=label_list[scheme], color=colors[scheme])
        axes[i].plot(np.log(runtime),
                     err, 'o', ms=4, color=colors[scheme])

    axes[i].set_xlabel('Run time in Seconds (Log Scale)')
    axes[i].legend()
    axes[i].grid()
    axes[i].set_title(f'Feller Ratio = {FR[i]}, Average Log MC Error = {np.round(np.mean(MC_errs), 2)}')
    axes[i].set_ylabel('RMSE (Log Scale)')

plt.savefig('Strong Convergence (1)')






