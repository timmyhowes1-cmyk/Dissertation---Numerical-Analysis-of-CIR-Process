from Useful_Functions import *
from Explicit_Schemes import *
from Implicit_Schemes import *
import matplotlib.pyplot as plt

x = 0.03
a = 0.02
FR = [0.25, 0.45, 0.75, 1.15]
k = 0.4
T = 1
max_index = 13
N_MC = 20000
colors = {Partial_Truncation: 'orange', Full_Truncation: 'blue', Partial_Reflection: 'black', Reflection: 'red', Implicit_1: 'grey', Implicit_2: 'purple',
                  Modified_Milstein: 'green', Adjusted_Modified_Milstein: 'green', Truncated_Milstein: 'palevioletred'}
label_list = {Partial_Truncation: 'Partial Truncation', Full_Truncation: 'Full Truncation', Partial_Reflection: 'Partial Reflection', Reflection: 'Reflection', Implicit_1: 'Implicit 1',
                      Implicit_2: 'Implicit 2', Modified_Milstein: 'Modified Milstein',
                      Adjusted_Modified_Milstein: '(Truncated) Modified Milstein', Truncated_Milstein: 'Truncated Milstein'}

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1]]

for i in range(len(FR)):
    sigma = np.sqrt(2 * a / FR[i])

    scheme_list = [Partial_Truncation, Partial_Reflection]

    dt = [2 ** (-i) for i in range(1, max_index + 1) if 2 ** (-i) <= 1 / 4]

    for scheme in scheme_list:
        neg_probs = []
        runtimes = []
        for j in range(len(dt)):

            dW = Wiener_inc(dt[j], N_MC=N_MC, T=T)

            # simulate and time
            X1, X2, t = scheme(dW, Antithetic=True, x=x, N_MC=N_MC, T=T, a=a, sigma=sigma, k=k)

            _, prob_of_neg1 = prob_of_nonpositive(X1)
            _, prob_of_neg2 = prob_of_nonpositive(X2)

            neg_probs.append(prob_of_neg1 / 2 + prob_of_neg2 / 2)
            runtimes.append(t)

        axes[i].plot(np.log(runtimes), neg_probs, 'o', label=label_list[scheme], color=colors[scheme])

    axes[i].set_xlabel('Run time in Seconds (Log Scale)')
    axes[i].legend()
    axes[i].grid()
    axes[i].set_title(f'Feller Ratio = {FR[i]}')
    axes[i].set_ylabel('Probability of Going Negative')

plt.savefig('Negative Probability')