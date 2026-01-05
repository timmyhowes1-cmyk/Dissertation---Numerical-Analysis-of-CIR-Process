from Useful_Functions import *
from Implicit_Schemes import *
from Explicit_Schemes import *

a = 0.012
k = 0.3
x = 0.04
T = 1
dt = [1/2, 1/4, 1/8, 1/16, 1/32]
N_MC = 5000000
Feller_ratios = [0.3, 0.75, 1.1]
moments = [1, 2, 3, 4]
schemes = [FTE, Modified_Milstein, Truncated_Milstein]
colors = {FTE: 'green', Modified_Milstein: 'blue', Truncated_Milstein: 'orange'}

for i in range(len(Feller_ratios)):
    fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20, 20))
    sigma = np.sqrt(2 * a / Feller_ratios[i])

    if Feller_ratios[i] <= 0.5 and (Modified_Milstein in schemes):
        schemes.remove(Modified_Milstein)

    elif Feller_ratios[i] > 0.5 and (Modified_Milstein not in schemes):
        schemes.append(Modified_Milstein)
        schemes[1], schemes[2] = schemes[2], schemes[1]

    for j in range(4):
        MC_err = []

        for s in range(len(schemes)):
            err, run_time, line, MC_error = Moments_Error(schemes[s], dt, Antithetic=True, moment=moments[j], T=T, N_MC=N_MC, x=x, a=a,
                                                          sigma=sigma, k=k)
            MC_err.append(np.log(MC_error))
            ax[j].plot(np.log(run_time), line(np.log(run_time)), '-.', label=str(schemes[s].__name__) + ', Log MC_Error=' + str(np.round(np.log(MC_error), 2)), color=colors[schemes[s]])
            ax[j].plot(np.log(run_time), np.log(err), 'o', color=colors[schemes[s]])

        ax[j].hlines(y=np.mean(MC_err), xmin=np.log(run_time[0])-1, xmax=np.log(run_time[-1])+1, color='red', label='Average Log MC Error')
        ax[j].set_xlabel('Runtime in Seconds (Log Scale)')
        ax[j].set_ylabel('Moment Error (Log Scale)')

        if j==0:
            ax[j].set_title(f'{moments[j]}st Moment Error, FR = {Feller_ratios[i]}')
        elif j==1 or j==2:
            ax[j].set_title(f'{moments[j]}rd Moment Error, FR = {Feller_ratios[i]}')
        else:
            ax[j].set_title(f'{moments[j]}th Moment Error, FR = {Feller_ratios[i]}')

        ax[j].grid()
        ax[j].legend()

    plt.savefig('Moment Error FR = ' + str(Feller_ratios[i]) + '.png')
