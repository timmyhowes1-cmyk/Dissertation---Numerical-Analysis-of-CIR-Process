# Useful Helper Functions

import numpy as np
import matplotlib.pyplot as plt


def Wiener_inc(dt, N_MC=2000, T=1, ro=0):
    '''
    Generates Wiener increments into an array of size T/dt x N_MC
    :param dt: (float) Timestep size
    :param N_MC: (int) Number of Monte Carlo samples
    :param T: (float) Terminal time
    :param ro: (float) Correlation of Weiner processes. Between -1 and 1
    :return: (ndarray) Wiener increments
    '''
    dW1 = np.zeros([N_MC, int(T / dt)])
    dW1[:, 1:] = np.sqrt(dt) * np.random.standard_normal(size=(N_MC, int(T / dt) - 1))
    if ro != 0:
        dW2 = np.zeros([N_MC, int(T / dt)])
        dW2[:, 1:] = np.sqrt(dt) * np.random.standard_normal(size=(N_MC, int(T / dt) - 1))
        return ro * dW1 + np.sqrt(1 - ro**2) * dW2, dW1
    else:
        return dW1


def prob_of_nonpositive(X):
    '''
    Calculates probability of scheme hitting 0 or going negative.
    :param X: (ndarray) Simulated CIR processes.
    :return:
    '''

    negative_prob = len([1 for i in range(X.shape[0]) if X[i, -1] < 0]) / X.shape[0]
    zero_prob = len([1 for i in range(X.shape[0]) if X[i, -1] == 0]) / X.shape[0]

    return zero_prob, negative_prob


def error_analysis(dW, true_X1, true_X2, scheme, max_timestep=1/16, min_timestep_index=14, T=1, N_MC=2000, x=0.1, a=0.1, sigma=0.2, k=0.5):
    '''
    Calculates Strong error approximations for a given scheme.
    :param scheme: (function name) Scheme to implement.
    :param max_timestep: (float) biggest timestep to use
    :param min_timestep_index: (int) index which makes largest timestep 2 ** (-min_timestep_index)
    :param T: (float) Terminal time
    :param N_MC: (int) Number of Monte Carlo samples
    :param x: (nonnegative float) initial value for CIR process
    :param a: (nonnegative float) parameter for CIR process
    :param sigma: (nonnegative float) parameter for CIR process
    :param k: (nonnegative float) parameter for CIR process
    :return: strong_err: (list) log of L^2 Error for each timestep
    '''

    dt = [2 ** (-i) for i in range(1, min_timestep_index + 1) if 2 ** (-i) <= max_timestep]
    true_dt = T / dW.shape[1]

    strong_err = []
    run_time = []

    for i in range(len(dt)):
        step = int(dt[i] / true_dt)
        new_dW = np.transpose([np.sum(dW[:, (step*j):(step*(j + 1))], axis=1) for j in range(int((true_dt**(-1)) / step))])

        # simulate and time
        X1, X2, t = scheme(new_dW, Antithetic=True, x=x, N_MC=N_MC, T=T,  a=a, sigma=sigma, k=k)

        run_time.append(t)

        strong_err.append(np.sqrt(np.mean(((true_X1[:, -1] - X1[:, -1]) ** 2) / 2 + ((true_X2[:, -1] - X2[:, -1]) ** 2) / 2)))

    MC_err = np.log(np.sqrt(np.var((X1[:, -1] / 2 + X2[:, -1] / 2))/ N_MC))
    return np.log(strong_err), run_time, MC_err

def True_Bond_price(T=1, sigma=0.2, a=0.1, k=0.5, x=0.1):
    h = np.sqrt(k ** 2 + 2 * (sigma ** 2))
    A = ((2 * h * np.exp((k + h) * T / 2)) / (2 * h + (k + h)*(np.exp(h * T) - 1))) ** (2 * a / (sigma ** 2))
    B = (2 * (np.exp(h * T) - 1)) / (2*h + (k + h) * (np.exp(h * T) - 1))
    return A * np.exp(-B * x)

def Bond_error(true_price, scheme, dt, T=1, N_MC=2000, x=0.1, a=0.1, sigma=0.2, k=0.5):

    bond_prices = []
    err = []
    run_time = []

    for i in range(len(dt)):
        dW = Wiener_inc(dt[i], N_MC=N_MC, T=T)

        # simulate and time
        X1, X2, t = scheme(dW, Antithetic=True, x=x, N_MC=N_MC, T=T, a=a, sigma=sigma, k=k)

        # approximate integrated process
        integral1 = np.zeros(N_MC)
        integral2 = np.zeros(N_MC)
        for n in range(1, X1.shape[1]):
            # use trapazoidal rule
            integral1 += (dt[i] / 2) * (X1[:, n] + X1[:, n - 1])
            integral2 += (dt[i] / 2) * (X2[:, n] + X2[:, n - 1])

        bond_prices.append(np.mean(np.exp(-integral1) / 2 + np.exp(-integral2) / 2))
        run_time.append(t)
        err.append(np.abs(bond_prices[-1] - true_price))

    MC_error = np.sqrt(np.var(np.exp(-integral1) / 2 + np.exp(-integral2) / 2) / N_MC)
    return err, run_time, MC_error

def true_moments(moment, degree, non_centrality, c):
    if moment == 1:
        true_moment = (degree + non_centrality) / c
    elif moment == 2:
        true_moment = ((degree + non_centrality) ** 2 + 2 * (degree + 2 * non_centrality)) / (c ** 2)
    elif moment == 3:
        true_moment = ((degree + non_centrality) ** 3 + 6 * (degree + non_centrality) * (degree + 2 * non_centrality) + 8 * (degree + 3 * non_centrality)) / (c ** 3)
    elif moment == 4:
        true_moment = ((degree + non_centrality) ** 4 + 12 * (degree + non_centrality) ** 2 * (degree + 2 * non_centrality) + 4 * (11 * degree ** 2 + 44 * degree * non_centrality + 36 * non_centrality ** 2) + 48 * (degree + 4 * non_centrality)) / (c ** 4)
    return true_moment

def Moments_Error(scheme, dt, Antithetic=True, moment = 1, T=1, N_MC=2000, x=0.1, a=0.1, sigma=0.2, k=0.5):
    err = []
    run_time = []

    if k != 0:
        c = 4 / ((sigma ** 2) * (1 - np.exp(-k * T)) / k)
    else:
        c = 4 / (T * (sigma ** 2))
    d = c * np.exp(-k * T)
    degree = 4 * a / (sigma ** 2)
    non_centrality = x * d
    true = true_moments(moment, degree, non_centrality, c)

    if not Antithetic:
        for i in range(len(dt)):
            dW = Wiener_inc(dt[i], N_MC=N_MC, T=T)

            # simulate and time
            X_sim, t = scheme(dW, Antithetic, x, N_MC, T, a, sigma, k)

            run_time.append(t)
            err.append(np.abs(np.mean(X_sim[:, -1] ** moment) - true))

        MC_error = np.sqrt(np.var(X_sim[:, -1] ** moment) / N_MC)
        _, line = line_of_best_fit(np.log(run_time), np.log(err))

    else:
        for i in range(len(dt)):
            dW = Wiener_inc(dt[i], N_MC=N_MC, T=T)

            # simulate and time
            X1, X2, t = scheme(dW, Antithetic, x, N_MC, T, a, sigma, k)

            run_time.append(t)
            err.append(np.abs(np.mean((X1[:, -1] ** moment + X2[:, -1] ** moment) / 2) - true))

        MC_error = np.sqrt(np.var((X1[:, -1] ** moment + X2[:, -1] ** moment) / 2) / N_MC)
        _, line = line_of_best_fit(np.log(run_time), np.log(err))

    return err, run_time, line, MC_error

def simulate_Heston(dt, scheme, ro=0.5, S0=100, r=0.02, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    dW2, dW1 = Wiener_inc(dt, N_MC=N_MC, T=T, ro=ro)
    S1 = np.zeros([N_MC, dW1.shape[1]])
    S1[:, 0] = S0
    S2 = np.zeros([N_MC, dW1.shape[1]])
    S2[:, 0] = S0
    V1, V2, runtime = scheme(dW2, Antithetic=True, x=x, N_MC=N_MC, T=T, a=a, sigma=sigma, k=k)

    for i in range(1, S1.shape[1]):
        S1[:, i] = S1[:, i-1] * np.exp((r - 0.5 * V1[:, i-1]) * dt + np.sqrt(V1[:, i-1]) * dW1[:, i])
        S2[:, i] = S2[:, i - 1] * np.exp((r - 0.5 * V2[:, i - 1]) * dt - np.sqrt(V2[:, i - 1]) * dW1[:, i])

    return S1, S2, runtime

def Heston_analysis(true_price, scheme, dt, K=90, ro=0.5, r=0.02, S0=100, T=1, N_MC=2000, x=0.1, a=0.1, sigma=0.2, k=0.5):
    #dt = [2 ** (-i) for i in range(1, min_timestep_index + 1) if 2 ** (-i) <= max_timestep]
    err = []
    run_time = []

    for i in range(len(dt)):

        # simulate and time
        S1, S2, t = simulate_Heston(dt[i], scheme, ro=ro, r=r, S0=S0, x=x, N_MC=N_MC, T=T,  a=a, sigma=sigma, k=k)

        run_time.append(t)

        estimated_price = np.exp(-r * T) * np.mean((np.maximum(S1[:, -1] - K, 0) + np.maximum(S2[:, -1] - K, 0)) / 2)

        err.append(np.abs(true_price - estimated_price))

    MC_error = np.sqrt(np.var(np.exp(-r * T) * (np.maximum(S1[:, -1] - K, 0) + np.maximum(S2[:, -1] - K, 0)) / 2) / N_MC)

    _, line = line_of_best_fit(np.log(run_time), np.log(err))

    return err, run_time, line, MC_error

def line_of_best_fit(X, Y, degree=1):
    coeff = np.polyfit(X, Y, degree)
    return coeff[0], np.poly1d(coeff)

# Heston Functions
def d(phi, ro, b, u, sigma):
    return np.sqrt((complex(0, ro * sigma * phi) - b) ** 2 - sigma**2 * (complex(0, 2 * u * phi) - phi**2))

def g(phi, ro, b, u, sigma):
    return (b - complex(0, ro * sigma * phi) - d(phi, ro, b, u, sigma)) / (b - complex(0, ro * sigma * phi) + d(phi, ro, b, u, sigma))

def C(phi, T, r, ro, a, sigma, b, u):
    return complex(0, r * phi * T) + a / (sigma**2) * (T * (b - complex(0, ro * sigma * phi) - d(phi, ro, b, u, sigma))
                                                       - 2 * np.log((1 - g(phi, ro, b, u, sigma) * np.exp(-T * d(phi, ro, b, u, sigma))) / (1 - g(phi, ro, b, u, sigma))))

def D(phi, T, ro, sigma, b, u):
    return ((b - complex(0, ro * sigma * phi) - d(phi, ro, b, u, sigma)) / (sigma**2)) * ((1 - np.exp(-T * d(phi, ro, b, u, sigma))) / (1 - g(phi, ro, b, u, sigma) * np.exp(-T * d(phi, ro, b, u, sigma))))

def psi(phi, T, r, ro, a, sigma, b, u, S0, v0):
    return np.exp(C(phi, T, r, ro, a, sigma, b, u) + v0 * D(phi, T, ro, sigma, b, u) + complex(0, S0 * phi))

def Heston_call_option(ro, r, S0, K, v0, T,  a, sigma, k):

    b = [k - ro * sigma, k]
    u = [1/2, -1/2]

    phi_values, step_size = np.linspace(0.0000001, 175, 8000, endpoint=True, retstep=True)

    P1 = 1/2
    P2 = 1/2

    for i in range(1, len(phi_values)):
        if np.round(np.abs(step_size / 2 * ((np.exp(complex(0, -phi_values[i-1] * np.log(K))) * psi(phi=phi_values[i-1], T=T, r=r, ro=ro, a=a, sigma=sigma, b=b[0], u=u[0], S0=np.log(S0), v0=v0) / complex(0, phi_values[i - 1])).real
                               + (np.exp(complex(0, -phi_values[i] * np.log(K))) * psi(phi=phi_values[i], T=T, r=r, ro=ro, a=a, sigma=sigma, b=b[0], u=u[0], S0=np.log(S0), v0=v0) / complex(0, phi_values[i])).real) / np.pi), 15) > 0:
            P1 += step_size / 2 * ((np.exp(complex(0, -phi_values[i-1] * np.log(K))) * psi(phi=phi_values[i-1], T=T, r=r, ro=ro, a=a, sigma=sigma, b=b[0], u=u[0], S0=np.log(S0), v0=v0) / complex(0, phi_values[i - 1])).real
                               + (np.exp(complex(0, -phi_values[i] * np.log(K))) * psi(phi=phi_values[i], T=T, r=r, ro=ro, a=a, sigma=sigma, b=b[0], u=u[0], S0=np.log(S0), v0=v0) / complex(0, phi_values[i])).real) / np.pi
            y1 = 0
        else:
            y1 = 1
        if np.round(np.abs(step_size / 2 * ((np.exp(complex(0, -phi_values[i - 1] * np.log(K))) * psi(phi=phi_values[i - 1], T=T, r=r, ro=ro, a=a, sigma=sigma, b=b[1], u=u[1], S0=np.log(S0), v0=v0) / complex(0, phi_values[i - 1])).real
                               + (np.exp(complex(0, -phi_values[i] * np.log(K))) * psi(phi=phi_values[i], T=T, r=r, ro=ro, a=a, sigma=sigma, b=b[1], u=u[1], S0=np.log(S0), v0=v0) / complex(0, phi_values[i])).real) / np.pi), 15) > 0:
            P2 += step_size / 2 * ((np.exp(complex(0, -phi_values[i - 1] * np.log(K))) * psi(phi=phi_values[i - 1], T=T, r=r, ro=ro, a=a, sigma=sigma, b=b[1], u=u[1], S0=np.log(S0), v0=v0) / complex(0, phi_values[i-1])).real
                                   + (np.exp(complex(0, -phi_values[i] * np.log(K))) * psi(phi=phi_values[i], T=T, r=r, ro=ro, a=a, sigma=sigma, b=b[1], u=u[1], S0=np.log(S0), v0=v0) / complex(0, phi_values[i])).real) / np.pi
            y2 = 0
        else:
            y2 = 1

        if y1 and y2:
            break

    return S0 * P1 - K * np.exp(-r * T) * P2



