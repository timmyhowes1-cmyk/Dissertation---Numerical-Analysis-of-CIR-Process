# Implementation of modified Explicit Schemes

import time
import numpy as np


# Absorption Scheme
def Absorption(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
    Implements explicit scheme for CIR process that takes max value between 0 and X.
    :param dW: (ndarray) Wiener increments to use for simulation
    :param x: (nonnegative float) initial value for CIR process
    :param N_MC: (int) Number of Monte Carlo samples
    :param T: (float) Terminal time
    :param a: (nonnegative float) parameter for CIR process
    :param sigma: (nonnegative float) parameter for CIR process
    :param k: (nonnegative float) parameter for CIR process
    :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = np.maximum(0, X[:, (i-1)] + (a - k * X[:, (i-1)]) * (T / N) + sigma * np.sqrt(X[:, (i-1)]) * dW[:, i])
        t1 = time.time()

        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = np.maximum(0, X1[:, (i - 1)] + (a - k * X1[:, (i - 1)]) * (T / N) + sigma * np.sqrt(
                X1[:, (i - 1)]) * dW[:, i])
            X2[:, i] = np.maximum(0, X2[:, (i - 1)] + (a - k * X2[:, (i - 1)]) * (T / N) + sigma * np.sqrt(
                X2[:, (i - 1)]) * dW2[:, i])
        t1 = time.time()

        return X1, X2, t1 - t0


# Partial Truncation Scheme from Deelbean and Deelstra
def Partial_Truncation(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
    Implements explicit scheme for CIR process that takes max value between 0 and X in square root term.
    :param dW: (ndarray) Wiener increments to use for simulation
    :param x: (nonnegative float) initial value for CIR process
    :param N_MC: (int) Number of Monte Carlo samples
    :param T: (float) Terminal time
    :param a: (nonnegative float) parameter for CIR process
    :param sigma: (nonnegative float) parameter for CIR process
    :param k: (nonnegative float) parameter for CIR process
    :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = X[:, (i-1)] + (a - k * X[:, (i-1)]) * (T / N) + sigma * np.sqrt(np.maximum(X[:, (i-1)], 0)) * dW[:, i]
        t1 = time.time()

        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = X1[:, (i-1)] + (a - k * X1[:, (i-1)]) * (T / N) + sigma * np.sqrt(np.maximum(X1[:, (i-1)], 0)) * dW[:, i]
            X2[:, i] = X2[:, (i-1)] + (a - k * X2[:, (i-1)]) * (T / N) + sigma * np.sqrt(np.maximum(X2[:, (i-1)], 0)) * dW2[:, i]

        t1 = time.time()

        return X1, X2, t1 - t0

# FTE Scheme from 'Strong order 1/2 convergence of full truncation Euler approximations to the
# # Cox–Ingersoll–Ross process - Cozma, Reisinger'
def Full_Truncation(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
    Implements explicit scheme for CIR process that takes max between 0 and X in both drift and diffusion terms.
    :param dW: (ndarray) Wiener increments to use for simulation
    :param x: (nonnegative float) initial value for CIR process
    :param N_MC: (int) Number of Monte Carlo samples
    :param T: (float) Terminal time
    :param a: (nonnegative float) parameter for CIR process
    :param sigma: (nonnegative float) parameter for CIR process
    :param k: (nonnegative float) parameter for CIR process
    :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = X[:, (i-1)] + (a - k * np.maximum(X[:, (i-1)], 0)) * (T / N) + sigma * np.sqrt(np.maximum(X[:, (i-1)], 0)) * dW[:, i]
        X = np.maximum(0, X)
        t1 = time.time()
        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = X1[:, (i-1)] + (a - k * np.maximum(X1[:, (i-1)], 0)) * (T / N) + sigma * np.sqrt(np.maximum(X1[:, (i-1)], 0)) * dW[:, i]
            X2[:, i] = X2[:, (i-1)] + (a - k * np.maximum(X2[:, (i-1)], 0)) * (T / N) + sigma * np.sqrt(np.maximum(X2[:, (i-1)], 0)) * dW2[:, i]
        X1 = np.maximum(0, X1)
        X2 = np.maximum(0, X2)
        t1 = time.time()
        return X1, X2, t1 - t0


# Partial Reflection Scheme from 'Convergence of monte carlo simulations involving the mean-reverting square root
# process. - Higham, D.J., Mao, X'
def Partial_Reflection(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
    Implements explicit scheme for CIR process that takes absolute value in square root term.
    :param dW: (ndarray) Wiener increments to use for simulation
    :param x: (nonnegative float) initial value for CIR process
    :param N_MC: (int) Number of Monte Carlo samples
    :param T: (float) Terminal time
    :param a: (nonnegative float) parameter for CIR process
    :param sigma: (nonnegative float) parameter for CIR process
    :param k: (nonnegative float) parameter for CIR process
    :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]
    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = X[:, (i - 1)] + (a - k * X[:, (i - 1)]) * (T / N) + sigma * np.sqrt(np.abs(X[:, (i - 1)])) * dW[:, i]
        t1 = time.time()

        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = X1[:, (i - 1)] + (a - k * X1[:, (i - 1)]) * (T / N) + sigma * np.sqrt(np.abs(X1[:, (i - 1)])) * dW[:, i]
            X2[:, i] = X2[:, (i - 1)] + (a - k * X2[:, (i - 1)]) * (T / N) + sigma * np.sqrt(np.abs(X2[:, (i - 1)])) * dW2[:, i]
        t1 = time.time()

        return X1, X2, t1 - t0


# Reflection Scheme from 'Euler scheme for SDEs with non-Lipschitz diffusion coefficient: strong
# convergence. - Berkaoui, A., Bossy, M., Diop, A'
def Reflection(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
    Implements explicit scheme for CIR process that reflects the process when it goes lower than 0.
    :param dW: (ndarray) Wiener increments to use for simulation
    :param Antithetic: (bool) whether to implement antithetic variate technique or not
    :param x: (nonnegative float) initial value for CIR process
    :param N_MC: (int) Number of Monte Carlo samples
    :param T: (float) Terminal time
    :param a: (nonnegative float) parameter for CIR process
    :param sigma: (nonnegative float) parameter for CIR process
    :param k: (nonnegative float) parameter for CIR process
    :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = np.abs(X[:, (i - 1)] + (a - k * X[:, (i - 1)]) * (T / N) + sigma * np.sqrt(X[:, (i - 1)]) * dW[:, i])
        t1 = time.time()

        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = np.abs(X1[:, (i - 1)] + (a - k * X1[:, (i - 1)]) * (T / N) + sigma * np.sqrt(X1[:, (i - 1)]) * dW[:, i])
            X2[:, i] = np.abs(X2[:, (i - 1)] + (a - k * X2[:, (i - 1)]) * (T / N) + sigma * np.sqrt(X2[:, (i - 1)]) * dW2[:, i])
        t1 = time.time()

        return X1, X2, t1 - t0


# Normal Milstein Scheme
def Milstein(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements Milstein explicit scheme for CIR process.
        :param dW: (ndarray) Wiener increments to use for simulation
        :param x: (nonnegative float) initial value for CIR process
        :param N_MC: (int) Number of Monte Carlo samples
        :param T: (float) Terminal time
        :param a: (nonnegative float) parameter for CIR process
        :param sigma: (nonnegative float) parameter for CIR process
        :param k: (nonnegative float) parameter for CIR process
        :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]
    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = X[:, i-1] + (a - k * X[:, i-1]) * T / N + sigma * np.sqrt(X[:, i-1]) * dW[:, i] \
                      + 1 / 4 * sigma ** 2 * (dW[:, i-1] ** 2 - T / N)
        t1 = time.time()

        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = X1[:, i-1] + (a - k * X1[:, i-1]) * T / N + sigma * np.sqrt(X1[:, i-1]) * dW[:, i] \
                      + 1 / 4 * sigma ** 2 * (dW[:, i-1] ** 2 - T / N)
            X2[:, i] = X2[:, i-1] + (a - k * X2[:, i-1]) * T / N + sigma * np.sqrt(X2[:, i-1]) * dW2[:, i] \
                      + 1 / 4 * sigma ** 2 * (dW2[:, i-1] ** 2 - T / N)
        t1 = time.time()

        return X1, X2, t1 - t0

# Milstein Scheme from 'On the discretization schemes for the CIR (and Bessel squared) processes. - Alfonsi'
def Modified_Milstein(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements modified Milstein explicit scheme for CIR process.
        :param dW: (ndarray) Wiener increments to use for simulation
        :param x: (nonnegative float) initial value for CIR process
        :param N_MC: (int) Number of Monte Carlo samples
        :param T: (float) Terminal time
        :param a: (nonnegative float) parameter for CIR process
        :param sigma: (nonnegative float) parameter for CIR process
        :param k: (nonnegative float) parameter for CIR process
        :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = ((1 - k * T / (2 * N)) * np.sqrt(X[:, i-1]) + (sigma * dW[:, i]) / (2 * (1 - k * T / N))) ** 2 + (a - sigma**2 / 4) * T / N
        t1 = time.time()
        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = ((1 - k * T / (2 * N)) * np.sqrt(X1[:, i-1]) + (sigma * dW[:, i]) / (2 * (1 - k * T / N))) ** 2 + (a - sigma**2 / 4) * T / N
            X2[:, i] = ((1 - k * T / (2 * N)) * np.sqrt(X2[:, i-1]) + (sigma * dW2[:, i]) / (2 * (1 - k * T / N))) ** 2 + (a - sigma**2 / 4) * T / N
        t1 = time.time()
        return X1, X2, t1 - t0

def Adjusted_Modified_Milstein(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements modified Milstein explicit scheme for CIR process.
        :param dW: (ndarray) Wiener increments to use for simulation
        :param x: (nonnegative float) initial value for CIR process
        :param N_MC: (int) Number of Monte Carlo samples
        :param T: (float) Terminal time
        :param a: (nonnegative float) parameter for CIR process
        :param sigma: (nonnegative float) parameter for CIR process
        :param k: (nonnegative float) parameter for CIR process
        :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = np.maximum(0, ((1 - k * T / (2 * N)) * np.sqrt(np.maximum(0, X[:, i-1])) + (sigma * dW[:, i]) / (2 * (1 - k * T / N))) ** 2 + (a - sigma**2 / 4) * T / N)
        t1 = time.time()
        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = np.maximum(0, ((1 - k * T / (2 * N)) * np.sqrt(np.maximum(0, X1[:, i-1])) + (sigma * dW[:, i]) / (2 * (1 - k * T / N))) ** 2 + (a - sigma**2 / 4) * T / N)
            X2[:, i] = np.maximum(0, ((1 - k * T / (2 * N)) * np.sqrt(np.maximum(0, X2[:, i-1])) + (sigma * dW2[:, i]) / (2 * (1 - k * T / N))) ** 2 + (a - sigma**2 / 4) * T / N)
        t1 = time.time()
        return X1, X2, t1 - t0


# Adaptive Milstein scheme from 'Fast strong approximation Monte Carlo schemes for stochastic volatility models - C. Kahl & P. Jäckel'
def Adaptive_Milstein(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements Adaptive Milstein explicit scheme for CIR process.
        :param dW: (ndarray) Wiener increments to use for simulation
        :param x: (nonnegative float) initial value for CIR process
        :param N_MC: (int) Number of Monte Carlo samples
        :param T: (float) Terminal time
        :param a: (nonnegative float) parameter for CIR process
        :param sigma: (nonnegative float) parameter for CIR process
        :param k: (nonnegative float) parameter for CIR process
        :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = X[:, i-1] + (a - k * X[:, i-1]) * T / N + sigma * np.sqrt(X[:, i-1]) * dW[:, i] \
                      + 1 / 4 * sigma ** 2 * T / N * (dW[:, i] ** 2 - 1)
        t1 = time.time()

        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = X1[:, i-1] + (a - k * X1[:, i-1]) * T / N + sigma * np.sqrt(X1[:, i-1]) * dW[:, i] \
                      + 1 / 4 * sigma ** 2 * T / N * (dW[:, i] ** 2 - 1)
            X2[:, i] = X2[:, i-1] + (a - k * X2[:, i-1]) * T / N + sigma * np.sqrt(X2[:, i-1]) * dW2[:, i] \
                      + 1 / 4 * sigma ** 2 * T / N * (dW2[:, i] ** 2 - 1)
        t1 = time.time()

        return X1, X2, t1 - t0

# Pathwise Adapted Linearisation Quartic scheme from 'Fast strong approximation Monte Carlo schemes for stochastic volatility models - C. Kahl & P. Jäckel'
def Pathwise_Adapted_Linearisation_Quartic(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements Pathwise Adapted Linearisation Quartic explicit scheme for CIR process.
        :param dW: (ndarray) Wiener increments to use for simulation
        :param x: (nonnegative float) initial value for CIR process
        :param N_MC: (int) Number of Monte Carlo samples
        :param T: (float) Terminal time
        :param a: (nonnegative float) parameter for CIR process
        :param sigma: (nonnegative float) parameter for CIR process
        :param k: (nonnegative float) parameter for CIR process
        :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = X[:, i-1] + (a - sigma**2 / 4 - k * X[:, i-1] + sigma * dW[:, i] / (T / N) * np.sqrt(X[:, i-1])) * T / N * (1 + T / N * (sigma * dW[:, i] / (T / N) - 2 * k * np.sqrt(X[:, i-1])) / (4 * np.sqrt(X[:, i-1])))
        t1 = time.time()

        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = X1[:, i-1] + (a - sigma**2 / 4 - k * X1[:, i-1] + sigma * dW[:, i] / (T / N) * np.sqrt(X1[:, i-1])) * T / N * (1 + T / N * (sigma * dW[:, i] / (T / N) - 2 * k * np.sqrt(X1[:, i-1])) / (4 * np.sqrt(X1[:, i-1])))
            X2[:, i] = X2[:, i-1] + (a - sigma**2 / 4 - k * X2[:, i-1] + sigma * dW2[:, i] / (T / N) * np.sqrt(X2[:, i-1])) * T / N * (1 + T / N * (sigma * dW2[:, i] / (T / N) - 2 * k * np.sqrt(X2[:, i-1])) / (4 * np.sqrt(X2[:, i-1])))
        t1 = time.time()

        return X1, X2, t1 - t0


# Pathwise Adapted Linearisation Quadratic scheme from 'Fast strong approximation Monte Carlo schemes for stochastic volatility models - C. Kahl & P. Jäckel'
def Pathwise_Adapted_Linearisation_Quadratic(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements Pathwise Adapted Linearisation Quadratic explicit scheme for CIR process.
        :param dW: (ndarray) Wiener increments to use for simulation
        :param x: (nonnegative float) initial value for CIR process
        :param N_MC: (int) Number of Monte Carlo samples
        :param T: (float) Terminal time
        :param a: (nonnegative float) parameter for CIR process
        :param sigma: (nonnegative float) parameter for CIR process
        :param k: (nonnegative float) parameter for CIR process
        :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = X[:, i-1] + (a - sigma**2 / 4 - k * X[:, i-1] + sigma * dW[:, i] / (T / N) * np.sqrt(X[:, i-1])) * T / N * (1 + T / N * (sigma * dW[:, i] / (T / N) - 2 * k * np.sqrt(X[:, i-1])) / (4 * np.sqrt(X[:, i-1]))
                                                                                                                                  + (k * (X[:, i-1] * (4 * k * np.sqrt(X[:, i-1]) - 3 * sigma * dW[:, i] / (T / N))) - sigma * dW[:, i] / (T / N) * (a - sigma**2 / 4)) / (24 * np.sqrt(X[:, i-1])**3) * (T / N)**2
                                                                                                                                  + (k * (3 * sigma * dW[:, i] / (T / N) * k * (a / k - sigma**2 / (4 * k))**2
                                                                                                                                          + k * X[:, i-1]**2 * (7 * sigma * dW[:, i] / (T / N) - 8 * k * np.sqrt(X[:, i-1]))
                                                                                                                                          + 2 * sigma * dW[:, i] / (T / N) * (a / k - sigma**2 / (4 * k)) * np.sqrt(X[:, i-1]) * (sigma * dW[:, i] / (T / N) + k * np.sqrt(X[:, i-1])))) / (192 * np.sqrt(X[:, i-1] ** 5)) * (T / N) ** 3)
        t1 = time.time()

        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = X1[:, i-1] + (a - sigma**2 / 4 - k * X1[:, i-1] + sigma * dW[:, i] / (T / N) * np.sqrt(X1[:, i-1])) * T / N * (1 + T / N * (sigma * dW[:, i] / (T / N) - 2 * k * np.sqrt(X1[:, i-1])) / (4 * np.sqrt(X1[:, i-1]))
                                                                                                                                  + (k * (X1[:, i-1] * (4 * k * np.sqrt(X1[:, i-1]) - 3 * sigma * dW[:, i] / (T / N))) - sigma * dW[:, i] / (T / N) * (a - sigma**2 / 4)) / (24 * np.sqrt(X1[:, i-1])**3) * (T / N)**2
                                                                                                                                  + (k * (3 * sigma * dW[:, i] / (T / N) * k * (a / k - sigma**2 / (4 * k))**2
                                                                                                                                          + k * X1[:, i-1]**2 * (7 * sigma * dW[:, i] / (T / N) - 8 * k * np.sqrt(X1[:, i-1]))
                                                                                                                                          + 2 * sigma * dW[:, i] / (T / N) * (a / k - sigma**2 / (4 * k)) * np.sqrt(X1[:, i-1]) * (sigma * dW[:, i] / (T / N) + k * np.sqrt(X1[:, i-1])))) / (192 * np.sqrt(X1[:, i-1] ** 5)) * (T / N) ** 3)
            X2[:, i] = X2[:, i-1] + (a - sigma**2 / 4 - k * X2[:, i-1] + sigma * dW2[:, i] / (T / N) * np.sqrt(X2[:, i-1])) * T / N * (1 + T / N * (sigma * dW2[:, i] / (T / N) - 2 * k * np.sqrt(X2[:, i-1])) / (4 * np.sqrt(X2[:, i-1]))
                                                                                                                                  + (k * (X2[:, i-1] * (4 * k * np.sqrt(X2[:, i-1]) - 3 * sigma * dW2[:, i] / (T / N))) - sigma * dW2[:, i] / (T / N) * (a - sigma**2 / 4)) / (24 * np.sqrt(X2[:, i-1])**3) * (T / N)**2
                                                                                                                                  + (k * (3 * sigma * dW2[:, i] / (T / N) * k * (a / k - sigma**2 / (4 * k))**2
                                                                                                                                          + k * X2[:, i-1]**2 * (7 * sigma * dW2[:, i] / (T / N) - 8 * k * np.sqrt(X2[:, i-1]))
                                                                                                                                          + 2 * sigma * dW2[:, i] / (T / N) * (a / k - sigma**2 / (4 * k)) * np.sqrt(X2[:, i-1]) * (sigma * dW2[:, i] / (T / N) + k * np.sqrt(X2[:, i-1])))) / (192 * np.sqrt(X2[:, i-1] ** 5)) * (T / N) ** 3)
        t1 = time.time()

        return X1, X2, t1 - t0

# Truncated Milstein scheme from 'STRONG CONVERGENCE RATES FOR COX-INGERSOLL-ROSS
# PROCESSES – FULL PARAMETER RANGE - MARIO HEFTER AND ANDRE HERZWURM'
def Truncated_Milstein(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements Pathwise Adapted Linearisation Quartic explicit scheme for CIR process.
        :param dW: (ndarray) Wiener increments to use for simulation
        :param x: (nonnegative float) initial value for CIR process
        :param N_MC: (int) Number of Monte Carlo samples
        :param T: (float) Terminal time
        :param a: (nonnegative float) parameter for CIR process
        :param sigma: (nonnegative float) parameter for CIR process
        :param k: (nonnegative float) parameter for CIR process
        :return: X (ndarray): values for simulated CIR processes
    '''

    N = dW.shape[1]

    if not Antithetic:
        # initialize process
        X = np.zeros([N_MC, N])
        X[:, 0] = x

        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = np.maximum(0, np.maximum(np.sqrt(T / N * sigma**2 / 4), dW[:, i] * sigma / 2 + np.sqrt(np.maximum(X[:, i-1], T / N * sigma**2 / 4))) ** 2 + (a - sigma**2 / 4 - k * X[:, i-1]) * T / N)
        t1 = time.time()
        return X, t1 - t0

    else:
        X1 = np.zeros([N_MC, N])
        X2 = np.zeros([N_MC, N])
        X1[:, 0] = x
        X2[:, 0] = x
        dW2 = -dW
        # simulate and time process
        t0 = time.time()
        for i in range(1, N):
            X1[:, i] = np.maximum(0, np.maximum(np.sqrt(T / N * sigma**2 / 4), dW[:, i] * sigma / 2 + np.sqrt(np.maximum(X1[:, i-1], T / N * sigma**2 / 4))) ** 2 + (a - sigma**2 / 4 - k * X1[:, i-1]) * T / N)
            X2[:, i] = np.maximum(0, np.maximum(np.sqrt(T / N * sigma**2 / 4), dW2[:, i] * sigma / 2 + np.sqrt(np.maximum(X2[:, i-1], T / N * sigma**2 / 4))) ** 2 + (a - sigma**2 / 4 - k * X2[:, i-1]) * T / N)
        t1 = time.time()
    return X1, X2, t1 - t0
