# Implementation of Implicit Schemes

import time
import numpy as np


# Implicit scheme as proposed in 'Credit default swap calibration and derivatives pricing with the SSRD
# stochastic intensity model - Brigo and Alfonsi'
def Implicit_1(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements implicit scheme for CIR process that is solution to polynomial.
        Defined when sigma^2 < 2a, 1+kT/n > 0 and X > 0.
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
            X[:, i] = ((sigma * dW[:, i] + np.sqrt((sigma * dW[:, i])**2 +
                                                   4 * (X[:, i-1] + T / N * (a - 0.5 * sigma**2)) * (1 + k * T / N))) / (2 * (1 + k * T / N))) ** 2
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
            X1[:, i] = ((sigma * dW[:, i] + np.sqrt((sigma * dW[:, i])**2 +
                                                   4 * (X1[:, i-1] + T / N * (a - 0.5 * sigma**2)) * (1 + k * T / N))) / (2 * (1 + k * T / N))) ** 2
            X2[:, i] = ((sigma * dW2[:, i] + np.sqrt((sigma * dW2[:, i])**2 +
                                                   4 * (X2[:, i-1] + T / N * (a - 0.5 * sigma**2)) * (1 + k * T / N))) / (2 * (1 + k * T / N))) ** 2
        t1 = time.time()

        return X1, X2, t1 - t0


# Square root implicit scheme from 'On the discretization schemes for the CIR (and Bessel squared) processes. - Alfonsi, A'
def Implicit_2(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
        Implements Square root implicit scheme for CIR process.
        Defined when sigma^2 < 2a, 1+kT/n > 0 and X > 0.
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

        # simulate process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = ((0.5 * sigma * dW[:, i] + np.sqrt(X[:, i-1]) + np.sqrt((0.5 * sigma * dW[:, i] + np.sqrt(X[:, i-1]))**2 +
                                                   4 * (1 + k * T / (2 * N)) * T / N * 0.5 * (a - 0.25 * sigma**2))) / (2 * (1 + k * T / (2 * N)))) ** 2
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
            X1[:, i] = ((0.5 * sigma * dW[:, i] + np.sqrt(X1[:, i-1]) + np.sqrt((0.5 * sigma * dW[:, i] + np.sqrt(X1[:, i-1]))**2 +
                                                   4 * (1 + k * T / (2 * N)) * T / N * 0.5 * (a - 0.25 * sigma**2))) / (2 * (1 + k * T / (2 * N)))) ** 2
            X2[:, i] = ((0.5 * sigma * dW2[:, i] + np.sqrt(X2[:, i-1]) + np.sqrt((0.5 * sigma * dW2[:, i] + np.sqrt(X2[:, i-1]))**2 +
                                                   4 * (1 + k * T / (2 * N)) * T / N * 0.5 * (a - 0.25 * sigma**2))) / (2 * (1 + k * T / (2 * N)))) ** 2
        t1 = time.time()

        return X1, X2, t1 - t0


# Balanced Implicit scheme from 'Fast strong approximation Monte Carlo schemes for stochastic volatility models - C. Kahl & P. Jäckel'
def Balanced_Implicit(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
           Implements Balanced Implicit scheme for CIR process.
           Defined when sigma^2 < 2a, 1+kT/n > 0 and X > 0.
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

        # simulate process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = (X[:, i-1] + (a - k * X[:, i-1]) * T / N + sigma * np.sqrt(X[:, i-1]) * dW[:, i]
                       + X[:, i-1] * (k * T / N + sigma * np.sqrt(X[:, i-1]) * np.abs(dW[:, i]))) / (1 + k * T / N + sigma * np.sqrt(X[:, i-1]) * np.abs(dW[:, i]))
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
            X1[:, i] = (X1[:, i-1] + (a - k * X1[:, i-1]) * T / N + sigma * np.sqrt(X1[:, i-1]) * dW[:, i]
                       + X1[:, i-1] * (k * T / N + sigma * np.sqrt(X1[:, i-1]) * np.abs(dW[:, i]))) / (1 + k * T / N + sigma * np.sqrt(X1[:, i-1]) * np.abs(dW[:, i]))
            X2[:, i] = (X2[:, i-1] + (a - k * X2[:, i-1]) * T / N + sigma * np.sqrt(X2[:, i-1]) * dW2[:, i]
                       + X2[:, i-1] * (k * T / N + sigma * np.sqrt(X2[:, i-1]) * np.abs(dW2[:, i]))) / (1 + k * T / N + sigma * np.sqrt(X2[:, i-1]) * np.abs(dW2[:, i]))
        t1 = time.time()

        return X1, X2, t1 - t0


def Balanced_Milstein(dW, Antithetic=False, x=0.1, N_MC=2000, T=1,  a=0.1, sigma=0.2, k=0.5):
    '''
           Implements Balanced Milstein Method for CIR process.
           Defined when sigma^2 < 2a, 1+kT/n > 0 and X > 0.
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

        # simulate process
        t0 = time.time()
        for i in range(1, N):
            X[:, i] = (X[:, i-1] + (a - k * X[:, i-1]) * T / N + sigma * np.sqrt(X[:, i-1]) * dW[:, i]
                       + 1 / 4 * sigma ** 2 * (dW[:, i] ** 2 - T / N)
                       + X[:, i-1] * k * T / N) / (1 + k * T / N)
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
            X1[:, i] = (X1[:, i-1] + (a - k * X1[:, i-1]) * T / N + sigma * np.sqrt(X1[:, i-1]) * dW[:, i]
                       + 1 / 4 * sigma ** 2 * (dW[:, i] ** 2 - T / N)
                       + X1[:, i-1] * k * T / N) / (1 + k * T / N)
            X2[:, i] = (X2[:, i-1] + (a - k * X2[:, i-1]) * T / N + sigma * np.sqrt(X2[:, i-1]) * dW2[:, i]
                       + 1 / 4 * sigma ** 2 * (dW2[:, i] ** 2 - T / N)
                       + X2[:, i-1] * k * T / N) / (1 + k * T / N)
        t1 = time.time()

        return X1, X2, t1 - t0
