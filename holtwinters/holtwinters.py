"""
Holt Winters seasonal timeseries forecasting

https://www.otexts.org/fpp/7/5
"""

import numpy as np
import scipy.optimize
import time

def initial(Y, m):
  """
  Initial holt winters values
  http://robjhyndman.com/hyndsight/hw-initialization

  :param Y: array. Input array of timeseries values.
  :param m: int. Number of samples in a season. E.g. monthly = 12, weekly = 7
  :returns: 3 initial values
  """
  a = [np.mean(Y[:m])]
  b = [(sum(Y[m:2 * m]) - sum(Y[:m])) / m ** 2]
  s = [Y[i] / a[0] for i in range(m)]
  return a, b, s

def RMSE(params, *args):
  """
  Root-mean-square error
  """
  alpha, beta, gamma = params
  Y, m = args
  a, b, s = initial(Y, m)
  size = len(Y)

  for i in range(size - 1):
    a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
    s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])

  y = np.clip((np.array(a) + np.array(b)) * s[:size], 0, np.iinfo(np.int64).max)
  rmse = np.sqrt(np.mean((np.asarray(Y) - y) ** 2))

  return rmse

def multiplicative(Y, m, fc, alpha = None, beta = None, gamma = None):
  """
  Multiplicative Holt Winters

  :param Y: array. Input array of timeseries values.
  :param m: int. Number of samples in a season. E.g. monthly = 12, weekly = 7
  :param fc: int. Number of samples to forecast
  :returns: forecasts of len(fc), smoothed input, rmse
  """
  if (alpha == None or beta == None or gamma == None):
    initial_values = np.array([0.01, 0.9, 0.01])
    boundaries = [(0.01, 0.99), (0.01, 0.99), (0.01, 0.99)]

    parameters = scipy.optimize.fmin_l_bfgs_b(RMSE,
      x0=initial_values, args=(Y, m), bounds=boundaries, approx_grad=True)
    alpha, beta, gamma = parameters[0]

  a, b, s = initial(Y, m)
  size = len(Y)

  for i in range(size + fc):
    if i == len(Y):
      forecast = (a[-1] + b[-1]) * s[-m]
      Y.append(0 if forecast < 0 else forecast)
    a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
    b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
    s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])

  y = np.clip((np.array(a) + np.array(b))[:size] * s[:size], 0, np.iinfo(np.int64).max)
  rmse = np.sqrt(sum([(m - n) ** 2 for m, n in zip(Y[:size], y[:-fc])]) / size)

  return Y[-fc:], y[:size], rmse
