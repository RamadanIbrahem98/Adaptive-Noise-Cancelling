import numpy as np

def f(x, y, a=None, iter=255, mu=0.00001):
  if a is None:
    # TODO: calculate this parameter list from the x and y data
    a = np.random.normal(0, 1, 6)
    # a = np.array([3.69638615894025, 2.65632500794996, -16.0183501652287, -6.64080641879074, 36.5091899503955, -20.6057199653232])

  errors = []
  coefficients = []

  for i in range(iter):
    xhat = np.convolve(y, a, mode='full')
    xhat = xhat[:-len(a) + 1]
    err = x - xhat
    error = err[i]
    errors.append(error)
    temp_coefficients = []
    for idx, ai in enumerate(a):
      temp_coefficients.append(ai + mu * error * y[i - idx] if i - idx >= 0 else ai)
    coefficients.append(temp_coefficients)
    a = np.array(temp_coefficients)

  return a, errors, coefficients, xhat
