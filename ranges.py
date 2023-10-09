import numpy as np

def get_arrays(N=200):
  tj_vec = np.linspace(0.05, 0.99, N) # T/J
  x_vec  = np.linspace(0.0, 1.0, N)
  return [tj_vec, x_vec] 

def get_arrays_order(N=200):
  tj_vec = np.linspace(0.0001, 0.9999, N) # T/J
  Delta_vec  = np.linspace(0.0, 1, N)
  return [tj_vec, Delta_vec]