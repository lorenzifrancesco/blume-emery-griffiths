
import numpy as np
from scipy.optimize import root
from tqdm import tqdm
import threading

def get_delta(bDelta, bK, x):
  # FIX: there was also the -bK(1-x) term
  return 1+1/2*np.exp(bDelta)

# betaA...
def get_bA(d, tj):
  return d/2-1/(2*tj) 

def get_bB(d):
  return 1/8*(d**2-d**3/3)

def get_bC(d):
  return 1/6*(1/2*d**3-3/8*d**4+3/40*d**5)

def get_M_Delta(tj, x, bK, prev):
  def eqs(pars):
    M, Delta = pars
    return [
      1-x - 2*np.cosh(M/tj)/(np.exp(Delta/tj - bK*(1-x)) +2*np.cosh(M/tj)),
      M - 2*np.sinh(M/tj)/(np.exp(Delta/tj - bK*(1-x)) +2*np.cosh(M/tj)),
    ] 
  initial_guesses = prev
  root_solution = root(eqs, initial_guesses)
  if not root_solution.success:
    # print("WARNING: root finding failed!")
    # print(root_solution.message)
    pass
  sol = root_solution.x
  # print("Solved! results are M =", sol[0], "and Delta =", sol[1])
  return sol


def get_M_Delta_separate(tj, x, bK, prev):
  def Meq(M):
    return 1-x-M/np.tanh(M/tj)
  initial_guesses = prev[0]
  root_solution = root(Meq, initial_guesses)
  if not root_solution.success:
    # print("WARNING: root finding failed!")
    # print(root_solution.message)
    pass
  M = root_solution.x
  def Deltaeq(Delta):
    return 1-x - 2*np.cosh(M/tj)/(np.exp(Delta/tj - bK*(1-x)) +2*np.cosh(M/tj))
  # print("Solved! results are M =", sol[0], "and Delta =", sol[1])
  initial_guesses = prev[1]
  root_solution = root(Deltaeq, initial_guesses)
  if not root_solution.success:
    # print("WARNING: root finding failed!")
    # print(root_solution.message)
    pass
  Delta = root_solution.x
  # if tj<x/2:
  #   Delta = 0.7-Delta
  return [M, Delta]

def get_M_x(tj, Delta, bK, prev):
  def eqs(pars):
    M, x = pars
    return [
      1-x - 2*np.cosh(M/tj)/(np.exp(Delta/tj - bK*(1-x)) +2*np.cosh(M/tj)),
      M - 2*np.sinh(M/tj)/(np.exp(Delta/tj - bK*(1-x)) +2*np.cosh(M/tj)),
    ] 
  initial_guesses = prev 
  root_solution = root(eqs, initial_guesses)

  if not root_solution.success:
    print("WARNING: root finding failed!")
    print(root_solution.message)
  sol = root_solution.x
  # print("Solved! results are M =", sol[0], "and Delta =", sol[1])
  return sol

def get_x(tj, Delta, bK, M, prev):
  def eq(x):
    return 1-x - 2*np.cosh(M/tj)/(np.exp( (Delta/tj - bK * (1-x))) + 2 * np.cosh(M/tj))
  root_solution = root(eq, prev)
  return root_solution.x

def get_M_t(x, Delta, K, prev):
  def eqs(pars):
    M, tj = pars
    return [
      1-x - 2*np.cosh(M/tj)/(np.exp(Delta/tj - K/tj*(1-x)) +2*np.cosh(M/tj)),
      M - 2*np.sinh(M/tj)/(np.exp(Delta/tj - K/tj*(1-x)) +2*np.cosh(M/tj)),
    ] 
  initial_guesses = prev 
  root_solution = root(eqs, initial_guesses)

  if not root_solution.success:
    print("WARNING: root finding failed!")
    print(root_solution.message)
    status = "x"
  else: 
    status = "+"
  sol = root_solution.x

  return sol, status

def get_M(tj, Delta, bK, prev):
  def eq(M):
    return M - 2*np.sinh(M/tj)/(np.exp(Delta/tj - bK * (M*np.cosh(M/tj)/np.sinh(M/tj)) ) + 2 * np.cosh(M/tj))
  root_solution = root(eq, prev)
  return root_solution.x
