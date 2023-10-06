
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

def get_M(tj, Delta, bK, prev):
  def eq(M):
    return M - 2*np.sinh(M/tj)/(np.exp(Delta/tj - bK * (M*np.cosh(M/tj)/np.sinh(M/tj)) ) + 2 * np.cosh(M/tj))
  root_solution = root(eq, prev)
  return root_solution.x

def compute_orders(
    N=200, 
    K_over_J=0.0,
    num_threads=1,
    reset=True):  
  print("\n============= defining vectors =============\n")
  [tj_vec, Delta_vec] = get_arrays_order(N=N)
  M      = np.ones((N, N))
  x      = np.zeros((N, N))
  suffix = '_' + K_over_J.__str__().replace(".", "_")
  if not os.path.isfile('order_M'+suffix+'.npy') or (reset): # check only the first file
    print("\n============= start computing =============\n")
    # Define a worker function to fill the matrices
    def worker(start, end):
        for iD in tqdm(range(start, end)):
          D = Delta_vec[iD]
          prev = [1, 0.5]
          for it, tj in enumerate(tj_vec):
            bK = K_over_J * 1 / tj
            if it == 0:
               pluss = 0
            start_point = [(2*M[it-1, iD]-M[it-1, iD]), x[it, iD-1]]
            [M[it, iD], x[it, iD]] = get_M_x(tj, D, bK, prev = start_point)
    chunk_size = len(Delta_vec) // num_threads
    threads = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(Delta_vec)
        thread = threading.Thread(target=worker, args=(start, end))
        threads.append(thread)
        thread.start()
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    # Save the matrix to the file
    print("\n============= saving =============\n")
    np.save('order_M'+suffix+'.npy', M)
    np.save('order_x'+suffix+'.npy', x)
  else:
    print("\n============= loading =============\n")
    # Load the matrix from the file
    M = np.load('order_M'+suffix+'.npy')
    x = np.load('order_x'+suffix+'.npy')
  return [M, x]
