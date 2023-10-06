import numpy as np
from scipy.optimize import root
from tqdm import tqdm
import threading

from plotting_utils import *
from solver import *

import matplotlib.pyplot as plt

def sweep_T(
    N=200, 
    K_over_J=0.0):
  print("\n============= defining vectors =============")
  [tj_vec, Delta_vec] = get_arrays_order(N=N)
  Delta = 0.45
  x_results = 99 * np.ones_like(tj_vec)
  prev = [0.1]
  for it, tj in enumerate(tj_vec):
    bK = K_over_J /tj
    M = get_M(tj, Delta, bK, prev=0.5)
    x_results[it] = get_x(tj, Delta, bK, M, prev=prev)
    prev = x_results[it]
  plt.scatter(x_results, tj_vec, marker="+", s=10)
  plt.savefig("media/sweep_T.pdf", format="pdf", bbox_inches='tight')
  return

def sweep_x(
    N=200, 
    K_over_J=0.0):
  print("\n============= defining vectors =============")
  [tj_vec, Delta_vec] = get_arrays_order(N=N)
  [tj_vec, x_vec] = get_arrays(N=N)
  Delta = 0.5
  T_results = 99 * np.ones_like(x_vec)
  success = ""

  prev = [0.9, 0.1]
  for ix, x in enumerate(x_vec):
    prev_status = get_M_t(x, Delta, K_over_J, prev=prev)
    [prev, success] = prev_status
    [M, T_results[ix]] = prev 
  for ix, x in enumerate(x_vec):
    if M>0:
      cc = "red"
    else:
      cc = "blue"
    plt.scatter(x, T_results[ix], marker=success, color=cc, s=90)
  plt.xlabel("x")
  plt.ylabel("T/J")
  plt.legend(labels=["Delta = "+ str(Delta)])
  plt.savefig("media/sweep_x.pdf", format="pdf", bbox_inches='tight')
  return

def compute_parameters(
    N=200, 
    K_over_J=0.0,
    num_threads=1,
    reset=True):
  print("\n============= defining vectors =============")
  [tj_vec, x_vec] = get_arrays(N=N)
  M      = np.zeros((N, N))
  bDelta = np.zeros((N, N))
  Delta =  np.zeros((N, N))
  #
  bA      = np.zeros((N, N))
  bB      = np.zeros((N, N))
  bC      = np.zeros((N, N))
  suffix = '_' + K_over_J.__str__().replace(".", "_")
  # 
  if not os.path.isfile('M_matrix'+suffix+'.npy') or (reset): # check only the first file
    print("\n============= start computing =============")
    # Define a worker function to fill the matrices
    def worker(start, end):
        vertical = True
        if vertical:
          for ix in tqdm(range(start, end)):
            x = x_vec[ix]
            # TODO: initial conditions are really sensible!!!
            # working for low K/J
            start_point = [(1-x)*1.2, 0.5]
            # working for high K/J
            # start_point = [(1-x)*1.2, 2]
            for it, tj in enumerate(tj_vec):
                bK = K_over_J / tj
                [M[it][ix], Delta[it][ix]] = get_M_Delta_separate(tj, x, bK, prev = start_point)
                start_point = [M[it][ix], np.abs(Delta[it][ix])]
                bDelta[it][ix] = Delta[it][ix] / tj
                d = get_delta(bDelta[it][ix], bK, x)
                bA[it][ix] = get_bA(d, tj) * tj
                bB[it][ix] = get_bB(d) * tj
                bC[it][ix] = get_bC(d) * tj
        else:
          for it, tj in tqdm(enumerate(tj_vec)):  
            for ix in range(start, end):
              x = x_vec[ix]
              if it == 0 :
                start_point = [(1-x)*1.5, 2]
              else:
                start_point = [M[it-1][ix-1], Delta[it-1][ix]]
              bK = K_over_J * 1 / tj
              [M[it][ix], Delta[it][ix]] = get_M_Delta_separate(tj, x, bK, prev = start_point)
              start_point = [M[it][ix], Delta[it][ix]]
              bDelta[it][ix] = Delta[it][ix] / tj
              d = get_delta(bDelta[it][ix], bK, x)
              bA[it][ix] = get_bA(d, tj) * tj
              bB[it][ix] = get_bB(d) * tj
              bC[it][ix] = get_bC(d) * tj
    # Calculate the range of indices each thread should handle
    chunk_size = len(x_vec) // num_threads
    threads = []
    #
    # Create and start the threads
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(x_vec)
        thread = threading.Thread(target=worker, args=(start, end))
        threads.append(thread)
        thread.start()
    #
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    #
    # Save the matrix to the file
    print("\n============= saving =============")
    np.save('M_matrix'+suffix+'.npy', M)
    np.save('Delta_matrix'+suffix+'.npy', Delta)
    np.save('bA_matrix'+suffix+'.npy', bA)
    np.save('bB_matrix'+suffix+'.npy', bB)
    np.save('bC_matrix'+suffix+'.npy', bC)
    #
  else:
    print("\n============= loading =============")
    # Load the matrix from the file
    M = np.load('M_matrix'+suffix+'.npy')
    Delta = np.load('Delta_matrix'+suffix+'.npy')
    bA = np.load('bA_matrix'+suffix+'.npy')
    bB = np.load('bB_matrix'+suffix+'.npy')
    bC = np.load('bC_matrix'+suffix+'.npy')
  return [M, Delta, bA, bB, bC]

def compute_Mx(
    N=400, 
    K_over_J=0.0,
    num_threads=1,
    reset=True):
  print("\n============= defining vectors =============")
  [tj_vec, Delta_vec] = get_arrays_order(N=N)
  M      = np.zeros((N, N))
  x = np.zeros((N, N))
  suffix = '_' + K_over_J.__str__().replace(".", "_")
  # 
  if not os.path.isfile('Mx_matrix'+suffix+'.npy') or (reset): # check only the first file
    print("\n============= start computing =============")
    # Define a worker function to fill the matrices
    def worker(start, end):
        vertical = True
        if vertical:
          for id in tqdm(range(start, end)):
            Delta = Delta_vec[id]
            # TODO: initial conditions are really sensible!!!
            # working for low K/J
            start_point = [0.6, 0.01]

            if Delta < 0.60 and Delta>0.40:
              start_point = [(0.6-Delta)/0.2 * 1, 0.01]
            # working for high K/J
            # start_point = [(1-x)*1.2, 2]
            for it, tj in enumerate(tj_vec):
                bK = K_over_J / tj
                idfwd = N-it-1
                idfwd = it
                [M[idfwd][id], x[idfwd][id]] = get_M_x(tj, Delta, bK, prev = start_point)
                start_point = [M[idfwd][id], np.abs(x[idfwd][id])]
        else:
          assert False

    # Calculate the range of indices each thread should handle
    chunk_size = len(Delta_vec) // num_threads
    threads = []
    #
    # Create and start the threads
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(Delta_vec)
        thread = threading.Thread(target=worker, args=(start, end))
        threads.append(thread)
        thread.start()
    #
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    #
    # Save the matrix to the file
    print("\n============= saving =============")
    np.save('Mx_matrix'+suffix+'.npy', M)
    np.save('x_matrix'+suffix+'.npy', x)
    #
  else:
    print("\n============= loading =============")
    # Load the matrix from the file
    M = np.load('Mx_matrix'+suffix+'.npy')
    x = np.load('x_matrix'+suffix+'.npy')
  return [M, x]

   

def plot_all(matrices, K_over_J=0.0):
  suffix = '_' + K_over_J.__str__().replace(".", "_")
  [M, Delta, bA, bB, bC] = matrices
  bDelta =  np.zeros_like(Delta)
  a = np.zeros_like(Delta)
  b = np.zeros_like(Delta)
  c = np.zeros_like(Delta)

  [tj_vec, x_vec] = get_arrays()

  for it, tj in enumerate(tj_vec):
    bDelta[it, :] = tj * Delta[it, :]
    a[it, :] = tj * bA[it, :]
    b[it, :] = tj * bB[it, :]
    c[it, :] = tj * bC[it, :]


  print("\n============= plotting =============")
  #
  # M and Delta as for MF EQUATIONS
  plot_heatmap(M,        name="M"+suffix+".pdf")
  plot_contour(Delta,    name="Delta"+suffix+".pdf")
  plot_heatmap(Delta,    name="Delta_heat"+suffix+".pdf", level=0.49)
  plot_heatmap(a, name="a"+suffix+".pdf")
  plot_heatmap(b, name="b"+suffix+".pdf")
  plot_heatmap(b/c, name="b_over_c"+suffix+".pdf")
  plot_heatmap(b**2-4*a*c, name="discriminant"+suffix+".pdf")
  # EXPANSION COEFFICIENTS
  # print("\n\tmaximum of bC", np.max(bC), "\n\tminimum of bC", np.min(bC))
  # plot_sign(bC, name="bC"+suffix+".pdf")
  return


def plot_implicit(fn, bbox=(-2.5,2.5)):
    xmin, xmax, ymin, ymax, zmin, zmax = bbox*3
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    A = np.linspace(xmin, xmax, 100) # resolution of the contour
    B = np.linspace(xmin, xmax, 15) # number of slices
    A1,A2 = np.meshgrid(A,A) # grid on which the contour is plotted
    for z in B: # plot contours in the XY plane
        X,Y = A1,A2
        Z = fn(X,Y,z)
        cset = ax.contour(X, Y, Z+z, [z], zdir='z')
        # [z] defines the only level to plot for this contour for this value of z
    for y in B: # plot contours in the XZ plane
        X,Z = A1,A2
        Y = fn(X,y,Z)
        cset = ax.contour(X, Y+y, Z, [y], zdir='y')
    for x in B: # plot contours in the YZ plane
        Y,Z = A1,A2
        X = fn(x,Y,Z)
        cset = ax.contour(X+x, Y, Z, [x], zdir='x')
    ax.set_zlim3d(zmin,zmax)
    ax.set_xlim3d(xmin,xmax)
    ax.set_ylim3d(ymin,ymax)

def run(
      N=1000):
  K_over_J_list = [0.0]
  for K_over_J in K_over_J_list:
    print("\n\n°°°°°° computing for K/J =", K_over_J, " °°°°°°")

    # matrices = compute_parameters(K_over_J = K_over_J, reset=True)
    # plot_all(matrices, K_over_J=K_over_J)

    # ORDERS
    orders = compute_Mx(K_over_J = K_over_J, reset=True, N=N)
    plot_heatmap(orders[0], name="order_M.pdf", midline=False)
    plot_heatmap(orders[1], name="order_x.pdf", midline=False)
    plot_surface(orders[1])
    return

sweep_x()