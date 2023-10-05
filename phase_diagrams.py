import numpy as np
from scipy.optimize import root
from tqdm import tqdm
import threading

from plotting_utils import *
from solver import *


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
                bK = K_over_J * 1 / tj
                [M[it][ix], Delta[it][ix]] = get_M_Delta_separate(tj, x, bK, prev = start_point)
                start_point = [M[it][ix], Delta[it][ix]]
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


def plot_all(matrices, K_over_J=0.0):
  suffix = '_' + K_over_J.__str__().replace(".", "_")
  [M, Delta, bA, bB, bC] = matrices
  print("\n============= plotting =============")
  #
  # M and Delta as for MF EQUATIONS
  deltozzo = bB**2 - 4*bA*bC
  plot_heatmap(deltozzo, name="heat_first_PT"+suffix +".pdf")
  plot_heatmap(M,        name="M"+suffix+".pdf")
  plot_contour(Delta,    name="Delta"+suffix+".pdf")
  plot_heatmap(Delta,    name="Delta_heat"+suffix+".pdf")
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

K_over_J_list = [0.0]
for K_over_J in K_over_J_list:
  print("\n\n°°°°°° computing for K/J =", K_over_J, " °°°°°°")

  matrices = compute_parameters(K_over_J = K_over_J, reset=True)
  plot_all(matrices, K_over_J=K_over_J)

  # ORDERS
  # orders = compute_orders(K_over_J = K_over_J, reset=True, N=200)
  # plot_surface(orders[0], name="order_M.pdf")
  # plot_contour(orders[0], name="order_M.pdf")
  # plot_contour(orders[1], name="order_x.pdf")