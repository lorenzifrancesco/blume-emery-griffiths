import numpy as np
from scipy.optimize import root
from tqdm import tqdm
import threading
from plotting_utils import *
from solver import *

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

def sweep_T(
    Delta = 0.1,
    N=200, 
    K_over_J=0.0):
  print("\n============= defining vectors =============")
  [tj_vec, x_vec] = get_arrays(N=N)
  x_results = 99 * np.ones_like(tj_vec)
  success = ""

  prev = [0.0]
  for it, tj in enumerate(tj_vec):
    bK = K_over_J /tj
    M = get_M(tj, Delta, bK, prev=0.5)
    x_results[it] = get_x(tj, Delta, bK, M, prev=prev)
    # prev = x_results[it]
  plt.scatter(x_results, tj_vec, marker="+", s=10)
  ax = plt.gca()
  ax.set_xlim([x_vec[0], x_vec[-1]])
  ax.set_ylim([tj_vec[0], tj_vec[-1]])
  plt.xlabel("x")
  plt.ylabel("T")
  plt.legend(labels=["Delta = "+ str(Delta)])
  plt.plot(x_vec, 1-x_vec, color="black")
  plt.scatter(2/3, 1/3, c='mediumseagreen', marker='+', s=150)
  plt.savefig("media/sweep_T_"+str(Delta)+".pdf", format="pdf", bbox_inches='tight')
  return


def sweep_x(
    Delta=0.1,
    N=200, 
    K_over_J=0.0):
  print("\n============= defining vectors =============")
  [tj_vec, x_vec] = get_arrays(N=N)
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
    plt.scatter(x, T_results[ix], marker=success, color=cc, s=30, alpha=0.3)
  ax = plt.gca()
  ax.set_xlim([x_vec[0], x_vec[-1]])
  ax.set_ylim([tj_vec[0], tj_vec[-1]])
  plt.xlabel("x")
  plt.ylabel("T")
  plt.legend(labels=["Delta = "+ str(Delta)])
  plt.plot(x_vec, 1-x_vec, color="black")
  plt.scatter(2/3, 1/3, c='mediumseagreen', marker='+', s=150)
  plt.savefig("media/sweep_x_"+str(Delta)+".pdf", format="pdf", bbox_inches='tight')
  return


def compute_parameters(
    N=200, 
    K_over_J=0.0,
    num_threads=1,
    reset=True, 
    prev_mem = False):
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
        if K_over_J < 1:
          vertical = True
        else: 
          vertical = False
        if vertical:
          for ix in tqdm(range(start, end)):
            x = x_vec[ix]
            # TODO: initial conditions are really sensible!!!
            # working for low K/J 
            # [M, Delta]
            start_point = [(1-x)*1.2, 0.6]
            # working for high K/J
            # start_point = [(1-x)*1.2, 2]
            for it, tj in enumerate(tj_vec):
                bK = K_over_J / tj
                M[it][ix], Delta[it][ix] = get_M_Delta_separate(tj, x, bK, prev = start_point)
                if prev_mem:
                  start_point = [M[it][ix], np.abs(Delta[it][ix])]
                bDelta[it][ix] = Delta[it][ix] / tj
                d = get_delta(bDelta[it][ix], bK, x)
                bA[it][ix] = get_bA(d, tj) * tj
                bB[it][ix] = get_bB(d) * tj
                bC[it][ix] = get_bC(d) * tj
        else:
          for it, tj in tqdm(enumerate(tj_vec)):  
            for ix in range(start, end):
              start_point = [(1-x_vec[ix])*1.5, 2.5]
              x = x_vec[ix]
              bK = K_over_J * 1 / tj
              M[it][ix], Delta[it][ix] = get_M_Delta_separate(tj, x, bK, prev = start_point)
              if prev_mem:
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
    #
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

   
def process_Delta(
      Delta, 
      spinodal = True,
      debug = False, 
      tj_max = 0.33, 
      discriminant=None,
      K_over_J = 0.0, 
      reset = False):
  N = np.shape(Delta)[0]
  suffix = '_' + K_over_J.__str__().replace(".", "_")
  
  if not os.path.isfile('Beautiful'+suffix+'.npy') or reset:
    N_max = int(np.ceil(N*tj_max))
    # find the spinodal
    [tj_vec, x_vec] = get_arrays(N=N)
    Delta_dx = np.zeros_like(Delta)
    for ix, x in enumerate(x_vec[1:]):
      ix += 1
      Delta_dx[:, ix] = Delta[:, ix]-Delta[:, ix-1]
    Delta_dx[:, 0] = Delta_dx[:, 1]


    # find the spinodal line
    spinodal = np.zeros_like(Delta)
    change_sign = 1
    print("\n\n--------------- Finding spinodal ---------------")
    for it, tj in tqdm(enumerate(tj_vec[:N_max]), ascii=' >='):
      for ix, x in enumerate(x_vec[1:]):
        ix += 1
        change_sign = Delta_dx[it, ix] * Delta_dx[it, ix-1]
        if change_sign < 0.0:
          ix += 1
          # mem = Delta[it, ix]      
          spinodal[it, ix] = 1

          while Delta_dx[it, ix] * Delta_dx[it, ix-1] > 0:
            ix += 1
          spinodal[it, ix] = 1
          break

    Delta_dx[it, ix] = 99 # number greater than zero
    # find the phase separation
    separation = np.zeros_like(Delta)
    skip = 0
    if isinstance(discriminant, type(None)):
      counter = np.zeros(100)
      print("\n\n--------------- Maxwellizing the Delta ---------------\n")
      for it in tqdm(range(len(tj_vec[skip:N_max]))):
        it += skip
        locations = (np.diff(np.sign(Delta_dx[it, :])) != 0)*1
        stationary = np.where(locations == 1)[0]
        if debug:
          plt.clf()
          plt.plot(x_vec, Delta_dx[it, :])
          plt.savefig("debug.pdf", format="pdf", bbox_inches='tight')
          print("\rstationary:", stationary)
        if len(stationary) < 2:
          print("\rdidn't  two stationary point, passing to next temp...")
        else:
          if len(stationary) % 2 != 0:
            print("\rOdd number of stationary points, dropping the last...")
            stationary = stationary[:-1]
          
          # iterate Maxwell over all couples of stationary points
          stationary_all = stationary
          stationary_aux = np.concatenate(([0], stationary_all, [N]))
          print("no of stationary points:", len(stationary_aux))
          for idx in range(int(len(stationary_all)/2)):  
            extremes = [stationary_aux[2*idx], N]
            stationary = [stationary_all[2*idx], stationary_all[2*idx+1]]
            H_idx = stationary[0]
            L_idx = stationary[1]
            top = Delta[it, H_idx]
            bottom = Delta[it, L_idx]
            select_level = (top + bottom)/2
            # run a certain number of bisections
            lx_point = 66
            rx_point = 80
            for i in list(range(40)):
              # intersections = np.where((np.diff(np.sign(Delta[it, :] - select_level)) != 0)*1 == 1)[0]
              intersections = []
              prev = Delta[it, 0]-select_level
              ix = extremes[0]
              while ix < extremes[1]:
                if (Delta[it, ix]-select_level) * prev < 0:
                  intersections.append(ix)
                  for _ in range(1): # backoff for 5 steps
                    ix += 1
                  prev = Delta[it, np.min([ix, N-1])]-select_level
                else:
                  prev = Delta[it, ix]-select_level
                  ix += 1
                
              # print(intersections)
              # for ii in intersections:
              #   Delta[it, ii] = 99
              counter[len(intersections)] +=1
              if len(intersections) == 3:
                print("\rFound 3 intersections...")
                lx_point = intersections[0]
                md_point = intersections[1]
                rx_point = intersections[2]
              elif len(intersections) <= 1:
                print("\rInsufficient number of intersections, breaking...")
                lx_point = 556
                rx_point = 550
                break
                lx_point = 0
                md_point = intersections[0]
                rx_point = N-1
              elif len(intersections) == 2:
                print("\r 2 intersections, managing...")
                if H_idx < intersections[1]: 
                  lx_point = intersections[0]
                  md_point = intersections[1]
                  rx_point = N-1
                else:
                  lx_point = 0
                  md_point = intersections[0]
                  rx_point = intersections[1]
              else:
                # lx_point = L_idx
                # rx_point = H_idx
                # print("\r More than 3 intersections?? BREAKING !!!...")
                # break
                print("\r More than 3 intersections?? dropping last one...")
                intersections = intersections[:-1]
                print("diocane : ", len(intersections))
                lx_point = intersections[0]
                md_point = intersections[1]
                rx_point = intersections[2]
              lx_sum = np.sum(Delta[it, lx_point:md_point] - select_level)
              rx_sum = np.sum(select_level - Delta[it, md_point:rx_point])
              if lx_sum > rx_sum:
                bottom = select_level
              else:
                top = select_level 
              select_level = (bottom + top) / 2
            Delta[it, lx_point:rx_point] = select_level
            separation[it, lx_point] = 1
            separation[it, rx_point] = 1
      print("number of hits: ", counter)
    else:
      for it, tj in tqdm(enumerate(tj_vec[skip:N_max]), ascii=' >='):
        # detect a change of sign in the discriminant
        locations = (np.diff(np.sign(discriminant[it, :])) != 0)*1
        start_point = np.where(locations == 1)[0]
        print("start:", start_point)
        start_point=start_point[0]

        # select the Delta value at the change of sign
        value = Delta[it, start_point]
        # find the endpoint and flatten
        ix = start_point + 1
        while ix<N and Delta[it, ix] - value > 0 :
          Delta[it, ix] = value
          ix += 1
        
        if ix<N:
          ix += 1 
          Delta[it, ix] = value

          while ix<N and Delta[it, ix] - value < 0:
            Delta[it, ix] = value
            ix += 1
    np.save('Beautiful'+suffix+'.npy', (Delta, spinodal, separation))
  else:
    print("\n Loading the beautiful Delta...")
    Delta, spinodal, separation = np.load('Beautiful'+suffix+'.npy')
  return Delta, spinodal, separation


def run(  
    K_over_J_list = [0.0, 2.88],
    N=1000, 
    reset=False):

  for K_over_J in K_over_J_list:
    # T-x plot
    print("\n\n°°°°°° computing for K/J =", K_over_J, " °°°°°°")
    matrices = compute_parameters(
      K_over_J = K_over_J, 
      reset=reset, 
      N=N
      )
    
    [tj_vec, x_vec] = get_arrays(N=N)
    [M, Delta_treated, bA, bB, bC] = matrices
    suffix = '_' + K_over_J.__str__().replace(".", "_")
    plot_heatmap(M, name="M"+suffix+".pdf", levels=[], title=r"$M$")
    plot_heatmap(Delta_treated, name="Delta_bare"+suffix+".pdf", levels=[0.4763], clamp = [2, 2.2] )

    plt.clf()
    tj_list = np.array([25, 147, 265])
    tj_list = np.clip(tj_list, 0, np.shape(Delta_treated)[0]-1)
    gnuplot = mpl.colormaps["gnuplot"]
    scaled = (tj_list - tj_list.min()) / (tj_list.max()-tj_list.min()) * 0.8
    for iii, tj_idx in enumerate(tj_list):
      plt.plot(x_vec, Delta_treated[tj_idx, :], color=gnuplot(scaled[iii]))
      plt.xlabel(r"$x$")
      plt.ylabel(r"$\Delta$")
    ax = plt.gca()
    ax.set_ylim([-0.5, 1.5])
    plt.scatter(2/3, 2/3*np.log(2), c='red', marker='x', s=100)
    plt.legend([r'$T = {:.2f}$'.format(tj_vec[_]) for _ in tj_list])
    plt.savefig("media/isotherms.pdf", format="pdf", bbox_inches='tight')
    

    if K_over_J == 0.0:
      plt.clf()
      it = 20
      tj_list = np.array([it])
      gnuplot = mpl.colormaps["gnuplot"]
      scaled = (tj_list - tj_list.min()) / (tj_list.max()-tj_list.min()) * 0.8
      Ddx = np.diff(Delta_treated)
      extremes = np.where((np.diff(np.sign(Ddx[it, :])) != 0)*1 == 1)[0]
      assert len(extremes)==2
      H_idx = extremes[0]
      L_idx = extremes[1]
      top = Delta_treated[it, H_idx]
      bottom = Delta_treated[it, L_idx]
      select_level = (top + bottom)/2
      for _ in range(10):
        intersections = []
        prev = Delta_treated[it, 0]-select_level
        ix = 0
        while ix < len(x_vec):
          if (Delta_treated[it, ix]-select_level) * prev < 0:
            intersections.append(ix)
            for _ in range(1): # backoff for 5 steps
              ix += 1
            prev = Delta_treated[it, np.min([ix, N-1])]-select_level
          else:
            prev = Delta_treated[it, ix]-select_level
            ix += 1
        lx_point = intersections[0]
        md_point = intersections[1]
        rx_point = intersections[2]
        lx_sum = np.sum(Delta_treated[it, lx_point:md_point] - select_level)
        rx_sum = np.sum(select_level - Delta_treated[it, md_point:rx_point])
        if lx_sum > rx_sum:
          bottom = select_level
        else:
          top = select_level 
        select_level = (bottom + top) / 2
        print(select_level)
      x_range = np.linspace(x_vec[lx_point], x_vec[rx_point],  100)
      plt.plot(x_vec, Delta_treated[it, :], color=gnuplot(0.0))
      plt.plot(x_range, select_level*np.ones_like(x_range), ls="dashed", lw=1.8)

      plt.xlabel(r"$x$")
      plt.ylabel(r"$\Delta$")
      ax = plt.gca()
      ax.set_ylim([-0.5, 1.5])
      plt.legend([r'$T = {:.2f}$'.format(tj_vec[it]) ,r'$\Delta = {:.2f}$'.format(select_level)])
      plt.savefig("media/maxwell.pdf", format="pdf", bbox_inches='tight')

    bDelta =  np.zeros_like(Delta_treated)
    a = np.zeros_like(Delta_treated)
    b = np.zeros_like(Delta_treated)
    c = np.zeros_like(Delta_treated)
    discriminant = np.zeros_like(Delta_treated)

    for it, tj in enumerate(tj_vec):
      bDelta[it, :] = Delta_treated[it, :] / tj
      a[it, :] = tj * bA[it, :]
      b[it, :] = tj * bB[it, :]
      c[it, :] = tj * bC[it, :]
    discriminant = b**2 - 4*a*c


    print("\n\n°°°°°° Treating Delta °°°°°°")
    # process Delta
    tj_max = 1.0
    if K_over_J == 0.0:
      tj_max = 0.33
      clampD = [-0.2, 0.9] 
    elif K_over_J < 1:
      tj_max = 0.7
      clampD = [0.0, 1] 
    elif K_over_J<3:
      clampD = [1.8, 2.2] 
    else:
      clampD = [1.8, 3.2] 

    Delta_treated, spinodal, separation = process_Delta(
      matrices[1], 
      tj_max = tj_max,
      discriminant = None,
      K_over_J=K_over_J
      )

    
    print("\n\n°°°°°° Plotting °°°°°°")

    print("\n============= plotting =============")
    plot_contour(Delta_treated,
                 name="Delta_contour"+suffix+".pdf", 
                 levels=np.linspace(-0.5, 1.5, 50), 
                 clamp = [0.])
    plot_heatmap(a,
                 name="a"+suffix+".pdf", 
                 levels=[0.0], 
                 clamp = [-0.2, 0.4])
    plot_heatmap(separation,
                 name="spinsept"+suffix+".pdf", 
                 levels=[], 
                 midline=False)
    plot_heatmap(discriminant,
                 name="discriminant"+suffix+".pdf", 
                 levels=[0.0], 
                 clamp = [-0.2, 0.2])
    plot_heatmap(Delta_treated,    
                 name="Delta_heat"+suffix+".pdf", 
                 levels = [0.4763], 
                 clamp = [2, 2.2])
    plt.clf()
    print(np.shape(x_vec))
    print(np.shape(tj_vec))
    print(np.shape(spinodal))
    spinodal_heigth = np.ones_like(x_vec) * -100
    separation_heigth = np.ones_like(x_vec) * -100
    for ix in range(len(x_vec)):
      spin_try = np.where(spinodal[:, ix] == 1)[0] 
      sepa_try = np.where(separation[:, ix] == 1)[0] 
      if len(spin_try) > 0:
        spinodal_heigth[ix] = tj_vec[spin_try[0]]
      if len(sepa_try) > 0:
        separation_heigth[ix] = tj_vec[sepa_try[0]]
    spinodal_heigth[spinodal_heigth<0.2] = -100
    separation_heigth[separation_heigth<0.2] = -100
    spinodal_heigth[0] = 0.0
    separation_heigth[0] = 0.0
    spinodal_heigth[N-1] = 0.0
    separation_heigth[N-1] = 0.0
    sp_plus = np.array(spinodal_heigth, copy=True)
    sp_minu = np.array(spinodal_heigth, copy=True)
    se_plus = np.array(separation_heigth, copy=True)
    se_minu = np.array(separation_heigth, copy=True)

    for ix in range(len(x_vec[1:])):
      ix += 1
      if np.abs(sp_plus[ix]-sp_plus[ix-1]) > 90:
        sp_plus[ix] = sp_plus[ix-1]
      if np.abs(se_plus[ix]-se_plus[ix-1]) > 90:
        se_plus[ix] = se_plus[ix-1]

    for ix in range(len(x_vec[1:])):
      ix += 1
      ix = N-1-ix
      if np.abs(sp_minu[ix]-sp_minu[ix+1]) > 90:
        sp_minu[ix] = sp_minu[ix+1]
      if np.abs(se_minu[ix]-se_minu[ix+1]) > 90:
        se_minu[ix] = se_minu[ix+1]

    spinodal_heigth = (sp_minu+sp_plus)/2 
    separation_heigth = (se_minu+se_plus)/2

    plt.plot(x_vec, separation_heigth, label=r"First order transition", lw = 0.5)
    plt.plot(x_vec, spinodal_heigth, label=r"Spinodal line", lw = 0.5, ls="dashed")
    stopx = int(round(N * 0.67))
    plt.plot(x_vec[:stopx], 1-x_vec[:stopx], label=r"Second order transition", lw = 0.5)
    plt.xlabel(r"$x$")
    plt.ylabel(r"$T$")
    ax = plt.gca()
    ax.set_xlim([x_vec[0], x_vec[-1]])
    ax.set_ylim([tj_vec[0], tj_vec[-1]])
    plt.legend(loc="best")
    plt.savefig("media/lines.pdf", format="pdf", bbox_inches='tight')

  return


run(
  K_over_J_list=[2.88],
  N=1000,
  reset=False
  )


# for Delta in [0.2, 0.3, 0.4, 0.48, 0.51]:
#   print("################# COMPUTING ", Delta)
#   sweep_T(Delta)