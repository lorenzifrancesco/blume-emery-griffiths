import matplotlib.pyplot as plt
import os.path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
from ranges import *
# WARN: since Colab cannot find the LaTeX library, in the plot preparation phase
# it is better to have the script in local and uncomment the following lines for
# have LaTeX fonts
#
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "Helvetica"
})
#
# reproduce figures with the (T/j, x) plane and different K
# bSomething means \beta * Something
#
# solve (3.10) always valid in the assumption M!=0

def plot_sign(mat, name="lower.pdf"):
  N = np.shape(mat)[0]
  [tj_vec, x_vec] = get_arrays(N=N)
  extent = [x_vec.min(), x_vec.max(), tj_vec.min(), tj_vec.max()]
  mask = np.zeros((N, N))
  mask[mat > 0.0] = 1
  fig, ax = plt.subplots()
  # plt.contour(mask, cmap='hot', interpolation='nearest', extent=extent, levels=200)
  plt.imshow(np.flip(mask, axis=0), cmap='hot', interpolation='nearest', extent=extent)
  plt.scatter(2/3, 1/3, c='red', marker='+')
  plt.xlabel('x')
  plt.ylabel('T/J')
  skip = 20
  plt.xticks(ticks=x_vec[::skip],  labels=[format(x, '.2f') for x in x_vec[::skip]])
  plt.yticks(ticks=tj_vec[::skip], labels=[format(x, '.2f') for x in tj_vec[::skip]])
  plt.colorbar()
  # plt.title(name[:-4])
  if False:
    plt.show()
  print("saving plot", name, "...")
  plt.savefig("media/"+name, format="pdf", bbox_inches='tight')
  return 0


def plot_heatmap(mat, 
                 name="lower.pdf", 
                 type="normal", 
                 level = 0.0, 
                 midline=True, 
                 x_name = "x"):
  N = np.shape(mat)[0]
  if type=="normal":
    [tj_vec, x_vec] = get_arrays(N=N)
  else:
    [tj_vec, x_vec] = get_arrays_order(N=N)
  extent = [x_vec.min(), x_vec.max(), tj_vec.min(), tj_vec.max()]
  fig, ax = plt.subplots()
  mat = np.clip(mat, -1, 1)
  plt.imshow(np.flip(mat, axis = 0), extent = extent, cmap='hot', interpolation='nearest', vmin=np.min(mat), vmax=np.max(mat))
  plt.colorbar()
  # SET THE CORRECT LEVELS
  plt.contour(x_vec, tj_vec, mat, colors='cyan',  levels=[level])
  if midline:
    plt.plot(x_vec, 1-x_vec, linestyle="dotted", color="white")

  plt.scatter(2/3, 1/3, c='mediumseagreen', marker='+', s=150)
  plt.plot(tj_vec, np.ones_like(x_vec)*1/3, color="green")
  plt.xlabel(x_name)
  plt.ylabel('T/J')
  skip = 20
  plt.xticks(ticks=x_vec[::skip],  labels=[format(x, '.2f') for x in x_vec[::skip]])
  plt.yticks(ticks=tj_vec[::skip], labels=[format(x, '.2f') for x in tj_vec[::skip]])
  # plt.title(name[:-4])
  if False:
    plt.show()
  print("saving plot ", name, "...")
  plt.savefig("media/"+name, format="pdf", bbox_inches='tight')
  return 0


def plot_contour(mat, 
                 name="lower.pdf", 
                 type="normal", 
                 levels = 200):
  N = np.shape(mat)[0]
  if type=="normal":
    [tj_vec, x_vec] = get_arrays(N=N)
  else:
    [tj_vec, x_vec] = get_arrays_order(N=N)
  extent = [x_vec.min(), x_vec.max(), tj_vec.min(), tj_vec.max()]
  fig, ax = plt.subplots()
  np.clip(mat, -10, 10)
  plt.contour(x_vec, tj_vec, mat, levels=levels)
  plt.scatter(2/3, 1/3, c='red', marker='+')
  plt.xlabel('x')
  plt.ylabel('T/J')
  skip = 20
  plt.xticks(ticks=x_vec[::skip],  labels=[format(x, '.2f') for x in x_vec[::skip]])
  plt.yticks(ticks=tj_vec[::skip], labels=[format(x, '.2f') for x in tj_vec[::skip]])
  plt.colorbar()
  # plt.title(name[:-4])
  if False:
    plt.show()
  print("saving plot", name, "...")
  plt.savefig("media/"+name, format="pdf", bbox_inches='tight')
  return 0


def plot_surface(mat, name="surf.pdf", type="normal"):
  N = np.shape(mat)[0]
  if type=="normal":
    [tj_vec, x_vec] = get_arrays(N=N)
  else:
    [tj_vec, x_vec] = get_arrays_order(N=N)
  
  X, Y = np.meshgrid(tj_vec, x_vec)
  Z = mat
  # Clip z values to be between -1 and 1
  Z = np.clip(Z, -10, 10)
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z, cmap='viridis')
  # Add color bar
  fig.colorbar(surf)
  # Set axis labels
  ax.set_xlabel('Delta/J')
  ax.set_ylabel('T/J')
  ax.set_zlabel('x')
  # Set plot title
  # plt.title(name[:-4])
  # Show the plot
  plt.show()
  return

def plot_project_Delta_x(
    Delta, 
    name="projection.pdf"):
  N = np.shape(Delta)[0]
  [tj_vec, x_vec] = get_arrays(N=N)
  hot = mpl.colormaps["hot"]
  for it, tj in enumerate(tj_vec):
    plt.scatter(x_vec, Delta[it, :])
  plt.show()
  print("saving plot", name, "...")
  plt.savefig("media/"+name, format="pdf", bbox_inches='tight')
  return
