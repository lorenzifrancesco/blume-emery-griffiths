import matplotlib.pyplot as plt
import os.path
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
import matplotlib as mpl
import numpy as np
import matplotlib.ticker as mticker
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
# reproduce figures with the (T, x) plane and different K
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
  plt.ylabel('T')
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
                 levels = [0.0], 
                 midline=True, 
                 x_name = r"$x$",
                 title=r"$\Delta$",
                 clamp = [-10, 10]):
  N = np.shape(mat)[0]
  if type=="normal":
    [tj_vec, x_vec] = get_arrays(N=N)
  else:
    [tj_vec, x_vec] = get_arrays_order(N=N)
  extent = [x_vec.min(), x_vec.max(), tj_vec.min(), tj_vec.max()]
  fig, ax = plt.subplots()
  mat = np.clip(mat, clamp[0], clamp[1])
  plt.imshow(np.flip(mat, axis = 0), extent = extent, cmap='hot', interpolation='nearest', vmin=np.min(mat), vmax=np.max(mat))
  lowl = np.max([clamp[0], np.min(mat)])
  higl = np.min([clamp[1], np.max(mat)])
  cbar = plt.colorbar(
                    ticks=[lowl, higl]+levels,
                    format=mticker.FixedFormatter([r"$<{:.2f}$".format(lowl), r"$>{:.2f}$".format(higl)]+[r"${:.2f}$".format(_) for _ in levels]),
                    extend='both'
                    )  
  cbar.set_label(title, rotation=0)
  # TODO REMOVE THE LABEL NONSENSE
  label_try=0.0
  try:
    if len(levels)==1:
      label_try = r"${:.2f}$".format(levels[0])
  except:
    pass
  print(label_try)
  plt.contour(x_vec, tj_vec, mat, colors='cyan',  levels=levels, linewidths=[0.5], extent=extent)
  if midline:
    # beware, this goes beyond the bottom line
    plt.plot(x_vec, np.clip(1-x_vec, tj_vec.min(), tj_vec.max()), linestyle="dotted", color="white", lw=1)

  plt.scatter(2/3, 1/3, c='blue', marker='+', lw=0.5, s=200)
  # plt.plot(tj_vec, np.ones_like(x_vec)*1/3, color="green")
  plt.xlabel(x_name)
  plt.ylabel(r'$T$')
  number_of_ticks = 7
  skip = int(np.round(N/number_of_ticks))
  plt.xticks(ticks=x_vec[::skip],  labels=[r'${:.2f}$'.format(x) for x in x_vec[::skip]])
  plt.yticks(ticks=tj_vec[::skip], labels=[r'${:.2f}$'.format(x) for x in tj_vec[::skip]])
  # plt.title(name[:-4])
  if False:
    plt.show()
  print("saving plot ", name, "...")
  plt.savefig("media/"+name, format="pdf", bbox_inches='tight')
  return 0


def plot_contour(mat, 
                 name="lower.pdf", 
                 type="normal", 
                 levels = 200,
                 K_over_J = 0.0, 
                 clamp = [-10, 10], 
                 title=r"$\Delta$"):
  N = np.shape(mat)[0]
  if type=="normal":
    [tj_vec, x_vec] = get_arrays(N=N)
  else:
    [tj_vec, x_vec] = get_arrays_order(N=N)
  extent = [x_vec.min(), x_vec.max(), tj_vec.min(), tj_vec.max()]
  fig, ax = plt.subplots()
  # np.clip(mat, clamp[0], clamp[1])
  plt.contour(x_vec, tj_vec, mat, levels=levels, extent=extent)
  plt.scatter(2/3, 1/3, c='red', marker='+')
  plt.xlabel(r'$x$')
  plt.ylabel(r'$T$')
  number_of_ticks = 7
  skip = int(np.round(N/number_of_ticks))
  plt.xticks(ticks=x_vec[::skip],  labels=[r'${:.2f}$'.format(x) for x in x_vec[::skip]])
  plt.yticks(ticks=tj_vec[::skip], labels=[r'${:.2f}$'.format(x) for x in tj_vec[::skip]])
  cbar = plt.colorbar()
  cbar.set_label(title, rotation=0)
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
  ax.set_xlabel(r'$Delta/J$')
  ax.set_ylabel(r'$T$')
  ax.set_zlabel(r'$x$')
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
