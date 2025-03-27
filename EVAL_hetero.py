import matplotlib.pyplot as plt
from matplotlib import cm
import random
import numpy as np
import json
import os
from matplotlib.colors import LinearSegmentedColormap

# Create a figure with 3D ax
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')


# Define dimensions: NX= sensing, NY=Excess, NZ=strategy
Nx, Ny, Nz = 6, 11, 7
X, Y, Z = np.meshgrid(np.arange(1,Nx), np.arange(-5, 6), np.arange(Nz))


#BASE DATA
fata = np.zeros((Ny, Nx, Nz))

### aggressive / mixed #0=defend, 1=engage, 2=fight
# for x in range(Nx-1):
#     for y in range(Ny):
#         if x <= 1: fata[y,x,0] = random.choices([1, 2], weights=[2, 1], k=1)[0]
#         else:
#             if y<=3:
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[5, 3, 1], k=1)[0]
#             elif y in range(4,7):
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[3, 5, 1], k=1)[0]
#             else:
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[1, 2, 2], k=1)[0]

#engage
# for x in range(Nx-1):
#     for y in range(Ny):
#         if x <= 1: fata[y,x,0] = random.choices([1, 2], weights=[1,3], k=1)[0]
#         else:
#             if y<=3:
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[5, 3, 2], k=1)[0]
#             elif y in range(4,7):
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[3, 5, 2], k=1)[0]
#             else:
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[1, 2, 3], k=1)[0]


# np.save('def.npy', fata)

agg_ = np.load(os.path.join(os.getcwd(), "patterns_new", "aggr2.npy"))
def_ = np.load(os.path.join(os.getcwd(), "patterns_new", "def1.npy"))
eng_ = np.load(os.path.join(os.getcwd(), "patterns_new", "eng2.npy"))
mix_ = np.load(os.path.join(os.getcwd(), "patterns_new", "mix1.npy"))

agg_[1,2,0] = 0

data = np.stack((agg_[:, :,0], eng_[:, :, 0], def_[:, :, 0], mix_[:, :, 0]), axis=2)

# data = fata
dmin = data.min()
dptp = np.ptp(data)

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap',
    [(0, 0, 0.5), (0.27, 0.51, 0.71), (0.7, 0.7, 0.7)]
)

def colorize(d, cmap=custom_cmap):
    shape = d.shape
    # Normalize the data:
    # v = (d - dmin) / (dptp + 1e-10)
    normalized = ((d - dmin) / (dptp + 1e-10)).flatten()
    # Map normalized values to RGBA colors:
    colored = cmap(normalized)
    return colored.reshape((*shape, 4))

# def colorize(d, cmap=cm.viridis):
#     shape = d.shape
#     return cmap(((d-dmin)/(dptp+1e-10)).flatten()).reshape((*shape, 4))

# Plot contour surfaces
C = ax.plot_surface(
    #X[:, :, Nz//2], Y[:, :, Nz//2], Z[:, :, Nz//2],
    X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
    facecolors=colorize(data[:, :, 0]),
    shade=False,
    edgecolor='black',
    lw=0.2
)
C = ax.plot_surface(
    #X[:, :, Nz//2], Y[:, :, Nz//2], Z[:, :, Nz//2],
    X[:, :, 2], Y[:, :, 2], Z[:, :, 2],
    facecolors=colorize(data[:, :, 1]),
    shade=False,
    edgecolor='black',
    lw=0.2
)
C = ax.plot_surface(
    #X[:, :, Nz//2], Y[:, :, Nz//2], Z[:, :, Nz//2],
    X[:, :, 4], Y[:, :, 4], Z[:, :, 4],
    facecolors=colorize(data[:, :, 2]),
    shade=False,
    edgecolor='black',
    lw=0.2
)
C = ax.plot_surface(
    #X[:, :, Nz//2], Y[:, :, Nz//2], Z[:, :, Nz//2],
    X[:, :, 6], Y[:, :, 6], Z[:, :, 6],
    facecolors=colorize(data[:, :, 3]),
    shade=False,
    edgecolor='black',
    lw=0.2
)


# Set limits of the plot from coord limits
xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])


# Set labels and zticks
ax.set(
    xlabel='Sensing',
    xticks=[1,2,3,4,5],
    ylabel='Combat Difference',
    yticks=[-5,-4,-3,-2,-1,0,1,2,3,4,5],
    zlabel='Opp Strategy',
    zticks=[0,2,4,6],
    zticklabels=['attack', 'engage', 'defend', 'mixed'],
)
#ax.set_xlabel(xlabel='SEnsing',labelpad=20)
#ax.legend()

#CHANGES VIEWING
ax.view_init(elev=10, azim=30)

# Show Figure
plt.savefig('res00.png', bbox_inches="tight")