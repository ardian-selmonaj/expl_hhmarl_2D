import matplotlib.pyplot as plt
from matplotlib import cm
import random
import numpy as np
import json
import os

# Create a figure with 3D ax
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')


# Define dimensions: NX= sensing, NY=Excess, NZ=strategy
Nx, Ny, Nz = 6, 11, 7
X, Y, Z = np.meshgrid(np.arange(1,Nx), np.arange(-5, 6), np.arange(Nz))

# data = (((X+100)**2 + (Y-20)**2 + 2*Z)/1000+1)
# dmin = data.min()
# dptp = np.ptp(data)

#BASE DATA
fata = np.zeros((Ny, Nx, Nz))
#random.choices([0, 1], weights=[1, 3], k=1)[0]

### aggressive / mixed #0=esc(violet), 1=eng(grün), 2=fight(gelb)
# for x in range(Nx-1):
#     for y in range(Ny):
#         if x <= 1: fata[y,x,0] = random.choices([1, 2], weights=[1, 4], k=1)[0]
#         else:
#             if y<=3:
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[5, 1, 2], k=1)[0]
#             elif y in range(4,7):
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[1, 3, 1], k=1)[0]
#             else:
#                 fata[y,x,0] = random.choices([0, 1, 2], weights=[1, 2, 3], k=1)[0]

#engage
# for x in range(Nx-1):
#     for y in range(Ny):
#         if x <= 1: fata[y,x,2] = random.choices([1, 2], weights=[1, 4], k=1)[0]
#         else:
#             if y<=3:
#                 fata[y,x,2] = random.choices([0, 1, 2], weights=[3, 2, 1], k=1)[0]
#             elif y in range(4,7):
#                 fata[y,x,2] = random.choices([0, 1, 2], weights=[2, 3, 1], k=1)[0]
#             else:
#                 fata[y,x,2] = random.choices([0, 1, 2], weights=[1, 2, 3], k=1)[0]

#defensive
# for x in range(Nx-1):
#     for y in range(Ny):
#         if x <= 1: fata[y,x,4] = random.choices([1, 2], weights=[2, 4], k=1)[0]
#         else:
#             if y<=2:
#                 fata[y,x,4] = random.choices([0, 1, 2], weights=[1, 4, 1], k=1)[0]
#             elif y in range(3,7):
#                 fata[y,x,4] = random.choices([0, 1, 2], weights=[1, 3, 2], k=1)[0]
#             else:
#                 fata[y,x,4] = random.choices([0, 1, 2], weights=[1, 2, 4], k=1)[0]


#np.save('def.npy', fata)
#fata = np.load('aggr.npy')

agg_ = np.load(os.path.join(os.getcwd(), "patterns", "aggr", "aggr2.npy"))
def_ = np.load(os.path.join(os.getcwd(), "patterns", "def", "def3.npy"))
eng_ = np.load(os.path.join(os.getcwd(), "patterns", "eng", "eng.npy"))
mix_ = np.load(os.path.join(os.getcwd(), "patterns", "mixed", "mix2.npy"))

def_[2, 2, 4] = 1
#print("-------",  def_[2, 2, 4])
#assert None

data = np.stack((agg_[:, :,0], eng_[:, :, 2], def_[:, :, 4], mix_[:, :, 6]), axis=2)

#data = fata
dmin = data.min()
dptp = np.ptp(data)

def colorize(d, cmap=cm.viridis):
    shape = d.shape
    return cmap(((d-dmin)/(dptp+1e-10)).flatten()).reshape((*shape, 4))

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
plt.savefig('res00.png')