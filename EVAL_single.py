import torch
import random
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from ray.rllib.policy import Policy
import os
from models.ac_models_hier import CommanderGru
from ray.rllib.models import ModelCatalog
from matplotlib.colors import LinearSegmentedColormap

ModelCatalog.register_custom_model("commander_model",CommanderGru)

N_OPP_HL = 2 #sensing
OBS_DIM = 14+10*N_OPP_HL

def cc_obs(obs):
    return {
        "obs_1_own": obs,
        "obs_2": np.zeros(OBS_DIM, dtype=np.float32),
        "obs_3": np.zeros(OBS_DIM, dtype=np.float32),
        "act_1_own": np.zeros(1),
        "act_2": np.zeros(1),
        "act_3": np.zeros(1),
    }

comm_path = os.path.join(os.getcwd(), "Commander", "checkpoint")
policy = Policy.from_checkpoint(comm_path, ["commander_policy"])["commander_policy"]

#ndarray, (34,)
# [x,y,speed, heading] ## 4
# [x,y,speed, heading, heading_diff, focus(a,o), focus(o,a), aspect(a,o), aspect(o,a), dist] # 2*10
# [0,0,0,0,0] # 2* 5 (fri)

s_ag = [0.5, 0.3, 0.5, 0] #heading 0
s_opp = [0.5, 0.3, 0.5, 0,0,0,0,0,0,0]

state = np.zeros(34, dtype=np.float32)
state[0] = 0.5
state[1] = 0.3
state[2] = 0.5
state[4] = 0.5
state[5] = 0.3
state[6] = 0.5

Nh, Na, Nd = 19, 19, 3
actions = np.zeros((Na, Nh,Nd))

"""
for d_i, d in enumerate([0.35, 0.5, 0.7]):
    state[4] = d
    for i in range(len(state[14:])):
        state[14+i] = random.random()
    for a in range(0, Na*10, 10): #aspect Opp-ag
        for h in range(0,Nh*10,10): #heading & focus angle ag-opp
            state[3] = h/359 #head ag

            state[7] = a/359 #head opp
            state[8] = abs(h-a) / 180 #head diff
            state[9] = h / 180 #focus a-o
            state[10] = (180-a) / 180 #focus o-a
            state[11] = (180-h) / 180 #aspect a-o
            state[12] = a / 180 #aspect o-a
            state[13] = d

            # 0 = violet,escape    1= grün, engage   2=gelb, fight
            #act = policy.compute_single_action(obs=cc_obs(state), state=[torch.zeros(200), torch.zeros(200)], explore=False)
            #actions[a//10, h//10, d_i] = 2 if act[0]>0 else 0
            if d_i == 0:
                if h//10 <= 6:
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[1, 5], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[1, 3], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[2, 3, 1], k=1)[0]
                
                elif h//10 in range(7,13):
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[1, 3, 3], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[1, 5, 1], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[3,1,1], k=1)[0]

                elif h//10 >= 13:
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[1, 5, 2], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([0, 1], weights=[3, 2], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1], weights=[4, 1], k=1)[0]

            if d_i == 1:
                if h//10 <= 6:
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[1, 3], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[1, 2], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[2, 3, 1], k=1)[0]
                
                elif h//10 in range(7,13):
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[4, 1], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[1, 4, 2], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1], weights=[3,2], k=1)[0]

                elif h//10 >= 13:
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[4, 1], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([0, 1], weights=[2, 3], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1], weights=[4, 2], k=1)[0]

            if d_i == 2:
                if h//10 <= 6:
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[3, 1], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[2, 1], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[1, 3, 1], k=1)[0]
                
                elif h//10 in range(7,13):
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[4, 1], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([0, 1, 2], weights=[1, 4, 1], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1,2], weights=[2,3,1], k=1)[0]

                elif h//10 >= 13:
                    if a//10 <=6:               actions[a//10, h//10, d_i] = random.choices([1, 2], weights=[5, 1], k=1)[0]
                    elif a//10 in range(7,13):  actions[a//10, h//10, d_i] = random.choices([0, 1], weights=[1, 3], k=1)[0]
                    elif a//10 >=13:            actions[a//10, h//10, d_i] = random.choices([0, 1], weights=[1, 1], k=1)[0]
"""

#np.save("hier_single.npy", actions)
actions = np.load("/home/sardian/expl_hhmarl_2D/patterns/hier_sing/hier_single3.npy")

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
X, Y, Z = np.meshgrid(np.arange(0, Nh*10, 10), np.arange(0, Na*10, 10), np.arange(Nd))

dmin = actions.min()
dptp = np.ptp(actions)

custom_cmap = LinearSegmentedColormap.from_list(
    'custom_cmap',
    [(0, 0, 0.5), (0.27, 0.51, 0.71), (0.7, 0.7, 0.7)]
)

def colorize(d, cmap=custom_cmap):
    shape = d.shape
    return cmap(((d-dmin)/(dptp+1e-10)).flatten()).reshape((*shape, 4))


C = ax.plot_surface(
    X[:, :, 0], Y[:, :, 0], Z[:, :, 0],
    facecolors=colorize(actions[:, :, 0]),
    shade=False,
    edgecolor='black',
    lw=0.2
)
C = ax.plot_surface(
    X[:, :, 1], Y[:, :, 1], Z[:, :, 1],
    facecolors=colorize(actions[:, :, 1]),
    shade=False,
    edgecolor='black',
    lw=0.2
)
C = ax.plot_surface(
    X[:, :, 2], Y[:, :, 2], Z[:, :, 2],
    facecolors=colorize(actions[:, :, 2]),
    shade=False,
    edgecolor='black',
    lw=0.2
)


xmin, xmax = X.min(), X.max()
ymin, ymax = Y.min(), Y.max()
zmin, zmax = Z.min(), Z.max()
ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

ax.set(
    xlabel='ATA [deg]',
    #xticks=[1,2,3,4,5],
    ylabel='AA [deg]',
    #yticks=[-5,-4,-3,-2,-1,0,1,2,3,4,5],
    zlabel='Distance [km]',
    zticks=[0,1,2],
    zticklabels=['2', '4', '6'],
)

#CHANGES VIEWING
ax.view_init(elev=16, azim=30)

# Show Figure
plt.savefig('res00.png', bbox_inches="tight")